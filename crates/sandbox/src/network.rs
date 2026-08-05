//! Network proxy with domain allowlist and IP-range enforcement.
//!
//! Implements the `PROXY_ALLOWLIST` network mode: a local HTTP(S) proxy that
//! only forwards requests to allowlisted domains, resolving names through
//! `trust-dns-resolver` and blocking resolved IPs that fall inside configured
//! CIDR ranges. Supports plain HTTP forwarding and HTTPS `CONNECT` tunneling,
//! both streamed bidirectionally (no whole-body buffering).
//!
//! The three network modes are:
//! - `NONE` — no network access at all (the proxy rejects everything).
//! - `PROXY_ALLOWLIST` — only allowlisted domains, enforced by this proxy.
//! - `HOST` — direct host networking (the proxy is not used).

use crate::policy::NetworkPolicy;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, RwLock};
use tracing::{debug, error, info, warn};
use trust_dns_resolver::TokioAsyncResolver;

/// The three supported network isolation modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkMode {
    /// No network access.
    None,
    /// Only allowlisted domains via the local proxy.
    ProxyAllowlist,
    /// Direct host network access.
    Host,
}

/// Map a policy network mode to the proxy's runtime mode.
pub fn mode_from_policy(policy: &NetworkPolicy) -> NetworkMode {
    match policy {
        NetworkPolicy::None => NetworkMode::None,
        NetworkPolicy::ProxyAllowlist(_) => NetworkMode::ProxyAllowlist,
        NetworkPolicy::Host => NetworkMode::Host,
    }
}

/// A single CIDR block that the proxy refuses to forward to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IpRange {
    /// The original CIDR string, e.g. `"192.168.0.0/16"`.
    pub cidr: String,
    /// The network address (masked).
    pub network: IpAddr,
    /// Prefix length in bits.
    pub prefix: u8,
}

impl IpRange {
    /// Parse a CIDR string such as `"10.0.0.0/8"` or `"fe80::/10"`.
    pub fn parse(cidr: &str) -> Result<IpRange, String> {
        let (addr_str, prefix_str) = cidr
            .split_once('/')
            .ok_or_else(|| format!("invalid CIDR (expected 'ip/prefix'): {cidr}"))?;
        let network: IpAddr = addr_str
            .parse()
            .map_err(|_| format!("invalid IP address in CIDR: {addr_str}"))?;
        let prefix: u8 = prefix_str
            .parse()
            .map_err(|_| format!("invalid prefix length: {prefix_str}"))?;
        let max = match network {
            IpAddr::V4(_) => 32,
            IpAddr::V6(_) => 128,
        };
        if prefix > max {
            return Err(format!("prefix {prefix} exceeds {max} for {addr_str}"));
        }
        Ok(IpRange {
            cidr: cidr.to_string(),
            network,
            prefix,
        })
    }

    /// Does this range contain the given IP?
    pub fn contains(&self, ip: &IpAddr) -> bool {
        match (&self.network, ip) {
            (IpAddr::V4(network), IpAddr::V4(ip)) => {
                let mask = if self.prefix == 0 {
                    0u32
                } else {
                    u32::MAX << (32 - self.prefix as u32)
                };
                (u32::from(*network) & mask) == (u32::from(*ip) & mask)
            }
            (IpAddr::V6(network), IpAddr::V6(ip)) => {
                let mask = if self.prefix == 0 {
                    0u128
                } else {
                    u128::MAX << (128 - self.prefix as u32)
                };
                (u128::from(*network) & mask) == (u128::from(*ip) & mask)
            }
            _ => false,
        }
    }
}

/// Configuration for the proxy server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Local bind address. Use port `0` to let the OS pick a free port.
    pub bind_addr: SocketAddr,
    /// Timeout for connecting to the upstream target.
    pub connect_timeout_ms: u64,
    /// Maximum lifetime of a proxied connection (guards against hangs).
    pub session_timeout_ms: u64,
    /// Maximum number of concurrently active proxied connections.
    pub max_concurrent: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:0".parse().expect("valid socket address"),
            connect_timeout_ms: 5_000,
            session_timeout_ms: 60_000,
            max_concurrent: 64,
        }
    }
}

/// A single audited proxy decision.
#[derive(Debug, Clone, Serialize)]
pub struct ProxyAuditEntry {
    pub timestamp: DateTime<Utc>,
    pub method: String,
    pub host: String,
    pub target_ip: Option<IpAddr>,
    /// `allowed`, `blocked_domain`, `blocked_ip`, `dns_failed`, `connect_failed`, `timeout`, `rejected`
    pub decision: String,
}

/// Handle used to stop a running proxy.
pub struct ProxyHandle {
    shutdown_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl ProxyHandle {
    /// Signal shutdown and await the accept loop.
    pub async fn stop(self) -> Result<(), String> {
        let _ = self.shutdown_tx.send(true);
        self.task
            .await
            .map_err(|e| format!("network proxy task error: {e}"))
    }

    /// Abort the accept loop without graceful shutdown.
    pub fn abort(self) {
        self.task.abort();
    }
}

/// Shared per-connection context cloned into every handler task.
struct ConnCtx {
    allowed_domains: Arc<dashmap::DashSet<String>>,
    blocked_ranges: Arc<RwLock<Vec<IpRange>>>,
    mode: Arc<RwLock<NetworkMode>>,
    resolver: TokioAsyncResolver,
    config: NetworkConfig,
    audit: Arc<DashMap<String, Vec<ProxyAuditEntry>>>,
    active: Arc<AtomicUsize>,
}

/// Network proxy for sandboxed processes with domain allowlist enforcement.
#[derive(Clone)]
pub struct NetworkProxy {
    config: NetworkConfig,
    allowed_domains: Arc<dashmap::DashSet<String>>,
    blocked_ranges: Arc<RwLock<Vec<IpRange>>>,
    mode: Arc<RwLock<NetworkMode>>,
    resolver: TokioAsyncResolver,
    audit: Arc<DashMap<String, Vec<ProxyAuditEntry>>>,
    active: Arc<AtomicUsize>,
}

impl NetworkProxy {
    /// Create a new network proxy bound to the given address.
    pub async fn new(config: NetworkConfig) -> Result<Self, String> {
        let resolver = TokioAsyncResolver::tokio_from_system_conf()
            .map_err(|e| format!("failed to create DNS resolver: {e}"))?;
        Ok(Self {
            config,
            allowed_domains: Arc::new(dashmap::DashSet::new()),
            blocked_ranges: Arc::new(RwLock::new(Vec::new())),
            mode: Arc::new(RwLock::new(NetworkMode::ProxyAllowlist)),
            resolver,
            audit: Arc::new(DashMap::new()),
            active: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Set the allowed domains from a network policy.
    pub fn set_allowed_domains(&self, policy: &NetworkPolicy) {
        self.allowed_domains.clear();
        if let NetworkPolicy::ProxyAllowlist(domains) = policy {
            for domain in domains {
                self.allowed_domains.insert(domain.trim().to_lowercase());
            }
        }
    }

    /// Configure the proxy from a policy and a set of blocked CIDR ranges.
    pub async fn apply_policy(&self, policy: &NetworkPolicy, blocked_ranges: &[IpRange]) {
        self.set_allowed_domains(policy);
        *self.blocked_ranges.write().await = blocked_ranges.to_vec();
        *self.mode.write().await = mode_from_policy(policy);
    }

    /// Check whether a domain is allowed under the current mode.
    pub async fn is_domain_allowed(&self, domain: &str) -> bool {
        match *self.mode.read().await {
            NetworkMode::Host => true,
            NetworkMode::None => false,
            NetworkMode::ProxyAllowlist => matches_allowlist(&self.allowed_domains, domain),
        }
    }

    /// Check whether an IP falls inside a configured blocked range.
    pub async fn is_ip_blocked(&self, ip: &IpAddr) -> bool {
        let ranges = self.blocked_ranges.read().await;
        ranges.iter().any(|r| r.contains(ip))
    }

    /// Start the proxy server, returning a handle that can stop it.
    pub async fn start(&self) -> Result<ProxyHandle, String> {
        let listener = TcpListener::bind(self.config.bind_addr)
            .await
            .map_err(|e| format!("failed to bind proxy: {e}"))?;
        let bound = listener.local_addr().map_err(|e| format!("local addr: {e}"))?;

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let ctx = Arc::new(ConnCtx {
            allowed_domains: self.allowed_domains.clone(),
            blocked_ranges: self.blocked_ranges.clone(),
            mode: self.mode.clone(),
            resolver: self.resolver.clone(),
            config: self.config.clone(),
            audit: self.audit.clone(),
            active: self.active.clone(),
        });

        info!("network proxy listening on {bound}");
        let task = tokio::spawn(accept_loop(listener, shutdown_rx, ctx));
        Ok(ProxyHandle { shutdown_tx, task })
    }

    /// Snapshot of all audit entries, newest last.
    pub fn audit_entries(&self) -> Vec<ProxyAuditEntry> {
        let mut out = Vec::new();
        for entries in self.audit.iter() {
            out.extend(entries.value().iter().cloned());
        }
        out.sort_by_key(|e| e.timestamp);
        out
    }

    /// Number of concurrently active proxied connections.
    pub fn active_connections(&self) -> usize {
        self.active.load(Ordering::SeqCst)
    }
}

async fn accept_loop(
    listener: TcpListener,
    mut shutdown_rx: watch::Receiver<bool>,
    ctx: Arc<ConnCtx>,
) {
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, peer)) => {
                        let ctx = ctx.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, peer, ctx).await {
                                debug!("proxy connection handler: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        error!("proxy accept error: {e}");
                    }
                }
            }
        }
    }
    info!("network proxy stopped");
}

async fn handle_connection(
    stream: TcpStream,
    peer: SocketAddr,
    ctx: Arc<ConnCtx>,
) -> Result<(), String> {
    let active = ctx.active.fetch_add(1, Ordering::SeqCst) + 1;
    let _guard = ActiveGuard {
        active: ctx.active.clone(),
    };
    if active > ctx.config.max_concurrent {
        warn!(
            %peer,
            active, max = ctx.config.max_concurrent,
            "proxy connection rejected: too many active connections"
        );
        ctx.audit
            .entry(peer.to_string())
            .or_default()
            .push(ProxyAuditEntry {
                timestamp: Utc::now(),
                method: "CONNECT".to_string(),
                host: peer.to_string(),
                target_ip: None,
                decision: "rejected".to_string(),
            });
        return Ok(());
    }

    let session_timeout = Duration::from_millis(ctx.config.session_timeout_ms);
    match tokio::time::timeout(session_timeout, handle_stream(stream, peer, ctx.clone())).await {
        Ok(result) => result,
        Err(_) => {
            ctx.audit
                .entry(peer.to_string())
                .or_default()
                .push(ProxyAuditEntry {
                    timestamp: Utc::now(),
                    method: "session".to_string(),
                    host: peer.to_string(),
                    target_ip: None,
                    decision: "timeout".to_string(),
                });
            Ok(())
        }
    }
}

struct ActiveGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn handle_stream(
    stream: TcpStream,
    peer: SocketAddr,
    ctx: Arc<ConnCtx>,
) -> Result<(), String> {
    let (mut reader, mut writer) = stream.into_split();

    let mut request_line = Vec::new();
    let n = read_line(&mut reader, &mut request_line).await?;
    if n == 0 {
        return Ok(());
    }
    let line = String::from_utf8_lossy(&request_line).trim().to_string();
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        write_error(&mut writer, "400 Bad Request", "Malformed request line").await?;
        return Ok(());
    }
    let method = parts[0].to_ascii_uppercase();
    let target = parts[1].to_string();

    // Read headers until the blank line.
    let mut headers: Vec<Vec<u8>> = Vec::new();
    loop {
        let mut h = Vec::new();
        let n = read_line(&mut reader, &mut h).await?;
        if n == 0 {
            break;
        }
        if h == b"\r\n" || h == b"\n" {
            break;
        }
        headers.push(h);
    }

    let (host, port) = if method == "CONNECT" {
        parse_authority(&target)?
    } else {
        parse_http_target(&target)?
    };

    if !ctx.is_allowed(&host).await {
        warn!(%host, %peer, "proxy blocked domain");
        ctx.audit
            .entry(host.clone())
            .or_default()
            .push(ProxyAuditEntry {
                timestamp: Utc::now(),
                method: method.clone(),
                host: host.clone(),
                target_ip: None,
                decision: "blocked_domain".to_string(),
            });
        write_error(&mut writer, "403 Forbidden", "Domain not allowed").await?;
        return Ok(());
    }

    let ips = match ctx.resolve(&host).await {
        Ok(ips) => ips,
        Err(e) => {
            warn!(%host, %peer, error = %e, "proxy DNS resolution failed");
            ctx.audit
                .entry(host.clone())
                .or_default()
                .push(ProxyAuditEntry {
                    timestamp: Utc::now(),
                    method: method.clone(),
                    host: host.clone(),
                    target_ip: None,
                    decision: "dns_failed".to_string(),
                });
            write_error(&mut writer, "502 Bad Gateway", "DNS resolution failed").await?;
            return Ok(());
        }
    };
    if ips.is_empty() {
        ctx.audit
            .entry(host.clone())
            .or_default()
            .push(ProxyAuditEntry {
                timestamp: Utc::now(),
                method: method.clone(),
                host: host.clone(),
                target_ip: None,
                decision: "dns_failed".to_string(),
            });
        write_error(&mut writer, "502 Bad Gateway", "No addresses resolved").await?;
        return Ok(());
    }

    if let Some(blocked) = ctx.first_blocked(&ips).await {
        warn!(%host, ip = %blocked, "proxy blocked IP range");
        ctx.audit
            .entry(host.clone())
            .or_default()
            .push(ProxyAuditEntry {
                timestamp: Utc::now(),
                method: method.clone(),
                host: host.clone(),
                target_ip: Some(blocked),
                decision: "blocked_ip".to_string(),
            });
        write_error(&mut writer, "403 Forbidden", "IP address blocked").await?;
        return Ok(());
    }

    let addr = SocketAddr::new(ips[0], port);
    let connect_timeout = Duration::from_millis(ctx.config.connect_timeout_ms);
    let mut target = match tokio::time::timeout(connect_timeout, TcpStream::connect(addr)).await {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            ctx.audit
                .entry(host.clone())
                .or_default()
                .push(ProxyAuditEntry {
                    timestamp: Utc::now(),
                    method: method.clone(),
                    host: host.clone(),
                    target_ip: Some(addr.ip()),
                    decision: "connect_failed".to_string(),
                });
            write_error(
                &mut writer,
                "502 Bad Gateway",
                &format!("Connection failed: {e}"),
            )
            .await?;
            return Ok(());
        }
        Err(_) => {
            write_error(&mut writer, "504 Gateway Timeout", "Connection timed out").await?;
            return Ok(());
        }
    };

    ctx.audit
        .entry(host.clone())
        .or_default()
        .push(ProxyAuditEntry {
            timestamp: Utc::now(),
            method: method.clone(),
            host: host.clone(),
            target_ip: Some(addr.ip()),
            decision: "allowed".to_string(),
        });

    if method == "CONNECT" {
        writer
            .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            .await
            .map_err(|e| format!("write CONNECT response: {e}"))?;
        writer
            .flush()
            .await
            .map_err(|e| format!("flush CONNECT response: {e}"))?;
        tunnel(&mut reader, &mut writer, &mut target).await?;
    } else {
        target
            .write_all(&request_line)
            .await
            .map_err(|e| format!("forward request line: {e}"))?;
        for h in &headers {
            target
                .write_all(h)
                .await
                .map_err(|e| format!("forward header: {e}"))?;
        }
        target
            .write_all(b"\r\n")
            .await
            .map_err(|e| format!("forward header terminator: {e}"))?;
        target
            .flush()
            .await
            .map_err(|e| format!("flush forwarded request: {e}"))?;
        tunnel(&mut reader, &mut writer, &mut target).await?;
    }
    Ok(())
}

/// Bidirectionally stream bytes between the client and the target until one
/// side closes. No buffering, safe for arbitrary payloads.
async fn tunnel(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    target: &mut TcpStream,
) -> Result<(), String> {
    let (mut tr, mut tw) = target.split();
    let client_to_target = tokio::io::copy(reader, &mut tw);
    let target_to_client = tokio::io::copy(&mut tr, writer);
    let _ = tokio::join!(client_to_target, target_to_client);
    Ok(())
}

fn matches_allowlist(domains: &dashmap::DashSet<String>, domain: &str) -> bool {
    if domains.is_empty() {
        return false;
    }
    let d = domain.to_lowercase();
    if domains.contains(&d) {
        return true;
    }
    for allowed in domains.iter() {
        if let Some(suffix) = allowed.strip_prefix("*.") {
            // `*.example.com` matches `example.com` and `sub.example.com` but
            // not `badexample.com`.
            if d == suffix || d.ends_with(&format!(".{suffix}")) {
                return true;
            }
        }
    }
    false
}

async fn read_line(reader: &mut tokio::net::tcp::OwnedReadHalf, buf: &mut Vec<u8>) -> Result<usize, String> {
    buf.clear();
    loop {
        let mut byte = [0u8; 1];
        let n = reader
            .read(&mut byte)
            .await
            .map_err(|e| format!("read: {e}"))?;
        if n == 0 {
            return Ok(0);
        }
        buf.push(byte[0]);
        if byte[0] == b'\n' {
            return Ok(buf.len());
        }
        if buf.len() >= 64 * 1024 {
            return Err("header line too long".to_string());
        }
    }
}

async fn write_error(
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
    status: &str,
    message: &str,
) -> Result<(), String> {
    let body = format!("{message}\r\n");
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    writer
        .write_all(response.as_bytes())
        .await
        .map_err(|e| format!("write error response: {e}"))?;
    let _ = writer.shutdown().await;
    Ok(())
}

impl ConnCtx {
    async fn is_allowed(&self, host: &str) -> bool {
        match *self.mode.read().await {
            NetworkMode::Host => true,
            NetworkMode::None => false,
            NetworkMode::ProxyAllowlist => matches_allowlist(&self.allowed_domains, host),
        }
    }

    async fn resolve(&self, host: &str) -> Result<Vec<IpAddr>, String> {
        if let Ok(ip) = host.parse::<IpAddr>() {
            return Ok(vec![ip]);
        }
        let response = self
            .resolver
            .lookup_ip(host)
            .await
            .map_err(|e| format!("DNS resolution failed: {e}"))?;
        Ok(response.iter().collect())
    }

    async fn first_blocked(&self, ips: &[IpAddr]) -> Option<IpAddr> {
        let ranges = self.blocked_ranges.read().await;
        for ip in ips {
            if ranges.iter().any(|r| r.contains(ip)) {
                return Some(*ip);
            }
        }
        None
    }
}

fn parse_http_target(target: &str) -> Result<(String, u16), String> {
    let without_scheme = target
        .strip_prefix("http://")
        .or_else(|| target.strip_prefix("https://"))
        .unwrap_or(target);
    let authority = without_scheme.split('/').next().unwrap_or("");
    split_host_port(authority, 80)
}

fn parse_authority(target: &str) -> Result<(String, u16), String> {
    split_host_port(target, 443)
}

fn split_host_port(s: &str, default_port: u16) -> Result<(String, u16), String> {
    if let Some(rest) = s.strip_prefix('[') {
        let end = rest
            .find(']')
            .ok_or_else(|| format!("unterminated IPv6 address: {s}"))?;
        let host = format!("[{}]", &rest[..end]);
        let after = &rest[end + 1..];
        if let Some(port_str) = after.strip_prefix(':') {
            let port: u16 = port_str
                .parse()
                .map_err(|_| format!("invalid port: {port_str}"))?;
            Ok((host, port))
        } else {
            Ok((host, default_port))
        }
    } else if let Some((host, port_str)) = s.rsplit_once(':') {
        if let Ok(port) = port_str.parse::<u16>() {
            return Ok((host.to_string(), port));
        }
        Ok((s.to_string(), default_port))
    } else {
        Ok((s.to_string(), default_port))
    }
}
