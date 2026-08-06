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

use crate::default_allowlist::default_allowlist_source;
use crate::domain_validation::{
    DomainStatus, domain_matches, validate_domain_pattern,
};
use crate::package_bundles::expand_package_bundle;
use crate::policy::NetworkPolicy;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, watch};
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
    use_default_allowlist: Arc<AtomicUsize>,
    enabled_bundles: Arc<RwLock<Vec<String>>>,
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
    use_default_allowlist: Arc<AtomicUsize>,
    enabled_bundles: Arc<RwLock<Vec<String>>>,
    bound_addr: Arc<RwLock<Option<SocketAddr>>>,
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
            use_default_allowlist: Arc::new(AtomicUsize::new(0)),
            enabled_bundles: Arc::new(RwLock::new(Vec::new())),
            bound_addr: Arc::new(RwLock::new(None)),
        })
    }

    /// The address the proxy is actually bound to, after [`NetworkProxy::start`]
    /// has run. `None` before the proxy is started.
    pub async fn bound_addr(&self) -> Option<SocketAddr> {
        *self.bound_addr.read().await
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

    /// Whether the built-in developer allowlist (github / search / docs) is
    /// also honoured during allowlist checks. Off by default so callers opt
    /// into the default posture explicitly.
    pub fn set_default_allowlist_enabled(&self, enabled: bool) {
        self.use_default_allowlist
            .store(usize::from(enabled), Ordering::SeqCst);
    }

    /// The package-manager bundle ids whose domains are also honoured during
    /// allowlist checks (see [`crate::package_bundles`]). Empty by default.
    pub async fn set_enabled_bundles(&self, bundle_ids: &[String]) {
        let mut bundles = self.enabled_bundles.write().await;
        bundles.clear();
        bundles.extend(bundle_ids.iter().cloned());
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
            NetworkMode::ProxyAllowlist => {
                self.allowlist_hits(domain).await
            }
        }
    }

    /// The default-allowlist + package-bundle + explicit allowlist check used
    /// by the proxy and by external callers.
    async fn allowlist_hits(&self, domain: &str) -> bool {
        if self.use_default_allowlist.load(Ordering::SeqCst) != 0
            && default_allowlist_source(domain).is_some()
        {
            return true;
        }
        let bundles = self.enabled_bundles.read().await;
        if !bundles.is_empty() {
            let d = domain.to_lowercase();
            for id in bundles.iter() {
                for bundled in expand_package_bundle(id) {
                    if domain_matches(&bundled, &d) {
                        return true;
                    }
                }
            }
        }
        matches_allowlist(&self.allowed_domains, domain)
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
        let bound = listener
            .local_addr()
            .map_err(|e| format!("local addr: {e}"))?;
        *self.bound_addr.write().await = Some(bound);

        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let ctx = Arc::new(ConnCtx {
            allowed_domains: self.allowed_domains.clone(),
            blocked_ranges: self.blocked_ranges.clone(),
            mode: self.mode.clone(),
            resolver: self.resolver.clone(),
            config: self.config.clone(),
            audit: self.audit.clone(),
            active: self.active.clone(),
            use_default_allowlist: self.use_default_allowlist.clone(),
            enabled_bundles: self.enabled_bundles.clone(),
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
    // Domain validation closes the SSRF-style gap: IP literals, non-FQDNs and
    // broad wildcards are never matched even if a policy listed them verbatim.
    let decision = validate_domain_pattern(domain);
    if decision.status != DomainStatus::Allowed {
        return false;
    }
    let d = decision.normalized;
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

async fn read_line(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
    buf: &mut Vec<u8>,
) -> Result<usize, String> {
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
            NetworkMode::ProxyAllowlist => {
                if self.use_default_allowlist.load(Ordering::SeqCst) != 0
                    && default_allowlist_source(host).is_some()
                {
                    return true;
                }
                let bundles = self.enabled_bundles.read().await;
                if !bundles.is_empty() {
                    let d = host.to_lowercase();
                    for id in bundles.iter() {
                        for bundled in expand_package_bundle(id) {
                            if domain_matches(&bundled, &d) {
                                return true;
                            }
                        }
                    }
                }
                matches_allowlist(&self.allowed_domains, host)
            }
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

/// A sliding-window rate limiter keyed by an arbitrary string (client address,
/// domain, user id, ...).
///
/// The limiter allows at most `limit` events per `window` duration per key.
/// Buckets are stored in a `DashMap` and lazily evicted once they fall out of
/// the window, so memory stays bounded by the number of active keys.
#[derive(Clone)]
pub struct RateLimiter {
    buckets: Arc<dashmap::DashMap<String, RateBucket>>,
    limit: usize,
    window: Duration,
}

#[derive(Debug, Clone)]
struct RateBucket {
    events: Vec<DateTime<Utc>>,
}

impl RateLimiter {
    /// Create a limiter allowing `limit` events per `window`.
    pub fn new(limit: usize, window: Duration) -> Self {
        Self {
            buckets: Arc::new(dashmap::DashMap::new()),
            limit,
            window,
        }
    }

    /// A limiter for the default proxy policy: 120 requests per minute per key.
    pub fn default_proxy() -> Self {
        Self::new(120, Duration::from_secs(60))
    }

    /// Allow a single request from `key`. Returns `true` when within the limit.
    pub fn allow(&self, key: &str) -> bool {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::from_std(self.window).unwrap_or_default();
        let mut bucket = self
            .buckets
            .entry(key.to_string())
            .or_insert(RateBucket { events: Vec::new() });
        bucket.events.retain(|t| *t > cutoff);
        if bucket.events.len() >= self.limit {
            return false;
        }
        bucket.events.push(now);
        true
    }

    /// Check whether `key` is currently rate-limited without consuming a slot.
    pub fn is_limited(&self, key: &str) -> bool {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::from_std(self.window).unwrap_or_default();
        let bucket = self.buckets.get(key);
        match bucket {
            Some(b) => b.events.iter().filter(|t| **t > cutoff).count() >= self.limit,
            None => false,
        }
    }

    /// Number of requests allowed to `key` in the current window.
    pub fn remaining(&self, key: &str) -> usize {
        let now = Utc::now();
        let cutoff = now - chrono::Duration::from_std(self.window).unwrap_or_default();
        let bucket = self.buckets.get(key);
        let count = match bucket {
            Some(b) => b.events.iter().filter(|t| **t > cutoff).count(),
            None => 0,
        };
        self.limit.saturating_sub(count)
    }

    /// Remove a key (frees its bucket).
    pub fn reset(&self, key: &str) {
        self.buckets.remove(key);
    }

    /// The configured per-window limit.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The configured window.
    pub fn window(&self) -> Duration {
        self.window
    }
}

/// A token bucket for outbound connection throttling.
///
/// Distinct from [`RateLimiter`] (which is per-key): this throttles the total
/// number of connections the proxy makes per unit time, e.g. 50 connect
/// attempts per second, smoothing bursts. Tokens refill continuously based on
/// elapsed time since the last access.
#[derive(Clone)]
pub struct TokenBucket {
    state: Arc<std::sync::Mutex<BucketState>>,
    capacity: f64,
    refill_per_sec: f64,
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last_refill: std::time::Instant,
}

impl TokenBucket {
    /// Create a bucket with `capacity` tokens, refilling `refill_per_sec`
    /// tokens per second.
    pub fn new(capacity: u64, refill_per_sec: u64) -> Self {
        Self {
            state: Arc::new(std::sync::Mutex::new(BucketState {
                tokens: capacity as f64,
                last_refill: std::time::Instant::now(),
            })),
            capacity: capacity as f64,
            refill_per_sec: refill_per_sec as f64,
        }
    }

    /// Try to consume one token, refilling first. Returns `true` if a token
    /// was available.
    pub fn try_consume(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        if elapsed > 0.0 {
            state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
            state.last_refill = now;
        }
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill the bucket to capacity.
    pub fn refill(&self) {
        let mut state = self.state.lock().unwrap();
        state.tokens = self.capacity;
        state.last_refill = std::time::Instant::now();
    }

    /// Add `n` tokens (capped at capacity).
    pub fn add(&self, n: u64) {
        let mut state = self.state.lock().unwrap();
        state.tokens = (state.tokens + n as f64).min(self.capacity);
        state.last_refill = std::time::Instant::now();
    }

    /// Remaining tokens (current instantaneous balance).
    pub fn available(&self) -> u64 {
        let state = self.state.lock().unwrap();
        state.tokens.floor() as u64
    }
}

/// The result of a domain allowlist check, including the matched rule and a
/// human-readable reason for audit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct DomainCheck {
    pub allowed: bool,
    pub domain: String,
    /// The allowlist rule that matched (for allow), or the reason for denial.
    pub detail: String,
}

/// A static domain allowlist with wildcard support.
#[derive(Debug, Clone, Default)]
pub struct DomainAllowlist {
    domains: Arc<dashmap::DashSet<String>>,
}

impl DomainAllowlist {
    /// Create an empty allowlist.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from an iterator of domains.
    pub fn from_domains<I: IntoIterator<Item = String>>(domains: I) -> Self {
        let set = Arc::new(dashmap::DashSet::new());
        for d in domains {
            set.insert(d.trim().to_lowercase());
        }
        Self { domains: set }
    }

    /// Add a domain.
    pub fn add(&self, domain: &str) {
        self.domains.insert(domain.trim().to_lowercase());
    }

    /// Check a domain against the allowlist.
    pub fn check(&self, domain: &str) -> DomainCheck {
        let d = domain.to_lowercase();
        if self.domains.is_empty() {
            return DomainCheck {
                allowed: false,
                domain: domain.to_string(),
                detail: "allowlist is empty; nothing is allowed".to_string(),
            };
        }
        if self.domains.contains(&d) {
            return DomainCheck {
                allowed: true,
                domain: domain.to_string(),
                detail: format!("exact match for '{d}'"),
            };
        }
        for allowed in self.domains.iter() {
            if let Some(suffix) = allowed.strip_prefix("*.") {
                if d == suffix || d.ends_with(&format!(".{suffix}")) {
                    return DomainCheck {
                        allowed: true,
                        domain: domain.to_string(),
                        detail: format!("wildcard match for '{}'", &*allowed),
                    };
                }
            }
        }
        // Fallback to the domain-validation matching rules, which normalize
        // trailing dots and reject malformed patterns (blocked patterns never
        // match, so this is strictly additive to the fast path above).
        for allowed in self.domains.iter() {
            if domain_matches(&allowed, domain) {
                return DomainCheck {
                    allowed: true,
                    domain: domain.to_string(),
                    detail: format!("validated match for '{}'", &*allowed),
                };
            }
        }
        DomainCheck {
            allowed: false,
            domain: domain.to_string(),
            detail: format!("'{d}' is not in the allowlist"),
        }
    }

    /// Number of domains in the allowlist.
    pub fn len(&self) -> usize {
        self.domains.len()
    }

    /// Is the allowlist empty?
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
    }
}

/// DNS resolution checks that complement the allowlist.
///
/// Verifies that a hostname resolves and that every resolved address passes
/// the SSRF guard (not in a blocked CIDR). This is used by the proxy before
/// opening an upstream connection.
pub struct DnsChecker {
    resolver: trust_dns_resolver::TokioAsyncResolver,
    blocked: Arc<RwLock<Vec<IpRange>>>,
}

impl DnsChecker {
    /// Create a DNS checker from the system resolver configuration.
    pub async fn new() -> Result<Self, String> {
        let resolver = trust_dns_resolver::TokioAsyncResolver::tokio_from_system_conf()
            .map_err(|e| format!("failed to create DNS resolver: {e}"))?;
        Ok(Self {
            resolver,
            blocked: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Set the blocked CIDR ranges.
    pub async fn set_blocked_ranges(&self, ranges: Vec<IpRange>) {
        *self.blocked.write().await = ranges;
    }

    /// Resolve a hostname to a list of addresses.
    pub async fn resolve(&self, host: &str) -> Result<Vec<IpAddr>, String> {
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

    /// Resolve a hostname and check every address against the blocked ranges.
    /// Returns `Ok(())` only if at least one address resolves and none are
    /// blocked. Returns the resolved addresses on success so the caller can
    /// connect to one.
    pub async fn resolve_and_check(&self, host: &str) -> Result<Vec<IpAddr>, String> {
        let ips = self.resolve(host).await?;
        if ips.is_empty() {
            return Err(format!("no addresses resolved for '{host}'"));
        }
        let ranges = self.blocked.read().await;
        for ip in &ips {
            if ranges.iter().any(|r| r.contains(ip)) {
                return Err(format!(
                    "resolved address {ip} for '{host}' is inside a blocked CIDR range"
                ));
            }
        }
        Ok(ips)
    }

    /// Check whether a specific IP is blocked.
    pub async fn is_ip_blocked(&self, ip: &IpAddr) -> bool {
        let ranges = self.blocked.read().await;
        ranges.iter().any(|r| r.contains(ip))
    }
}

/// A per-request audit logger that captures the full lifecycle of a proxied
/// request: received, allowed, connected, completed, with timing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProxyRequestLog {
    pub id: String,
    pub peer: String,
    pub method: String,
    pub host: String,
    pub port: u16,
    pub target_ip: Option<IpAddr>,
    pub allowed: bool,
    pub decision: String,
    pub started_at: DateTime<Utc>,
    pub connected_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub bytes_up: u64,
    pub bytes_down: u64,
    pub error: Option<String>,
}

/// A ring buffer of recent proxy request logs, bounded by capacity.
#[derive(Clone)]
pub struct RequestLogBuffer {
    logs: Arc<tokio::sync::RwLock<std::collections::VecDeque<ProxyRequestLog>>>,
    capacity: usize,
}

impl RequestLogBuffer {
    /// Create a buffer holding up to `capacity` recent logs.
    pub fn new(capacity: usize) -> Self {
        Self {
            logs: Arc::new(tokio::sync::RwLock::new(
                std::collections::VecDeque::with_capacity(capacity),
            )),
            capacity,
        }
    }

    /// Append a log, dropping the oldest when at capacity.
    pub async fn push(&self, log: ProxyRequestLog) {
        let mut logs = self.logs.write().await;
        if logs.len() >= self.capacity {
            logs.pop_front();
        }
        logs.push_back(log);
    }

    /// Snapshot all logs (oldest first).
    pub async fn snapshot(&self) -> Vec<ProxyRequestLog> {
        self.logs.read().await.iter().cloned().collect()
    }

    /// Recent logs matching a decision.
    pub async fn by_decision(&self, decision: &str) -> Vec<ProxyRequestLog> {
        self.logs
            .read()
            .await
            .iter()
            .filter(|l| l.decision == decision)
            .cloned()
            .collect()
    }

    /// Number of logs held.
    pub async fn len(&self) -> usize {
        self.logs.read().await.len()
    }

    /// Is the buffer empty?
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

/// Convenience constructor for a default SSRF-prevention blocked-range list
/// parsed from [`crate::policy::default_blocked_cidrs`].
pub fn default_blocked_ranges() -> Result<Vec<IpRange>, String> {
    crate::policy::default_blocked_cidrs()
        .iter()
        .map(|c| IpRange::parse(c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_blocks_after_limit() {
        let limiter = RateLimiter::new(3, Duration::from_secs(60));
        assert!(limiter.allow("client-1"));
        assert!(limiter.allow("client-1"));
        assert!(limiter.allow("client-1"));
        assert!(!limiter.allow("client-1"));
        assert_eq!(limiter.remaining("client-1"), 0);
        // A different key is not limited.
        assert!(limiter.allow("client-2"));
    }

    #[test]
    fn token_bucket_consumes() {
        let bucket = TokenBucket::new(2, 1);
        assert!(bucket.try_consume());
        assert!(bucket.try_consume());
        assert!(!bucket.try_consume());
        bucket.refill();
        assert!(bucket.try_consume());
    }

    #[test]
    fn domain_allowlist_wildcards() {
        let wl = DomainAllowlist::from_domains(vec![
            "example.com".to_string(),
            "*.sub.example.com".to_string(),
        ]);
        assert!(wl.check("example.com").allowed);
        assert!(wl.check("api.sub.example.com").allowed);
        assert!(!wl.check("badexample.com").allowed);
        assert!(!wl.check("other.org").allowed);
    }

    #[test]
    fn empty_allowlist_denies_all() {
        let wl = DomainAllowlist::new();
        assert!(!wl.check("example.com").allowed);
    }

    #[test]
    fn ip_range_contains() {
        let r = IpRange::parse("192.168.0.0/16").unwrap();
        assert!(r.contains(&"192.168.1.1".parse().unwrap()));
        assert!(!r.contains(&"10.0.0.1".parse().unwrap()));
    }

    #[test]
    fn default_blocked_ranges_parse() {
        let ranges = default_blocked_ranges().unwrap();
        assert!(ranges.iter().any(|r| r.cidr == "127.0.0.0/8"));
    }

    #[tokio::test]
    async fn request_log_buffer_rotates() {
        let buf = RequestLogBuffer::new(2);
        for i in 0..3 {
            buf.push(ProxyRequestLog {
                id: format!("r{i}"),
                peer: "127.0.0.1:1".to_string(),
                method: "CONNECT".to_string(),
                host: "example.com".to_string(),
                port: 443,
                target_ip: None,
                allowed: true,
                decision: "allowed".to_string(),
                started_at: Utc::now(),
                connected_at: None,
                completed_at: None,
                bytes_up: 0,
                bytes_down: 0,
                error: None,
            })
            .await;
        }
        assert_eq!(buf.len().await, 2);
        let logs = buf.snapshot().await;
        assert_eq!(logs[0].id, "r1");
        assert_eq!(logs[1].id, "r2");
    }

    #[test]
    fn matches_allowlist_rejects_invalid_hosts() {
        let set = dashmap::DashSet::new();
        set.insert("example.com".to_string());
        // Valid host matches.
        assert!(matches_allowlist(&set, "example.com"));
        // IP literals and non-FQDNs are blocked even when listed verbatim.
        set.insert("127.0.0.1".to_string());
        assert!(!matches_allowlist(&set, "127.0.0.1"));
        set.insert("localhost".to_string());
        assert!(!matches_allowlist(&set, "localhost"));
        // Wildcard patterns still match their suffix at query time (broad
        // wildcard rejection happens at allowlist ingestion, not matching).
        let wide = dashmap::DashSet::new();
        wide.insert("*.com".to_string());
        assert!(matches_allowlist(&wide, "anything.com"));
        assert!(!matches_allowlist(&wide, "anything.org"));
    }

    #[tokio::test]
    async fn default_allowlist_opt_in() {
        let proxy = NetworkProxy::new(NetworkConfig::default()).await.unwrap();
        proxy.apply_policy(&NetworkPolicy::ProxyAllowlist(vec![]), &[]).await;
        // Off by default: an empty allowlist denies github.com.
        assert!(!proxy.is_domain_allowed("github.com").await);
        // Opt in: the built-in developer allowlist is honoured.
        proxy.set_default_allowlist_enabled(true);
        assert!(proxy.is_domain_allowed("github.com").await);
        assert!(proxy.is_domain_allowed("docs.python.org").await);
        // Unrelated hosts are still denied.
        assert!(!proxy.is_domain_allowed("evil.example.com").await);
    }

    #[tokio::test]
    async fn package_bundles_opt_in() {
        let proxy = NetworkProxy::new(NetworkConfig::default()).await.unwrap();
        proxy.apply_policy(&NetworkPolicy::ProxyAllowlist(vec![]), &[]).await;
        assert!(!proxy.is_domain_allowed("pypi.org").await);
        proxy
            .set_enabled_bundles(&["python-package-install".to_string()])
            .await;
        assert!(proxy.is_domain_allowed("pypi.org").await);
        assert!(proxy.is_domain_allowed("files.pythonhosted.org").await);
        assert!(!proxy.is_domain_allowed("registry.npmjs.org").await);
    }

    #[tokio::test]
    async fn host_mode_ignores_allowlist_flags() {
        let proxy = NetworkProxy::new(NetworkConfig::default()).await.unwrap();
        proxy.apply_policy(&NetworkPolicy::Host, &[]).await;
        proxy.set_default_allowlist_enabled(true);
        proxy
            .set_enabled_bundles(&["python-package-install".to_string()])
            .await;
        assert!(proxy.is_domain_allowed("anything.example.com").await);
    }
}
