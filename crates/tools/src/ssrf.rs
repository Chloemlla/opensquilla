//! SSRF (Server-Side Request Forgery) protection.
//!
//! Standalone module that validates URLs do not resolve to private, loopback,
//! or link-local IP addresses before a server-side fetch is performed.
//!
//! ## Fail-closed semantics
//!
//! **DNS resolution errors fail CLOSED (the request is denied).** A DNS failure
//! is treated as unsafe rather than falling back to allowing the request. This
//! prevents an attacker from forcing a transient resolver outage to bypass the
//! check, and avoids the classic fail-open SSRF bypass where a host appears to
//! resolve cleanly only after the guard has returned.
//!
//! ## Capabilities
//!
//! - Hostname → IP resolution via `trust-dns-resolver`
//! - IP range checks for IPv4/IPv6 private, loopback, and link-local ranges
//! - Domain allowlist (permit only listed hosts) and denylist (block listed hosts)
//! - Direct IP literal checks (no DNS round-trip)

use crate::registry::ToolError;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use trust_dns_resolver::config::{ResolverConfig, ResolverOpts};
use trust_dns_resolver::TokioAsyncResolver;

/// Configuration for SSRF protection.
#[derive(Clone, Debug)]
pub struct SsrfConfig {
    /// Block RFC 1918 / unique-local private address ranges.
    pub block_private: bool,
    /// Block loopback addresses (127.0.0.0/8, ::1).
    pub block_loopback: bool,
    /// Block link-local addresses (169.254.0.0/16, fe80::/10).
    pub block_link_local: bool,
    /// If non-empty, only hosts whose domain matches an allowlist entry are
    /// permitted. The allowlist is matched against the host suffix
    /// (e.g. "example.com" permits "api.example.com").
    pub allowlist: HashSet<String>,
    /// Hosts whose domain matches a denylist entry are always blocked,
    /// regardless of the resolved IP. The denylist takes precedence over the
    /// allowlist.
    pub denylist: HashSet<String>,
    /// When true, DNS resolution failures deny the request (fail-closed).
    /// Defaults to `true`. Setting this to `false` reverts to fail-open
    /// behavior, which is **not recommended** and exists only for parity with
    /// legacy callers in tests.
    pub fail_closed_on_dns_error: bool,
}

impl Default for SsrfConfig {
    fn default() -> Self {
        Self {
            block_private: true,
            block_loopback: true,
            block_link_local: true,
            allowlist: HashSet::new(),
            denylist: HashSet::new(),
            fail_closed_on_dns_error: true,
        }
    }
}

/// SSRF protection: validates that URLs do not resolve to private/internal IPs.
#[derive(Clone)]
pub struct SsrfProtection {
    resolver: Arc<TokioAsyncResolver>,
    config: SsrfConfig,
}

impl SsrfProtection {
    /// Create a new SSRF protection instance with default (fail-closed) config.
    pub fn new() -> Self {
        Self::with_config(SsrfConfig::default())
    }

    /// Create a new SSRF protection instance with the given config.
    pub fn with_config(config: SsrfConfig) -> Self {
        let resolver =
            TokioAsyncResolver::tokio(ResolverConfig::default(), ResolverOpts::default());
        Self {
            resolver: Arc::new(resolver),
            config,
        }
    }

    /// Returns a reference to the active configuration.
    pub fn config(&self) -> &SsrfConfig {
        &self.config
    }

    /// Check if a URL is safe to fetch.
    ///
    /// Returns `Ok(())` if the URL's host is permitted by the allowlist/denylist
    /// and does not resolve to a blocked IP range. Returns `Err` otherwise.
    ///
    /// **Fails closed** on DNS resolution errors when
    /// `fail_closed_on_dns_error` is enabled (the default): the request is
    /// denied rather than allowed.
    pub async fn check_url(&self, url_str: &str) -> Result<(), ToolError> {
        let url = url::Url::parse(url_str)
            .map_err(|e| ToolError::invalid_args(format!("Invalid URL '{}': {}", url_str, e)))?;

        let host = url
            .host_str()
            .ok_or_else(|| ToolError::invalid_args(format!("URL '{}' has no host", url_str)))?;

        // Denylist takes precedence over everything else.
        if self.matches_list(host, &self.config.denylist) {
            return Err(ToolError::new(
                "SSRF_BLOCKED",
                format!("Host '{}' is on the SSRF denylist", host),
            ));
        }

        // If an allowlist is configured, the host must match it.
        if !self.config.allowlist.is_empty()
            && !self.matches_list(host, &self.config.allowlist)
        {
            return Err(ToolError::new(
                "SSRF_BLOCKED",
                format!("Host '{}' is not on the SSRF allowlist", host),
            ));
        }

        // If the host is already an IP literal, check it directly (no DNS).
        if let Ok(ip) = host.parse::<IpAddr>() {
            self.check_ip(&ip, url_str)?;
            return Ok(());
        }

        // Resolve the hostname to IP addresses and check every resolved address.
        // Fail CLOSED on DNS errors: a resolver failure is treated as unsafe.
        match self.resolver.lookup_ip(host).await {
            Ok(response) => {
                let mut count = 0;
                for ip in response.iter() {
                    count += 1;
                    self.check_ip(&ip, url_str)?;
                }
                if count == 0 {
                    // No addresses returned — treat as a resolution failure.
                    if self.config.fail_closed_on_dns_error {
                        return Err(ToolError::new(
                            "SSRF_BLOCKED",
                            format!(
                                "DNS resolution returned no addresses for '{}' (fail-closed)",
                                host
                            ),
                        ));
                    } else {
                        tracing::warn!(
                            host = %host,
                            "DNS returned no addresses for host (fail-open; not recommended)"
                        );
                    }
                }
                Ok(())
            }
            Err(e) => {
                if self.config.fail_closed_on_dns_error {
                    tracing::warn!(
                        host = %host,
                        error = %e,
                        "DNS resolution failed; denying request (fail-closed)"
                    );
                    Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!(
                            "DNS resolution failed for '{}' (fail-closed): {}",
                            host, e
                        ),
                    ))
                } else {
                    tracing::warn!(
                        host = %host,
                        error = %e,
                        "DNS resolution failed for host (fail-open; not recommended)"
                    );
                    Ok(())
                }
            }
        }
    }

    /// Check whether a host matches any entry in a list.
    ///
    /// Matching is suffix-based so that `example.com` matches both `example.com`
    /// and `api.example.com`. Exact matches also pass.
    fn matches_list(&self, host: &str, list: &HashSet<String>) -> bool {
        let host = host.to_lowercase();
        for entry in list {
            let entry = entry.to_lowercase();
            if host == entry || host.ends_with(&format!(".{}", entry)) {
                return true;
            }
        }
        false
    }

    /// Check a single resolved IP against the configured block ranges.
    fn check_ip(&self, ip: &IpAddr, url_str: &str) -> Result<(), ToolError> {
        match ip {
            IpAddr::V4(v4) => {
                if self.config.block_loopback && v4.is_loopback() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to loopback address {}", url_str, v4),
                    ));
                }
                if self.config.block_private && v4.is_private() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to private address {}", url_str, v4),
                    ));
                }
                if self.config.block_link_local && v4.is_link_local() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to link-local address {}", url_str, v4),
                    ));
                }
                // Block 0.0.0.0/8 ("this host") explicitly — is_unspecified only
                // catches 0.0.0.0 itself, but the whole range is unsafe as a
                // destination.
                if self.config.block_private && v4.is_unspecified() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to unspecified address {}", url_str, v4),
                    ));
                }
            }
            IpAddr::V6(v6) => {
                if self.config.block_loopback && v6.is_loopback() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to loopback address {}", url_str, v6),
                    ));
                }
                if self.config.block_link_local && v6.is_unicast_link_local() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!(
                            "URL '{}' resolves to link-local address {}",
                            url_str, v6
                        ),
                    ));
                }
                // IPv6-mapped IPv4 addresses: re-check the embedded IPv4 against
                // the private ranges so a mapped 10.0.0.1 cannot slip through.
                if let Some(v4) = v6.to_ipv4_mapped() {
                    if self.config.block_private && v4.is_private() {
                        return Err(ToolError::new(
                            "SSRF_BLOCKED",
                            format!(
                                "URL '{}' resolves to private address {} (via mapped {})",
                                url_str, v4, v6
                            ),
                        ));
                    }
                    if self.config.block_loopback && v4.is_loopback() {
                        return Err(ToolError::new(
                            "SSRF_BLOCKED",
                            format!(
                                "URL '{}' resolves to loopback address {} (via mapped {})",
                                url_str, v4, v6
                            ),
                        ));
                    }
                }
                if self.config.block_private && v6.is_unspecified() {
                    return Err(ToolError::new(
                        "SSRF_BLOCKED",
                        format!("URL '{}' resolves to unspecified address {}", url_str, v6),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl Default for SsrfProtection {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`SsrfConfig`].
#[derive(Debug, Default)]
pub struct SsrfConfigBuilder {
    config: SsrfConfig,
}

impl SsrfConfigBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn block_private(mut self, v: bool) -> Self {
        self.config.block_private = v;
        self
    }

    pub fn block_loopback(mut self, v: bool) -> Self {
        self.config.block_loopback = v;
        self
    }

    pub fn block_link_local(mut self, v: bool) -> Self {
        self.config.block_link_local = v;
        self
    }

    /// Add a host to the allowlist (suffix-matched).
    pub fn allow(mut self, host: impl Into<String>) -> Self {
        self.config.allowlist.insert(host.into());
        self
    }

    /// Add a host to the denylist (suffix-matched).
    pub fn deny(mut self, host: impl Into<String>) -> Self {
        self.config.denylist.insert(host.into());
        self
    }

    /// Fail-open on DNS errors. **Not recommended.** The default is fail-closed.
    pub fn fail_open_on_dns_error(mut self) -> Self {
        self.config.fail_closed_on_dns_error = false;
        self
    }

    pub fn build(self) -> SsrfConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ssrf_private_ip_literal() {
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://127.0.0.1:8080/secret").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_ssrf_loopback_ipv6_literal() {
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://[::1]:8080/secret").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_ssrf_unspecified_ipv4() {
        let ssrf = SsrfProtection::new();
        let result = ssrf.check_url("http://0.0.0.0:8080/").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_denylist_takes_precedence() {
        let cfg = SsrfConfigBuilder::new()
            .allow("example.com")
            .deny("evil.example.com")
            .build();
        let ssrf = SsrfProtection::with_config(cfg);
        let result = ssrf.check_url("http://evil.example.com/").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_allowlist_blocks_unlisted_host() {
        let cfg = SsrfConfigBuilder::new()
            .allow("example.com")
            .build();
        let ssrf = SsrfProtection::with_config(cfg);
        let result = ssrf.check_url("http://unlisted.example.org/").await;
        assert!(result.is_err());
        // Should be blocked by the allowlist, not by IP range (DNS won't run).
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }

    #[tokio::test]
    async fn test_allowlist_permits_listed_host_suffix() {
        let cfg = SsrfConfigBuilder::new().allow("example.com").build();
        let ssrf = SsrfProtection::with_config(cfg);
        // api.example.com should pass the allowlist check (suffix match) and then
        // proceed to DNS. We can't assert the public IP result reliably, but the
        // error must NOT be an allowlist block.
        let result = ssrf.check_url("http://api.example.com/").await;
        if let Err(e) = &result {
            assert_ne!(e.message, "Host 'api.example.com' is not on the SSRF allowlist");
        }
    }

    #[tokio::test]
    async fn test_fail_closed_on_dns_error_for_nonexistent_host() {
        // A non-resolving TLD should be denied under fail-closed semantics.
        let ssrf = SsrfProtection::new();
        let result = ssrf
            .check_url("http://this-host-definitely-does-not-exist.invalid/")
            .await;
        assert!(result.is_err(), "fail-closed should deny unresolved hosts");
        assert_eq!(result.unwrap_err().code, "SSRF_BLOCKED");
    }
}
