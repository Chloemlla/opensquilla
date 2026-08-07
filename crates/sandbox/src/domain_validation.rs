//! Domain normalization and safety checks for sandbox managed network.
//!
//! Port of `src/opensquilla/sandbox/domain_validation.py`. Provides the
//! validation boundary for allowlist entries: a pattern must be a well-formed
//! DNS name (not an IP literal, not a broad wildcard) before it is accepted
//! into the managed-network allowlist. This closes the SSRF-style gap where a
//! caller could smuggle `127.0.0.1`, `*.com` or a non-FQDN into the allowlist.
//!
//! The module is pure (no I/O) and is consumed by
//! [`crate::default_allowlist()`] and the network policy layer.

/// Whether a domain pattern was accepted into the allowlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainStatus {
    /// The pattern passed validation and may be used as an allowlist entry.
    Allowed,
    /// The pattern was rejected; the `reason` on [`DomainDecision`] explains
    /// why.
    Blocked,
}

/// The outcome of validating a domain pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainDecision {
    /// `Allowed` or `Blocked`.
    pub status: DomainStatus,
    /// The normalized domain (lowercase, trailing dot stripped, port removed).
    pub normalized: String,
    /// Machine-readable reason (`empty_domain`, `ip_literal`,
    /// `broad_wildcard`, `invalid_domain`, `invalid_port`, `not_fqdn`,
    /// `invalid_wildcard`, `exact_domain`, `wildcard_domain`).
    pub reason: String,
}

impl DomainDecision {
    fn allowed(normalized: impl Into<String>, reason: &str) -> Self {
        Self {
            status: DomainStatus::Allowed,
            normalized: normalized.into(),
            reason: reason.to_string(),
        }
    }

    fn blocked(normalized: impl Into<String>, reason: &str) -> Self {
        Self {
            status: DomainStatus::Blocked,
            normalized: normalized.into(),
            reason: reason.to_string(),
        }
    }
}

/// Only wildcard suffixes that are safe to broaden are permitted. A wildcard
/// like `*.pythonhosted.org` is scoped to a single trusted CDN vendor, whereas
/// `*.com` or `*.example.com` would swallow unrelated hosts.
const ALLOWED_WILDCARD_SUFFIXES: &[&str] = &["pythonhosted.org"];

const DNS_LABEL_CHARS: &str = "abcdefghijklmnopqrstuvwxyz0123456789-";

/// Normalize a raw host/user input to a lowercase host string.
///
/// - strips URL schemes and path components,
/// - removes a trailing `:port` (non-URL forms),
/// - keeps bracketed IPv6 literals intact (`[::1]:80` -> `[::1]`),
/// - returns `""` for empty input or URLs carrying IPv6 brackets.
pub fn normalize_domain(raw: &str) -> String {
    let text = raw.trim().to_lowercase();
    if text.is_empty() {
        return String::new();
    }
    if text.contains("://") {
        let after = &text[text.find("://").unwrap() + 3..];
        let netloc = after.split(['/', '?', '#']).next().unwrap_or("");
        if netloc.contains('[') || netloc.contains(']') {
            return String::new();
        }
        let (hostname, _) = split_authority_host_port(netloc);
        return hostname;
    }
    let mut host = text.split('/').next().unwrap_or("").to_string();
    if host.starts_with('[') {
        if let Some(end) = host.find(']') {
            host = host[..=end].to_string();
        }
    } else if host.matches(':').count() == 1 {
        if let Some((h, port)) = host.rsplit_once(':') {
            if is_valid_port(port) {
                host = h.to_string();
            }
        }
    }
    host
}

/// Validate a domain allowlist pattern.
///
/// The returned [`DomainDecision`] carries the normalized pattern (which the
/// caller should store) and a reason suitable for audit logs. Blocked reasons
/// are stable machine-readable strings matching the Python port.
pub fn validate_domain_pattern(raw: &str) -> DomainDecision {
    let (normalized, extraction_error) = extract_validation_host(raw);
    if normalized.is_empty() {
        return DomainDecision::blocked(normalized, "empty_domain");
    }
    if let Some(err) = extraction_error {
        return DomainDecision::blocked(normalized, err);
    }
    if is_ip_literal(&normalized) {
        return DomainDecision::blocked(normalized, "ip_literal");
    }
    if let Some(suffix) = normalized.strip_prefix("*.") {
        if suffix.matches('.').count() < 1 {
            return DomainDecision::blocked(normalized, "broad_wildcard");
        }
        if !is_valid_dns_name(suffix) {
            return DomainDecision::blocked(normalized, "invalid_domain");
        }
        if !ALLOWED_WILDCARD_SUFFIXES.contains(&suffix) {
            return DomainDecision::blocked(normalized, "broad_wildcard");
        }
        return DomainDecision::allowed(normalized, "wildcard_domain");
    }
    if normalized.contains('*') {
        return DomainDecision::blocked(normalized, "invalid_wildcard");
    }
    let normalized = normalize_exact_validation_host(&normalized);
    if normalized.is_empty() {
        return DomainDecision::blocked(normalized, "empty_domain");
    }
    if is_ip_literal(&normalized) {
        return DomainDecision::blocked(normalized, "ip_literal");
    }
    if !normalized.contains('.') {
        return DomainDecision::blocked(normalized, "not_fqdn");
    }
    if !is_valid_dns_name(&normalized) {
        return DomainDecision::blocked(normalized, "invalid_domain");
    }
    DomainDecision::allowed(normalized, "exact_domain")
}

/// Match a host against an allowlist pattern.
///
/// The pattern is validated first (blocked patterns never match). A
/// `*.suffix` pattern matches only hosts that end with `.suffix` (the bare
/// suffix itself is not matched). Exact patterns require equality.
pub fn domain_matches(pattern: &str, host: &str) -> bool {
    let decision = validate_domain_pattern(pattern);
    if decision.status != DomainStatus::Allowed {
        return false;
    }
    let normalized_pattern = decision.normalized;
    let (mut normalized_host, extraction_error) = extract_validation_host(host);
    if extraction_error.is_some() {
        return false;
    }
    normalized_host = normalize_exact_validation_host(&normalized_host);
    if is_ip_literal(&normalized_host) || !is_valid_dns_name(&normalized_host) {
        return false;
    }
    if let Some(suffix) = normalized_pattern.strip_prefix("*.") {
        return normalized_host.ends_with(&format!(".{suffix}"));
    }
    normalized_host == normalized_pattern
}

/// Split `authority` (the `host[:port]` part of a URL) into host and optional
/// port. Bracketed IPv6 (`[::1]:80`) is handled. Returns the hostname with
/// brackets removed.
fn split_authority_host_port(authority: &str) -> (String, Option<String>) {
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']').unwrap_or(rest.len());
        let host = &rest[..end];
        let after = &rest[end + 1..];
        if let Some(port) = after.strip_prefix(':') {
            (host.to_string(), Some(port.to_string()))
        } else {
            (host.to_string(), None)
        }
    } else if authority.matches(':').count() == 1 {
        if let Some((h, p)) = authority.rsplit_once(':') {
            return (h.to_string(), Some(p.to_string()));
        }
        (authority.to_string(), None)
    } else {
        (authority.to_string(), None)
    }
}

/// Extract the validation host from a raw input, mirroring Python's
/// `_extract_validation_host`. Returns `(host, error)` where `error` is
/// `None` when the input parsed cleanly.
fn extract_validation_host(raw: &str) -> (String, Option<&'static str>) {
    let text = raw.trim().to_lowercase();
    if text.is_empty() {
        return (String::new(), None);
    }
    if let Some(pos) = text.find("://") {
        let after = &text[pos + 3..];
        let netloc = after.split(['/', '?', '#']).next().unwrap_or("");
        if netloc.contains('[') || netloc.contains(']') {
            let (hostname, _) = split_authority_host_port(netloc);
            return (hostname, Some("invalid_domain"));
        }
        let (hostname, port) = split_authority_host_port(netloc);
        if let Some(p) = port {
            if !is_valid_port(&p) {
                return (hostname, Some("invalid_port"));
            }
        }
        return (hostname, None);
    }

    let host = text.split('/').next().unwrap_or("");
    if host.starts_with('[') {
        let Some(end) = host.find(']') else {
            return (host.to_string(), Some("invalid_domain"));
        };
        let bracketed_host = host[..=end].to_string();
        let remainder = &host[end + 1..];
        if !remainder.is_empty() {
            if !remainder.starts_with(':') {
                return (host.to_string(), Some("invalid_domain"));
            }
            if !is_valid_port(&remainder[1..]) {
                return (bracketed_host, Some("invalid_port"));
            }
        }
        return (bracketed_host, None);
    }
    if host.matches(':').count() == 1 {
        if let Some((host_part, port)) = host.rsplit_once(':') {
            if !is_valid_port(port) {
                return (host_part.to_string(), Some("invalid_port"));
            }
            return (host_part.to_string(), None);
        }
    }
    (host.to_string(), None)
}

/// Strip a single trailing dot for exact hosts (FQDN form). Wildcards keep
/// their dot because they start with `*.`.
fn normalize_exact_validation_host(value: &str) -> String {
    if value.ends_with('.') && !value.starts_with("*.") {
        value[..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

/// Validate a port string (`0..65535`, 1-5 digits).
fn is_valid_port(value: &str) -> bool {
    if value.is_empty() || value.len() > 5 || !value.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    match value.parse::<u32>() {
        Ok(port) => port <= 65535,
        Err(_) => false,
    }
}

/// Validate a DNS name: 1-253 chars, dot-separated labels of 1-63 chars,
/// alphanumeric + `-`, no leading/trailing hyphen.
fn is_valid_dns_name(value: &str) -> bool {
    if value.is_empty() || value.len() > 253 {
        return false;
    }
    for label in value.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        if label.starts_with('-') || label.ends_with('-') {
            return false;
        }
        if !label.chars().all(|c| DNS_LABEL_CHARS.contains(c)) {
            return false;
        }
    }
    true
}

/// True when the value is an IP literal (IPv4, IPv6, or a numeric IPv4 alias
/// such as `127.1`).
fn is_ip_literal(value: &str) -> bool {
    let candidate = value.trim_matches(['[', ']']);
    if candidate.parse::<std::net::IpAddr>().is_ok() {
        return true;
    }
    is_ipv4_numeric_alias(candidate)
}

/// True when the value looks like a numeric IPv4 alias: 1-4 dot-separated
/// labels, each a decimal integer or `0x` hex integer.
fn is_ipv4_numeric_alias(value: &str) -> bool {
    let labels: Vec<&str> = value.split('.').collect();
    if !(1..=4).contains(&labels.len()) {
        return false;
    }
    labels.iter().all(|l| is_numeric_label(l))
}

fn is_numeric_label(label: &str) -> bool {
    if label.is_empty() {
        return false;
    }
    if let Some(hex) = label
        .strip_prefix("0x")
        .or_else(|| label.strip_prefix("0X"))
    {
        !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit())
    } else {
        label.chars().all(|c| c.is_ascii_digit())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_domain_forms() {
        assert_eq!(normalize_domain("https://GitHub.com/foo"), "github.com");
        assert_eq!(normalize_domain("  EXAMPLE.COM:8080  "), "example.com");
        assert_eq!(normalize_domain("example.com:8080"), "example.com");
        assert_eq!(
            normalize_domain("example.com:notaport"),
            "example.com:notaport"
        );
        assert_eq!(normalize_domain("[::1]:80"), "[::1]");
        assert_eq!(normalize_domain("http://[::1]:80/"), "");
        assert_eq!(normalize_domain(""), "");
        assert_eq!(normalize_domain("  "), "");
    }

    #[test]
    fn validate_exact_domain() {
        let d = validate_domain_pattern("example.com");
        assert_eq!(d.status, DomainStatus::Allowed);
        assert_eq!(d.normalized, "example.com");
        assert_eq!(d.reason, "exact_domain");
    }

    #[test]
    fn validate_rejects_ip_literal() {
        for raw in ["127.0.0.1", "10.0.0.1", "::1", "127.1", "0x7f000001"] {
            let d = validate_domain_pattern(raw);
            assert_eq!(d.status, DomainStatus::Blocked, "for {raw}");
            assert_eq!(d.reason, "ip_literal", "for {raw}");
        }
    }

    #[test]
    fn validate_rejects_broad_wildcard() {
        let d = validate_domain_pattern("*.com");
        assert_eq!(d.reason, "broad_wildcard");
        let d = validate_domain_pattern("*.example.com");
        assert_eq!(d.reason, "broad_wildcard");
        let d = validate_domain_pattern("*");
        assert!(d.status == DomainStatus::Blocked);
    }

    #[test]
    fn validate_allows_pythonhosted_wildcard() {
        let d = validate_domain_pattern("*.pythonhosted.org");
        assert_eq!(d.status, DomainStatus::Allowed);
        assert_eq!(d.reason, "wildcard_domain");
    }

    #[test]
    fn validate_rejects_bad_dns() {
        // `bad_domain` has no dot, so it is rejected as `not_fqdn` before the
        // DNS-name check runs (mirroring the Python ordering).
        assert_eq!(validate_domain_pattern("bad_domain").reason, "not_fqdn");
        assert_eq!(
            validate_domain_pattern("-leading.com").reason,
            "invalid_domain"
        );
        assert_eq!(
            validate_domain_pattern("trailing-.com").reason,
            "invalid_domain"
        );
        assert_eq!(validate_domain_pattern("a..b.com").reason, "invalid_domain");
        assert_eq!(
            validate_domain_pattern("under_score.com").reason,
            "invalid_domain"
        );
    }

    #[test]
    fn validate_rejects_not_fqdn() {
        let d = validate_domain_pattern("localhost");
        assert_eq!(d.status, DomainStatus::Blocked);
        assert_eq!(d.reason, "not_fqdn");
    }

    #[test]
    fn validate_strips_trailing_dot() {
        let d = validate_domain_pattern("example.com.");
        assert_eq!(d.status, DomainStatus::Allowed);
        assert_eq!(d.normalized, "example.com");
    }

    #[test]
    fn domain_matches_semantics() {
        assert!(domain_matches("github.com", "github.com"));
        assert!(!domain_matches("github.com", "evilgithub.com"));
        assert!(domain_matches(
            "*.pythonhosted.org",
            "files.pythonhosted.org"
        ));
        // Bare suffix does not match a wildcard.
        assert!(!domain_matches("*.pythonhosted.org", "pythonhosted.org"));
        // Invalid pattern never matches.
        assert!(!domain_matches("127.0.0.1", "127.0.0.1"));
        assert!(!domain_matches("*.com", "anything.com"));
        // Port and scheme are stripped from the host.
        assert!(domain_matches(
            "example.com",
            "https://example.com:8443/path"
        ));
    }

    #[test]
    fn invalid_port_reason() {
        let d = validate_domain_pattern("example.com:99999");
        assert_eq!(d.status, DomainStatus::Blocked);
        assert_eq!(d.reason, "invalid_port");
    }

    #[test]
    fn empty_domain_reason() {
        let d = validate_domain_pattern("");
        assert_eq!(d.reason, "empty_domain");
    }
}
