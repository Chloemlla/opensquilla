//! Authentication module for the gateway.
//!
//! Supports three authentication modes:
//! - **Token auth**: Bearer token validation via a shared secret or API key.
//! - **Open auth**: No authentication required (for development / public
//!   endpoints).
//! - **Loopback upgrade**: Requests originating from localhost are
//!   automatically upgraded to a privileged session without explicit auth.
//!
//! The module also provides the principal/scope model ported from the Python
//! `auth.py` (scope resolvers for token vs open mode), a rotatable
//! [`TokenStore`], and IP-based access control for allow/deny lists.

use opensquilla_core::error::{AppError, AppResult};
use opensquilla_core::types::UserId;
use std::collections::BTreeSet;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

/// The authentication mode the gateway is configured with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMode {
    /// Require a valid Bearer token.
    Token,
    /// Allow all requests without authentication.
    Open,
    /// Automatically authenticate requests from loopback addresses.
    Loopback,
}

impl AuthMode {
    /// Parse an auth mode from a string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "token" => AuthMode::Token,
            "open" | "none" => AuthMode::Open,
            "loopback" => AuthMode::Loopback,
            _ => AuthMode::Token,
        }
    }

    /// The stable string identifier of the mode.
    pub fn as_str(&self) -> &'static str {
        match self {
            AuthMode::Token => "token",
            AuthMode::Open => "none",
            AuthMode::Loopback => "loopback",
        }
    }
}

impl fmt::Display for AuthMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The outcome of an authentication attempt.
#[derive(Debug, Clone)]
pub struct AuthResult {
    /// The authenticated user identifier.
    pub user_id: UserId,
    /// Whether this session has elevated (admin) privileges.
    pub is_admin: bool,
    /// The authentication method used.
    pub method: AuthMethod,
}

/// The authentication method that was used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    Token,
    Open,
    Loopback,
}

/// Authentication provider.
#[derive(Debug, Clone)]
pub struct AuthProvider {
    mode: AuthMode,
    token: Option<String>,
}

impl AuthProvider {
    /// Create a new `AuthProvider` with the given mode and optional token.
    pub fn new(mode: AuthMode, token: Option<String>) -> Self {
        Self { mode, token }
    }

    /// Create a token-based auth provider.
    pub fn with_token(token: impl Into<String>) -> Self {
        Self {
            mode: AuthMode::Token,
            token: Some(token.into()),
        }
    }

    /// Create an open auth provider (no authentication).
    pub fn open() -> Self {
        Self {
            mode: AuthMode::Open,
            token: None,
        }
    }

    /// Create a loopback auth provider.
    pub fn loopback() -> Self {
        Self {
            mode: AuthMode::Loopback,
            token: None,
        }
    }

    /// Authenticate a request based on the `Authorization` header value and
    /// the remote address.
    ///
    /// Returns `Ok(AuthResult)` on success or `Err(AppError)` on failure.
    pub fn authenticate(
        &self,
        auth_header: Option<&str>,
        remote_addr: &str,
    ) -> AppResult<AuthResult> {
        match self.mode {
            AuthMode::Open => Ok(AuthResult {
                user_id: UserId::new(),
                is_admin: true,
                method: AuthMethod::Open,
            }),
            AuthMode::Loopback => {
                if is_loopback(remote_addr) {
                    Ok(AuthResult {
                        user_id: UserId::new(),
                        is_admin: true,
                        method: AuthMethod::Loopback,
                    })
                } else {
                    self.authenticate_with_token(auth_header)
                }
            }
            AuthMode::Token => self.authenticate_with_token(auth_header),
        }
    }

    fn authenticate_with_token(&self, auth_header: Option<&str>) -> AppResult<AuthResult> {
        let header = auth_header
            .and_then(|h| h.strip_prefix("Bearer "))
            .ok_or_else(|| AppError::unauthorized("Missing or malformed Authorization header"))?;

        let expected = self.token.as_deref().unwrap_or("");

        if header == expected {
            Ok(AuthResult {
                user_id: UserId::new(),
                is_admin: true,
                method: AuthMethod::Token,
            })
        } else {
            Err(AppError::unauthorized("Invalid token"))
        }
    }

    /// Return the current auth mode.
    pub fn mode(&self) -> &AuthMode {
        &self.mode
    }

    /// Return `true` if the provider has a token configured.
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }
}

/// Check whether a remote address string represents a loopback connection.
pub fn is_loopback(addr: &str) -> bool {
    addr == "127.0.0.1"
        || addr == "::1"
        || addr == "localhost"
        || addr.starts_with("127.")
        || addr.starts_with("0:0:0:0:0:0:0:1")
        || addr == "0.0.0.0"
}

/// Check whether a bind host is a loopback address.
///
/// `0.0.0.0` (all interfaces) is deliberately *not* considered a loopback
/// bind so that a public-facing gateway never auto-upgrades a client to the
/// local-owner principal.
pub fn is_loopback_bind(host: &str) -> bool {
    host == "127.0.0.1"
        || host == "::1"
        || host == "localhost"
        || host == "0:0:0:0:0:0:0:1"
        || host.starts_with("127.")
}

impl Default for AuthProvider {
    fn default() -> Self {
        Self::open()
    }
}

// ---------------------------------------------------------------------------
// Principal / scope model (ported from the Python gateway auth.py)
// ---------------------------------------------------------------------------

/// The default scope set granted to a locally-proven gateway operator.
pub const CLI_DEFAULT_OPERATOR_SCOPES: &[&str] = &[
    "operator.read",
    "operator.write",
    "operator.admin",
    "pairing",
];

/// The narrower scope set granted to a remote (non-loopback) operator.
pub const REMOTE_OPERATOR_SCOPES: &[&str] = &["operator.read", "operator.write", "approvals"];

/// The scope set granted to a `node` role.
pub const NODE_DEFAULT_SCOPES: &[&str] = &["node.read", "node.write"];

/// A server-computed identity credential, immutable for the lifetime of a
/// connection.
///
/// `is_owner` flags the caller as a locally-proven gateway owner. It is
/// advisory only — authorization decisions consult `scopes`.
#[derive(Debug, Clone)]
pub struct AuthPrincipal {
    /// The role claim (`operator` or `node`).
    pub role: String,
    /// The server-computed scope set.
    pub scopes: BTreeSet<String>,
    /// Whether the caller is a locally-proven gateway owner.
    pub is_owner: bool,
    /// Whether the caller authenticated with a token.
    pub authenticated: bool,
}

impl AuthPrincipal {
    /// Create a new principal.
    pub fn new(
        role: impl Into<String>,
        scopes: impl IntoIterator<Item = impl AsRef<str>>,
        is_owner: bool,
        authenticated: bool,
    ) -> Self {
        Self {
            role: role.into(),
            scopes: scopes.into_iter().map(|s| s.as_ref().to_string()).collect(),
            is_owner,
            authenticated,
        }
    }

    /// Whether the principal holds a specific scope.
    pub fn has_scope(&self, scope: &str) -> bool {
        self.scopes.contains(scope)
    }

    /// Whether the principal holds any of the given scopes.
    pub fn has_any_scope(&self, scopes: &[&str]) -> bool {
        scopes.iter().any(|s| self.scopes.contains(*s))
    }

    /// Whether the principal is a node role.
    pub fn is_node(&self) -> bool {
        self.role == "node"
    }
}

/// Normalize a declared operator scope list so a token declared with
/// `["operator.write"]` behaves identically to `["operator.write",
/// "operator.read"]`, and `operator.admin` additionally implies pairing.
pub fn normalize_operator_scopes(scopes: &[String]) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    for scope in scopes {
        set.insert(scope.clone());
        match scope.as_str() {
            "operator.write" => {
                set.insert("operator.read".to_string());
            }
            "operator.admin" => {
                set.insert("operator.read".to_string());
                set.insert("operator.write".to_string());
                set.insert("pairing".to_string());
            }
            _ => {}
        }
    }
    set
}

/// Authentication-related configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// The authentication mode.
    pub mode: AuthMode,
    /// The shared token, when token auth is enabled.
    pub token: Option<String>,
    /// The scope set declared for token-bearing operators.
    pub token_scopes: Vec<String>,
    /// Roles accepted by the gateway.
    pub allowed_roles: Vec<String>,
    /// Debug mode: grants the declared `token_scopes` regardless of peer.
    pub debug: bool,
    /// The host the gateway is bound to (used for loopback proximity).
    pub bind_host: String,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            mode: AuthMode::Open,
            token: None,
            token_scopes: CLI_DEFAULT_OPERATOR_SCOPES
                .iter()
                .map(|s| s.to_string())
                .collect(),
            allowed_roles: vec!["operator".to_string(), "node".to_string()],
            debug: false,
            bind_host: "127.0.0.1".to_string(),
        }
    }
}

/// A strategy for computing a [`AuthPrincipal`] for a given auth mode.
pub trait ScopeResolver: Send + Sync + fmt::Debug {
    /// Resolve a principal from the connect parameters, or return an error
    /// string describing why authentication failed.
    fn resolve(
        &self,
        config: &AuthConfig,
        auth_params: &serde_json::Value,
        role_claim: &str,
        peer_ip: Option<&str>,
    ) -> std::result::Result<AuthPrincipal, String>;
}

/// Token-mode resolver: validates the shared token and computes scopes from
/// the configured `token_scopes`, ignoring client-declared scopes.
#[derive(Debug, Default)]
pub struct TokenScopeResolver;

impl ScopeResolver for TokenScopeResolver {
    fn resolve(
        &self,
        config: &AuthConfig,
        auth_params: &serde_json::Value,
        role_claim: &str,
        peer_ip: Option<&str>,
    ) -> std::result::Result<AuthPrincipal, String> {
        let provided = auth_params
            .get("token")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let expected = config
            .token
            .as_deref()
            .ok_or_else(|| "No token configured".to_string())?;
        if provided != expected {
            return Err("Invalid token".to_string());
        }
        if !config.allowed_roles.iter().any(|r| r == role_claim) {
            return Err(format!("Invalid role: {role_claim:?}"));
        }

        if role_claim == "node" {
            return Ok(AuthPrincipal::new("node", NODE_DEFAULT_SCOPES, false, true));
        }

        let scopes = normalize_operator_scopes(&config.token_scopes);
        // Owner flag follows proximity, not the token: a shared token used
        // from a LAN peer should not claim ownership.
        let is_owner =
            is_loopback_bind(&config.bind_host) && peer_ip.map(is_loopback).unwrap_or(false);

        Ok(AuthPrincipal::new(role_claim, scopes, is_owner, true))
    }
}

/// Open-mode resolver: no authentication, with loopback-scoped admin upgrade.
#[derive(Debug, Default)]
pub struct OpenScopeResolver;

impl ScopeResolver for OpenScopeResolver {
    fn resolve(
        &self,
        config: &AuthConfig,
        _auth_params: &serde_json::Value,
        role_claim: &str,
        peer_ip: Option<&str>,
    ) -> std::result::Result<AuthPrincipal, String> {
        if !config.allowed_roles.iter().any(|r| r == role_claim) {
            return Err(format!("Invalid role: {role_claim:?}"));
        }

        if role_claim == "node" {
            return Ok(AuthPrincipal::new(
                "node",
                NODE_DEFAULT_SCOPES,
                false,
                false,
            ));
        }

        let local_owner =
            is_loopback_bind(&config.bind_host) && peer_ip.map(is_loopback).unwrap_or(false);

        let scopes: BTreeSet<String> = if config.debug {
            normalize_operator_scopes(&config.token_scopes)
        } else if local_owner {
            CLI_DEFAULT_OPERATOR_SCOPES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            REMOTE_OPERATOR_SCOPES
                .iter()
                .map(|s| s.to_string())
                .collect()
        };

        Ok(AuthPrincipal::new(role_claim, scopes, local_owner, false))
    }
}

/// Resolve a principal for the given auth configuration.
///
/// Returns `None` when authentication fails or the auth mode is unsupported.
/// `peer_ip` is the caller's IP as observed at the transport layer.
pub fn resolve_auth(
    config: &AuthConfig,
    auth_params: &serde_json::Value,
    role_claim: &str,
    peer_ip: Option<&str>,
) -> Option<AuthPrincipal> {
    let resolver: &dyn ScopeResolver = match config.mode {
        AuthMode::Token => &TokenScopeResolver,
        AuthMode::Open => &OpenScopeResolver,
        AuthMode::Loopback => {
            // Loopback upgrade: a loopback peer on a loopback-bound gateway is
            // treated as the local owner; anything else falls back to token.
            if is_loopback_bind(&config.bind_host) && peer_ip.map(is_loopback).unwrap_or(false) {
                &OpenScopeResolver
            } else {
                &TokenScopeResolver
            }
        }
    };
    resolver.resolve(config, auth_params, role_claim, peer_ip).ok()
}

// ---------------------------------------------------------------------------
// Token rotation
// ---------------------------------------------------------------------------

/// A shared-token store that supports rotation with a grace period for the
/// previous token.
#[derive(Debug, Clone)]
pub struct TokenStore {
    current: Arc<parking_lot::RwLock<String>>,
    previous: Arc<parking_lot::RwLock<Option<String>>>,
}

impl TokenStore {
    /// Create a new token store holding the given token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            current: Arc::new(parking_lot::RwLock::new(token.into())),
            previous: Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// The current (newest) token.
    pub fn current(&self) -> String {
        self.current.read().clone()
    }

    /// Whether the candidate matches the current or the previous token.
    pub fn validate(&self, candidate: &str) -> bool {
        if *self.current.read() == candidate {
            return true;
        }
        self.previous.read().as_deref() == Some(candidate)
    }

    /// Rotate to a new token, retaining the old one for the grace period.
    /// Returns the previous token.
    pub fn rotate(&self, new_token: impl Into<String>) -> String {
        let new_token = new_token.into();
        let mut current = self.current.write();
        let mut previous = self.previous.write();
        let old = current.clone();
        *previous = Some(current.clone());
        *current = new_token;
        old
    }

    /// Clear the previous-token grace window.
    pub fn clear_previous(&self) {
        *self.previous.write() = None;
    }
}

// ---------------------------------------------------------------------------
// IP-based access control
// ---------------------------------------------------------------------------

/// A compiled IP match pattern (exact address, CIDR, or wildcard).
#[derive(Debug, Clone)]
enum IpPattern {
    /// Matches any address (`*`).
    Any,
    /// Matches a single address.
    Exact(IpAddr),
    /// IPv4 CIDR block.
    CidrV4(Ipv4Addr, u8),
    /// IPv6 CIDR block.
    CidrV6(Ipv6Addr, u8),
}

impl IpPattern {
    /// Parse a pattern string: `*`, `192.168.1.5`, `10.0.0.0/8`, or an IPv6
    /// address / CIDR.
    fn parse(s: &str) -> Option<IpPattern> {
        let s = s.trim();
        if s.is_empty() || s == "*" {
            return Some(IpPattern::Any);
        }
        if let Some((addr, prefix)) = s.split_once('/') {
            if let Ok(v4) = addr.parse::<Ipv4Addr>() {
                let p: u8 = prefix.parse::<u8>().ok()?;
                return Some(IpPattern::CidrV4(v4, p.min(32)));
            }
            if let Ok(v6) = addr.parse::<Ipv6Addr>() {
                let p: u8 = prefix.parse::<u8>().ok()?;
                return Some(IpPattern::CidrV6(v6, p.min(128)));
            }
            return None;
        }
        if let Ok(v4) = s.parse::<Ipv4Addr>() {
            return Some(IpPattern::Exact(IpAddr::V4(v4)));
        }
        if let Ok(v6) = s.parse::<Ipv6Addr>() {
            return Some(IpPattern::Exact(IpAddr::V6(v6)));
        }
        None
    }

    /// Whether this pattern matches the given address.
    fn matches(&self, ip: &IpAddr) -> bool {
        match self {
            IpPattern::Any => true,
            IpPattern::Exact(expected) => expected == ip,
            IpPattern::CidrV4(net, prefix) => {
                let IpAddr::V4(addr) = ip else {
                    return false;
                };
                let a = u32::from(*addr);
                let n = u32::from(*net);
                let mask = if *prefix == 0 {
                    0
                } else {
                    u32::MAX << (32 - *prefix as u32)
                };
                (a & mask) == (n & mask)
            }
            IpPattern::CidrV6(net, prefix) => {
                let IpAddr::V6(addr) = ip else {
                    return false;
                };
                let a = addr.octets();
                let n = net.octets();
                let full_bytes = (*prefix / 8) as usize;
                let rem = *prefix % 8;
                for i in 0..full_bytes {
                    if a[i] != n[i] {
                        return false;
                    }
                }
                if rem > 0 {
                    let mask = 0xFFu8 << (8 - rem);
                    if (a[full_bytes] & mask) != (n[full_bytes] & mask) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// IP allow/deny access control.
///
/// By default all addresses are allowed; a deny list wins over the allow
/// list. When an allow list is present, only matching addresses are allowed.
#[derive(Debug, Clone, Default)]
pub struct IpAccessControl {
    allowlist: Vec<IpPattern>,
    denylist: Vec<IpPattern>,
    default_allow: bool,
}

impl IpAccessControl {
    /// Create a permissive access-control list.
    pub fn new() -> Self {
        Self {
            allowlist: Vec::new(),
            denylist: Vec::new(),
            default_allow: true,
        }
    }

    /// Allow the given pattern (exact IP or CIDR).
    pub fn allow(&mut self, pattern: &str) -> std::result::Result<&mut Self, String> {
        let compiled =
            IpPattern::parse(pattern).ok_or_else(|| format!("Invalid IP pattern '{pattern}'"))?;
        self.allowlist.push(compiled);
        Ok(self)
    }

    /// Deny the given pattern (exact IP or CIDR).
    pub fn deny(&mut self, pattern: &str) -> std::result::Result<&mut Self, String> {
        let compiled =
            IpPattern::parse(pattern).ok_or_else(|| format!("Invalid IP pattern '{pattern}'"))?;
        self.denylist.push(compiled);
        Ok(self)
    }

    /// Flip the default to deny when no allow list matches.
    pub fn default_deny(mut self) -> Self {
        self.default_allow = false;
        self
    }

    /// Whether the given IP string is allowed.
    pub fn allows(&self, ip: &str) -> bool {
        match ip.parse::<IpAddr>() {
            Ok(addr) => self.allows_addr(&addr),
            Err(_) => self.default_allow,
        }
    }

    /// Whether the given address is allowed.
    pub fn allows_addr(&self, ip: &IpAddr) -> bool {
        if self.denylist.iter().any(|p| p.matches(ip)) {
            return false;
        }
        if self.allowlist.is_empty() {
            return self.default_allow;
        }
        self.allowlist.iter().any(|p| p.matches(ip))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_auth_success() {
        let provider = AuthProvider::with_token("secret123");
        let result = provider.authenticate(Some("Bearer secret123"), "10.0.0.1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_token_auth_failure() {
        let provider = AuthProvider::with_token("secret123");
        let result = provider.authenticate(Some("Bearer wrong"), "10.0.0.1");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_header() {
        let provider = AuthProvider::with_token("secret123");
        let result = provider.authenticate(None, "10.0.0.1");
        assert!(result.is_err());
    }

    #[test]
    fn test_open_auth() {
        let provider = AuthProvider::open();
        let result = provider.authenticate(None, "10.0.0.1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_loopback_upgrade() {
        let provider = AuthProvider::loopback();
        let result = provider.authenticate(None, "127.0.0.1");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().method, AuthMethod::Loopback);
    }

    #[test]
    fn test_loopback_fallback_to_token() {
        let provider = AuthProvider {
            mode: AuthMode::Loopback,
            token: Some("key".into()),
        };
        let result = provider.authenticate(Some("Bearer key"), "10.0.0.1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_is_loopback() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("::1"));
        assert!(is_loopback("localhost"));
        assert!(!is_loopback("10.0.0.1"));
        assert!(!is_loopback("192.168.1.1"));
    }

    #[test]
    fn test_is_loopback_bind() {
        assert!(is_loopback_bind("127.0.0.1"));
        assert!(is_loopback_bind("::1"));
        assert!(!is_loopback_bind("0.0.0.0"));
        assert!(!is_loopback_bind("192.168.1.5"));
    }

    #[test]
    fn test_normalize_operator_scopes() {
        let scopes = vec!["operator.write".to_string()];
        let normalized = normalize_operator_scopes(&scopes);
        assert!(normalized.contains("operator.write"));
        assert!(normalized.contains("operator.read"));

        let admin = vec!["operator.admin".to_string()];
        let normalized = normalize_operator_scopes(&admin);
        assert!(normalized.contains("operator.admin"));
        assert!(normalized.contains("operator.write"));
        assert!(normalized.contains("operator.read"));
        assert!(normalized.contains("pairing"));
    }

    #[test]
    fn test_principal_scope_checks() {
        let principal = AuthPrincipal::new("operator", CLI_DEFAULT_OPERATOR_SCOPES, true, false);
        assert!(principal.has_scope("operator.admin"));
        assert!(principal.has_scope("pairing"));
        assert!(principal.has_any_scope(&["operator.read", "unknown"]));
        assert!(!principal.has_scope("node.read"));
        assert!(!principal.is_node());
    }

    #[test]
    fn test_token_resolver_success() {
        let config = AuthConfig {
            mode: AuthMode::Token,
            token: Some("secret".into()),
            bind_host: "127.0.0.1".into(),
            ..Default::default()
        };
        let principal = resolve_auth(
            &config,
            &serde_json::json!({"token": "secret"}),
            "operator",
            Some("127.0.0.1"),
        );
        let principal = principal.expect("resolved");
        assert!(principal.authenticated);
        assert!(principal.is_owner);
        assert!(principal.has_scope("operator.admin"));
    }

    #[test]
    fn test_token_resolver_remote_not_owner() {
        let config = AuthConfig {
            mode: AuthMode::Token,
            token: Some("secret".into()),
            bind_host: "127.0.0.1".into(),
            ..Default::default()
        };
        let principal = resolve_auth(
            &config,
            &serde_json::json!({"token": "secret"}),
            "operator",
            Some("10.0.0.5"),
        );
        let principal = principal.expect("resolved");
        // A shared token used from a LAN peer must not claim ownership.
        assert!(!principal.is_owner);
        assert!(principal.authenticated);
    }

    #[test]
    fn test_token_resolver_rejects_bad_token() {
        let config = AuthConfig {
            mode: AuthMode::Token,
            token: Some("secret".into()),
            ..Default::default()
        };
        let principal = resolve_auth(
            &config,
            &serde_json::json!({"token": "wrong"}),
            "operator",
            Some("127.0.0.1"),
        );
        assert!(principal.is_none());
    }

    #[test]
    fn test_open_resolver_local_owner() {
        let config = AuthConfig {
            mode: AuthMode::Open,
            bind_host: "127.0.0.1".into(),
            ..Default::default()
        };
        let principal = resolve_auth(
            &config,
            &serde_json::json!({}),
            "operator",
            Some("127.0.0.1"),
        );
        let principal = principal.expect("resolved");
        assert!(principal.is_owner);
        assert!(principal.has_scope("operator.admin"));
        assert!(!principal.authenticated);
    }

    #[test]
    fn test_open_resolver_remote_gets_limited_scopes() {
        let config = AuthConfig {
            mode: AuthMode::Open,
            bind_host: "0.0.0.0".into(),
            ..Default::default()
        };
        // A remote peer on a public bind must not get admin scopes.
        let principal = resolve_auth(
            &config,
            &serde_json::json!({}),
            "operator",
            Some("10.0.0.9"),
        );
        let principal = principal.expect("resolved");
        assert!(!principal.is_owner);
        assert!(!principal.has_scope("operator.admin"));
        assert!(principal.has_scope("approvals"));
    }

    #[test]
    fn test_token_store_validate_and_rotate() {
        let store = TokenStore::new("v1");
        assert!(store.validate("v1"));
        assert!(!store.validate("wrong"));

        let old = store.rotate("v2");
        assert_eq!(old, "v1");
        assert_eq!(store.current(), "v2");
        // Both the new and previous tokens validate during the grace period.
        assert!(store.validate("v2"));
        assert!(store.validate("v1"));

        store.clear_previous();
        assert!(store.validate("v2"));
        assert!(!store.validate("v1"));
    }

    #[test]
    fn test_ip_access_control_exact() {
        let mut acl = IpAccessControl::new();
        acl.deny("192.168.1.10").unwrap();
        assert!(!acl.allows("192.168.1.10"));
        assert!(acl.allows("192.168.1.11"));
    }

    #[test]
    fn test_ip_access_control_cidr_allow() {
        let mut acl = IpAccessControl::new();
        acl.allow("10.0.0.0/8").unwrap();
        acl = acl.default_deny();
        assert!(acl.allows("10.1.2.3"));
        assert!(!acl.allows("11.0.0.1"));
        assert!(!acl.allows("not-an-ip"));
    }

    #[test]
    fn test_ip_access_control_deny_wins() {
        let mut acl = IpAccessControl::new();
        acl.allow("10.0.0.0/8").unwrap();
        acl.deny("10.0.0.5").unwrap();
        acl = acl.default_deny();
        assert!(acl.allows("10.0.0.6"));
        assert!(!acl.allows("10.0.0.5"));
    }

    #[test]
    fn test_ip_pattern_wildcard() {
        let mut acl = IpAccessControl::new();
        acl.allow("*").unwrap();
        acl = acl.default_deny();
        assert!(acl.allows("203.0.113.7"));
        assert!(acl.allows("::1"));
    }
}
