//! Scope definitions and checking.
//!
//! Mirrors the Python `scopes.py` module. Defines the permission scopes that
//! protect RPC methods, provides scope hierarchy (`admin` > `user` >
//! `read_only`), and maps RPC method names to the scopes they require.

use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};

/// The set of permission scopes recognized by the gateway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Read-only access: viewing sessions, config, logs, catalog.
    ReadOnly,
    /// Standard user access: creating/editing sessions, sending turns.
    User,
    /// Administrative access: changing provider credentials, scopes, and
    /// system-level settings.
    Admin,
    /// Internal/gateway-to-gateway scope for system services.
    System,
}

impl Scope {
    /// The stable string identifier of the scope.
    pub fn as_str(&self) -> &'static str {
        match self {
            Scope::ReadOnly => "read_only",
            Scope::User => "user",
            Scope::Admin => "admin",
            Scope::System => "system",
        }
    }

    /// Parse a scope from its string identifier.
    pub fn parse(s: &str) -> Result<Self, AppError> {
        match s.to_ascii_lowercase().as_str() {
            "read_only" | "readonly" | "read" => Ok(Scope::ReadOnly),
            "user" => Ok(Scope::User),
            "admin" => Ok(Scope::Admin),
            "system" => Ok(Scope::System),
            other => Err(AppError::bad_request(format!("Unknown scope '{other}'"))),
        }
    }

    /// Return `true` if `other` is granted by this scope under the hierarchy
    /// `admin` > `user` > `read_only`. `system` is its own tier that does not
    /// imply or get implied by the others.
    pub fn grants(&self, other: Scope) -> bool {
        use Scope::*;
        match (*self, other) {
            (Admin, ReadOnly) | (Admin, User) | (Admin, Admin) => true,
            (User, ReadOnly) | (User, User) => true,
            (ReadOnly, ReadOnly) => true,
            (System, System) => true,
            _ => false,
        }
    }

    /// Return the scope implied by this one in the hierarchy.
    pub fn implied_scopes(&self) -> Vec<Scope> {
        use Scope::*;
        match *self {
            Admin => vec![Admin, User, ReadOnly],
            User => vec![User, ReadOnly],
            ReadOnly => vec![ReadOnly],
            System => vec![System],
        }
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The access level of an authenticated principal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Principal {
    scope: Scope,
}

impl Principal {
    /// Create a principal with the given scope.
    pub fn new(scope: Scope) -> Self {
        Self { scope }
    }

    /// The principal's effective scope.
    pub fn scope(&self) -> Scope {
        self.scope
    }

    /// Check whether this principal may access a required scope.
    pub fn can_access(&self, required: Scope) -> bool {
        self.scope.grants(required)
    }
}

/// A static registry mapping RPC method names (or prefixes) to required
/// scopes. Prefixes end with `.*` and match any method with that prefix.
#[derive(Debug, Clone, Default)]
pub struct ScopeRegistry {
    entries: Vec<(String, Scope)>,
}

impl ScopeRegistry {
    /// Create a new registry.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register a scope requirement for an exact method name.
    pub fn register(&mut self, method: &str, scope: Scope) {
        self.entries.push((method.to_string(), scope));
    }

    /// Register a scope requirement for a method prefix (e.g. `"session.*"`).
    pub fn register_prefix(&mut self, prefix: &str, scope: Scope) {
        let key = if prefix.ends_with('*') {
            prefix.to_string()
        } else {
            format!("{prefix}*")
        };
        self.entries.push((key, scope));
    }

    /// Look up the scope required for a method.
    ///
    /// Exact matches win over prefix matches. If no entry matches, `None` is
    /// returned (callers decide the default — typically `User`).
    pub fn scope_for(&self, method: &str) -> Option<Scope> {
        // Exact match first.
        for (key, scope) in &self.entries {
            if key == method {
                return Some(*scope);
            }
        }
        // Prefix match: keys ending in `*`.
        for (key, scope) in &self.entries {
            if let Some(prefix) = key.strip_suffix('*') {
                if method.starts_with(prefix) {
                    return Some(*scope);
                }
            }
        }
        None
    }

    /// Check whether a principal may call a method.
    ///
    /// If no scope is registered for the method, the default scope is used.
    pub fn authorize(
        &self,
        principal: &Principal,
        method: &str,
        default: Scope,
    ) -> Result<(), AppError> {
        let required = self.scope_for(method).unwrap_or(default);
        if principal.can_access(required) {
            Ok(())
        } else {
            Err(AppError::forbidden(format!(
                "Principal with scope '{scope}' cannot call '{method}' (requires '{required}')",
                scope = principal.scope()
            )))
        }
    }
}

/// Build a default scope registry for the standard RPC method families.
pub fn default_scope_registry() -> ScopeRegistry {
    let mut reg = ScopeRegistry::new();
    // Read-only method families.
    reg.register_prefix("session.list", Scope::ReadOnly);
    reg.register_prefix("session.get", Scope::ReadOnly);
    reg.register_prefix("chat.history", Scope::ReadOnly);
    reg.register_prefix("config.get", Scope::ReadOnly);
    reg.register_prefix("config.list", Scope::ReadOnly);
    reg.register_prefix("models.list", Scope::ReadOnly);
    reg.register_prefix("tools.list", Scope::ReadOnly);
    reg.register_prefix("usage.get", Scope::ReadOnly);
    reg.register_prefix("logs.check", Scope::ReadOnly);
    reg.register_prefix("doctor.run", Scope::ReadOnly);
    reg.register_prefix("memory.search", Scope::ReadOnly);
    reg.register_prefix("routing.get_hold", Scope::ReadOnly);
    reg.register_prefix("router.history", Scope::ReadOnly);
    reg.register_prefix("skills.list", Scope::ReadOnly);
    reg.register_prefix("channels.list", Scope::ReadOnly);
    reg.register_prefix("cron.list", Scope::ReadOnly);
    reg.register_prefix("proposals.list", Scope::ReadOnly);
    reg.register_prefix("meta_runs.list", Scope::ReadOnly);
    reg.register_prefix("agents.list", Scope::ReadOnly);
    reg.register_prefix("workspaces.list", Scope::ReadOnly);

    // User-scoped method families.
    reg.register_prefix("session.create", Scope::User);
    reg.register_prefix("session.update", Scope::User);
    reg.register_prefix("session.archive", Scope::User);
    reg.register_prefix("session.delete", Scope::User);
    reg.register_prefix("chat.send", Scope::User);
    reg.register_prefix("chat.attach", Scope::User);
    reg.register_prefix("chat.delete", Scope::User);
    reg.register_prefix("config.set", Scope::User);
    reg.register_prefix("config.update", Scope::User);
    reg.register_prefix("routing.hold", Scope::User);
    reg.register_prefix("routing.release", Scope::User);
    reg.register_prefix("skills.install", Scope::User);
    reg.register_prefix("skills.enable", Scope::User);
    reg.register_prefix("cron.create", Scope::User);
    reg.register_prefix("cron.update", Scope::User);
    reg.register_prefix("channels.connect", Scope::User);
    reg.register_prefix("channels.disconnect", Scope::User);
    reg.register_prefix("memory.refresh", Scope::User);
    reg.register_prefix("tools.run", Scope::User);

    // Admin-scoped method families.
    reg.register_prefix("secrets.*", Scope::Admin);
    reg.register_prefix("onboarding.*", Scope::Admin);
    reg.register_prefix("migration.*", Scope::Admin);
    reg.register_prefix("system.*", Scope::Admin);
    reg.register_prefix("diagnostics.*", Scope::Admin);
    reg.register_prefix("agents.delete", Scope::Admin);
    reg.register_prefix("users.*", Scope::Admin);

    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_parse_roundtrip() {
        for scope in [Scope::ReadOnly, Scope::User, Scope::Admin, Scope::System] {
            assert_eq!(Scope::parse(scope.as_str()).unwrap(), scope);
        }
        assert!(Scope::parse("bogus").is_err());
    }

    #[test]
    fn test_scope_hierarchy() {
        assert!(Scope::Admin.grants(Scope::User));
        assert!(Scope::Admin.grants(Scope::ReadOnly));
        assert!(Scope::User.grants(Scope::ReadOnly));
        assert!(!Scope::User.grants(Scope::Admin));
        assert!(!Scope::ReadOnly.grants(Scope::User));
        assert!(!Scope::ReadOnly.grants(Scope::Admin));
        assert!(Scope::Admin.grants(Scope::Admin));
        assert!(!Scope::System.grants(Scope::User));
        assert!(!Scope::User.grants(Scope::System));
    }

    #[test]
    fn test_implied_scopes() {
        assert_eq!(
            Scope::Admin.implied_scopes(),
            vec![Scope::Admin, Scope::User, Scope::ReadOnly]
        );
        assert_eq!(
            Scope::User.implied_scopes(),
            vec![Scope::User, Scope::ReadOnly]
        );
        assert_eq!(Scope::ReadOnly.implied_scopes(), vec![Scope::ReadOnly]);
    }

    #[test]
    fn test_principal_access() {
        let read_only = Principal::new(Scope::ReadOnly);
        assert!(read_only.can_access(Scope::ReadOnly));
        assert!(!read_only.can_access(Scope::User));

        let admin = Principal::new(Scope::Admin);
        assert!(admin.can_access(Scope::User));
        assert!(admin.can_access(Scope::Admin));
    }

    #[test]
    fn test_registry_exact_and_prefix() {
        let mut reg = ScopeRegistry::new();
        reg.register("session.get", Scope::ReadOnly);
        reg.register("chat.send", Scope::User);
        reg.register_prefix("secrets", Scope::Admin);

        assert_eq!(reg.scope_for("session.get"), Some(Scope::ReadOnly));
        assert_eq!(reg.scope_for("chat.send"), Some(Scope::User));
        assert_eq!(reg.scope_for("secrets.set_key"), Some(Scope::Admin));
        assert_eq!(reg.scope_for("unknown.method"), None);
    }

    #[test]
    fn test_authorize_granted_and_denied() {
        let reg = default_scope_registry();
        let user = Principal::new(Scope::User);
        let read_only = Principal::new(Scope::ReadOnly);

        // User can call a user-scoped method.
        assert!(reg.authorize(&user, "chat.send", Scope::User).is_ok());
        // Read-only principal is denied for user-scoped methods.
        assert!(reg.authorize(&read_only, "chat.send", Scope::User).is_err());
        // Read-only principal can call read-only methods.
        assert!(
            reg.authorize(&read_only, "session.list", Scope::User)
                .is_ok()
        );
        // Admin can do anything user-scoped.
        let admin = Principal::new(Scope::Admin);
        assert!(reg.authorize(&admin, "chat.send", Scope::User).is_ok());
        // Secrets are admin-only.
        assert!(reg.authorize(&user, "secrets.list", Scope::User).is_err());
        assert!(
            reg.authorize(&admin, "secrets.set_key", Scope::User)
                .is_ok()
        );
    }

    #[test]
    fn test_authorize_default_scope() {
        let reg = ScopeRegistry::new();
        let read_only = Principal::new(Scope::ReadOnly);
        // Unknown method falls back to the provided default.
        assert!(
            reg.authorize(&read_only, "unknown.method", Scope::User)
                .is_err()
        );
        assert!(
            reg.authorize(&read_only, "unknown.method", Scope::ReadOnly)
                .is_ok()
        );
    }

    #[test]
    fn test_default_registry_covers_families() {
        let reg = default_scope_registry();
        assert_eq!(reg.scope_for("session.create"), Some(Scope::User));
        assert_eq!(reg.scope_for("secrets.get"), Some(Scope::Admin));
        assert_eq!(reg.scope_for("models.list"), Some(Scope::ReadOnly));
    }
}
