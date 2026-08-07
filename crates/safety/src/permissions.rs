use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Risk level assigned to an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum RiskLevel {
    /// Safe to execute without confirmation.
    Safe,
    /// Requires user confirmation before execution.
    Confirm,
    /// Only allowed for admin users.
    AdminOnly,
}

/// A permission entry in the matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Unique identifier for the permission.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Risk level of this operation.
    pub risk_level: RiskLevel,
    /// Category for grouping related permissions.
    pub category: String,
}

/// The permission matrix mapping operations to their risk levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionMatrix {
    permissions: HashMap<String, Permission>,
    overrides: HashMap<String, RiskLevel>,
}

impl PermissionMatrix {
    /// Create a new permission matrix with default entries.
    pub fn new() -> Self {
        let mut matrix = Self {
            permissions: HashMap::new(),
            overrides: HashMap::new(),
        };
        matrix.load_defaults();
        matrix
    }

    /// Load the default set of permissions.
    fn load_defaults(&mut self) {
        let defaults = vec![
            // File system operations
            Permission {
                name: "file.read".to_string(),
                description: "Read files from the local filesystem".to_string(),
                risk_level: RiskLevel::Safe,
                category: "filesystem".to_string(),
            },
            Permission {
                name: "file.write".to_string(),
                description: "Write files to the local filesystem".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "filesystem".to_string(),
            },
            Permission {
                name: "file.delete".to_string(),
                description: "Delete files from the local filesystem".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "filesystem".to_string(),
            },
            Permission {
                name: "file.execute".to_string(),
                description: "Execute files on the local system".to_string(),
                risk_level: RiskLevel::AdminOnly,
                category: "filesystem".to_string(),
            },
            // Network operations
            Permission {
                name: "network.http".to_string(),
                description: "Make HTTP requests to external services".to_string(),
                risk_level: RiskLevel::Safe,
                category: "network".to_string(),
            },
            Permission {
                name: "network.websocket".to_string(),
                description: "Open WebSocket connections".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "network".to_string(),
            },
            Permission {
                name: "network.listen".to_string(),
                description: "Listen on network ports".to_string(),
                risk_level: RiskLevel::AdminOnly,
                category: "network".to_string(),
            },
            // System operations
            Permission {
                name: "system.info".to_string(),
                description: "Read system information".to_string(),
                risk_level: RiskLevel::Safe,
                category: "system".to_string(),
            },
            Permission {
                name: "system.process".to_string(),
                description: "Spawn and manage system processes".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "system".to_string(),
            },
            Permission {
                name: "system.env".to_string(),
                description: "Read environment variables".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "system".to_string(),
            },
            // Data operations
            Permission {
                name: "data.read".to_string(),
                description: "Read data from the knowledge base".to_string(),
                risk_level: RiskLevel::Safe,
                category: "data".to_string(),
            },
            Permission {
                name: "data.write".to_string(),
                description: "Write data to the knowledge base".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "data".to_string(),
            },
            Permission {
                name: "data.delete".to_string(),
                description: "Delete data from the knowledge base".to_string(),
                risk_level: RiskLevel::AdminOnly,
                category: "data".to_string(),
            },
            // Provider operations
            Permission {
                name: "provider.switch".to_string(),
                description: "Switch between AI providers".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "provider".to_string(),
            },
            Permission {
                name: "provider.configure".to_string(),
                description: "Modify provider configuration".to_string(),
                risk_level: RiskLevel::AdminOnly,
                category: "provider".to_string(),
            },
            // Sandbox operations
            Permission {
                name: "sandbox.create".to_string(),
                description: "Create a new sandbox environment".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "sandbox".to_string(),
            },
            Permission {
                name: "sandbox.exec".to_string(),
                description: "Execute commands in a sandbox".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "sandbox".to_string(),
            },
            // Channel operations
            Permission {
                name: "channel.send".to_string(),
                description: "Send messages through configured channels".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "channel".to_string(),
            },
            // Memory operations
            Permission {
                name: "memory.read".to_string(),
                description: "Read memory entries".to_string(),
                risk_level: RiskLevel::Safe,
                category: "memory".to_string(),
            },
            Permission {
                name: "memory.write".to_string(),
                description: "Write to memory".to_string(),
                risk_level: RiskLevel::Confirm,
                category: "memory".to_string(),
            },
        ];

        for perm in defaults {
            self.permissions.insert(perm.name.clone(), perm);
        }
    }

    /// Look up the risk level of a permission.
    pub fn check(&self, permission_name: &str) -> Option<RiskLevel> {
        // Check user overrides first
        if let Some(level) = self.overrides.get(permission_name) {
            return Some(*level);
        }
        // Fall back to default
        self.permissions.get(permission_name).map(|p| p.risk_level)
    }

    /// Check if a permission is safe to execute without confirmation.
    pub fn is_safe(&self, permission_name: &str) -> bool {
        self.check(permission_name) == Some(RiskLevel::Safe)
    }

    /// Check if a permission requires confirmation.
    pub fn requires_confirmation(&self, permission_name: &str) -> bool {
        self.check(permission_name) == Some(RiskLevel::Confirm)
    }

    /// Check if a permission is admin-only.
    pub fn is_admin_only(&self, permission_name: &str) -> bool {
        self.check(permission_name) == Some(RiskLevel::AdminOnly)
    }

    /// Override the default risk level for a permission.
    pub fn override_permission(&mut self, name: &str, level: RiskLevel) {
        self.overrides.insert(name.to_string(), level);
    }

    /// Remove an override for a permission.
    pub fn remove_override(&mut self, name: &str) {
        self.overrides.remove(name);
    }

    /// Register a new custom permission.
    pub fn register(&mut self, permission: Permission) {
        self.permissions.insert(permission.name.clone(), permission);
    }

    /// List all registered permissions.
    pub fn list(&self) -> Vec<&Permission> {
        self.permissions.values().collect()
    }

    /// List permissions by category.
    pub fn list_by_category(&self, category: &str) -> Vec<&Permission> {
        self.permissions
            .values()
            .filter(|p| p.category == category)
            .collect()
    }

    /// Get all available categories.
    pub fn categories(&self) -> Vec<String> {
        let mut cats: Vec<String> = self
            .permissions
            .values()
            .map(|p| p.category.clone())
            .collect();
        cats.sort();
        cats.dedup();
        cats
    }

    /// Get the permission details.
    pub fn get_permission(&self, name: &str) -> Option<&Permission> {
        self.permissions.get(name)
    }
}

impl Default for PermissionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

/// The functional scope of a permission, used for grouping and policy routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PermissionScope {
    Tool,
    File,
    Network,
    Subprocess,
    Config,
}

impl PermissionScope {
    /// Parse a scope from its string token (`tool`, `file`, `network`,
    /// `subprocess`, `config`).
    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_lowercase().as_str() {
            "tool" => Some(PermissionScope::Tool),
            "file" | "filesystem" => Some(PermissionScope::File),
            "network" => Some(PermissionScope::Network),
            "subprocess" | "process" | "shell" => Some(PermissionScope::Subprocess),
            "config" | "configuration" => Some(PermissionScope::Config),
            _ => None,
        }
    }

    /// The canonical string token for this scope.
    pub fn as_str(&self) -> &'static str {
        match self {
            PermissionScope::Tool => "tool",
            PermissionScope::File => "file",
            PermissionScope::Network => "network",
            PermissionScope::Subprocess => "subprocess",
            PermissionScope::Config => "config",
        }
    }
}

/// The action required by a permission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionAction {
    Allow,
    Confirm,
    Deny,
}

/// Context for a permission check: who is asking and under what conditions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PermissionContext {
    /// The requesting user's role (`admin`, `user`, `guest`, ...).
    pub user_role: String,
    /// Whether the action runs inside a sandbox.
    pub sandboxed: bool,
    /// Whether the caller is running with elevated privileges.
    pub elevated: bool,
    /// The scopes the caller is permitted to touch, if scoped.
    pub scopes: Vec<PermissionScope>,
}

impl PermissionContext {
    /// Create an empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the requesting user's role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.user_role = role.into();
        self
    }

    /// Mark the action as running inside a sandbox.
    pub fn in_sandbox(mut self) -> Self {
        self.sandboxed = true;
        self
    }

    /// Mark the caller as elevated.
    pub fn elevated(mut self) -> Self {
        self.elevated = true;
        self
    }

    /// Restrict the caller to the given scopes.
    pub fn with_scopes(mut self, scopes: Vec<PermissionScope>) -> Self {
        self.scopes = scopes;
        self
    }
}

/// The result of a permission check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionCheck {
    /// The required action for this permission.
    pub action: PermissionAction,
    /// The permission (or action) that was checked.
    pub permission: String,
    /// The risk level assigned to the permission.
    pub risk_level: RiskLevel,
    /// A human-readable reason for the decision.
    pub reason: String,
}

impl PermissionCheck {
    /// Returns true when the action is allowed outright.
    pub fn is_allowed(&self) -> bool {
        self.action == PermissionAction::Allow
    }

    /// Returns true when the action requires user confirmation.
    pub fn needs_confirmation(&self) -> bool {
        self.action == PermissionAction::Confirm
    }

    /// Returns true when the action is denied.
    pub fn is_denied(&self) -> bool {
        self.action == PermissionAction::Deny
    }
}

impl PermissionMatrix {
    /// Classify an action string into a risk level.
    ///
    /// Exact permission entries and overrides take precedence; unknown actions
    /// fall back to a deterministic verb-based heuristic.
    pub fn classify_action(&self, action: &str) -> RiskLevel {
        if let Some(level) = self.check(action) {
            return level;
        }
        classify_action_heuristic(action)
    }

    /// Check whether an action is permitted for a given context.
    ///
    /// * `Safe` actions are always allowed.
    /// * `Confirm` actions are allowed when the context is elevated, otherwise
    ///   they require confirmation.
    /// * `AdminOnly` actions are allowed only for elevated contexts or an
    ///   `admin` user role, otherwise denied.
    pub fn check_permission(&self, action: &str, context: &PermissionContext) -> PermissionCheck {
        let risk = self.classify_action(action);
        let (required, reason) = match risk {
            RiskLevel::Safe => (
                PermissionAction::Allow,
                "Action is classified as safe".to_string(),
            ),
            RiskLevel::Confirm => {
                if context.elevated {
                    (
                        PermissionAction::Allow,
                        "Confirm-level action auto-allowed for elevated context".to_string(),
                    )
                } else {
                    (
                        PermissionAction::Confirm,
                        "Action requires user confirmation".to_string(),
                    )
                }
            }
            RiskLevel::AdminOnly => {
                let admin =
                    context.elevated || context.user_role.trim().eq_ignore_ascii_case("admin");
                if admin {
                    (
                        PermissionAction::Allow,
                        "Admin-only action allowed for privileged caller".to_string(),
                    )
                } else {
                    (
                        PermissionAction::Deny,
                        "Admin-only action denied for unprivileged caller".to_string(),
                    )
                }
            }
        };

        PermissionCheck {
            action: required,
            permission: action.to_string(),
            risk_level: risk,
            reason,
        }
    }

    /// Whether the given action requires user confirmation.
    pub fn require_confirmation(&self, action: &str) -> bool {
        self.classify_action(action) == RiskLevel::Confirm
    }

    /// Classify a whole scope to a risk level using the scope's most dangerous
    /// verb. Useful for policy defaults when the operator has not pinned a
    /// permission.
    pub fn classify_scope(&self, scope: PermissionScope) -> RiskLevel {
        match scope {
            PermissionScope::Tool => RiskLevel::Confirm,
            PermissionScope::File => RiskLevel::Confirm,
            PermissionScope::Network => RiskLevel::Safe,
            PermissionScope::Subprocess => RiskLevel::AdminOnly,
            PermissionScope::Config => RiskLevel::AdminOnly,
        }
    }

    /// Enforce a permission for a sandbox boundary: deny any action that is not
    /// at least `Allow`/`Confirm` for the context.
    ///
    /// This is the integration point sandbox policy layers call before
    /// executing an action.
    pub fn enforce(&self, action: &str, context: &PermissionContext) -> PermissionCheck {
        let check = self.check_permission(action, context);
        // Inside a sandbox, even a safe action is bounded by the caller's scopes.
        if context.sandboxed {
            let scope_token = action.split('.').next().unwrap_or("");
            if let Some(scope) = PermissionScope::parse(scope_token) {
                if !context.scopes.is_empty() && !context.scopes.contains(&scope) {
                    return PermissionCheck {
                        action: PermissionAction::Deny,
                        permission: action.to_string(),
                        risk_level: check.risk_level,
                        reason: format!(
                            "Action '{}' is outside the caller's permitted scopes",
                            action
                        ),
                    };
                }
            }
        }
        check
    }
}

/// Deterministic verb-based risk heuristic for unknown actions.
fn classify_action_heuristic(action: &str) -> RiskLevel {
    let lower = action.trim().to_lowercase();
    if lower.is_empty() {
        return RiskLevel::Confirm;
    }
    // Destructive operations are admin-only.
    if contains_any(
        &lower,
        &["delete", "remove", "drop", "truncate", "purge", "wipe"],
    ) {
        return RiskLevel::AdminOnly;
    }
    // Process/subprocess execution is admin-only.
    if contains_any(
        &lower,
        &[
            "execute",
            "exec",
            "spawn",
            "subprocess",
            "shell",
            "run_command",
        ],
    ) {
        return RiskLevel::AdminOnly;
    }
    // Configuration mutation is admin-only.
    if contains_any(
        &lower,
        &[
            "configure",
            "config.set",
            "install",
            "uninstall",
            "modify_config",
        ],
    ) {
        return RiskLevel::AdminOnly;
    }
    // Mutating / sending operations require confirmation.
    if contains_any(
        &lower,
        &[
            "write",
            "edit",
            "create",
            "append",
            "update",
            "send",
            "post",
            "upload",
            "delete_data",
        ],
    ) {
        return RiskLevel::Confirm;
    }
    // Pure reads and info are safe.
    if contains_any(
        &lower,
        &["read", "list", "get", "info", "query", "search", "status"],
    ) {
        return RiskLevel::Safe;
    }
    RiskLevel::Confirm
}

/// Case-insensitive substring check over a set of needles.
fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(*n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_permissions() {
        let matrix = PermissionMatrix::new();
        assert_eq!(matrix.check("file.read"), Some(RiskLevel::Safe));
        assert_eq!(matrix.check("file.write"), Some(RiskLevel::Confirm));
        assert_eq!(matrix.check("file.execute"), Some(RiskLevel::AdminOnly));
    }

    #[test]
    fn test_override() {
        let mut matrix = PermissionMatrix::new();
        assert_eq!(matrix.check("file.write"), Some(RiskLevel::Confirm));
        matrix.override_permission("file.write", RiskLevel::Safe);
        assert_eq!(matrix.check("file.write"), Some(RiskLevel::Safe));
    }

    #[test]
    fn test_unknown_permission() {
        let matrix = PermissionMatrix::new();
        assert_eq!(matrix.check("unknown.operation"), None);
    }

    #[test]
    fn test_categories() {
        let matrix = PermissionMatrix::new();
        let cats = matrix.categories();
        assert!(cats.contains(&"filesystem".to_string()));
        assert!(cats.contains(&"network".to_string()));
    }
}

#[cfg(test)]
mod expansion_tests {
    use super::*;

    #[test]
    fn test_classify_action_exact() {
        let matrix = PermissionMatrix::new();
        assert_eq!(matrix.classify_action("file.read"), RiskLevel::Safe);
        assert_eq!(matrix.classify_action("file.write"), RiskLevel::Confirm);
        assert_eq!(matrix.classify_action("file.execute"), RiskLevel::AdminOnly);
    }

    #[test]
    fn test_classify_action_heuristic() {
        let matrix = PermissionMatrix::new();
        assert_eq!(
            matrix.classify_action("file.delete_all"),
            RiskLevel::AdminOnly
        );
        assert_eq!(
            matrix.classify_action("network.send_message"),
            RiskLevel::Confirm
        );
        assert_eq!(
            matrix.classify_action("network.list_endpoints"),
            RiskLevel::Safe
        );
        assert_eq!(
            matrix.classify_action("subprocess.spawn"),
            RiskLevel::AdminOnly
        );
        assert_eq!(
            matrix.classify_action("completely.unknown.thing"),
            RiskLevel::Confirm
        );
    }

    #[test]
    fn test_check_permission_safe_allowed() {
        let matrix = PermissionMatrix::new();
        let check = matrix.check_permission("file.read", &PermissionContext::new());
        assert!(check.is_allowed());
        assert_eq!(check.risk_level, RiskLevel::Safe);
    }

    #[test]
    fn test_check_permission_confirm_needs_confirmation() {
        let matrix = PermissionMatrix::new();
        let check = matrix.check_permission("file.write", &PermissionContext::new());
        assert!(check.needs_confirmation());
    }

    #[test]
    fn test_check_permission_confirm_elevated_auto_allows() {
        let matrix = PermissionMatrix::new();
        let context = PermissionContext::new().elevated();
        let check = matrix.check_permission("file.write", &context);
        assert!(check.is_allowed());
    }

    #[test]
    fn test_check_permission_admin_only_denied_for_user() {
        let matrix = PermissionMatrix::new();
        let context = PermissionContext::new().with_role("user");
        let check = matrix.check_permission("file.execute", &context);
        assert!(check.is_denied());
    }

    #[test]
    fn test_check_permission_admin_only_allowed_for_admin() {
        let matrix = PermissionMatrix::new();
        let context = PermissionContext::new().with_role("admin");
        let check = matrix.check_permission("file.execute", &context);
        assert!(check.is_allowed());
    }

    #[test]
    fn test_require_confirmation() {
        let matrix = PermissionMatrix::new();
        assert!(matrix.require_confirmation("file.write"));
        assert!(!matrix.require_confirmation("file.read"));
    }

    #[test]
    fn test_enforce_scope_bounding() {
        let matrix = PermissionMatrix::new();
        // Sandboxed caller with only tool scopes cannot touch files.
        let context = PermissionContext::new()
            .in_sandbox()
            .with_scopes(vec![PermissionScope::Tool]);
        let check = matrix.enforce("file.read", &context);
        assert!(check.is_denied());
        assert!(check.reason.contains("outside"));
    }

    #[test]
    fn test_scope_parse_round_trip() {
        for scope in [
            PermissionScope::Tool,
            PermissionScope::File,
            PermissionScope::Network,
            PermissionScope::Subprocess,
            PermissionScope::Config,
        ] {
            assert_eq!(PermissionScope::parse(scope.as_str()), Some(scope));
        }
        assert_eq!(PermissionScope::parse("bogus"), None);
    }
}
