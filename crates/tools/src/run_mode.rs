//! Request-scoped sandbox run mode helpers.
//!
//! Mirrors the Python `opensquilla.tools.run_mode` module. The Rust crate has
//! no sandbox runtime dependency, so this module ports the environment levers
//! and the pure context-based classification. Callers that integrate a real
//! sandbox runtime can feed the sandbox-enabled state through
//! [`full_host_access_for_context`].

use crate::context::{RunMode, ToolContext};

/// The valid run modes.
pub const VALID_RUN_MODES: &[&str] = &["standard", "trusted", "full"];

/// Env lever: a configured-but-disabled sandbox grants Full Host Access.
const SANDBOX_DISABLED_FULL_HOST_ENV: &str = "OPENSQUILLA_SANDBOX_DISABLED_FULL_HOST";
/// Values that turn the fallback off.
const SANDBOX_DISABLED_FULL_HOST_OFF: &[&str] = &["0", "false", "no", "off", "disabled"];

/// Whether a configured-but-disabled sandbox implies Full Host Access.
///
/// On by default: a runtime configured with `sandbox=False` grants Full Host
/// Access semantics to every tool call. Embedded deployments that disable the
/// sandbox but still rely on the workspace policy layers can set
/// `OPENSQUILLA_SANDBOX_DISABLED_FULL_HOST=off` so run-mode semantics come
/// from the tool context alone. Reads fail safe to the default when the value
/// is unrecognized.
pub fn sandbox_disabled_full_host_fallback() -> bool {
    let raw = std::env::var(SANDBOX_DISABLED_FULL_HOST_ENV)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    !SANDBOX_DISABLED_FULL_HOST_OFF.contains(&raw.as_str())
}

/// Return Full Host Access state without consulting approval storage.
///
/// `sandbox_enabled` reports whether the runtime's sandbox is enabled:
/// - `Some(true)` — sandboxed; Full mode comes only from the context.
/// - `Some(false)` with the fallback lever on — Full Host Access.
/// - `Some(false)` with the fallback lever off — not Full.
/// - `None` — unknown runtime; falls back to the context alone.
pub fn full_host_access_for_context(
    ctx: &ToolContext,
    sandbox_enabled: Option<bool>,
) -> bool {
    if let Some(enabled) = sandbox_enabled {
        if !enabled {
            if sandbox_disabled_full_host_fallback() {
                return true;
            }
        }
    }
    ctx.full_host_access_active()
}

/// Return the active Standard/Trusted/Full mode for the scoped tool call.
///
/// Reads the task-scoped [`crate::context::ToolContext`] set by the dispatch
/// engine; `None` when no context is scoped (direct tool invocation).
pub fn current_run_mode() -> Option<RunMode> {
    crate::context::current_tool_context().and_then(|ctx| ctx.current_run_mode())
}

/// True when the current scoped tool call should use Full Host Access.
pub fn full_host_access_active() -> bool {
    crate::context::with_current_tool_context(|ctx| full_host_access_for_context(ctx, None))
        .unwrap_or(false)
}

/// True when the current scoped tool call is in Managed Execution mode.
pub fn trusted_sandbox_active() -> bool {
    let mode = current_run_mode();
    !full_host_access_active() && mode == Some(RunMode::Trusted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::RunMode;

    /// Serialize env-lever mutations so parallel tests do not race.
    fn with_env_var(value: Option<&str>, f: impl FnOnce()) {
        static MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = MUTEX.lock().unwrap();
        match value {
            Some(value) => unsafe {
                std::env::set_var(SANDBOX_DISABLED_FULL_HOST_ENV, value);
            },
            None => unsafe {
                std::env::remove_var(SANDBOX_DISABLED_FULL_HOST_ENV);
            },
        }
        f();
        unsafe {
            std::env::remove_var(SANDBOX_DISABLED_FULL_HOST_ENV);
        }
    }

    #[test]
    fn full_host_access_from_context_mode() {
        let mut ctx = ToolContext::owner();
        assert!(!full_host_access_for_context(&ctx, Some(true)));

        ctx.run_mode = Some(RunMode::Full);
        assert!(full_host_access_for_context(&ctx, Some(true)));
    }

    #[test]
    fn disabled_sandbox_with_fallback_implies_full() {
        with_env_var(None, || {
            let ctx = ToolContext::owner();
            // Fallback is on by default when the env is unset.
            assert!(full_host_access_for_context(&ctx, Some(false)));
        });
    }

    #[test]
    fn disabled_sandbox_without_fallback_stays_context_only() {
        with_env_var(Some("off"), || {
            let mut ctx = ToolContext::owner();
            assert!(!full_host_access_for_context(&ctx, Some(false)));

            ctx.run_mode = Some(RunMode::Full);
            assert!(full_host_access_for_context(&ctx, Some(false)));
        });
    }

    #[test]
    fn unrecognized_fallback_value_fails_safe_to_default() {
        with_env_var(Some("banana"), || {
            assert!(sandbox_disabled_full_host_fallback());
        });
    }

    #[test]
    fn fallback_env_parsing() {
        with_env_var(Some("off"), || {
            assert!(!sandbox_disabled_full_host_fallback());
        });
        with_env_var(Some("FALSE"), || {
            assert!(!sandbox_disabled_full_host_fallback());
        });
        with_env_var(None, || {
            assert!(sandbox_disabled_full_host_fallback());
        });
    }

    #[tokio::test]
    async fn scoped_run_mode_helpers() {
        let mut ctx = ToolContext::owner();
        assert_eq!(
            crate::context::run_with_tool_context(Some(ctx.clone()), async {
                current_run_mode()
            })
            .await,
            None
        );
        ctx.run_mode = Some(RunMode::Full);
        let (mode, full, trusted) = crate::context::run_with_tool_context(
            Some(ctx.clone()),
            async { (current_run_mode(), full_host_access_active(), trusted_sandbox_active()) },
        )
        .await;
        assert_eq!(mode, Some(RunMode::Full));
        assert!(full);
        assert!(!trusted);

        let mut trusted_ctx = ToolContext::owner();
        trusted_ctx.run_mode = Some(RunMode::Trusted);
        let (full, trusted) = crate::context::run_with_tool_context(
            Some(trusted_ctx),
            async { (full_host_access_active(), trusted_sandbox_active()) },
        )
        .await;
        assert!(!full);
        assert!(trusted);

        // Outside a scoped block the helpers are inactive.
        assert_eq!(current_run_mode(), None);
        assert!(!full_host_access_active());
    }
}
