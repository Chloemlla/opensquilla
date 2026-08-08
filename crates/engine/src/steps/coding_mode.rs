//! Coding-mode enforcement step.
//!
//! Mirrors the Python `engine/steps/coding_mode.py` step. When the operator
//! toggle is ON and the required code-task launch tools are available, the
//! step injects a directive into the system prompt that steers code changes
//! through the code-task plugin instead of letting the agent hand-edit files.
//!
//! Coding mode is an operator toggle, not an intent classifier. No per-message
//! detection is performed. Restricted surfaces (missing launch tools) receive
//! neither the directive nor the skill.

use crate::pipeline::PipelineContext;
use crate::steps::{PipelineStep, StepAction};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{Message, MessageRole};
use tracing::{debug, info, instrument};

/// Tools required to launch code-task.
const CODE_TASK_REQUIRED_TOOLS: &[&str] = &["background_process", "exec_command", "process"];

/// Configuration for the coding-mode step.
#[derive(Debug, Clone, Default)]
pub struct CodingModeConfig {
    /// Master switch. When false the step is a complete no-op.
    pub enabled: bool,
    /// A resolved, PATH-independent code-task command prefix. When `None` the
    /// step emits the "unavailable" directive.
    pub code_task_command: Option<String>,
}

/// Pre-turn pipeline step that enforces coding mode.
#[derive(Debug)]
pub struct CodingModeStep {
    config: CodingModeConfig,
    /// Names of tools available to the agent this turn.
    available_tools: Vec<String>,
}

impl CodingModeStep {
    /// Create a new step with default configuration (disabled).
    pub fn new() -> Self {
        Self {
            config: CodingModeConfig::default(),
            available_tools: Vec::new(),
        }
    }

    /// Create a step from a full configuration.
    pub fn with_config(config: CodingModeConfig) -> Self {
        Self {
            config,
            available_tools: Vec::new(),
        }
    }

    /// Register the set of tools available this turn.
    pub fn with_available_tools(mut self, tools: Vec<String>) -> Self {
        self.available_tools = tools;
        self
    }

    /// Return the required code-task tools that are absent from the available
    /// tool surface.
    fn missing_tools(&self) -> Vec<&'static str> {
        CODE_TASK_REQUIRED_TOOLS
            .iter()
            .filter(|t| !self.available_tools.iter().any(|a| a == *t))
            .copied()
            .collect()
    }

    /// Build the coding-mode directive text, substituting the resolved
    /// code-task command or emitting the unavailable directive.
    fn build_directive(&self) -> String {
        match &self.config.code_task_command {
            None => CODING_MODE_UNAVAILABLE_DIRECTIVE.to_string(),
            Some(cmd) => CODING_MODE_DIRECTIVE_TEMPLATE.replace("__CODE_TASK_CMD__", cmd),
        }
    }
}

impl Default for CodingModeStep {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PipelineStep for CodingModeStep {
    #[instrument(skip(self), fields(step = %self.name()))]
    async fn execute(&self, ctx: &mut PipelineContext) -> Result<StepAction> {
        if !self.config.enabled {
            debug!("coding_mode disabled, skipping");
            return Ok(StepAction::Continue);
        }

        let missing = self.missing_tools();
        if !missing.is_empty() {
            // Do not leave a conflicting, mandatory code-task instruction on
            // restricted callers such as ordinary channel users.
            ctx.set_metadata("enforce_coding_mode__applied", "false");
            info!(missing = ?missing, "coding_mode.skipped");
            return Ok(StepAction::Continue);
        }

        let directive = self.build_directive();

        // Append the directive to the last system message, or insert a fresh
        // system message when none exists.
        if let Some(idx) = ctx
            .messages
            .iter()
            .rposition(|m| m.role == MessageRole::System)
        {
            let existing = ctx.messages[idx].text_content();
            let combined = if existing.is_empty() {
                directive.clone()
            } else {
                format!("{existing}{directive}")
            };
            ctx.messages[idx] = Message::text(MessageRole::System, combined);
        } else {
            ctx.add_message(Message::system(directive));
        }

        // Pin code-task so a relevance filter cannot drop it from
        // <available_skills>.
        let pinned = ctx
            .get_metadata("pinned_skills")
            .cloned()
            .unwrap_or_default();
        let mut pinned_set: Vec<String> = if pinned.is_empty() {
            Vec::new()
        } else {
            pinned.split(',').map(|s| s.trim().to_string()).collect()
        };
        if !pinned_set.iter().any(|s| s == "code-task") {
            pinned_set.push("code-task".to_string());
        }
        ctx.set_metadata("pinned_skills", pinned_set.join(","));
        ctx.set_metadata("coding_mode", "true");
        info!("coding_mode.enforced");

        Ok(StepAction::Continue)
    }

    fn name(&self) -> &str {
        "coding_mode"
    }
}

const CODING_MODE_DIRECTIVE_TEMPLATE: &str = "\n\n[CODING MODE - ACTIVE]\n\
The operator has enabled coding mode for this session. For ANY request to \
WRITE or CHANGE code you MUST do the work THROUGH code-task - NEVER by \
typing the code directly in your reply, and never by hand-editing files. \
Choose the matching case. (1) The user NAMES a real repository (a \
filesystem path or a git URL) - fix a bug, add/implement a feature, edit \
a file, resolve a GitHub issue - run\n\
    __CODE_TASK_CMD__ solve --repo <url-or-path> \
(--issue N | --task-file <path>) --shallow --yes\n\
Use that command EXACTLY as written above - it is resolved to run in THIS \
environment regardless of PATH. Do NOT replace it with a bare \
`opensquilla`, do NOT `pip install` OpenSquilla, and if that command fails \
to run, STOP and report that code-task / the environment is broken instead \
of working around it by hand.\n\
For building an app or UI from scratch (e.g. an Electron + React \
desktop app), add --verification-mode build with NO --repo (code-task \
scaffolds the new app's workspace itself) so code-task verifies the \
app compiles and packages instead of running red->green tests.\n\
(2) The user asks for SELF-CONTAINED, TESTABLE code FROM SCRATCH and names \
NO repo (e.g. 'write a python function that maps A-Z to pitches', 'write a \
script that parses a log'): this STILL goes through code-task - do NOT \
answer by typing the code in your reply - run\n\
    __CODE_TASK_CMD__ solve --task-file <path> --verification-mode \
scratch --yes\n\
(no --repo); code-task scaffolds a throwaway project, writes the code plus a \
test, and verifies it green. ONLY answer inline (no code-task) for trivial \
one-liners, pseudocode, or conceptual / non-deterministic / GUI- or \
network-dependent requests that cannot be expressed as a quick automated \
test.\n\
code-task runs for MANY minutes (often 20-40, up to ~90 on a heavy repo \
that must install dependencies), so it is a long-running task: ALWAYS \
launch it with background_process(timeout=5400) and then await it with \
process(action=\"wait\", session_id=..., timeout=5400). Do NOT run \
code-task with a blocking exec_command. Do not poll \
process(action=\"poll\") in a loop either - just wait for the result.\n\
Do NOT clone the repository yourself and do NOT hand-edit its files in \
this session: the file-editing tools (write_file, edit_file, apply_patch, \
execute_code, git_commit, create_*) are DISABLED while coding mode is on. \
Read-only requests (showing structure, explaining code) and ordinary \
conversation are answered normally.";

const CODING_MODE_UNAVAILABLE_DIRECTIVE: &str = "\n\n[CODING MODE - ACTIVE, but code-task is UNAVAILABLE]\n\
The operator enabled coding mode, which requires every code change to go \
through the code-task plugin - but `opensquilla code-task` cannot be run \
in this environment (the OpenSquilla CLI is not installed or not runnable \
here). For ANY request that would change code, STOP and tell the user that \
code-task is unavailable and the environment must be fixed. \
Do NOT try to `pip install` OpenSquilla yourself, do NOT clone the \
repository, and do NOT hand-edit files via the shell as a workaround - the \
in-session file-editing tools are disabled and a manual workaround skips \
code-task's isolation and verification. Read-only requests (showing \
structure, explaining code) and ordinary conversation are answered \
normally.";

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Message;

    #[tokio::test]
    async fn test_disabled_is_noop() {
        let step = CodingModeStep::new();
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert!(ctx.get_metadata("coding_mode").is_none());
    }

    #[tokio::test]
    async fn test_missing_tools_skips() {
        let step = CodingModeStep::with_config(CodingModeConfig {
            enabled: true,
            code_task_command: Some("opensquilla code-task".into()),
        });
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        let action = step.execute(&mut ctx).await.unwrap();
        assert!(matches!(action, StepAction::Continue));
        assert_eq!(
            ctx.get_metadata("enforce_coding_mode__applied")
                .map(String::as_str),
            Some("false")
        );
    }

    #[tokio::test]
    async fn test_injects_directive() {
        let step = CodingModeStep::with_config(CodingModeConfig {
            enabled: true,
            code_task_command: Some("opensquilla code-task".into()),
        })
        .with_available_tools(
            CODE_TASK_REQUIRED_TOOLS
                .iter()
                .map(|t| t.to_string())
                .collect(),
        );
        let mut ctx = PipelineContext::new("t1".into(), vec![Message::user("hi")]);
        step.execute(&mut ctx).await.unwrap();
        assert_eq!(
            ctx.get_metadata("coding_mode").map(String::as_str),
            Some("true")
        );
        assert!(
            ctx.messages
                .iter()
                .any(|m| m.text_content().contains("[CODING MODE"))
        );
    }
}
