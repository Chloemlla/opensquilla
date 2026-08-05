//! Input stage.
//!
//! Mirrors the Python `engine/turn_runner/input_stage.py` stage. It runs once
//! per turn, before provider resolution. It:
//!
//! * validates that the turn carries a user (or internal system-event) input,
//! * normalizes `system_event` input mode by prefixing an internal-event marker
//!   and supplying extra prompt context,
//! * enforces the message-count and per-input character caps,
//! * sanitizes the user input (control characters, excessive whitespace, length
//!   truncation),
//! * detects prompt-injection patterns (via `opensquilla-safety` when the
//!   `safety` feature is enabled),
//! * detects slash commands against the configured command registry,
//! * prepares the message list for the provider (coalescing trailing user
//!   messages so the model sees a single final instruction).

use crate::agent::TurnGenerator;
use crate::commands::CommandRegistry;
use crate::stages::{Stage, StageContext, StageError, StageOutput};
use async_trait::async_trait;
use opensquilla_core::error::Result;
use opensquilla_core::types::{ContentBlock, Message, MessageRole};
use tracing::{debug, info, instrument};

/// Input mode for the turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    /// A normal human user message.
    #[default]
    User,
    /// An internal scheduler/system event, not a human user.
    SystemEvent,
}

/// Configuration for the input stage.
#[derive(Debug, Clone)]
pub struct InputConfig {
    /// Maximum number of messages allowed in a single turn.
    pub max_messages: usize,
    /// Maximum number of characters a single user input may carry before it is
    /// truncated.
    pub max_input_chars: usize,
    /// Whether prompt-injection detection is enabled.
    pub injection_detection_enabled: bool,
    /// Whether slash-command detection is enabled.
    pub command_detection_enabled: bool,
    /// Whether control characters are stripped from the user input.
    pub sanitize_control_chars: bool,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_input_chars: 100_000,
            injection_detection_enabled: true,
            command_detection_enabled: true,
            sanitize_control_chars: true,
        }
    }
}

/// A per-turn report of what the input stage observed.
#[derive(Debug, Clone, Default)]
pub struct InputReport {
    /// The resolved input mode.
    pub mode: InputMode,
    /// Whether the input text was modified by sanitization.
    pub sanitized: bool,
    /// Whether a prompt-injection pattern was detected.
    pub injection_detected: bool,
    /// The severity of the highest-severity injection match, if any.
    pub injection_severity: Option<String>,
    /// The matching injection pattern name, if any.
    pub injection_pattern: Option<String>,
    /// Slash commands detected in the input.
    pub commands: Vec<String>,
    /// The message count after preparation.
    pub message_count: usize,
}

/// The input stage in the turn pipeline.
#[derive(Debug)]
pub struct InputStage {
    config: InputConfig,
    /// Prompt-injection guard (only compiled with the `safety` feature).
    #[cfg(feature = "safety")]
    injection_guard: opensquilla_safety::InjectionGuard,
    /// The slash-command registry used for command detection.
    commands: Option<CommandRegistry>,
    /// The most recent per-turn report.
    last_report: std::sync::Mutex<Option<InputReport>>,
}

impl InputStage {
    /// Create a new input stage with the given message cap.
    pub fn new(max_messages: usize) -> Self {
        Self {
            config: InputConfig {
                max_messages: max_messages.max(1),
                ..Default::default()
            },
            #[cfg(feature = "safety")]
            injection_guard: opensquilla_safety::InjectionGuard::new(),
            commands: None,
            last_report: std::sync::Mutex::new(None),
        }
    }

    /// Replace the stage configuration.
    pub fn with_config(mut self, config: InputConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the per-input character cap.
    pub fn with_max_chars(mut self, max: usize) -> Self {
        self.config.max_input_chars = max.max(1);
        self
    }

    /// Enable or disable prompt-injection detection.
    pub fn with_injection_detection(mut self, enabled: bool) -> Self {
        self.config.injection_detection_enabled = enabled;
        self
    }

    /// Enable or disable slash-command detection.
    pub fn with_command_detection(mut self, enabled: bool) -> Self {
        self.config.command_detection_enabled = enabled;
        self
    }

    /// Attach a slash-command registry used for command detection.
    pub fn with_commands(mut self, commands: CommandRegistry) -> Self {
        self.commands = Some(commands);
        self
    }

    /// The stage configuration.
    pub fn config(&self) -> &InputConfig {
        &self.config
    }

    /// The report from the most recent execution, if any.
    pub fn last_report(&self) -> Option<InputReport> {
        self.last_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Sanitize a raw user input string.
    ///
    /// Replaces control characters (except `\n` and `\t`) with spaces when
    /// configured, collapses runs of three or more blank lines down to two,
    /// trims surrounding whitespace, and truncates to the configured character
    /// cap. Returns the sanitized text and whether it was modified.
    pub fn sanitize(&self, text: &str) -> (String, bool) {
        let mut modified = false;
        let mut out = String::with_capacity(text.len());
        let mut newline_run = 0u32;
        for ch in text.chars() {
            if self.config.sanitize_control_chars && ch.is_control() && ch != '\n' && ch != '\t' {
                out.push(' ');
                newline_run = 0;
                modified = true;
                continue;
            }
            if ch == '\n' {
                newline_run += 1;
                if newline_run > 2 {
                    modified = true;
                    continue;
                }
                out.push('\n');
            } else {
                newline_run = 0;
                out.push(ch);
            }
        }
        let trimmed = out.trim();
        if trimmed.len() != out.len() {
            modified = true;
        }
        let mut final_text = trimmed.to_string();
        if final_text.chars().count() > self.config.max_input_chars {
            let mut truncated = String::new();
            for (i, c) in final_text.char_indices() {
                if i >= self.config.max_input_chars {
                    modified = true;
                    break;
                }
                truncated.push(c);
            }
            final_text = truncated;
        }
        (final_text, modified)
    }

    /// Detect prompt-injection patterns in the input.
    ///
    /// Returns `None` when detection is disabled or nothing matched.
    #[cfg(feature = "safety")]
    pub fn detect_injection(&self, text: &str) -> Option<opensquilla_safety::InjectionResult> {
        if !self.config.injection_detection_enabled {
            return None;
        }
        let result = self.injection_guard.scan(text);
        if result.detected { Some(result) } else { None }
    }

    /// Detect prompt-injection patterns in the input.
    ///
    /// This fallback compiles to a harmless no-op when the `safety` feature is
    /// disabled.
    #[cfg(not(feature = "safety"))]
    pub fn detect_injection(&self, _text: &str) -> Option<InjectionFallback> {
        None
    }

    /// Detect slash commands in the input.
    ///
    /// Returns the canonical names of every command that the leading token of
    /// the input resolves to.
    pub fn detect_commands(&self, text: &str) -> Vec<String> {
        if !self.config.command_detection_enabled {
            return Vec::new();
        }
        let Some(registry) = &self.commands else {
            return Vec::new();
        };
        let first_token = text.split_whitespace().next().unwrap_or_default();
        match registry.resolve(first_token) {
            Some(command) => vec![command.name.clone()],
            None => Vec::new(),
        }
    }

    /// Normalize the input message text for the given mode.
    ///
    /// For `SystemEvent` the message is wrapped in an internal-event envelope
    /// and a guidance fragment is returned for prompt assembly.
    pub fn normalize_input(&self, message: &str, mode: InputMode) -> (String, Option<String>) {
        match mode {
            InputMode::User => (message.to_string(), None),
            InputMode::SystemEvent => {
                let runtime = format!("[INTERNAL SYSTEM EVENT]\n{message}");
                let extra = "The next input is an internal scheduler event, not a human \
                             user message. Treat it as system-originated context."
                    .to_string();
                (runtime, Some(extra))
            }
        }
    }

    /// Validate the turn's message list, returning an error on violation.
    fn validate(&self, ctx: &StageContext) -> Option<StageError> {
        if ctx.messages.is_empty() {
            return Some(StageError {
                message: "No messages provided for the turn".to_string(),
                code: Some("EMPTY_MESSAGES".to_string()),
                stage: self.name().to_string(),
            });
        }
        if ctx.messages.len() > self.config.max_messages {
            return Some(StageError {
                message: format!(
                    "Message count {} exceeds maximum of {}",
                    ctx.messages.len(),
                    self.config.max_messages
                ),
                code: Some("MESSAGE_LIMIT".to_string()),
                stage: self.name().to_string(),
            });
        }
        None
    }

    /// Prepare the message list for the provider.
    ///
    /// Coalesces trailing consecutive user messages so the model sees a single
    /// final instruction; the turn's last message must be a user message for a
    /// `User`-mode turn.
    pub fn prepare_messages(&self, messages: Vec<Message>, mode: InputMode) -> Vec<Message> {
        if mode == InputMode::User && messages.is_empty() {
            return messages;
        }
        // Coalesce trailing user messages.
        let mut out = messages;
        while out.len() > 1 {
            let n = out.len();
            if out[n - 1].role == MessageRole::User && out[n - 2].role == MessageRole::User {
                let last = out.pop().unwrap();
                let prev = out.pop().unwrap();
                let combined = format!("{}\n\n{}", prev.text_content(), last.text_content());
                out.push(Message {
                    role: MessageRole::User,
                    content: vec![ContentBlock::Text(combined)],
                    name: None,
                    tool_call_id: None,
                    tool_calls: None,
                    tool_result: None,
                });
            } else {
                break;
            }
        }
        out
    }

    /// Sanitize and re-serialize the latest user message in place.
    ///
    /// Returns `true` when the message text was modified.
    fn sanitize_latest_user_message(&self, messages: &mut Vec<Message>) -> bool {
        let Some(pos) = messages.iter().rposition(|m| m.role == MessageRole::User) else {
            return false;
        };
        let msg = &mut messages[pos];
        let text = msg.text_content();
        let (clean, modified) = self.sanitize(&text);
        if !modified {
            return false;
        }
        // Rebuild the message content: replace the first text block with the
        // sanitized text and drop trailing empty text blocks.
        let mut cleaned_blocks: Vec<ContentBlock> = Vec::new();
        let mut replaced_text = false;
        for block in msg.content.clone() {
            match block {
                ContentBlock::Text(_) => {
                    if !replaced_text {
                        if !clean.is_empty() {
                            cleaned_blocks.push(ContentBlock::Text(clean.clone()));
                        }
                        replaced_text = true;
                    }
                    // subsequent text blocks are dropped
                }
                other => cleaned_blocks.push(other),
            }
        }
        if !replaced_text && !clean.is_empty() {
            cleaned_blocks.push(ContentBlock::Text(clean));
        }
        msg.content = cleaned_blocks;
        true
    }
}

impl Default for InputStage {
    fn default() -> Self {
        Self::new(100)
    }
}

#[async_trait]
impl Stage for InputStage {
    #[instrument(skip(self), fields(stage = %self.name()))]
    async fn execute(
        &self,
        ctx: &mut StageContext,
        _generator: &dyn TurnGenerator,
    ) -> Result<StageOutput> {
        debug!("input: validating and preparing input messages");

        if let Some(err) = self.validate(ctx) {
            return Ok(StageOutput::Error(err));
        }

        // Sanitize the latest user message.
        let sanitized = self.sanitize_latest_user_message(&mut ctx.messages);

        // Detect system-event turns: the final user message carries the
        // internal-event marker. In this engine the mode is inferred from the
        // message prefix; the harness may set it explicitly via the provider.
        let mode = if ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content().starts_with("[INTERNAL SYSTEM EVENT]"))
            .unwrap_or(false)
        {
            InputMode::SystemEvent
        } else {
            InputMode::User
        };

        // Detect prompt injection in the latest user input.
        let input_text = ctx
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.text_content())
            .unwrap_or_default();

        #[cfg(feature = "safety")]
        let (injection_detected, injection_severity, injection_pattern) =
            if let Some(result) = self.detect_injection(&input_text) {
                tracing::warn!(
                    turn_id = %ctx.turn_id,
                    severity = ?result.severity,
                    pattern = ?result.pattern,
                    "input stage: prompt-injection pattern detected"
                );
                (
                    true,
                    Some(format!("{:?}", result.severity)),
                    result.pattern.clone(),
                )
            } else {
                (false, None, None)
            };

        #[cfg(not(feature = "safety"))]
        let (injection_detected, injection_severity, injection_pattern) = (false, None, None);

        // Detect slash commands.
        let commands = self.detect_commands(&input_text);

        ctx.messages = self.prepare_messages(ctx.messages.clone(), mode);

        let command_count = commands.len();
        let report = InputReport {
            mode,
            sanitized,
            injection_detected,
            injection_severity,
            injection_pattern,
            commands,
            message_count: ctx.messages.len(),
        };
        *self.last_report.lock().unwrap_or_else(|e| e.into_inner()) = Some(report);

        info!(
            turn_id = %ctx.turn_id,
            message_count = ctx.messages.len(),
            mode = ?mode,
            sanitized = sanitized,
            injection_detected = injection_detected,
            commands = command_count,
            "input stage complete"
        );

        Ok(StageOutput::Continue)
    }

    fn name(&self) -> &str {
        "input"
    }
}

/// A type-only fallback for the non-`safety` build so the public
/// `detect_injection` signature stays uniform across feature sets.
#[cfg(not(feature = "safety"))]
#[derive(Debug, Clone)]
pub struct InjectionFallback;

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::Usage;

    fn context(messages: Vec<Message>) -> StageContext {
        StageContext {
            turn_id: "t1".to_string(),
            messages,
            current_model: String::new(),
            current_provider: String::new(),
            usage: Usage::default(),
            streaming_tx: None,
            tool_round: 0,
            max_tool_rounds: 10,
        }
    }

    #[test]
    fn test_normalize_user() {
        let stage = InputStage::new(100);
        let (runtime, extra) = stage.normalize_input("hello", InputMode::User);
        assert_eq!(runtime, "hello");
        assert!(extra.is_none());
    }

    #[test]
    fn test_normalize_system_event() {
        let stage = InputStage::new(100);
        let (runtime, extra) = stage.normalize_input("scheduler tick", InputMode::SystemEvent);
        assert!(runtime.starts_with("[INTERNAL SYSTEM EVENT]"));
        assert!(extra.is_some());
    }

    #[test]
    fn test_sanitize_strips_control_chars() {
        let stage = InputStage::new(100);
        let (clean, modified) = stage.sanitize("hello\x00world\x01");
        assert!(modified);
        assert_eq!(clean, "hello world");
    }

    #[test]
    fn test_sanitize_truncates() {
        let stage = InputStage::new(100).with_max_chars(5);
        let (clean, modified) = stage.sanitize("abcdefghij");
        assert!(modified);
        assert_eq!(clean, "abcde");
    }

    #[test]
    fn test_sanitize_collapses_blank_lines() {
        let stage = InputStage::new(100);
        let (clean, modified) = stage.sanitize("a\n\n\n\n\nb");
        assert!(modified);
        assert_eq!(clean, "a\n\nb");
    }

    #[test]
    fn test_prepare_coalesces_trailing_users() {
        let stage = InputStage::new(100);
        let msgs = vec![
            Message::user("first"),
            Message::assistant("response"),
            Message::user("second"),
            Message::user("third"),
        ];
        let out = stage.prepare_messages(msgs, InputMode::User);
        assert_eq!(out.len(), 3);
        let last = out.last().unwrap();
        assert_eq!(last.text_content(), "second\n\nthird");
    }

    #[test]
    fn test_sanitize_latest_user_message() {
        let stage = InputStage::new(100);
        let mut msgs = vec![Message::user("hi"), Message::user("clean\x00text")];
        let modified = stage.sanitize_latest_user_message(&mut msgs);
        assert!(modified);
        let last = msgs.last().unwrap();
        assert_eq!(last.text_content(), "clean text");
    }

    #[test]
    fn test_validate_empty() {
        let stage = InputStage::new(100);
        let ctx = context(Vec::new());
        let err = stage.validate(&ctx);
        assert!(err.is_some());
        assert_eq!(err.unwrap().code.as_deref(), Some("EMPTY_MESSAGES"));
    }

    #[test]
    fn test_validate_message_limit() {
        let stage = InputStage::new(2);
        let ctx = context(vec![
            Message::system("s"),
            Message::user("u1"),
            Message::user("u2"),
        ]);
        let err = stage.validate(&ctx);
        assert!(err.is_some());
        assert_eq!(err.unwrap().code.as_deref(), Some("MESSAGE_LIMIT"));
    }

    #[tokio::test]
    async fn test_execute_runs_end_to_end() {
        let stage = InputStage::new(100);
        let mut ctx = context(vec![Message::user("hello world")]);
        let generator = MockGenerator;
        let out = stage.execute(&mut ctx, &generator).await.unwrap();
        assert!(matches!(out, StageOutput::Continue));
        let report = stage.last_report().unwrap();
        assert_eq!(report.mode, InputMode::User);
        assert_eq!(report.message_count, 1);
    }

    #[derive(Debug)]
    struct MockGenerator;
    #[async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
            Ok(vec![Message::assistant("ok")])
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn provider_name(&self) -> &str {
            "mock"
        }
    }
}
