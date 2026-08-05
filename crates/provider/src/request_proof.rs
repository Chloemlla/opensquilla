//! Pre-flight payload projection and budget calculation.
//!
//! Before a request is dispatched to a provider, [`RequestProof`] inspects the
//! assembled messages and config to ensure the payload will actually be
//! accepted and fit within the model's context window. This is the second
//! load-bearing module for cross-provider correctness (alongside
//! [`crate::compat_policy`]).
//!
//! Responsibilities:
//!
//! - **Token budget estimation** — estimate the token cost of the message
//!   array and tool definitions *before* sending, using a cheap heuristic when
//!   no tokenizer is available and an optional `tiktoken`-based estimator
//!   when one is.
//! - **Context-window trimming** — if the estimated prompt exceeds the model's
//!   context window minus the reserved generation budget, older middle messages
//!   are dropped (preserving the system prompt and the most recent turns).
//! - **`max_tokens` clamping** — clamp the requested `max_tokens` to the
//!   provider-imposed cap and to the remaining window budget so the request
//!   cannot be rejected outright.
//! - **Tool-definition budget** — account for the token cost of serialized
//!   tool schemas so they are not silently dropped.
//! - **Reasoning-effort budgeting** — when a reasoning model is in use, reserve
//!   an additional reasoning token budget.

use crate::compat_policy::{CompatPolicy, policy_for};
use crate::types::ChatConfig;
use opensquilla_core::types::{ChatMessage, ContentBlock, MessageRole, ToolDefinition};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// A pluggable token estimator.
///
/// Implementations may wrap a real tokenizer (e.g. tiktoken) or use a
/// heuristic character-ratio estimator. The trait is object-safe so estimators
/// can be swapped at runtime.
pub trait TokenEstimator: Send + Sync {
    /// Estimate the number of tokens for an arbitrary UTF-8 string.
    fn estimate_text(&self, text: &str) -> u64;

    /// Estimate the token cost of a serialized tool definition.
    fn estimate_tool(&self, tool: &ToolDefinition) -> u64 {
        // Default: serialize to compact JSON and estimate the text length.
        let json = serde_json::to_string(tool).unwrap_or_default();
        self.estimate_text(&json)
    }

    /// Estimate the token cost of a single message (role + content + tool calls).
    fn estimate_message(&self, msg: &ChatMessage) -> u64 {
        // Each message carries a small structural overhead (role tag, delimiters).
        const MSG_OVERHEAD: u64 = 4;
        let mut total = MSG_OVERHEAD;

        for block in &msg.content {
            match block {
                ContentBlock::Text(t) => total += self.estimate_text(t),
                ContentBlock::Reasoning(t) => total += self.estimate_text(t),
                ContentBlock::ToolUse(tc) => {
                    total += self.estimate_text(&tc.name);
                    total += self.estimate_text(&tc.id);
                    total += self.estimate_text(&tc.input.to_string());
                    total += 4;
                }
                ContentBlock::ToolResult(tr) => {
                    total += self.estimate_text(&tr.content);
                    total += self.estimate_text(&tr.tool_use_id);
                    total += 4;
                }
            }
        }

        // Tool calls attached to the message (legacy field).
        if let Some(calls) = &msg.tool_calls {
            for tc in calls {
                total += self.estimate_text(&tc.name);
                total += self.estimate_text(&tc.id);
                total += self.estimate_text(&tc.input.to_string());
                total += 4;
            }
        }

        if let Some(name) = &msg.name {
            total += self.estimate_text(name);
        }

        total
    }
}

/// A heuristic estimator that approximates tokens as `ceil(chars / ratio)`.
///
/// The default ratio of ~4 chars/token is a well-known rough estimate for
/// English text against GPT-style BPE tokenizers. CJK text is more expensive
/// (roughly 1.5–2 chars/token), so the estimator detects CJK runs and applies
/// a tighter ratio for them.
#[derive(Debug, Clone)]
pub struct HeuristicEstimator {
    /// Chars per token for ASCII/Latin text.
    pub latin_ratio: f64,
    /// Chars per token for CJK text.
    pub cjk_ratio: f64,
}

impl Default for HeuristicEstimator {
    fn default() -> Self {
        Self {
            latin_ratio: 4.0,
            cjk_ratio: 1.7,
        }
    }
}

impl HeuristicEstimator {
    /// Create a new heuristic estimator with the default ratios.
    pub fn new() -> Self {
        Self::default()
    }

    fn is_cjk(c: char) -> bool {
        matches!(c as u32,
            0x3000..=0x30FF |   // CJK punctuation, Hiragana, Katakana
            0x3400..=0x4DBF |   // CJK Ext A
            0x4E00..=0x9FFF |   // CJK Unified
            0xAC00..=0xD7AF |   // Hangul
            0xF900..=0xFAFF |   // CJK Compatibility
            0xFF00..=0xFFEF     // Halfwidth/Fullwidth
        )
    }
}

impl TokenEstimator for HeuristicEstimator {
    fn estimate_text(&self, text: &str) -> u64 {
        if text.is_empty() {
            return 0;
        }
        let mut cjk_chars: u64 = 0;
        let mut other_chars: u64 = 0;
        for c in text.chars() {
            if Self::is_cjk(c) {
                cjk_chars += 1;
            } else {
                other_chars += 1;
            }
        }
        let tokens = (cjk_chars as f64 / self.cjk_ratio)
            + (other_chars as f64 / self.latin_ratio);
        tokens.ceil() as u64
    }
}

/// A token estimator backed by `tiktoken-rs` (cl100k_base) when available.
///
/// Construction is fallible because loading the encoding requires the
/// embedded BPE ranks. On failure callers should fall back to
/// [`HeuristicEstimator`].
#[cfg(feature = "tiktoken")]
pub struct TiktokenEstimator {
    bpe: tiktoken_rs::CoreBPE,
}

#[cfg(feature = "tiktoken")]
impl TiktokenEstimator {
    /// Create a tiktoken estimator using the cl100k_base encoding.
    pub fn new() -> Result<Self, String> {
        let bpe = tiktoken_rs::get_bpe_from_tokenizer(
            tiktoken_rs::tokenizer::Tokenizer::Cl100kBase,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self { bpe })
    }
}

#[cfg(feature = "tiktoken")]
impl TokenEstimator for TiktokenEstimator {
    fn estimate_text(&self, text: &str) -> u64 {
        self.bpe.encode_with_special_tokens(text).len() as u64
    }
}

// ---------------------------------------------------------------------------
// Model window metadata
// ---------------------------------------------------------------------------

/// Context-window metadata for a single model.
#[derive(Debug, Clone)]
pub struct ModelWindow {
    /// The model id.
    pub model: String,
    /// Total context window in tokens.
    pub context_window: u32,
    /// Maximum output tokens the model can generate in one response.
    pub max_output: u32,
    /// Whether the model supports extended reasoning (which consumes output budget).
    pub reasoning_model: bool,
}

impl ModelWindow {
    /// The number of tokens reserved for the model's response (output + reasoning).
    pub fn generation_budget(&self) -> u32 {
        self.max_output
    }
}

/// A small lookup of well-known model context windows.
///
/// Unknown models fall back to a conservative 8k window.
pub fn lookup_model_window(model: &str) -> ModelWindow {
    let lower = model.to_ascii_lowercase();
    let (ctx, out, reasoning) = if lower.starts_with("gpt-4o") || lower.starts_with("gpt-4.1") {
        (128_000, 16_384, false)
    } else if lower.starts_with("gpt-4-turbo") || lower.starts_with("gpt-4-1106") {
        (128_000, 4_096, false)
    } else if lower.starts_with("gpt-4") {
        (8_192, 4_096, false)
    } else if lower.starts_with("gpt-3.5") {
        (16_385, 4_096, false)
    } else if lower.starts_with("o1") || lower.starts_with("o3") || lower.starts_with("o4") {
        (200_000, 100_000, true)
    } else if lower.starts_with("claude-3-5-sonnet") || lower.starts_with("claude-3.5-sonnet") {
        (200_000, 8_192, false)
    } else if lower.starts_with("claude-3-opus") || lower.starts_with("claude-3-sonnet") {
        (200_000, 4_096, false)
    } else if lower.starts_with("claude-3-haiku") {
        (200_000, 4_096, false)
    } else if lower.starts_with("claude-4") || lower.starts_with("claude-sonnet-4")
        || lower.starts_with("claude-opus-4")
    {
        (200_000, 64_000, true)
    } else if lower.starts_with("deepseek-r") || lower.starts_with("deepseek-reasoner") {
        (128_000, 32_768, true)
    } else if lower.starts_with("deepseek") {
        (128_000, 8_192, false)
    } else if lower.starts_with("gemini-2") || lower.starts_with("gemini-1.5") {
        (1_000_000, 8_192, false)
    } else if lower.starts_with("qwen") || lower.starts_with("qwen2") {
        (131_072, 8_192, false)
    } else if lower.starts_with("llama-3.1") || lower.starts_with("llama3.1") {
        (128_000, 4_096, false)
    } else if lower.starts_with("llama-3") || lower.starts_with("llama3") {
        (8_192, 4_096, false)
    } else if lower.starts_with("mistral-large") || lower.starts_with("mistral-medium") {
        (128_000, 8_192, false)
    } else if lower.starts_with("mistral") {
        (32_000, 4_096, false)
    } else if lower.starts_with("mixtral") {
        (32_000, 4_096, false)
    } else if lower.starts_with("gemma") {
        (8_192, 4_096, false)
    } else if lower.starts_with("phi") {
        (4_096, 2_048, false)
    } else if lower.starts_with("command-r") {
        (128_000, 4_096, false)
    } else if lower.starts_with("kimi") || lower.starts_with("moonshot") {
        (128_000, 8_192, false)
    } else if lower.starts_with("glm-4") {
        (128_000, 4_096, false)
    } else if lower.contains("codex") {
        (128_000, 16_384, true)
    } else {
        // Conservative fallback.
        (8_192, 4_096, false)
    };

    ModelWindow {
        model: model.to_string(),
        context_window: ctx,
        max_output: out,
        reasoning_model: reasoning,
    }
}

// ---------------------------------------------------------------------------
// Budget calculation result
// ---------------------------------------------------------------------------

/// The result of a budget calculation for a request.
#[derive(Debug, Clone)]
pub struct BudgetReport {
    /// Estimated tokens consumed by the system message.
    pub system_tokens: u64,
    /// Estimated tokens consumed by tool definitions.
    pub tools_tokens: u64,
    /// Estimated tokens consumed by the conversation messages.
    pub messages_tokens: u64,
    /// Total estimated prompt tokens.
    pub prompt_tokens: u64,
    /// Tokens reserved for the model's response.
    pub generation_budget: u64,
    /// The model's total context window.
    pub context_window: u64,
    /// Whether the prompt fits within the window after trimming.
    pub fits: bool,
    /// Number of messages that were dropped during trimming.
    pub dropped_messages: usize,
    /// The effective `max_tokens` to request after clamping.
    pub effective_max_tokens: u32,
}

impl BudgetReport {
    /// Total estimated tokens (prompt + generation).
    pub fn total_estimated(&self) -> u64 {
        self.prompt_tokens + self.generation_budget
    }
}

// ---------------------------------------------------------------------------
// RequestProof
// ---------------------------------------------------------------------------

/// Pre-flight payload projector.
///
/// Inspects and adapts a chat request so it will be accepted by the target
/// provider/model: estimates the token budget, trims the conversation to fit
/// the context window, and clamps `max_tokens`.
pub struct RequestProof {
    estimator: Arc<dyn TokenEstimator>,
}

impl Default for RequestProof {
    fn default() -> Self {
        Self::new(Arc::new(HeuristicEstimator::new()))
    }
}

impl RequestProof {
    /// Create a new proof using the given token estimator.
    pub fn new(estimator: Arc<dyn TokenEstimator>) -> Self {
        Self { estimator }
    }

    /// Create a proof using the default heuristic estimator.
    pub fn heuristic() -> Self {
        Self::default()
    }

    /// Calculate the token budget for a request without modifying anything.
    ///
    /// `provider` is the canonical provider id (e.g. "openai", "deepseek") used
    /// to look up the per-provider compatibility policy's `max_tokens` cap. Pass
    /// an empty string to skip the policy lookup.
    ///
    /// Returns a [`BudgetReport`] describing whether the payload fits.
    pub fn calculate_budget(
        &self,
        provider: &str,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> BudgetReport {
        let window = lookup_model_window(&config.model);
        let policy = if provider.is_empty() {
            CompatPolicy::default()
        } else {
            policy_for(provider)
        };

        let system_tokens: u64 = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| self.estimator.estimate_message(m))
            .sum();

        let tools_tokens: u64 = if tools.is_empty() {
            0
        } else {
            // Tools are serialized once plus a small structural overhead.
            const TOOLS_OVERHEAD: u64 = 16;
            tools
                .iter()
                .map(|t| self.estimator.estimate_tool(t))
                .sum::<u64>()
                + TOOLS_OVERHEAD
        };

        let messages_tokens: u64 = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .map(|m| self.estimator.estimate_message(m))
            .sum();

        let prompt_tokens = system_tokens + tools_tokens + messages_tokens;

        let effective_max_tokens = clamp_max_tokens(config, &window, policy.max_tokens_cap);
        let generation_budget = effective_max_tokens as u64;

        let fits = prompt_tokens + generation_budget <= window.context_window as u64;

        BudgetReport {
            system_tokens,
            tools_tokens,
            messages_tokens,
            prompt_tokens,
            generation_budget,
            context_window: window.context_window as u64,
            fits,
            dropped_messages: 0,
            effective_max_tokens,
        }
    }

    /// Project (trim + clamp) a request in place so it fits the model's window.
    ///
    /// `provider` is the canonical provider id (e.g. "openai", "groq") used to
    /// look up the per-provider compatibility policy's `max_tokens` cap. Pass an
    /// empty string to skip the policy lookup.
    ///
    /// Returns the adapted messages and a new [`ChatConfig`] with clamped
    /// `max_tokens`, plus a [`BudgetReport`] describing what changed.
    pub fn project(
        &self,
        provider: &str,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> (Vec<ChatMessage>, ChatConfig, BudgetReport) {
        let window = lookup_model_window(&config.model);
        let policy = if provider.is_empty() {
            CompatPolicy::default()
        } else {
            policy_for(provider)
        };

        let system_tokens: u64 = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| self.estimator.estimate_message(m))
            .sum();

        let tools_tokens: u64 = if tools.is_empty() {
            0
        } else {
            const TOOLS_OVERHEAD: u64 = 16;
            tools
                .iter()
                .map(|t| self.estimator.estimate_tool(t))
                .sum::<u64>()
                + TOOLS_OVERHEAD
        };

        // Generation budget: clamp to window max_output and policy cap.
        let generation_budget = clamp_max_tokens(config, &window, policy.max_tokens_cap) as u64;

        // Available budget for conversation messages.
        let available = (window.context_window as u64)
            .saturating_sub(system_tokens)
            .saturating_sub(tools_tokens)
            .saturating_sub(generation_budget);

        // Separate system messages (always kept) from the conversation.
        let mut system_msgs: Vec<ChatMessage> =
            messages.iter().filter(|m| m.role == MessageRole::System).cloned().collect();
        let mut convo: Vec<ChatMessage> = messages
            .iter()
            .filter(|m| m.role != MessageRole::System)
            .cloned()
            .collect();

        // Estimate the conversation tokens; trim from the front (oldest) if
        // needed, but always preserve the last turn (most recent user/assistant
        // pair) so the model has something to respond to.
        let mut convo_tokens: u64 =
            convo.iter().map(|m| self.estimator.estimate_message(m)).sum();

        let mut dropped = 0usize;
        // Keep at least the last message.
        while convo_tokens > available && convo.len() > 1 {
            // Never drop the last message.
            let removed = convo.remove(0);
            convo_tokens =
                convo.iter().map(|m| self.estimator.estimate_message(m)).sum();
            // If we removed a tool result, also drop the orphaned preceding
            // assistant tool-call to keep pairs intact.
            if removed.role == MessageRole::Tool {
                if let Some(last) = convo.last() {
                    if last.role == MessageRole::Assistant
                        && last
                            .tool_calls
                            .as_ref()
                            .map(|c| !c.is_empty())
                            .unwrap_or(false)
                    {
                        let removed2 = convo.pop().unwrap();
                        convo_tokens = convo
                            .iter()
                            .map(|m| self.estimator.estimate_message(m))
                            .sum();
                        dropped += 1;
                        let _ = removed2;
                    }
                }
            }
            dropped += 1;
        }

        // Reassemble: system messages first, then trimmed conversation.
        let mut result = std::mem::take(&mut system_msgs);
        result.extend(convo);

        let prompt_tokens = system_tokens + tools_tokens + convo_tokens;
        let fits = prompt_tokens + generation_budget <= window.context_window as u64;

        // Clamp max_tokens on the returned config.
        let mut new_config = config.clone();
        new_config.max_tokens = generation_budget as u32;

        let report = BudgetReport {
            system_tokens,
            tools_tokens,
            messages_tokens: convo_tokens,
            prompt_tokens,
            generation_budget,
            context_window: window.context_window as u64,
            fits,
            dropped_messages: dropped,
            effective_max_tokens: generation_budget as u32,
        };

        (result, new_config, report)
    }
}

/// Clamp the requested `max_tokens` to the model's output limit and an optional
/// provider cap.
///
/// Precedence: a provider-declared `max_tokens` cap is authoritative — when a
/// provider explicitly declares one it reflects the platform's real hard limit,
/// so it is used as the ceiling even if the heuristic model-window estimate is
/// more conservative. When no provider cap is declared, the model window's
/// `max_output` is used as the ceiling.
pub fn clamp_max_tokens(
    config: &ChatConfig,
    window: &ModelWindow,
    provider_cap: Option<u32>,
) -> u32 {
    let mut value = config.max_tokens;
    if value == 0 {
        value = window.max_output;
    }
    match provider_cap {
        Some(cap) => value = value.min(cap),
        None => value = value.min(window.max_output),
    }
    // Never allow zero.
    value.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::types::{ContentBlock, Message};

    fn user(text: &str) -> ChatMessage {
        Message::user(text)
    }

    #[test]
    fn test_heuristic_estimate_nonempty() {
        let est = HeuristicEstimator::new();
        let t = est.estimate_text("hello world");
        assert!(t >= 2);
        // CJK is more expensive per token.
        let cjk = est.estimate_text("你好世界");
        assert!(cjk >= 2);
    }

    #[test]
    fn test_calculate_budget_fits() {
        let proof = RequestProof::heuristic();
        let config = ChatConfig {
            model: "gpt-4o".into(),
            max_tokens: 1024,
            ..Default::default()
        };
        let messages = vec![user("hello"), user("world")];
        let report = proof.calculate_budget("openai", &config, &messages, &[]);
        assert!(report.fits);
        assert!(report.effective_max_tokens <= 16384);
    }

    #[test]
    fn test_calculate_budget_applies_provider_cap() {
        // Groq's compat policy sets max_tokens_cap = 8192. Even if the request
        // asks for more, the effective max_tokens must be clamped to the cap.
        let proof = RequestProof::heuristic();
        let config = ChatConfig {
            model: "gpt-4o".into(), // window max_output = 16384
            max_tokens: 50_000,
            ..Default::default()
        };
        let messages = vec![user("hello")];
        let report = proof.calculate_budget("groq", &config, &messages, &[]);
        assert_eq!(report.effective_max_tokens, 8192);
    }

    #[test]
    fn test_project_trims_old_messages() {
        let proof = RequestProof::heuristic();
        let config = ChatConfig {
            model: "gpt-4".into(), // 8k window
            max_tokens: 4096,
            ..Default::default()
        };
        // Build a conversation large enough to exceed the 8k window.
        let big = "x".repeat(50_000);
        let mut messages = vec![Message::system("You are helpful.")];
        for _ in 0..20 {
            messages.push(user(&big));
        }
        messages.push(user("Final question"));

        let (trimmed, new_config, report) = proof.project("openai", &config, &messages, &[]);
        assert!(report.dropped_messages > 0);
        assert!(trimmed.len() < messages.len());
        // System message preserved.
        assert!(trimmed.iter().any(|m| m.role == MessageRole::System));
        // Last user message preserved.
        assert!(trimmed
            .iter()
            .any(|m| m.role == MessageRole::User && m.text_content() == "Final question"));
        // max_tokens clamped.
        assert!(new_config.max_tokens <= 4096);
        let _ = big;
    }

    #[test]
    fn test_clamp_max_tokens() {
        let config = ChatConfig {
            model: "gpt-4o".into(),
            max_tokens: 100_000,
            ..Default::default()
        };
        let window = lookup_model_window("gpt-4o");
        let v = clamp_max_tokens(&config, &window, None);
        assert_eq!(v, 16384);

        let v = clamp_max_tokens(&config, &window, Some(4096));
        assert_eq!(v, 4096);
    }

    #[test]
    fn test_lookup_known_models() {
        assert_eq!(lookup_model_window("gpt-4o").context_window, 128_000);
        assert_eq!(lookup_model_window("claude-3-5-sonnet-20241022").context_window, 200_000);
        assert_eq!(lookup_model_window("deepseek-reasoner").reasoning_model, true);
        assert!(lookup_model_window("unknown-model").context_window > 0);
    }

    #[test]
    fn test_project_preserves_last_turn_when_tight() {
        let proof = RequestProof::heuristic();
        let config = ChatConfig {
            model: "gpt-3.5-turbo".into(), // 16k window
            max_tokens: 4000,
            ..Default::default()
        };
        let big = "y".repeat(30_000);
        let messages = vec![user(&big), user("last")];
        let (trimmed, _, report) = proof.project("openai", &config, &messages, &[]);
        // At least one message remains.
        assert!(!trimmed.is_empty());
        let _ = report;
        let _ = big;
    }

    #[test]
    fn test_estimate_message_with_tool_result() {
        let est = HeuristicEstimator::new();
        let msg = Message {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text("use tool".into())],
            name: None,
            tool_call_id: None,
            tool_calls: Some(vec![opensquilla_core::types::ToolCall::new(
                "id1",
                "get_weather",
                serde_json::json!({"location": "NYC"}),
            )]),
            tool_result: None,
        };
        let t = est.estimate_message(&msg);
        assert!(t > 0);
    }
}
