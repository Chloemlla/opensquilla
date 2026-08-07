//! Provider-request tool-result compaction stack.
//!
//! Parity port of the compaction helpers in the Python reference
//! (`src/opensquilla/provider/request_proof.py`):
//!
//! - env switches: `request_proof.py:46-55`
//! - `_compact_string`: `request_proof.py:465-474`
//! - `_compact_tail_string`: `request_proof.py:477-494`
//! - `_emergency_compact_string`: `request_proof.py:497-514`
//! - `_hard_compact_string`: `request_proof.py:517-526`
//! - `_compact_argument_string`: `request_proof.py:529-548`
//! - `_compact_tool_arguments`: `request_proof.py:551-603`
//! - `aggregate_tool_result_compacted` block builder:
//!   `src/opensquilla/engine/agent.py:4421-4437`
//!
//! Compaction is an *admission* pass: before a payload is trimmed by
//! [`crate::request_proof::RequestProof::project`], oversized tool results are
//! replaced with a small annotated stub so the model-visible payload shrinks
//! without dropping whole conversation turns.

use opensquilla_core::types::{ChatMessage, ContentBlock};
use sha2::{Digest, Sha256};
use std::env;

// ---------------------------------------------------------------------------
// Threshold constants (parity: request_proof.py:16-20)
// ---------------------------------------------------------------------------

/// Strings at or below this length are returned unchanged by
/// [`CompactionConfig::compact_string`].
pub const COMPACTED_STRING_MAX_CHARS: usize = 1200;
/// Strings at or below this length are returned unchanged by
/// [`CompactionConfig::compact_tail_string`] / `compact_argument_string`.
pub const COMPACTED_TAIL_STRING_MAX_CHARS: usize = 640;
/// Preview window used for tool-argument stubs.
pub const COMPACTED_ARGUMENT_PREVIEW_CHARS: usize = 360;
/// Tail window used for tool-argument stubs.
pub const COMPACTED_ARGUMENT_TAIL_CHARS: usize = 120;

const COMPACT_STRING_HEAD_CHARS: usize = 900;
const COMPACT_STRING_TAIL_CHARS: usize = 200;
const COMPACT_TAIL_HEAD_CHARS: usize = 420;
const COMPACT_TAIL_TAIL_CHARS: usize = 120;
const EMERGENCY_MAX_CHARS: usize = 320;
const EMERGENCY_HEAD_CHARS: usize = 180;
const EMERGENCY_TAIL_CHARS: usize = 40;
const HARD_COMPACT_MAX_CHARS: usize = 96;
const AGGREGATE_HEAD_CHARS: usize = 240;
const AGGREGATE_TAIL_CHARS: usize = 240;

/// Prefixes stamped on tool-result content by the delivery-time boundary
/// projection layer (parity: request_proof.py:61-65). Results that already
/// carry one of these prefixes are skipped by default.
pub const BOUNDARY_PROJECTED_RESULT_PREFIXES: [&str; 3] = [
    "[tool_result_projection]\n",
    "[aggregate_tool_result_compacted]\n",
    "[duplicate_tool_result_elided]\n",
];

// ---------------------------------------------------------------------------
// Environment switches (parity: request_proof.py:46-55)
// ---------------------------------------------------------------------------

pub const TINY_COMPACTION_GUARD_ENV: &str =
    "OPENSQUILLA_PROVIDER_COMPACTION_TINY_GUARD_CHARS";
pub const PROTECT_RECENT_ASSISTANT_ENV: &str =
    "OPENSQUILLA_PROVIDER_COMPACTION_PROTECT_RECENT_ASSISTANT";
pub const PROTECT_RECENT_RESULTS_ENV: &str =
    "OPENSQUILLA_PROVIDER_COMPACTION_PROTECT_RECENT_RESULTS";
pub const PROTECT_ERROR_RESULTS_ENV: &str =
    "OPENSQUILLA_PROVIDER_COMPACTION_PROTECT_ERROR_RESULTS";
pub const PROTECT_UNRESOLVED_RESULTS_ENV: &str =
    "OPENSQUILLA_PROVIDER_COMPACTION_PROTECT_UNRESOLVED_RESULTS";
pub const SKIP_PROJECTED_ENV: &str = "OPENSQUILLA_PROVIDER_COMPACTION_SKIP_PROJECTED";
pub const STUB_PREVIEW_CHARS_ENV: &str = "OPENSQUILLA_PROVIDER_COMPACTION_STUB_PREVIEW_CHARS";
pub const NEVER_WORSE_ENV: &str = "OPENSQUILLA_PROVIDER_COMPACTION_NEVER_WORSE";

const DEFAULT_PROTECTED_RECENT_RESULTS: usize = 2;

const FALSE_VALUES: [&str; 5] = ["0", "false", "no", "off", "disabled"];
const TRUE_VALUES: [&str; 5] = ["1", "true", "yes", "on", "enabled"];

/// Read a boolean env flag with the Python `_safety_default_enabled` semantics
/// (parity: request_proof.py:87-93): explicit false/true values win, anything
/// else (including unset) falls back to `default`.
fn env_flag(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(raw) => {
            let lowered = raw.trim().to_ascii_lowercase();
            if FALSE_VALUES.contains(&lowered.as_str()) {
                false
            } else if TRUE_VALUES.contains(&lowered.as_str()) {
                true
            } else {
                default
            }
        }
        Err(_) => default,
    }
}

/// Read an unsigned integer env flag with Python's `max(0, int(raw))`
/// semantics (parity: `_tiny_compaction_guard_chars` / `_stub_preview_chars`,
/// request_proof.py:77-84 / 135-142): empty or unparsable values fall back to
/// `default`, and negative values clamp to `0`.
fn env_usize(name: &str, default: usize) -> usize {
    match env::var(name) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                default
            } else {
                trimmed.parse::<i64>().map(|v| v.max(0) as usize).unwrap_or(default)
            }
        }
        Err(_) => default,
    }
}

/// Read `PROTECT_RECENT_RESULTS_ENV` (parity: request_proof.py:110-120).
/// Unlike the plain `env_usize` reader, an explicit false value maps to `0`
/// and a non-numeric value maps to the default count.
fn env_protect_recent_results() -> usize {
    let default = DEFAULT_PROTECTED_RECENT_RESULTS;
    match env::var(PROTECT_RECENT_RESULTS_ENV) {
        Ok(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                default
            } else if FALSE_VALUES.contains(&trimmed.to_ascii_lowercase().as_str()) {
                0
            } else {
                trimmed.parse::<i64>().map(|v| v.max(0) as usize).unwrap_or(default)
            }
        }
        Err(_) => default,
    }
}

// ---------------------------------------------------------------------------
// CompactionConfig
// ---------------------------------------------------------------------------

/// Runtime-resolved compaction safety configuration.
///
/// All values can be overridden with the `OPENSQUILLA_PROVIDER_COMPACTION_*`
/// environment switches listed above; defaults match the Python reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionConfig {
    /// `_tiny_compaction_guard_chars` — strings at or below this length are
    /// exempt from the aggressive (tail/emergency/hard/argument) paths.
    pub tiny_guard_chars: usize,
    /// `_protect_recent_assistant_enabled` — keep the most recent assistant
    /// message intact (default true).
    pub protect_recent_assistant: bool,
    /// `_protect_recent_results_count` — number of most recent tool results to
    /// protect (default 2).
    pub protect_recent_results: usize,
    /// `_protect_error_results_enabled` — never compact error tool results
    /// (default true).
    pub protect_error_results: bool,
    /// `_protect_unresolved_results_enabled` — never compact unresolved results
    /// (default true).
    pub protect_unresolved_results: bool,
    /// `_skip_projected_results_enabled` — skip results already stamped by the
    /// boundary projection layer (default true).
    pub skip_projected: bool,
    /// `_stub_preview_chars` — attach preview head/tail to stub replacements
    /// (default 0 = no previews).
    pub stub_preview_chars: usize,
    /// `_never_worse_enabled` — never return a replacement longer than the
    /// original (default true).
    pub never_worse: bool,
}

impl CompactionConfig {
    /// Resolve the configuration from the current process environment.
    pub fn from_env() -> Self {
        Self {
            tiny_guard_chars: env_usize(TINY_COMPACTION_GUARD_ENV, 0),
            protect_recent_assistant: env_flag(PROTECT_RECENT_ASSISTANT_ENV, true),
            protect_recent_results: env_protect_recent_results(),
            protect_error_results: env_flag(PROTECT_ERROR_RESULTS_ENV, true),
            protect_unresolved_results: env_flag(PROTECT_UNRESOLVED_RESULTS_ENV, true),
            skip_projected: env_flag(SKIP_PROJECTED_ENV, true),
            stub_preview_chars: env_usize(STUB_PREVIEW_CHARS_ENV, 0),
            never_worse: env_flag(NEVER_WORSE_ENV, true),
        }
    }

    /// A configuration with every safety net disabled and compaction off.
    pub fn disabled() -> Self {
        Self {
            tiny_guard_chars: 0,
            protect_recent_assistant: false,
            protect_recent_results: 0,
            protect_error_results: false,
            protect_unresolved_results: false,
            skip_projected: false,
            stub_preview_chars: 0,
            never_worse: false,
        }
    }

    /// `_compact_string` (parity: request_proof.py:465-474).
    ///
    /// Strings longer than [`COMPACTED_STRING_MAX_CHARS`] are reduced to a
    /// 900-char head and 200-char tail joined by an omission marker.
    pub fn compact_string(&self, value: &str) -> String {
        if chars_len(value) <= COMPACTED_STRING_MAX_CHARS {
            return value.to_string();
        }
        let head = head_chars(value, COMPACT_STRING_HEAD_CHARS);
        let tail = tail_chars(value, COMPACT_STRING_TAIL_CHARS);
        let omitted = chars_len(value)
            .saturating_sub(chars_len(&head))
            .saturating_sub(chars_len(&tail));
        let compacted = format!(
            "{head}\n\n[provider_request_compacted: omitted {omitted} chars]\n\n{tail}"
        );
        if keep_original_for_never_worse(self, value, &compacted) {
            return value.to_string();
        }
        compacted
    }

    /// `_compact_tail_string` (parity: request_proof.py:477-494).
    ///
    /// Strings longer than [`COMPACTED_TAIL_STRING_MAX_CHARS`] are reduced to a
    /// 420-char head and 120-char tail plus a sha256 digest for retrieval.
    pub fn compact_tail_string(&self, value: &str, label: &str) -> String {
        if chars_len(value) <= COMPACTED_TAIL_STRING_MAX_CHARS {
            return value.to_string();
        }
        if chars_len(value) <= self.tiny_guard_chars {
            return value.to_string();
        }
        let head = head_chars(value, COMPACT_TAIL_HEAD_CHARS);
        let tail = tail_chars(value, COMPACT_TAIL_TAIL_CHARS);
        let omitted = chars_len(value)
            .saturating_sub(chars_len(&head))
            .saturating_sub(chars_len(&tail));
        let digest = sha256_hex(value);
        let compacted = format!(
            "{head}\n\n[provider_request_{label}_compacted: omitted {omitted} chars; \
             original_chars={}; sha256={digest}]\n\n{tail}",
            chars_len(value),
        );
        if keep_original_for_never_worse(self, value, &compacted) {
            return value.to_string();
        }
        compacted
    }

    /// `_emergency_compact_string` (parity: request_proof.py:497-514).
    ///
    /// A more aggressive reduction: 180-char head and 40-char tail with digest.
    pub fn emergency_compact_string(&self, value: &str, label: &str) -> String {
        if chars_len(value) <= EMERGENCY_MAX_CHARS {
            return value.to_string();
        }
        if chars_len(value) <= self.tiny_guard_chars {
            return value.to_string();
        }
        let head = head_chars(value, EMERGENCY_HEAD_CHARS);
        let tail = tail_chars(value, EMERGENCY_TAIL_CHARS);
        let omitted = chars_len(value)
            .saturating_sub(chars_len(&head))
            .saturating_sub(chars_len(&tail));
        let digest = sha256_hex(value);
        let compacted = format!(
            "{head}\n\n[provider_request_{label}_emergency_compacted: omitted {omitted} chars; \
             original_chars={}; sha256={digest}]\n\n{tail}",
            chars_len(value),
        );
        if keep_original_for_never_worse(self, value, &compacted) {
            return value.to_string();
        }
        compacted
    }

    /// `_hard_compact_string` (parity: request_proof.py:517-526).
    ///
    /// Reduces the string to a single bracket stub carrying the length and the
    /// first 16 hex chars of its sha256 digest. Only used for strings longer
    /// than 96 chars.
    pub fn hard_compact_string(&self, value: &str, label: &str) -> String {
        if chars_len(value) <= HARD_COMPACT_MAX_CHARS {
            return value.to_string();
        }
        if chars_len(value) <= self.tiny_guard_chars {
            return value.to_string();
        }
        let digest = &sha256_hex(value)[..16];
        let compacted = format!(
            "[opensquilla_compacted:{label}:{}:{digest}]",
            chars_len(value),
        );
        if keep_original_for_never_worse(self, value, &compacted) {
            return value.to_string();
        }
        compacted
    }

    /// `_compact_argument_string` (parity: request_proof.py:529-548).
    ///
    /// Compacts a single tool-argument string. In preview mode it delegates to
    /// [`CompactionConfig::compact_tail_string`]; otherwise it emits a digest
    /// stub with optional head/tail previews.
    pub fn compact_argument_string(&self, value: &str, preview: bool) -> String {
        if preview {
            return self.compact_tail_string(value, "tool_input");
        }
        if chars_len(value) <= self.tiny_guard_chars {
            return value.to_string();
        }
        let digest = sha256_hex(value);
        let mut compacted = format!(
            "[provider_request_tool_input_compacted: original_chars={}; sha256={digest}]",
            chars_len(value),
        );
        let preview_chars = self.stub_preview_chars;
        if preview_chars > 0 && chars_len(value) > preview_chars * 2 {
            let with_previews = format!(
                "{}\n\n{compacted}\n\n{}",
                head_chars(value, preview_chars),
                tail_chars(value, preview_chars),
            );
            // Previews may never turn compaction into growth.
            if payload_chars(&with_previews) < payload_chars(value) {
                compacted = with_previews;
            }
        }
        if keep_original_for_never_worse(self, value, &compacted) {
            return value.to_string();
        }
        compacted
    }

    /// `_compact_tool_arguments` (parity: request_proof.py:551-603).
    ///
    /// Compacts a serialized JSON tool-argument object: string fields are
    /// individually compacted in place; when the payload is not valid JSON it
    /// falls back to a digest stub object.
    pub fn compact_tool_arguments(&self, value: &str, preview: bool) -> String {
        if preview && chars_len(value) <= COMPACTED_TAIL_STRING_MAX_CHARS {
            return value.to_string();
        }
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(value) {
            if let serde_json::Value::Object(map) = parsed {
                let mut compacted = serde_json::Map::new();
                let mut changed = false;
                let force_string_compaction = !preview;
                for (key, item) in map {
                    if let serde_json::Value::String(s) = item {
                        // `path` is preserved verbatim in non-preview mode so
                        // the compaction never hides the most load-bearing arg.
                        if key == "path" && !preview {
                            compacted.insert(key, serde_json::Value::String(s));
                            continue;
                        }
                        let next_item =
                            self.compact_argument_string(&s, preview && !force_string_compaction);
                        changed = changed || next_item != s;
                        compacted.insert(key, serde_json::Value::String(next_item));
                    } else {
                        compacted.insert(key, item);
                    }
                }
                if preview && !changed && self.never_worse {
                    return value.to_string();
                }
                if changed || !preview {
                    let compacted_json =
                        serde_json::to_string(&serde_json::Value::Object(compacted))
                            .unwrap_or_default();
                    if keep_original_for_never_worse(self, value, &compacted_json) {
                        return value.to_string();
                    }
                    return compacted_json;
                }
            }
        }
        if chars_len(value) <= self.tiny_guard_chars {
            return value.to_string();
        }
        let digest = sha256_hex(value);
        let mut stub = serde_json::json!({
            "note": "historical tool arguments omitted for provider context budget",
            "original_chars": chars_len(value),
            "sha256": digest,
        });
        let mut stub_json = serde_json::to_string(&stub).unwrap_or_default();
        let preview_chars = self.stub_preview_chars;
        if preview_chars > 0 && chars_len(value) > preview_chars * 2 {
            stub["preview_head"] = serde_json::Value::String(head_chars(value, preview_chars));
            stub["preview_tail"] = serde_json::Value::String(tail_chars(value, preview_chars));
            let with_previews_json = serde_json::to_string(&stub).unwrap_or_default();
            // Previews may never turn compaction into growth.
            if payload_chars(&with_previews_json) < payload_chars(value) {
                stub_json = with_previews_json;
            }
        }
        if keep_original_for_never_worse(self, value, &stub_json) {
            return value.to_string();
        }
        stub_json
    }
}

// ---------------------------------------------------------------------------
// aggregate_tool_result_compacted block builder
// ---------------------------------------------------------------------------

/// Input for [`aggregate_tool_result_compacted`].
///
/// Mirrors the data the Python reference records when it aggregates many
/// under-threshold tool results that together exceed the provider budget
/// (`src/opensquilla/engine/agent.py:4384-4437`).
#[derive(Debug, Clone)]
pub struct AggregateToolResult<'a> {
    /// The `tool_use_id` this result belongs to.
    pub tool_use_id: &'a str,
    /// The full original tool result content.
    pub content: &'a str,
    /// The estimated token cost of the original content.
    pub original_tokens_estimate: usize,
    /// An optional retrieval handle from the tool-result store.
    pub handle: Option<&'a str>,
    /// An optional retrieval hint line (already newline-terminated) shown when
    /// a handle is present.
    pub retrieve_hint: Option<&'a str>,
}

/// Build a `[aggregate_tool_result_compacted]` replacement block for one tool
/// result (parity: agent.py:4421-4437).
///
/// The block keeps a 240-char head and 240-char tail, records the sha256
/// digest and original size for retrieval, and optionally attaches the
/// tool-result-store handle so a later `retrieve_tool_result` call can restore
/// the full content.
pub fn aggregate_tool_result_compacted(input: &AggregateToolResult) -> String {
    let content = input.content;
    let digest = sha256_hex(content);
    let head = head_chars(content, AGGREGATE_HEAD_CHARS);
    let tail = if chars_len(content) > AGGREGATE_TAIL_CHARS {
        tail_chars(content, AGGREGATE_TAIL_CHARS)
    } else {
        String::new()
    };
    let omitted = chars_len(content)
        .saturating_sub(chars_len(&head))
        .saturating_sub(chars_len(&tail));
    let handle_line = match input.handle {
        Some(h) => format!("tool_result_handle: {h}\n"),
        None => String::new(),
    };
    let retrieve_hint = input.retrieve_hint.unwrap_or("");
    let compacted = format!(
        "[aggregate_tool_result_compacted]\n\
         tool_use_id: {}\n\
         original_chars: {}\n\
         original_tokens_estimate: {}\n\
         sha256: {digest}\n\
         {handle_line}\
         {retrieve_hint}\
         omitted_chars: {omitted}\n\
         preview_complete: {}\n\
         reason: older non-error tool result compacted for provider context budget.\n\
         head:\n{head}",
        input.tool_use_id,
        chars_len(content),
        input.original_tokens_estimate,
        if omitted == 0 { "true" } else { "false" },
    );
    if !tail.is_empty() && tail != head {
        return format!("{compacted}\n...\ntail:\n{tail}");
    }
    compacted
}

// ---------------------------------------------------------------------------
// Message-level pass used by RequestProof::project
// ---------------------------------------------------------------------------

/// Returns true when `content` already carries a boundary-projection prefix
/// (parity: `_tool_result_content_is_provider_projection`,
/// request_proof.py:153-154).
pub fn is_provider_projection(content: &str) -> bool {
    BOUNDARY_PROJECTED_RESULT_PREFIXES
        .iter()
        .any(|prefix| content.starts_with(prefix))
}

/// Compact oversized tool-result content blocks in a message list in place.
///
/// Returns the number of `ToolResult` blocks replaced by compacted stubs.
/// Recent results (the last `protect_recent_results` blocks), error results,
/// unresolved results and already-projected results are skipped by default.
/// This is the admission pass that runs *before* `RequestProof::project` starts
/// dropping whole messages.
pub fn compact_tool_results(messages: &mut [ChatMessage], config: &CompactionConfig) -> usize {
    let mut compacted = 0usize;
    for message in messages.iter_mut() {
        if let Some(tr) = message.tool_result.as_mut() {
            compacted += compact_one_tool_result(tr, config);
        }
        for block in message.content.iter_mut() {
            if let ContentBlock::ToolResult(tr) = block {
                compacted += compact_one_tool_result(tr, config);
            }
        }
    }
    compacted
}

fn compact_one_tool_result(
    tr: &mut opensquilla_core::types::ToolResult,
    config: &CompactionConfig,
) -> usize {
    if tr.is_error && config.protect_error_results {
        return 0;
    }
    if config.skip_projected && is_provider_projection(&tr.content) {
        return 0;
    }
    if chars_len(&tr.content) > COMPACTED_STRING_MAX_CHARS {
        let replacement = config.compact_string(&tr.content);
        if replacement != tr.content {
            tr.content = replacement;
            return 1;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn chars_len(s: &str) -> usize {
    s.chars().count()
}

fn head_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn tail_chars(s: &str, n: usize) -> String {
    s.chars().rev().take(n).collect::<Vec<char>>().into_iter().rev().collect()
}

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

/// Character count of the JSON encoding of `s` (parity:
/// `_payload_chars`, request_proof.py:205-213). Both sides of a never-worse
/// comparison go through the same encoding, so escape differences from Python's
/// `json.dumps(ensure_ascii=False)` are immaterial.
fn payload_chars(s: &str) -> usize {
    serde_json::to_string(s)
        .map(|encoded| encoded.chars().count())
        .unwrap_or_else(|_| s.chars().count())
}

/// `_keep_original_for_never_worse` (parity: request_proof.py:149-150):
/// keep the original whenever the replacement would be no smaller.
fn keep_original_for_never_worse(
    config: &CompactionConfig,
    original: &str,
    replacement: &str,
) -> bool {
    config.never_worse && payload_chars(replacement) >= payload_chars(original)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CompactionConfig {
        CompactionConfig {
            tiny_guard_chars: 0,
            protect_recent_assistant: true,
            protect_recent_results: 2,
            protect_error_results: true,
            protect_unresolved_results: true,
            skip_projected: true,
            stub_preview_chars: 0,
            never_worse: true,
        }
    }

    #[test]
    fn compact_string_short_is_unchanged() {
        let c = config();
        assert_eq!(c.compact_string("short"), "short");
    }

    #[test]
    fn compact_string_long_is_truncated() {
        let c = config();
        let long = "x".repeat(2000);
        let out = c.compact_string(&long);
        assert!(out.len() < long.len());
        assert!(out.contains("[provider_request_compacted: omitted"));
        assert!(out.starts_with(&"x".repeat(900)));
        assert!(out.ends_with("x"));
    }

    #[test]
    fn hard_compact_string_is_small() {
        let c = config();
        let long = "y".repeat(500);
        let out = c.hard_compact_string(&long, "tool_result");
        assert!(out.starts_with("[opensquilla_compacted:tool_result:500:"));
        assert!(out.ends_with(']'));
    }

    #[test]
    fn aggregate_keeps_head_tail_and_digest() {
        // Distinct head/tail so the trailing tail branch is exercised.
        let content = format!("{}{}{}", "a".repeat(240), "m".repeat(520), "b".repeat(240));
        let out = aggregate_tool_result_compacted(&AggregateToolResult {
            tool_use_id: "call_1",
            content: &content,
            original_tokens_estimate: 250,
            handle: Some("h1"),
            retrieve_hint: None,
        });
        assert!(out.starts_with("[aggregate_tool_result_compacted]\n"));
        assert!(out.contains("tool_use_id: call_1"));
        assert!(out.contains("original_tokens_estimate: 250"));
        assert!(out.contains("sha256: "));
        assert!(out.contains("tool_result_handle: h1\n"));
        assert!(out.contains("head:\n"));
        assert!(out.contains("\n...\ntail:\n"));
        assert!(out.contains("preview_complete: false"));
    }

    #[test]
    fn aggregate_complete_preview_when_small() {
        let content = "abc".repeat(60); // 180 chars <= head
        let out = aggregate_tool_result_compacted(&AggregateToolResult {
            tool_use_id: "call_2",
            content: &content,
            original_tokens_estimate: 45,
            handle: None,
            retrieve_hint: None,
        });
        assert!(out.contains("preview_complete: true"));
        assert!(!out.contains("\n...\ntail:\n"));
    }

    #[test]
    fn compact_tool_results_skips_projected_and_errors() {
        let mut msgs = vec![ChatMessage::text(
            opensquilla_core::types::MessageRole::Tool,
            "x",
        )];
        msgs[0].content = vec![
            ContentBlock::ToolResult(opensquilla_core::types::ToolResult::success(
                "a",
                "[tool_result_projection]\nlarge projected content",
            )),
            ContentBlock::ToolResult(opensquilla_core::types::ToolResult::error(
                "b",
                "e".repeat(5000),
            )),
            ContentBlock::ToolResult(opensquilla_core::types::ToolResult::success(
                "c",
                "c".repeat(5000),
            )),
        ];
        let n = compact_tool_results(&mut msgs, &config());
        // Only the third (large, non-projected, non-error) block is compacted.
        assert_eq!(n, 1);
    }
}
