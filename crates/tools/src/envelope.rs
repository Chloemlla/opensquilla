//! Tool failure envelope.
//!
//! Mirrors the Python `opensquilla.tools.envelope` module: converts any error
//! raised by a tool handler into a stable, user-facing JSON envelope. Keys are
//! fixed and exhaustive:
//!
//! ```json
//! {"status": "error", "tool": <str>, "error_class": <str>,
//!  "user_message": <str>, "retry_allowed": <bool>}
//! ```
//!
//! No raw `repr`/traceback text ever leaks into the envelope — `user_message`
//! is sanitised so it is safe to show to an end user or write to a channel
//! transcript.
//!
//! Because Rust has no exception objects, callers pass the exception's primary
//! class name plus its MRO-like class-name chain so retriable classification
//! and user-message selection behave the same way the Python side does.

use serde_json::{Value, json};

/// Class names whose instances represent transient infrastructure problems.
/// Subclass matching is handled by walking the error's MRO-like names.
const RETRIABLE_CLASSES: &[&str] = &[
    "TimeoutError",
    "TimeoutException",
    "ConnectError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "BrokenPipeError",
    "TransientError",
    "TypeError",
    "RetryableToolInputError",
];

/// Exception classes that are explicitly NOT retriable even if their name
/// coincidentally shares a token with a retriable class above.
const NEVER_RETRIABLE_CLASSES: &[&str] = &[
    "PermissionError",
    "PermissionDenied",
    "ValueError",
    "KeyError",
    "FileNotFoundError",
    "IsADirectoryError",
    "NotADirectoryError",
    "NotImplementedError",
];

/// Human-readable messages keyed by error class. Anything not in the table
/// falls back to a generic `The tool '<tool>' failed.` line.
fn user_message_for_class(class_name: &str) -> Option<&'static str> {
    Some(match class_name {
        "TimeoutError" => "The tool took too long to respond. Please try again.",
        "ConnectionError" => "A network error occurred while running the tool. Please try again.",
        "ConnectionResetError" => {
            "The connection was reset while running the tool. Please try again."
        }
        "ConnectionAbortedError" => {
            "The connection was aborted while running the tool. Please try again."
        }
        "ConnectionRefusedError" => {
            "The network connection was refused while running the tool. Please try again."
        }
        "BrokenPipeError" => "A pipe error interrupted the tool. Please try again.",
        "TransientError" => "A transient error occurred while running the tool. Please try again.",
        "PermissionError" => "The tool was not permitted to perform this action.",
        "PermissionDenied" => "The tool was not permitted to perform this action.",
        "ValueError" => "The tool received an invalid argument.",
        "TypeError" => {
            "The tool received an argument of the wrong type. Retry with arguments \
             that match the tool schema."
        }
        "KeyError" => "The tool was asked for a value it could not find.",
        "FileNotFoundError" => "The tool could not find the requested file.",
        "IsADirectoryError" => "The tool expected a file but received a directory.",
        "NotADirectoryError" => "The tool expected a directory but received a file.",
        "NotImplementedError" => "This tool does not support the requested operation.",
        "TimeoutException" => "The tool timed out while connecting or waiting for a response.",
        "ConnectError" => "The tool could not connect to the remote service.",
        "JSONDecodeError" => "The tool received an invalid response payload.",
        "ToolRunBudgetExceededError" => "The tool run budget for this turn is exhausted.",
        "SandboxBackendError" => {
            "The sandbox environment could not run this operation. Do not retry with another \
             tool; report the sandbox failure once."
        }
        "policy_denial" => "The action was blocked by policy. See user-facing reason for details.",
        _ => return None,
    })
}

/// Engine-authored messages — explicit overrides and curated (safe) error
/// payloads carry actionable, sometimes multi-line detail that a 500-char cap
/// would truncate mid-instruction.
const CURATED_USER_MESSAGE_MAX_CHARS: usize = 2000;
/// Generic canned lines are all far shorter than either cap.
const USER_MESSAGE_MAX_CHARS: usize = 500;
/// Truncation marker appended when a message is capped.
const TRUNCATION_MARKER: &str = "...[truncated]";

/// Env lever that opts policy-gate denial envelopes into a wider cap.
const POLICY_DENY_MAX_CHARS_ENV: &str = "OPENSQUILLA_TOOL_ENVELOPE_POLICY_DENY_MAX_CHARS";

fn policy_deny_max_chars() -> usize {
    let raw = std::env::var(POLICY_DENY_MAX_CHARS_ENV)
        .unwrap_or_default()
        .trim()
        .to_string();
    if raw.is_empty() {
        return 0;
    }
    raw.parse::<usize>().unwrap_or(0)
}

/// Options for building a tool failure envelope.
#[derive(Debug, Clone, Default)]
pub struct EnvelopeOptions {
    /// Build a policy-denial envelope (`error_class` defaults to
    /// `PolicyDenied`, `retry_allowed` is forced `false`).
    pub policy_denial: bool,
    /// Override the emitted `error_class` (used for policy-denial envelopes).
    pub error_class_override: Option<String>,
    /// Override the `user_message` verbatim.
    pub user_message_override: Option<String>,
    /// Whether the message is a curated safe-user message (widens the cap).
    pub curated: bool,
    /// Whether the raising error is marked as a policy-gate denial (applies
    /// the `OPENSQUILLA_TOOL_ENVELOPE_POLICY_DENY_MAX_CHARS` lever).
    pub policy_gate_denial: bool,
}

/// Decide whether an error represents a transient / retriable failure.
///
/// `mro_names` is the error class name followed by its superclass names;
/// never-retriable takes precedence even for oddly-named subclasses of
/// retriable classes.
pub fn is_retriable(mro_names: &[&str]) -> bool {
    for name in mro_names {
        if NEVER_RETRIABLE_CLASSES.contains(name) {
            return false;
        }
    }
    mro_names
        .iter()
        .any(|name| RETRIABLE_CLASSES.contains(name))
}

fn sanitise_user_message(tool_name: &str, class_name: &str, opts: &EnvelopeOptions) -> String {
    if opts.curated {
        return match opts.user_message_override.as_deref() {
            Some(message) if !message.trim().is_empty() => message.to_string(),
            _ => "The tool could not complete this action.".to_string(),
        };
    }
    if let Some(message) = user_message_for_class(class_name) {
        return message.to_string();
    }
    // Unknown classes: render a generic line that names the tool. Do NOT
    // interpolate the raw error message — operators may have put secrets in it.
    format!("The tool {tool_name:?} failed with an internal error.")
}

fn resolve_error_class(class_name: &str, opts: &EnvelopeOptions) -> String {
    if opts.policy_denial {
        return opts
            .error_class_override
            .clone()
            .unwrap_or_else(|| "PolicyDenied".to_string());
    }
    opts.error_class_override
        .clone()
        .unwrap_or_else(|| class_name.to_string())
}

/// Build the canonical tool-failure envelope for an error raised by `tool_name`.
///
/// `class_name` is the primary error class; `mro_names` additionally carries
/// superclass names for retriable classification. The returned dict has
/// exactly these keys: `status`, `tool`, `error_class`, `user_message`,
/// `retry_allowed`. `user_message` is guaranteed to contain neither a
/// traceback nor raw error text.
pub fn build_tool_failure_envelope(
    tool_name: &str,
    class_name: &str,
    mro_names: &[&str],
    opts: &EnvelopeOptions,
) -> Value {
    let tool = if tool_name.is_empty() {
        "<unknown>"
    } else {
        tool_name
    };

    let curated = opts.user_message_override.is_some() || opts.curated;
    let mut user_message = opts
        .user_message_override
        .clone()
        .unwrap_or_else(|| sanitise_user_message(tool, class_name, opts));
    // Defense in depth: strip traceback frames, collapse newlines.
    user_message = strip_traceback_frames(&user_message).trim().to_string();
    user_message = user_message.replace('\n', " ").trim().to_string();
    if user_message.is_empty() {
        user_message = format!("The tool {tool:?} failed.");
    }

    let mut max_chars = if curated {
        CURATED_USER_MESSAGE_MAX_CHARS
    } else {
        USER_MESSAGE_MAX_CHARS
    };
    if opts.policy_gate_denial {
        let override_max_chars = policy_deny_max_chars();
        // Caps that cannot fit the truncation marker would replace content
        // with marker text; treat them as off like the other invalid values.
        if override_max_chars > TRUNCATION_MARKER.len() {
            max_chars = override_max_chars;
        }
    }
    if user_message.chars().count() > max_chars {
        let truncated: String = user_message
            .chars()
            .take(max_chars.saturating_sub(TRUNCATION_MARKER.len()))
            .collect();
        user_message = format!("{truncated}{TRUNCATION_MARKER}");
    }

    json!({
        "status": "error",
        "tool": tool,
        "error_class": resolve_error_class(class_name, opts),
        "user_message": user_message,
        "retry_allowed": if opts.policy_denial { false } else { is_retriable(mro_names) },
    })
}

/// Strip Python-style traceback frame lines (`  File "...", line N, in ...`).
pub fn strip_traceback_frames(text: &str) -> String {
    let mut result = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("File \"")
            && trimmed.contains("\", line ")
            && trimmed.contains(", in ")
        {
            continue;
        }
        result.push(line);
    }
    result.join("\n")
}

/// Terminal denial statuses — all three signal "do not execute".
const TERMINAL_DENIAL_STATUSES: &[&str] = &["denied", "blocked", "approval_denied"];

/// Return `true` when `content` is a terminal denial payload.
///
/// Covers three shapes:
/// * `status=denied` — sandbox `DenialResult.to_dict`
/// * `status=blocked` — sensitive-path hard block
/// * `status=approval_denied` — a queued approval was rejected by the user
///
/// `approval_required` / `approval_pending` are explicitly *not* denials —
/// those are retry signals for the model.
pub fn is_denial_payload(content: &Value) -> bool {
    if let Some(status) = content.get("status").and_then(|v| v.as_str()) {
        return TERMINAL_DENIAL_STATUSES.contains(&status);
    }
    if let Some(text) = content.as_str() {
        if let Ok(payload) = serde_json::from_str::<Value>(text) {
            return payload
                .get("status")
                .and_then(|v| v.as_str())
                .map(|status| TERMINAL_DENIAL_STATUSES.contains(&status))
                .unwrap_or(false);
        }
    }
    false
}

/// Wrap a sandbox-style denial payload as a tool-visible envelope.
///
/// Forwards the denial fields verbatim and decorates them with the tool name
/// so callers can distinguish a sandbox denial from a handler failure on the
/// `status` field alone (`"denied"` vs `"error"`).
pub fn build_denial_envelope(denial: &Value, tool_name: &str) -> Value {
    let mut payload = if let Some(obj) = denial.as_object() {
        obj.clone()
    } else {
        serde_json::Map::new()
    };
    payload
        .entry("status".to_string())
        .or_insert_with(|| json!("denied"));
    payload.insert(
        "tool".to_string(),
        json!(if tool_name.is_empty() {
            "<unknown>"
        } else {
            tool_name
        }),
    );
    Value::Object(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_retriable_timeout() {
        let envelope = build_tool_failure_envelope(
            "web_fetch",
            "TimeoutError",
            &["TimeoutError", "OSError"],
            &EnvelopeOptions::default(),
        );
        assert_eq!(envelope["status"], "error");
        assert_eq!(envelope["tool"], "web_fetch");
        assert_eq!(envelope["error_class"], "TimeoutError");
        assert_eq!(envelope["retry_allowed"], true);
        assert!(
            envelope["user_message"]
                .as_str()
                .unwrap()
                .contains("too long")
        );
    }

    #[test]
    fn envelope_never_retriable_permission() {
        let envelope = build_tool_failure_envelope(
            "exec_command",
            "PermissionError",
            &["PermissionError", "OSError"],
            &EnvelopeOptions::default(),
        );
        assert_eq!(envelope["retry_allowed"], false);
        assert_eq!(
            envelope["user_message"],
            "The tool was not permitted to perform this action."
        );
    }

    #[test]
    fn envelope_unknown_class_is_generic_and_does_not_leak() {
        // A raw secret-bearing message must never appear in the envelope.
        let envelope = build_tool_failure_envelope(
            "git",
            "MysteryError",
            &["MysteryError"],
            &EnvelopeOptions::default(),
        );
        let text = envelope["user_message"].as_str().unwrap();
        assert!(text.contains("internal error"));
        assert!(!text.contains("SECRET"));
        assert_eq!(envelope["retry_allowed"], false);
    }

    #[test]
    fn envelope_never_retriable_wins_over_retriable_superclass() {
        // A ValueError subclass of TimeoutError must not be retriable.
        let envelope = build_tool_failure_envelope(
            "tool",
            "TimeoutError",
            &["TimeoutError", "ValueError"],
            &EnvelopeOptions::default(),
        );
        assert_eq!(envelope["retry_allowed"], false);
    }

    #[test]
    fn envelope_policy_denial_forces_retry_false_and_class() {
        let envelope = build_tool_failure_envelope(
            "write_file",
            "PermissionError",
            &["PermissionError"],
            &EnvelopeOptions {
                policy_denial: true,
                error_class_override: Some("PolicyDenied".to_string()),
                user_message_override: Some("Blocked by policy: write deny".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(envelope["error_class"], "PolicyDenied");
        assert_eq!(envelope["retry_allowed"], false);
        assert_eq!(envelope["user_message"], "Blocked by policy: write deny");
    }

    #[test]
    fn envelope_user_message_override_is_curated_and_capped_widely() {
        let long = "x".repeat(3000);
        let envelope = build_tool_failure_envelope(
            "tool",
            "ValueError",
            &["ValueError"],
            &EnvelopeOptions {
                user_message_override: Some(long.clone()),
                ..Default::default()
            },
        );
        let text = envelope["user_message"].as_str().unwrap();
        assert!(text.chars().count() <= CURATED_USER_MESSAGE_MAX_CHARS);
        assert!(text.ends_with(TRUNCATION_MARKER));
        // Non-curated generic messages use the tighter 500-char cap.
        let envelope2 = build_tool_failure_envelope(
            "tool",
            "UnknownClass",
            &["UnknownClass"],
            &EnvelopeOptions::default(),
        );
        let generic = envelope2["user_message"].as_str().unwrap();
        assert!(generic.chars().count() <= USER_MESSAGE_MAX_CHARS);
        assert_eq!(envelope["error_class"], "ValueError");
    }

    #[test]
    fn envelope_does_not_leak_override_class_without_override() {
        // An override provided but policy_denial false: error_class uses it.
        let envelope = build_tool_failure_envelope(
            "tool",
            "FileNotFoundError",
            &["FileNotFoundError"],
            &EnvelopeOptions {
                error_class_override: Some("NOT_FOUND".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(envelope["error_class"], "NOT_FOUND");
        assert_eq!(
            envelope["user_message"],
            "The tool could not find the requested file."
        );
    }

    #[test]
    fn strip_traceback_removes_frames() {
        let text = "Some error\n  File \"/usr/lib/x.py\", line 12, in <module>\n  File \"/a/b.py\", line 3, in run\nreal detail";
        let stripped = strip_traceback_frames(text);
        assert!(!stripped.contains("line 12, in"));
        assert!(!stripped.contains("line 3, in"));
        assert!(stripped.contains("Some error"));
        assert!(stripped.contains("real detail"));
    }

    #[test]
    fn denial_payload_detection() {
        assert!(is_denial_payload(
            &json!({"status": "denied", "message": "no"})
        ));
        assert!(is_denial_payload(&json!({"status": "blocked"})));
        assert!(is_denial_payload(&json!({"status": "approval_denied"})));
        assert!(!is_denial_payload(&json!({"status": "approval_required"})));
        assert!(!is_denial_payload(&json!({"status": "approval_pending"})));
        assert!(!is_denial_payload(&json!({"status": "error"})));
        assert!(!is_denial_payload(&json!("not a payload")));
        // String JSON payloads are parsed too.
        assert!(is_denial_payload(&json!("{\"status\": \"denied\"}")));
        assert!(!is_denial_payload(&json!("{\"status\": \"error\"}")));
        assert!(!is_denial_payload(&json!("not json")));
    }

    #[test]
    fn build_denial_envelope_decorates_tool() {
        let denial = json!({
            "reason": "sensitive_path",
            "suggested_next_step": "use workspace path",
            "level": "block",
            "action_fingerprint": "abc123",
            "message": "blocked",
            "retryable": true,
        });
        let envelope = build_denial_envelope(&denial, "read_file");
        assert_eq!(envelope["status"], "denied");
        assert_eq!(envelope["tool"], "read_file");
        assert_eq!(envelope["reason"], "sensitive_path");
        assert_eq!(envelope["retryable"], true);
    }

    #[test]
    fn build_denial_envelope_defaults_status() {
        let envelope = build_denial_envelope(&json!({"message": "x"}), "");
        assert_eq!(envelope["status"], "denied");
        assert_eq!(envelope["tool"], "<unknown>");
    }
}
