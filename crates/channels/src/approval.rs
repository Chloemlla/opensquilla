//! Surface-agnostic approval-prompt contract for chat channels.
//!
//! Mirrors the Python `channels/approval_prompt.py` rendering + parsing
//! surface. A channel-originated turn that hits an approval-gated tool blocks
//! on the approval queue; this module renders the plain-text prompt and parses
//! the originating user's `/approve` / `/deny` reply.

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::system_messages::{MessageKey, render_channel_message};

/// The decision a channel user makes about a pending approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approve,
    Deny,
    Always,
}

/// Universal text command: `/approve|deny <CODE> [always]`.
///
/// A leading slash avoids bare-word collisions with ordinary chat. The
/// optional trailing `always` selects the durable same-type grant.
fn text_command_re() -> Regex {
    Regex::new(r"(?i)^\s*/(approve|deny)\b\s*([0-9A-Za-z]{2,12})?(?:\s+(always))?\s*$")
        .expect("valid approval command regex")
}

/// Strip one leading mention-shaped token (`<@U123>`, `<@!U123>`) so
/// mention-gated groups still resolve `/approve AB12`.
fn leading_mention_re() -> Regex {
    Regex::new(r"^\s*<@!?[A-Za-z0-9|]+>\s*").expect("valid mention regex")
}

/// Canonicalise a user-typed code for case-insensitive lookup.
pub fn normalize_code(code: &str) -> String {
    code.trim().to_uppercase()
}

/// Render the plain-text approval prompt for one pending approval.
pub fn render_approval_prompt_text(
    command_or_tool: &str,
    short_code: &str,
    offer_always: bool,
    locale: &str,
    summary_label: &str,
) -> String {
    let command = if command_or_tool.trim().is_empty() {
        render_channel_message(MessageKey::ApprovalUnknownCommand, locale, &Default::default())
    } else {
        command_or_tool.to_string()
    };
    let label = if summary_label.trim().is_empty() || summary_label == "Command" {
        render_channel_message(MessageKey::ApprovalLabelCommand, locale, &Default::default())
    } else {
        summary_label.to_string()
    };
    let key = if offer_always {
        MessageKey::ApprovalPromptAlways
    } else {
        MessageKey::ApprovalPrompt
    };
    let mut values = std::collections::HashMap::new();
    values.insert("label", label);
    values.insert("command", command);
    values.insert("code", short_code.to_string());
    render_channel_message(key, locale, &values)
}

/// Recognise a plain-text approval action.
///
/// Accepts `/approve <code>`, `/deny <code>`, and `/approve <code> always`
/// (case-insensitive, with an optional leading mention token). Returns
/// `Some((normalized_code, decision))` or `None` when the input is not an
/// approval action. A missing code yields `None`.
pub fn parse_approval_action(text: &str) -> Option<(String, ApprovalDecision)> {
    let stripped = leading_mention_re().replace(text.trim(), "").to_string();
    let caps = text_command_re().captures(&stripped)?;
    let code = caps.get(2)?.as_str();
    if code.is_empty() {
        return None;
    }
    let verb = caps.get(1)?.as_str().to_lowercase();
    if verb != "approve" {
        return Some((normalize_code(code), ApprovalDecision::Deny));
    }
    if caps.get(3).is_some() {
        return Some((normalize_code(code), ApprovalDecision::Always));
    }
    Some((normalize_code(code), ApprovalDecision::Approve))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_approve() {
        let (code, decision) = parse_approval_action("/approve ab12").unwrap();
        assert_eq!(code, "AB12");
        assert_eq!(decision, ApprovalDecision::Approve);
    }

    #[test]
    fn parses_deny_case_insensitive() {
        let (code, decision) = parse_approval_action("/DENY K7PQ").unwrap();
        assert_eq!(code, "K7PQ");
        assert_eq!(decision, ApprovalDecision::Deny);
    }

    #[test]
    fn parses_approve_always() {
        let (code, decision) = parse_approval_action("/approve K7PQ always").unwrap();
        assert_eq!(code, "K7PQ");
        assert_eq!(decision, ApprovalDecision::Always);
    }

    #[test]
    fn strips_leading_mention() {
        let (code, decision) = parse_approval_action("<@U12345> /approve AB12").unwrap();
        assert_eq!(code, "AB12");
        assert_eq!(decision, ApprovalDecision::Approve);
    }

    #[test]
    fn non_approval_text_is_none() {
        assert!(parse_approval_action("approve the budget").is_none());
        assert!(parse_approval_action("hello").is_none());
        assert!(parse_approval_action("/approve").is_none());
    }

    #[test]
    fn renders_prompt_text_with_placeholders() {
        let text = render_approval_prompt_text("rm -rf", "AB12", false, "en", "Command");
        assert!(text.contains("rm -rf"));
        assert!(text.contains("AB12"));
        assert!(text.contains("/approve AB12"));
        assert!(!text.contains("/approve AB12 always"));
    }
}