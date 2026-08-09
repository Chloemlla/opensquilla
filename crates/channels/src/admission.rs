//! Authenticated admission decision for inbound channel events.
//!
//! Mirrors the Python `channels/admission.py` decision surface with a
//! desktop-first simplification: direct messages are admitted by default,
//! while group messages require an explicit bot mention (or a configured
//! channel admin sender). This is the single pre-dispatch admission gate.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::IncomingMessage;

/// Why an inbound channel event was admitted or rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionReason {
    DmAdmitted,
    DmDenied,
    GroupAdmitted,
    GroupDenied,
    NotMentionedInGroup,
    NotInAllowlist,
    PairingRequired,
    PairingRevoked,
    PrincipalMismatch,
}

/// The one pre-dispatch decision for an inbound channel event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAdmissionDecision {
    pub admit: bool,
    pub reason: AdmissionReason,
    pub is_group: bool,
    pub mentioned: bool,
    pub sender_id: String,
    pub pairing_id: Option<String>,
}

/// Whether the event is a group-style conversation (channel, thread, topic…).
pub fn is_group_event(msg: &IncomingMessage) -> bool {
    let meta = msg.metadata.as_object();
    if let Some(explicit) = meta.and_then(|m| m.get("is_group")).and_then(|v| v.as_bool()) {
        return explicit;
    }
    if let Some(kind) = meta.and_then(|m| m.get("conversation_kind")).and_then(|v| v.as_str()) {
        return matches!(kind, "group" | "group_dm" | "thread" | "topic");
    }
    false
}

/// Whether provider normalization marks the event as explicitly bot-addressed.
pub fn is_explicit_interaction(msg: &IncomingMessage) -> bool {
    if msg.sender_is_group_mentioned {
        return true;
    }
    let meta = msg.metadata.as_object();
    if let Some(kind) = meta.and_then(|m| m.get("conversation_kind")).and_then(|v| v.as_str()) {
        if kind == "interaction" {
            return true;
        }
    }
    if let Some(itype) = meta.and_then(|m| m.get("interaction_type")).and_then(|v| v.as_str()) {
        if !itype.trim().is_empty() {
            return true;
        }
    }
    meta.and_then(|m| m.get("approval_action"))
        .map_or(false, |v| v.is_object())
}

/// Whether the sender is a configured channel admin for this exact channel.
fn sender_is_channel_admin(
    sender_id: &str,
    channel_name: &str,
    channel_admin_senders: &HashMap<String, Vec<String>>,
) -> bool {
    channel_admin_senders
        .get(channel_name)
        .map_or(false, |senders| senders.iter().any(|s| s == sender_id))
}

/// Evaluate admission for an inbound message.
///
/// Simplified desktop-first policy: DMs are admitted by default; group
/// messages require an explicit mention, unless the sender is a configured
/// channel admin for the exact channel entry.
pub fn decide_channel_admission(
    channel_name: &str,
    msg: &IncomingMessage,
    channel_admin_senders: &HashMap<String, Vec<String>>,
) -> ChannelAdmissionDecision {
    let sender_id = msg.user_id.clone();
    let is_group = is_group_event(msg);
    let mentioned = is_explicit_interaction(msg);
    let is_admin =
        sender_is_channel_admin(&sender_id, channel_name, channel_admin_senders);

    if is_group {
        if mentioned || is_admin {
            return ChannelAdmissionDecision {
                admit: true,
                reason: AdmissionReason::GroupAdmitted,
                is_group,
                mentioned,
                sender_id,
                pairing_id: None,
            };
        }
        return ChannelAdmissionDecision {
            admit: false,
            reason: AdmissionReason::NotMentionedInGroup,
            is_group,
            mentioned,
            sender_id,
            pairing_id: None,
        };
    }

    // Direct message — admitted by default.
    ChannelAdmissionDecision {
        admit: true,
        reason: AdmissionReason::DmAdmitted,
        is_group,
        mentioned,
        sender_id,
        pairing_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::IncomingMessage;
    use chrono::Utc;
    use serde_json::json;
    use uuid::Uuid;

    fn msg(group: bool, mentioned: bool) -> IncomingMessage {
        let metadata = if group {
            json!({ "is_group": true })
        } else {
            json!({ "conversation_kind": "dm" })
        };
        IncomingMessage {
            id: Uuid::new_v4(),
            channel_id: "c1".into(),
            channel_type: crate::types::ChannelType::Slack,
            user_id: "U1".into(),
            user_name: Some("alice".into()),
            text: "hello".into(),
            thread_id: None,
            attachments: Vec::new(),
            timestamp: Utc::now(),
            raw: json!({}),
            metadata,
            provenance_authenticated: true,
            sender_is_group_mentioned: mentioned,
        }
    }

    #[test]
    fn dm_is_admitted_by_default() {
        let admins = HashMap::new();
        let d = decide_channel_admission("chan", &msg(false, false), &admins);
        assert!(d.admit);
        assert_eq!(d.reason, AdmissionReason::DmAdmitted);
        assert!(!d.is_group);
    }

    #[test]
    fn group_requires_mention() {
        let admins = HashMap::new();
        let denied = decide_channel_admission("chan", &msg(true, false), &admins);
        assert!(!denied.admit);
        assert_eq!(denied.reason, AdmissionReason::NotMentionedInGroup);

        let admitted = decide_channel_admission("chan", &msg(true, true), &admins);
        assert!(admitted.admit);
        assert_eq!(admitted.reason, AdmissionReason::GroupAdmitted);
    }

    #[test]
    fn group_admin_passes_without_mention() {
        let mut admins = HashMap::new();
        admins.insert("chan".to_string(), vec!["U1".to_string()]);
        let d = decide_channel_admission("chan", &msg(true, false), &admins);
        assert!(d.admit);
        assert_eq!(d.reason, AdmissionReason::GroupAdmitted);
    }
}