//! Per-adapter channel feedback strategy.
//!
//! Mirrors the Python `channels/stream_policy.py`. Dispatch infers
//! user-visible feedback from adapter capabilities plus an optional explicit
//! strategy override, rather than from method presence alone.

use serde::{Deserialize, Serialize};

/// How a channel user should be kept informed during a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelStreamMode {
    AdapterStream,
    TypingFinal,
    FinalOnly,
}

/// Resolved feedback policy for one channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelStreamPolicy {
    pub mode: ChannelStreamMode,
    pub relay_stream: bool,
    pub typing_keepalive: bool,
}

fn policy_for_mode(mode: ChannelStreamMode, has_typing: bool) -> ChannelStreamPolicy {
    ChannelStreamPolicy {
        mode,
        relay_stream: mode == ChannelStreamMode::AdapterStream,
        typing_keepalive: mode == ChannelStreamMode::TypingFinal && has_typing,
    }
}

/// Normalize an override string (e.g. `"adapter-stream"` → `"adapter_stream"`).
fn normalize_override(override_mode: &str) -> String {
    override_mode.trim().to_lowercase().replace('-', "_")
}

/// Resolve the channel stream policy from adapter capabilities and an
/// optional explicit strategy override.
pub fn resolve_channel_stream_policy(
    has_streaming: bool,
    has_typing: bool,
    override_mode: Option<&str>,
) -> ChannelStreamPolicy {
    match override_mode.map(normalize_override).as_deref() {
        Some("adapter_stream" | "stream" | "streaming" | "send_streaming") => policy_for_mode(
            if has_streaming {
                ChannelStreamMode::AdapterStream
            } else {
                ChannelStreamMode::FinalOnly
            },
            has_typing,
        ),
        Some("typing_final" | "typing" | "typing_indicator" | "placeholder") => {
            policy_for_mode(ChannelStreamMode::TypingFinal, has_typing)
        }
        Some("final_only" | "final" | "batch" | "none" | "off") => {
            policy_for_mode(ChannelStreamMode::FinalOnly, has_typing)
        }
        _ => {
            if has_streaming {
                policy_for_mode(ChannelStreamMode::AdapterStream, has_typing)
            } else if has_typing {
                policy_for_mode(ChannelStreamMode::TypingFinal, has_typing)
            } else {
                policy_for_mode(ChannelStreamMode::FinalOnly, has_typing)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_capability_defaults_to_adapter_stream() {
        let p = resolve_channel_stream_policy(true, true, None);
        assert_eq!(p.mode, ChannelStreamMode::AdapterStream);
        assert!(p.relay_stream);
    }

    #[test]
    fn typing_only_defaults_to_typing_final() {
        let p = resolve_channel_stream_policy(false, true, None);
        assert_eq!(p.mode, ChannelStreamMode::TypingFinal);
        assert!(!p.relay_stream);
        assert!(p.typing_keepalive);
    }

    #[test]
    fn no_capabilities_defaults_to_final_only() {
        let p = resolve_channel_stream_policy(false, false, None);
        assert_eq!(p.mode, ChannelStreamMode::FinalOnly);
        assert!(!p.relay_stream);
        assert!(!p.typing_keepalive);
    }

    #[test]
    fn explicit_override_wins() {
        let p = resolve_channel_stream_policy(false, false, Some("adapter-stream"));
        assert_eq!(p.mode, ChannelStreamMode::FinalOnly);

        let p = resolve_channel_stream_policy(true, true, Some("final_only"));
        assert_eq!(p.mode, ChannelStreamMode::FinalOnly);
        assert!(!p.relay_stream);
    }

    #[test]
    fn typing_keepalive_requires_has_typing() {
        let p = resolve_channel_stream_policy(false, false, Some("typing_final"));
        assert_eq!(p.mode, ChannelStreamMode::TypingFinal);
        assert!(!p.typing_keepalive);
    }
}