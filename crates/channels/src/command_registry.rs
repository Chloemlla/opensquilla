//! Channel-side slash-command registry.
//!
//! Mirrors the Python `channels/command_registry.py` dispatcher table. Lookup
//! keys are bare command names (leading slash stripped, lowercased). The
//! Rust desktop-first build ships a compact positioned table of the channel
//! surface commands; the lightweight commands that are weak/high-touch are
//! omitted.

use std::collections::HashMap;

/// A command maps its bare (slash-stripped, lowercase) name to an
/// `(rpc_method, params_hint)` pair consumed by the gateway dispatcher.
#[derive(Debug, Clone, Default)]
pub struct CommandRegistry {
    commands: HashMap<String, (String, String)>,
}

impl CommandRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a registry from a pre-populated command table.
    pub fn from_table(commands: HashMap<String, (String, String)>) -> Self {
        Self { commands }
    }

    /// The bare command names currently registered.
    pub fn command_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.commands.keys().cloned().collect();
        names.sort();
        names
    }

    /// Match inbound channel content against the registered command set.
    ///
    /// Returns `Some((name, rpc_method))` when `content`'s first whitespace
    /// token is a registered slash command (case-insensitive). Returns `None`
    /// for plain chat, a bare `/`, or an unknown command.
    pub fn match_command(&self, content: &str) -> Option<(String, String)> {
        let trimmed = content.trim();
        if trimmed.is_empty() || !trimmed.starts_with('/') || trimmed == "/" {
            return None;
        }
        let head = trimmed.split_whitespace().next()?;
        let bare = head[1..].to_lowercase();
        self.commands.get(&bare).cloned()
    }
}

/// Build the default channel command table.
///
/// Positioned from the Python channel surface: sandbox, compact, meta, new,
/// plus the universal help command. Each entry maps to its gateway RPC method
/// and a short params hint.
pub fn build_default() -> CommandRegistry {
    let mut commands = HashMap::new();
    commands.insert("sandbox".to_string(), ("sandbox.run_context.set".to_string(), "standard|trusted|full".to_string()));
    commands.insert("compact".to_string(), ("sessions.contextCompact".to_string(), String::new()));
    commands.insert("meta".to_string(), ("meta.list".to_string(), String::new()));
    commands.insert("new".to_string(), ("sessions.new".to_string(), String::new()));
    commands.insert("help".to_string(), ("commands.list".to_string(), String::new()));
    CommandRegistry::from_table(commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_command_case_insensitively() {
        let reg = build_default();
        let (name, method) = reg.match_command("/Sandbox trusted").unwrap();
        assert_eq!(name, "sandbox");
        assert_eq!(method, "sandbox.run_context.set");
    }

    #[test]
    fn bare_slash_and_plain_chat_do_not_match() {
        let reg = build_default();
        assert!(reg.match_command("/").is_none());
        assert!(reg.match_command("hello world").is_none());
        assert!(reg.match_command("").is_none());
    }

    #[test]
    fn unknown_command_does_not_match() {
        let reg = build_default();
        assert!(reg.match_command("/bogus args").is_none());
    }

    #[test]
    fn command_names_listed_bare() {
        let reg = build_default();
        let names = reg.command_names();
        assert!(names.contains(&"sandbox".to_string()));
        assert!(names.contains(&"help".to_string()));
    }
}