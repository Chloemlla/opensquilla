//! Slash command registry.
//!
//! Slash commands (e.g. `/clear`, `/model`, `/exit`) are defined declaratively
//! and registered in a `CommandRegistry`. The registry supports lookup,
//! listing (optionally filtered by surface), and resolution of aliases.
//!
//! This module is deliberately I/O-free: it only models commands and their
//! metadata. Execution is handled by the caller.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The surface on which a command may be invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Surface {
    /// The command is shown in interactive (chat) UIs.
    Visible,
    /// The command is hidden from the command list but still invokable.
    Hidden,
}

impl Surface {
    /// Returns true if the surface is visible in command listings.
    pub fn is_visible(self) -> bool {
        matches!(self, Surface::Visible)
    }
}

/// How a command is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionKind {
    /// The command runs entirely in-process (no external process).
    Local,
    /// The command dispatches to a remote service or external process.
    Remote,
}

/// The functional category of a slash command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCategory {
    /// Conversation management (clear, reset, history).
    Conversation,
    /// Model and provider configuration.
    Model,
    /// Tool and capability management.
    Tool,
    /// File and workspace operations.
    File,
    /// Memory and knowledge operations.
    Memory,
    /// Session lifecycle operations.
    Session,
    /// Agent and sub-agent operations.
    Agent,
    /// System and runtime operations (exit, shutdown).
    System,
    /// Help and discovery.
    Help,
    /// Plugin or extension operations.
    Plugin,
    /// Miscellaneous commands.
    Misc,
}

impl CommandCategory {
    /// Return a stable machine-readable name for this category.
    pub fn as_str(self) -> &'static str {
        match self {
            CommandCategory::Conversation => "conversation",
            CommandCategory::Model => "model",
            CommandCategory::Tool => "tool",
            CommandCategory::File => "file",
            CommandCategory::Memory => "memory",
            CommandCategory::Session => "session",
            CommandCategory::Agent => "agent",
            CommandCategory::System => "system",
            CommandCategory::Help => "help",
            CommandCategory::Plugin => "plugin",
            CommandCategory::Misc => "misc",
        }
    }
}

/// A single slash command definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// The canonical name of the command, without the leading slash.
    pub name: String,
    /// A short one-line description shown in help listings.
    pub description: String,
    /// Aliases that also resolve to this command (without leading slash).
    #[serde(default)]
    pub aliases: Vec<String>,
    /// The surface on which this command is available.
    #[serde(default)]
    pub surface: Surface,
    /// How the command is executed.
    #[serde(default)]
    pub execution_kind: ExecutionKind,
    /// The functional category of the command.
    #[serde(default)]
    pub category: CommandCategory,
    /// The number of arguments this command expects (0 = variadic).
    #[serde(default)]
    pub arity: usize,
}

impl Command {
    /// Create a minimal visible, local command.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            aliases: Vec::new(),
            surface: Surface::Visible,
            execution_kind: ExecutionKind::Local,
            category: CommandCategory::Misc,
            arity: 0,
        }
    }

    /// Add aliases to this command.
    pub fn alias(mut self, alias: impl Into<String>) -> Self {
        self.aliases.push(alias.into());
        self
    }

    /// Mark the command as hidden.
    pub fn hidden(mut self) -> Self {
        self.surface = Surface::Hidden;
        self
    }

    /// Mark the command as remotely executed.
    pub fn remote(mut self) -> Self {
        self.execution_kind = ExecutionKind::Remote;
        self
    }

    /// Set the command category.
    pub fn category(mut self, category: CommandCategory) -> Self {
        self.category = category;
        self
    }

    /// Set the expected argument count.
    pub fn arity(mut self, arity: usize) -> Self {
        self.arity = arity;
        self
    }

    /// The full invocation string including the leading slash.
    pub fn invocation(&self) -> String {
        format!("/{}", self.name)
    }
}

/// A registry of slash commands.
///
/// The registry maps both canonical names and aliases to command definitions.
/// It is not thread-safe by itself; wrap it in a `Mutex` or `RwLock` when
/// sharing across tasks.
#[derive(Debug, Clone, Default)]
pub struct CommandRegistry {
    /// Commands indexed by canonical name.
    commands: HashMap<String, Command>,
    /// Alias -> canonical name index.
    alias_index: HashMap<String, String>,
}

impl CommandRegistry {
    /// Create a new empty command registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a command and all of its aliases.
    ///
    /// Returns `Err` if the command name or any alias collides with an
    /// existing registration.
    pub fn register(&mut self, command: Command) -> Result<(), String> {
        if self.commands.contains_key(&command.name) {
            return Err(format!(
                "command '/{}' is already registered",
                command.name
            ));
        }
        for alias in &command.aliases {
            if self.commands.contains_key(alias)
                || self.alias_index.contains_key(alias)
            {
                return Err(format!(
                    "alias '/{}' collides with an existing registration",
                    alias
                ));
            }
        }

        // Insert aliases first (they must not collide with the canonical name).
        for alias in &command.aliases {
            self.alias_index.insert(alias.clone(), command.name.clone());
        }
        self.commands.insert(command.name.clone(), command);
        Ok(())
    }

    /// Look up a command by canonical name (without leading slash).
    pub fn get(&self, name: &str) -> Option<&Command> {
        self.commands
            .get(name)
            .or_else(|| self.alias_index.get(name).and_then(|n| self.commands.get(n)))
    }

    /// Resolve a command invocation string such as `/clear` or `/help`.
    ///
    /// The leading slash is optional. Returns `None` if the command is unknown.
    pub fn resolve(&self, input: &str) -> Option<&Command> {
        let trimmed = input.trim();
        let name = trimmed.strip_prefix('/').unwrap_or(trimmed);
        self.get(name)
    }

    /// List all commands, optionally restricted to visible surfaces.
    pub fn list(&self, visible_only: bool) -> Vec<&Command> {
        let mut commands: Vec<&Command> = self
            .commands
            .values()
            .filter(|c| !visible_only || c.surface.is_visible())
            .collect();
        commands.sort_by(|a, b| a.name.cmp(&b.name));
        commands
    }

    /// List commands in a specific category.
    pub fn by_category(&self, category: CommandCategory) -> Vec<&Command> {
        self.commands
            .values()
            .filter(|c| c.category == category)
            .collect()
    }

    /// Remove a command and its aliases by canonical name.
    pub fn unregister(&mut self, name: &str) -> Option<Command> {
        let command = self.commands.remove(name)?;
        for alias in &command.aliases {
            self.alias_index.remove(alias);
        }
        Some(command)
    }

    /// The total number of registered commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns true if the registry contains no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> CommandRegistry {
        let mut registry = CommandRegistry::new();
        registry
            .register(
                Command::new("clear", "Clear the conversation")
                    .category(CommandCategory::Conversation)
                    .alias("cls"),
            )
            .unwrap();
        registry
            .register(
                Command::new("model", "Switch model").category(CommandCategory::Model).remote(),
            )
            .unwrap();
        registry
            .register(Command::new("debug", "Toggle debug").hidden())
            .unwrap();
        registry
    }

    #[test]
    fn test_register_and_get() {
        let registry = sample_registry();
        assert_eq!(registry.get("clear").unwrap().name, "clear");
        assert_eq!(registry.get("cls").unwrap().name, "clear");
        assert!(registry.get("nope").is_none());
    }

    #[test]
    fn test_resolve_with_slash() {
        let registry = sample_registry();
        assert_eq!(registry.resolve("/model").unwrap().name, "model");
        assert_eq!(registry.resolve("/cls").unwrap().name, "clear");
        assert!(registry.resolve("/missing").is_none());
    }

    #[test]
    fn test_list_visible_only() {
        let registry = sample_registry();
        let visible = registry.list(true);
        assert!(visible.iter().all(|c| c.surface.is_visible()));
        assert!(visible.iter().any(|c| c.name == "clear"));
        assert!(visible.iter().all(|c| c.name != "debug"));
    }

    #[test]
    fn test_by_category() {
        let registry = sample_registry();
        let conv = registry.by_category(CommandCategory::Conversation);
        assert_eq!(conv.len(), 1);
        assert_eq!(conv[0].name, "clear");
    }

    #[test]
    fn test_duplicate_register_fails() {
        let mut registry = CommandRegistry::new();
        registry.register(Command::new("clear", "a")).unwrap();
        assert!(registry.register(Command::new("clear", "b")).is_err());
        assert!(registry.register(Command::new("clr", "c").alias("clear")).is_err());
    }

    #[test]
    fn test_unregister() {
        let mut registry = sample_registry();
        assert!(registry.unregister("clear").is_some());
        assert!(registry.get("cls").is_none());
        assert_eq!(registry.len(), 2);
    }
}
