//! Slash command directory RPC handlers.
//!
//! Provides `rpc_commands` for a read-only in-memory slash command directory.
//! Commands are registered at startup and looked up by name or category.

use std::collections::HashMap;
use std::sync::Arc;
use opensquilla_core::error::AppError;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::rpc::{rpc_handler, RpcRegistry};

/// The category of a slash command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandCategory {
    Session,
    Chat,
    Agent,
    Memory,
    Tool,
    System,
    Meta,
}

/// How a command is executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionKind {
    Internal,
    Rpc,
    Tool,
}

/// A slash command definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSpec {
    pub name: String,
    pub description: String,
    pub category: CommandCategory,
    pub execution: ExecutionKind,
    pub aliases: Vec<String>,
    pub args_hint: Option<String>,
}

/// In-memory command directory.
#[derive(Clone, Default)]
pub struct CommandDirectory {
    commands: Arc<Mutex<HashMap<String, CommandSpec>>>,
}

impl CommandDirectory {
    /// Create an empty directory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a directory seeded with the built-in commands.
    pub fn with_defaults() -> Self {
        let dir = Self::new();
        let commands = vec![
            CommandSpec {
                name: "/help".to_string(),
                description: "List available commands.".to_string(),
                category: CommandCategory::System,
                execution: ExecutionKind::Internal,
                aliases: vec!["/h".to_string()],
                args_hint: None,
            },
            CommandSpec {
                name: "/sessions".to_string(),
                description: "List and manage sessions.".to_string(),
                category: CommandCategory::Session,
                execution: ExecutionKind::Rpc,
                aliases: vec!["/s".to_string()],
                args_hint: Some("[list|new|switch <id>]".to_string()),
            },
            CommandSpec {
                name: "/clear".to_string(),
                description: "Clear the current conversation history.".to_string(),
                category: CommandCategory::Chat,
                execution: ExecutionKind::Rpc,
                aliases: vec![],
                args_hint: None,
            },
            CommandSpec {
                name: "/memory".to_string(),
                description: "Search and manage memories.".to_string(),
                category: CommandCategory::Memory,
                execution: ExecutionKind::Rpc,
                aliases: vec!["/mem".to_string()],
                args_hint: Some("[search <query>|add <text>]".to_string()),
            },
            CommandSpec {
                name: "/tools".to_string(),
                description: "List available tools.".to_string(),
                category: CommandCategory::Tool,
                execution: ExecutionKind::Rpc,
                aliases: vec![],
                args_hint: None,
            },
            CommandSpec {
                name: "/skills".to_string(),
                description: "List and manage skills.".to_string(),
                category: CommandCategory::Meta,
                execution: ExecutionKind::Rpc,
                aliases: vec![],
                args_hint: Some("[list|enable <id>|disable <id>]".to_string()),
            },
            CommandSpec {
                name: "/agent".to_string(),
                description: "Run a standalone agent turn.".to_string(),
                category: CommandCategory::Agent,
                execution: ExecutionKind::Internal,
                aliases: vec![],
                args_hint: Some("<prompt>".to_string()),
            },
            CommandSpec {
                name: "/cost".to_string(),
                description: "Show usage and cost summary.".to_string(),
                category: CommandCategory::System,
                execution: ExecutionKind::Rpc,
                aliases: vec![],
                args_hint: None,
            },
        ];
        for cmd in commands {
            dir.register(cmd);
        }
        dir
    }

    /// Register a command.
    pub fn register(&self, command: CommandSpec) {
        self.commands.lock().insert(command.name.clone(), command);
    }

    /// Look up a command by name or alias.
    pub fn get(&self, name: &str) -> Option<CommandSpec> {
        let commands = self.commands.lock();
        if let Some(cmd) = commands.get(name) {
            return Some(cmd.clone());
        }
        // Search aliases
        commands
            .values()
            .find(|c| c.aliases.iter().any(|a| a == name))
            .cloned()
    }

    /// List all commands, optionally filtered by category.
    pub fn list(&self, category: Option<CommandCategory>) -> Vec<CommandSpec> {
        let commands = self.commands.lock();
        let mut result: Vec<CommandSpec> = commands
            .values()
            .filter(|c| category.map_or(true, |cat| c.category == cat))
            .cloned()
            .collect();
        result.sort_by(|a, b| a.name.cmp(&b.name));
        result
    }

    /// Remove a command.
    pub fn remove(&self, name: &str) -> bool {
        self.commands.lock().remove(name).is_some()
    }
}

/// Register command directory RPC handlers on the given registry.
pub fn register_commands_handlers(registry: &mut RpcRegistry, directory: CommandDirectory) {
    let directory = Arc::new(directory);

    // commands.list — list all commands, optionally filtered by category
    registry.register(rpc_handler("commands.list", {
        let directory = directory.clone();
        move |params| {
            let directory = directory.clone();
            async move {
                let category = params
                    .get("category")
                    .and_then(|v| v.as_str())
                    .and_then(parse_category);
                let commands = directory.list(category);
                Ok(serde_json::json!({
                    "commands": commands,
                    "count": commands.len(),
                }))
            }
        }
    }));

    // commands.get — fetch a command by name or alias
    registry.register(rpc_handler("commands.get", {
        let directory = directory.clone();
        move |params| {
            let directory = directory.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?;
                match directory.get(name) {
                    Some(cmd) => Ok(serde_json::to_value(cmd)
                        .map_err(|e| AppError::internal(e.to_string()))?),
                    None => Err(AppError::not_found(format!("Command '{name}' not found"))),
                }
            }
        }
    }));

    // commands.register — register a new command
    registry.register(rpc_handler("commands.register", {
        let directory = directory.clone();
        move |params| {
            let directory = directory.clone();
            async move {
                let name = params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'name' parameter"))?;
                let description = params
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let category = params
                    .get("category")
                    .and_then(|v| v.as_str())
                    .and_then(parse_category)
                    .unwrap_or(CommandCategory::System);
                let execution = params
                    .get("execution")
                    .and_then(|v| v.as_str())
                    .and_then(parse_execution)
                    .unwrap_or(ExecutionKind::Internal);
                let aliases: Vec<String> = params
                    .get("aliases")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let args_hint = params
                    .get("args_hint")
                    .and_then(|v| v.as_str())
                    .map(String::from);

                let cmd = CommandSpec {
                    name: name.to_string(),
                    description,
                    category,
                    execution,
                    aliases,
                    args_hint,
                };
                directory.register(cmd.clone());
                Ok(serde_json::to_value(cmd)
                    .map_err(|e| AppError::internal(e.to_string()))?)
            }
        }
    }));

    // commands.categories — list all categories that have commands
    registry.register(rpc_handler("commands.categories", {
        let directory = directory.clone();
        move |_params| {
            let directory = directory.clone();
            async move {
                let commands = directory.list(None);
                let categories: Vec<String> = commands
                    .iter()
                    .map(|c| format!("{:?}", c.category).to_lowercase())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                Ok(serde_json::json!({"categories": categories}))
            }
        }
    }));

    // commands.resolve — resolve an alias to its canonical command name
    registry.register(rpc_handler("commands.resolve", {
        let directory = directory.clone();
        move |params| {
            let directory = directory.clone();
            async move {
                let alias = params
                    .get("alias")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'alias' parameter"))?;
                match directory.get(alias) {
                    Some(cmd) => Ok(serde_json::json!({
                        "alias": alias,
                        "name": cmd.name,
                        "found": true,
                    })),
                    None => Ok(serde_json::json!({
                        "alias": alias,
                        "name": null,
                        "found": false,
                    })),
                }
            }
        }
    }));
}

fn parse_category(s: &str) -> Option<CommandCategory> {
    match s.to_ascii_lowercase().as_str() {
        "session" => Some(CommandCategory::Session),
        "chat" => Some(CommandCategory::Chat),
        "agent" => Some(CommandCategory::Agent),
        "memory" => Some(CommandCategory::Memory),
        "tool" => Some(CommandCategory::Tool),
        "system" => Some(CommandCategory::System),
        "meta" => Some(CommandCategory::Meta),
        _ => None,
    }
}

fn parse_execution(s: &str) -> Option<ExecutionKind> {
    match s.to_ascii_lowercase().as_str() {
        "internal" => Some(ExecutionKind::Internal),
        "rpc" => Some(ExecutionKind::Rpc),
        "tool" => Some(ExecutionKind::Tool),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_commands_list_defaults() {
        let directory = CommandDirectory::with_defaults();
        let mut registry = RpcRegistry::new();
        register_commands_handlers(&mut registry, directory);

        let r = registry.dispatch("commands.list", serde_json::Value::Null).await;
        let resp = r.unwrap().unwrap();
        assert!(resp["count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_commands_get_and_resolve() {
        let directory = CommandDirectory::with_defaults();
        let mut registry = RpcRegistry::new();
        register_commands_handlers(&mut registry, directory);

        let r = registry
            .dispatch("commands.get", serde_json::json!({"name": "/help"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["name"], "/help");

        let r = registry
            .dispatch("commands.resolve", serde_json::json!({"alias": "/h"}))
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["name"], "/help");
    }

    #[tokio::test]
    async fn test_commands_register() {
        let directory = CommandDirectory::new();
        let mut registry = RpcRegistry::new();
        register_commands_handlers(&mut registry, directory);

        let params = serde_json::json!({
            "name": "/custom",
            "description": "A custom command",
            "category": "system",
            "execution": "internal",
        });
        let r = registry.dispatch("commands.register", params).await;
        assert!(r.unwrap().is_ok());
    }
}
