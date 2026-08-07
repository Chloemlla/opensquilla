//! # Agent commands
//!
//! Implements the `agent` subcommand for running autonomous agent turns
//! directly through the engine's `TurnRunner`. Unlike `chat`, which is a
//! conversational REPL, `agent run` executes a single goal-oriented task with
//! tool use, multi-step reasoning, and a configurable iteration budget.
//!
//! The agent command drives the TurnRunner in standalone mode (Mode B) — it
//! does not require a running gateway. It supports:
//! - `agent run <goal>` — execute a task with tool rounds
//! - `agent list` — list available agent profiles
//! - `agent show <name>` — show an agent profile
//! - `agent create <name>` — create a new agent profile
//! - `agent delete <name>` — delete an agent profile

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use opensquilla_core::config::Config;
use opensquilla_core::error::Result as CoreResult;
use opensquilla_core::types::{Message, MessageRole};
use opensquilla_engine::{TurnGenerator, TurnOutcome, TurnRunnerBuilder};
use opensquilla_provider::{ChatConfig, Provider, StreamEvent};
use opensquilla_session::{SessionMode, SessionStatus};
use ratatui::prelude::Stylize;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::table::{self, Alignment, Color, Column, KeyValue, Style, Table};
use crate::util;

/// Agent subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum AgentAction {
    /// Run an autonomous agent task.
    Run {
        goal: String,
        provider: Option<String>,
        model: Option<String>,
        max_rounds: u32,
        session: Option<String>,
        system_prompt: Option<String>,
        stream: bool,
        tools: Vec<String>,
    },
    /// List agent profiles.
    List,
    /// Show an agent profile.
    Show { name: String },
    /// Create a new agent profile.
    Create {
        name: String,
        system_prompt: String,
        provider: Option<String>,
        model: Option<String>,
        max_rounds: u32,
    },
    /// Delete an agent profile.
    Delete { name: String },
    /// Execute a skill as an agent.
    Skill {
        skill: String,
        input: Option<String>,
    },
}

/// An agent profile stored on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    pub name: String,
    pub system_prompt: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_rounds: u32,
    pub tools: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl Default for AgentProfile {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            system_prompt: "You are OpenSquilla, a helpful AI assistant.".to_string(),
            provider: None,
            model: None,
            max_rounds: 8,
            tools: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }
}

/// Directory where agent profiles are stored.
fn agents_dir() -> PathBuf {
    util::data_dir().join("agents")
}

/// Path to a specific agent profile.
fn agent_path(name: &str) -> PathBuf {
    agents_dir().join(format!("{name}.json"))
}

/// Load an agent profile by name.
fn load_profile(name: &str) -> Result<AgentProfile> {
    let path = agent_path(name);
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Agent '{name}' not found at {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse agent profile at {}", path.display()))
}

/// Save an agent profile.
fn save_profile(profile: &AgentProfile) -> Result<()> {
    let dir = agents_dir();
    std::fs::create_dir_all(&dir).ok();
    let path = agent_path(&profile.name);
    let json = serde_json::to_string_pretty(profile)
        .with_context(|| format!("Failed to serialize agent '{}'", profile.name))?;
    std::fs::write(&path, json)
        .with_context(|| format!("Failed to write agent profile to {}", path.display()))?;
    Ok(())
}

/// List all saved agent profiles.
fn list_profiles() -> Result<Vec<AgentProfile>> {
    let dir = agents_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut profiles = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(profile) = serde_json::from_str::<AgentProfile>(&contents) {
                    profiles.push(profile);
                }
            }
        }
    }
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(profiles)
}

/// Run an agent subcommand.
pub async fn run_agent(action: AgentAction) -> Result<()> {
    match action {
        AgentAction::Run {
            goal,
            provider,
            model,
            max_rounds,
            session,
            system_prompt,
            stream,
            tools,
        } => {
            agent_run(
                goal,
                provider,
                model,
                max_rounds,
                session,
                system_prompt,
                stream,
                tools,
            )
            .await
        }
        AgentAction::List => agent_list().await,
        AgentAction::Show { name } => agent_show(name).await,
        AgentAction::Create {
            name,
            system_prompt,
            provider,
            model,
            max_rounds,
        } => agent_create(name, system_prompt, provider, model, max_rounds).await,
        AgentAction::Delete { name } => agent_delete(name).await,
        AgentAction::Skill { skill, input } => agent_skill(skill, input).await,
    }
}

/// A `TurnGenerator` backed by an LLM provider with streaming support.
struct AgentTurnGenerator {
    provider: Arc<dyn Provider>,
    config: ChatConfig,
    stream: bool,
}

impl std::fmt::Debug for AgentTurnGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentTurnGenerator")
            .field("provider", &self.provider.name())
            .field("config", &self.config)
            .field("stream", &self.stream)
            .finish()
    }
}

#[async_trait]
impl TurnGenerator for AgentTurnGenerator {
    async fn generate(&self, messages: &[Message]) -> CoreResult<Vec<Message>> {
        if self.stream {
            let mut stream = self
                .provider
                .stream_chat(&self.config, messages, &[])
                .await
                .map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))?;
            let mut full = String::new();
            while let Some(event) = stream.next().await {
                match event.map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))? {
                    StreamEvent::Text { text } => {
                        print!("{text}");
                        use std::io::Write;
                        let _ = std::io::stdout().flush();
                        full.push_str(&text);
                    }
                    StreamEvent::Reasoning { reasoning } => {
                        eprint!("\x1b[2m{reasoning}\x1b[0m");
                    }
                    StreamEvent::ToolCall { name, .. } => {
                        eprintln!("\n\x1b[33m[tool] {name}\x1b[0m");
                    }
                    StreamEvent::Done { .. } => {}
                    StreamEvent::Error { message } => {
                        return Err(opensquilla_core::error::Error::Provider(message));
                    }
                }
            }
            println!();
            Ok(vec![Message::assistant(&full)])
        } else {
            let response = self
                .provider
                .send_message(&self.config, messages, &[])
                .await
                .map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))?;
            Ok(response.content)
        }
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

/// Run an autonomous agent task.
#[allow(clippy::too_many_arguments)]
pub async fn agent_run(
    goal: String,
    provider: Option<String>,
    model: Option<String>,
    max_rounds: u32,
    session: Option<String>,
    system_prompt: Option<String>,
    stream: bool,
    tools: Vec<String>,
) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let provider_name = provider.unwrap_or_else(|| util::default_provider(&config));
    let model_name = model.unwrap_or_else(|| util::default_model(&config));

    let active_session = util::resolve_or_create_session(&manager, session.as_deref()).await?;

    // Update session mode to Agent.
    {
        let mut s = active_session.clone();
        s.mode = SessionMode::Agent;
        s.status = SessionStatus::Active;
        s.updated_at = chrono::Utc::now();
        manager
            .storage()
            .update_session(&s)
            .map_err(|e| anyhow::anyhow!("Failed to update session: {e}"))?;
    }

    let system = system_prompt.unwrap_or_else(|| {
        "You are OpenSquilla, an autonomous AI agent. Complete the given task using available tools. Think step by step and verify your work.".to_string()
    });

    info!(
        "Agent run: session={} provider={} model={} max_rounds={}",
        active_session.id, provider_name, model_name, max_rounds
    );

    println!("Agent Run");
    println!();
    KeyValue::new()
        .entry("Session", active_session.id.to_string())
        .entry("Provider", provider_name.clone())
        .entry("Model", model_name.clone())
        .entry("Max rounds", max_rounds.to_string())
        .entry("Streaming", stream.to_string())
        .entry("Tools", tools.join(", "))
        .print();
    println!();
    println!("{}", "Goal:".bold());
    println!("  {goal}");
    println!();

    // Record the user goal.
    manager
        .add_message(&active_session.id, "user".to_string(), goal.clone(), 0)
        .map_err(|e| anyhow::anyhow!("Failed to record goal: {e}"))?;

    let registry = util::build_provider_registry(&config)?;
    let provider = registry
        .get(&provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' not found"))?;

    let chat_config = ChatConfig {
        model: model_name.clone(),
        temperature: 0.4,
        max_tokens: 4096,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream,
        extra: Default::default(),
    };

    let generator = AgentTurnGenerator {
        provider,
        config: chat_config,
        stream,
    };

    let runner_config = opensquilla_engine::TurnRunnerConfig {
        default_model: model_name.clone(),
        default_provider: provider_name.clone(),
        default_system_prompt: system,
        streaming_enabled: stream,
        ..opensquilla_engine::TurnRunnerConfig::default()
    };
    let stages = opensquilla_engine::turn_runner::default_stages(&runner_config);

    let mut builder = TurnRunnerBuilder::new();
    for stage in stages {
        builder = builder.add_stage(stage);
    }
    let runner = builder.max_tool_rounds(max_rounds).build();

    let messages = vec![
        Message::system(&runner_config.default_system_prompt),
        Message::user(&goal),
    ];

    let start = std::time::Instant::now();
    let outcome = runner
        .run_turn(messages, &generator)
        .await
        .context("Agent turn failed")?;
    let duration_ms = start.elapsed().as_millis();

    match &outcome {
        TurnOutcome::Complete {
            messages,
            usage,
            duration_ms: turn_ms,
        } => {
            // Record the assistant response.
            let response = last_assistant_text(messages);
            if !response.is_empty() {
                manager
                    .add_message(
                        &active_session.id,
                        "assistant".to_string(),
                        response.clone(),
                        usage.total_tokens,
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to record response: {e}"))?;
            }

            println!();
            println!("{}", "Result:".bold());
            println!("{response}");
            println!();
            println!(
                "{}  {} ms  |  {} tokens  |  {} tool round(s)",
                table::ok(),
                duration_ms,
                usage.total_tokens,
                max_rounds
            );
        }
        TurnOutcome::Halted { reason, .. } => {
            warn!("Agent halted: {reason}");
            println!();
            println!("{} Agent halted: {reason}", table::warn());
        }
        TurnOutcome::Error { message, .. } => {
            println!();
            println!("{} Agent error: {message}", table::fail());
        }
    }

    // Mark session as paused (idle).
    let mut s = active_session.clone();
    s.status = SessionStatus::Paused;
    s.updated_at = chrono::Utc::now();
    let _ = manager.storage().update_session(&s);

    Ok(())
}

/// List all saved agent profiles.
pub async fn agent_list() -> Result<()> {
    let profiles = list_profiles()?;
    if profiles.is_empty() {
        println!("No agent profiles found.");
        println!("Create one with: osq agent create <name>");
        return Ok(());
    }

    println!("Agent Profiles ({})", profiles.len());
    let mut table = Table::new()
        .border(table::TableBorder::Header)
        .column(Column::new("Name"))
        .column(Column::new("Provider"))
        .column(Column::new("Model"))
        .column(Column::new("Max Rounds").align(Alignment::Right))
        .column(Column::new("Tools").max_width(30));

    for p in &profiles {
        table = table.row_owned(vec![
            p.name.clone(),
            p.provider.clone().unwrap_or_else(|| "default".into()),
            p.model.clone().unwrap_or_else(|| "default".into()),
            p.max_rounds.to_string(),
            p.tools.join(", "),
        ]);
    }
    table.print();
    Ok(())
}

/// Show an agent profile.
pub async fn agent_show(name: String) -> Result<()> {
    let profile = load_profile(&name)?;
    println!("Agent: {}", profile.name);
    KeyValue::new()
        .entry(
            "Provider",
            profile
                .provider
                .clone()
                .unwrap_or_else(|| "(default)".into()),
        )
        .entry(
            "Model",
            profile.model.clone().unwrap_or_else(|| "(default)".into()),
        )
        .entry("Max rounds", profile.max_rounds.to_string())
        .entry("Tools", profile.tools.join(", "))
        .entry("Created", profile.created_at.to_rfc3339())
        .entry("Updated", profile.updated_at.to_rfc3339())
        .print();
    println!();
    println!("{}", "System prompt:".bold());
    println!("  {}", profile.system_prompt);
    Ok(())
}

/// Create a new agent profile.
pub async fn agent_create(
    name: String,
    system_prompt: String,
    provider: Option<String>,
    model: Option<String>,
    max_rounds: u32,
) -> Result<()> {
    if agent_path(&name).exists() {
        anyhow::bail!("Agent '{name}' already exists. Use a different name or delete it first.");
    }
    let now = chrono::Utc::now();
    let profile = AgentProfile {
        name: name.clone(),
        system_prompt,
        provider,
        model,
        max_rounds,
        tools: Vec::new(),
        created_at: now,
        updated_at: now,
    };
    save_profile(&profile)?;
    println!("Created agent profile: {name}");
    Ok(())
}

/// Delete an agent profile.
pub async fn agent_delete(name: String) -> Result<()> {
    let path = agent_path(&name);
    if !path.exists() {
        anyhow::bail!("Agent '{name}' not found");
    }
    std::fs::remove_file(&path).with_context(|| format!("Failed to delete agent '{name}'"))?;
    println!("Deleted agent profile: {name}");
    Ok(())
}

/// Execute a skill as an agent task.
pub async fn agent_skill(skill: String, input: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;

    // Load the skill to get its steps.
    let loader = build_skill_loader(&config).await?;
    let skill_def = loader
        .get_skill(&skill)
        .await
        .ok_or_else(|| anyhow::anyhow!("Skill '{skill}' not found"))?;

    println!(
        "Executing skill: {} ({} steps)",
        skill_def.name,
        skill_def.steps.len()
    );
    println!();

    // Build a goal from the skill's first step and the input.
    let goal = if let Some(input) = input {
        format!("{}: {input}", skill_def.description)
    } else {
        skill_def.description.clone()
    };

    // Run as a standard agent turn.
    agent_run(
        goal,
        None,
        None,
        4,
        None,
        Some(skill_def.description),
        false,
        Vec::new(),
    )
    .await
}

/// Build a skill loader (shared with the skills command module).
async fn build_skill_loader(config: &Config) -> Result<opensquilla_skills::loader::SkillLoader> {
    use opensquilla_skills::bundled::load_bundled_skills;
    use opensquilla_skills::loader::SkillLoader;
    use opensquilla_skills::types::SkillLayer;

    let loader = SkillLoader::new();
    if let Some(skills_cfg) = config.skills.as_ref() {
        for dir in &skills_cfg.skill_dirs {
            loader.register_layer_dir(SkillLayer::Extra, std::path::Path::new(dir).to_path_buf());
        }
    }
    let managed = util::skills_dir();
    std::fs::create_dir_all(&managed).ok();
    loader.register_layer_dir(SkillLayer::Managed, managed);
    let bundled = load_bundled_skills();
    loader.register_skills(bundled);
    loader
        .scan_all()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to scan skills: {e}"))?;
    Ok(loader)
}

/// Extract the text of the last assistant message.
fn last_assistant_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Assistant)
        .map(|m| m.text_content())
        .unwrap_or_default()
}
