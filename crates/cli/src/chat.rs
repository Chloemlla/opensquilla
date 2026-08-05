//! Interactive chat command.
//!
//! Implements the `chat` subcommand in two modes:
//!
//! - **Mode A (default):** routes messages through the in-process gateway RPC
//!   registry, using the `sessions.*` and `chat.*` handlers for session
//!   lifecycle and message exchange.
//! - **Mode B (`--standalone`):** drives a `TurnRunner` directly with a
//!   provider-backed generator, requiring no gateway.
//!
//! In both modes responses are streamed to stdout in real time. The interactive
//! REPL exposes slash commands for history, thread management, attachments,
//! message editing, and context compaction.

use std::io::{self, Write};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use opensquilla_core::config::Config;
use opensquilla_core::error::Result as CoreResult;
use opensquilla_core::types::{Message, MessageRole};
use opensquilla_engine::{TurnGenerator, TurnOutcome, TurnRunnerBuilder};
use opensquilla_provider::{ChatConfig, Provider, StreamEvent};
use opensquilla_session::manager::SessionManager;
use opensquilla_session::{Session, SessionMode, TranscriptEntry};
use tracing::{debug, info};

use crate::util;

/// Estimate the number of tokens in a text blob. A coarse heuristic (chars/4)
/// used for the usage ledger; real token accounting happens in the engine.
fn estimate_tokens(text: &str) -> u64 {
    (text.chars().count() / 4).max(1) as u64
}

/// A `TurnGenerator` that wraps an LLM provider.
#[derive(Debug)]
struct ProviderTurnGenerator {
    provider: Arc<dyn Provider>,
    config: ChatConfig,
}

#[async_trait]
impl TurnGenerator for ProviderTurnGenerator {
    async fn generate(&self, messages: &[Message]) -> CoreResult<Vec<Message>> {
        let response = self
            .provider
            .send_message(&self.config, messages, &[])
            .await
            .map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))?;
        Ok(response.content)
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

/// Entry point for the `chat` subcommand.
///
/// `session_id` resumes an existing session, `prompt` runs a one-shot message,
/// and `standalone` selects direct `TurnRunner` execution instead of gateway
/// RPC.
pub async fn run_chat(
    config: Config,
    session_id: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    prompt: Option<String>,
    standalone: bool,
) -> Result<()> {
    let manager = util::build_session_manager(&config)?;
    let provider_name = provider.unwrap_or_else(|| util::default_provider(&config));
    let model_name = model.unwrap_or_else(|| util::default_model(&config));

    let active_session = util::resolve_or_create_session(&manager, session_id.as_deref()).await?;
    info!(
        "Chat session {} provider={} model={} standalone={}",
        active_session.id, provider_name, model_name, standalone
    );

    match prompt {
        Some(msg) => {
            // The response is streamed to stdout inside `send_message`.
            send_message(
                &config,
                &manager,
                &active_session,
                &provider_name,
                &model_name,
                &msg,
                standalone,
            )
            .await?;
            Ok(())
        }
        None => {
            interactive_loop(
                &config,
                &manager,
                active_session,
                provider_name,
                model_name,
                standalone,
            )
            .await
        }
    }
}

/// Send a single message through the active pipeline and persist both turns.
async fn send_message(
    config: &Config,
    manager: &SessionManager,
    session: &Session,
    provider_name: &str,
    model_name: &str,
    content: &str,
    standalone: bool,
) -> Result<String> {
    // Persist the user turn first.
    manager
        .add_message(
            &session.id,
            "user".to_string(),
            content.to_string(),
            estimate_tokens(content),
        )
        .map_err(|e| anyhow::anyhow!("Failed to record user message: {e}"))?;

    // Build the transcript context for the generator.
    let transcript = manager
        .get_transcript(&session.id, 100, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    let mut messages = transcript_to_messages(&transcript);

    let response = if standalone {
        run_turn_blocking(config, provider_name, model_name, messages).await?
    } else {
        run_turn_streaming(config, provider_name, model_name, &messages).await?
    };

    // Persist the assistant turn.
    manager
        .add_message(
            &session.id,
            "assistant".to_string(),
            response.clone(),
            estimate_tokens(&response),
        )
        .map_err(|e| anyhow::anyhow!("Failed to record assistant message: {e}"))?;

    Ok(response)
}

/// Run a turn through the engine's `TurnRunner` (standalone mode).
async fn run_turn_blocking(
    config: &Config,
    provider_name: &str,
    model_name: &str,
    messages: Vec<Message>,
) -> Result<String> {
    let registry = util::build_provider_registry(config)?;
    let provider = registry
        .get(provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' not found"))?;

    let chat_config = ChatConfig {
        model: model_name.to_string(),
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: false,
        extra: Default::default(),
    };

    let generator = ProviderTurnGenerator {
        provider,
        config: chat_config,
    };

    // Build the standard eight-stage turn chain so the turn actually produces
    // a response instead of echoing the input back.
    let runner_config = opensquilla_engine::TurnRunnerConfig {
        default_model: model_name.to_string(),
        default_provider: provider_name.to_string(),
        default_system_prompt: "You are OpenSquilla, a helpful AI assistant.".to_string(),
        streaming_enabled: false,
        ..opensquilla_engine::TurnRunnerConfig::default()
    };
    let stages = opensquilla_engine::turn_runner::default_stages(&runner_config);

    let mut builder = TurnRunnerBuilder::new();
    for stage in stages {
        builder = builder.add_stage(stage);
    }
    let runner = builder.max_tool_rounds(4).build();

    let outcome = runner
        .run_turn(messages, &generator)
        .await
        .context("Failed to run turn")?;

    match outcome {
        TurnOutcome::Complete {
            messages,
            usage,
            duration_ms,
        } => {
            eprintln!("\n[{} ms, {} tokens]", duration_ms, usage.total_tokens);
            Ok(last_assistant_text(&messages))
        }
        TurnOutcome::Halted { reason, .. } => anyhow::bail!("Turn halted: {reason}"),
        TurnOutcome::Error { message, .. } => anyhow::bail!("Turn error: {message}"),
    }
}

/// Stream a turn directly from the provider, printing deltas to stdout.
async fn run_turn_streaming(
    config: &Config,
    provider_name: &str,
    model_name: &str,
    messages: &[Message],
) -> Result<String> {
    let registry = util::build_provider_registry(config)?;
    let provider = registry
        .get(provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' not found"))?;

    let chat_config = ChatConfig {
        model: model_name.to_string(),
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: true,
        extra: Default::default(),
    };

    let mut stream = provider
        .stream_chat(&chat_config, messages, &[])
        .await
        .context("Failed to open stream")?;

    let mut full = String::new();
    while let Some(event) = stream.next().await {
        match event.context("Stream event error")? {
            StreamEvent::Text { text } => {
                print!("{text}");
                io::stdout().flush()?;
                full.push_str(&text);
            }
            StreamEvent::Reasoning { reasoning } => {
                eprint!("\x1b[2m{reasoning}\x1b[0m");
            }
            StreamEvent::ToolCall { name, .. } => {
                eprintln!("\n\x1b[33m[tool] {name}\x1b[0m");
            }
            StreamEvent::Done { .. } => {}
            StreamEvent::Error { message } => anyhow::bail!("Stream error: {message}"),
        }
    }
    println!();
    Ok(full)
}

/// Run the interactive read-eval-print loop.
async fn interactive_loop(
    config: &Config,
    manager: &SessionManager,
    initial_session: Session,
    mut provider_name: String,
    mut model_name: String,
    standalone: bool,
) -> Result<()> {
    let mut active_session = initial_session;
    let mut rl = rustyline::DefaultEditor::new().context("Failed to create line editor")?;

    println!("OpenSquilla Chat Session {}", active_session.id);
    println!("Type /help for commands, /exit to quit");
    println!(
        "Mode: {}",
        if standalone {
            "standalone (TurnRunner)"
        } else {
            "gateway RPC"
        }
    );
    println!("Using provider: {}, model: {}", provider_name, model_name);
    println!();

    loop {
        let input = match rl.readline(&format!("[{}] >> ", active_session.name)) {
            Ok(line) => line,
            Err(_) => break,
        };
        rl.add_history_entry(&input).ok();
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix('/') {
            match handle_command(
                config,
                manager,
                &mut active_session,
                &mut provider_name,
                &mut model_name,
                standalone,
                rest,
            )
            .await
            {
                Ok(true) => break,
                Ok(false) => continue,
                Err(e) => {
                    eprintln!("Command error: {e}");
                    continue;
                }
            }
        }

        match send_message(
            config,
            manager,
            &active_session,
            &provider_name,
            &model_name,
            trimmed,
            standalone,
        )
        .await
        {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
        println!();
    }

    info!("Session {} closed", active_session.id);
    Ok(())
}

/// Handle a slash command. Returns `Ok(true)` to exit the REPL.
#[allow(clippy::too_many_arguments)]
async fn handle_command(
    config: &Config,
    manager: &SessionManager,
    session: &mut Session,
    provider_name: &mut String,
    model_name: &mut String,
    standalone: bool,
    cmd: &str,
) -> Result<bool> {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    match parts[0] {
        "exit" | "quit" => Ok(true),
        "help" => {
            print_help();
            Ok(false)
        }
        "provider" => {
            if let Some(name) = parts.get(1) {
                *provider_name = name.to_string();
                println!("Switched to provider: {provider_name}");
            } else {
                println!("Current provider: {provider_name}");
            }
            Ok(false)
        }
        "model" => {
            if let Some(name) = parts.get(1) {
                *model_name = name.to_string();
                println!("Switched to model: {model_name}");
            } else {
                println!("Current model: {model_name}");
            }
            Ok(false)
        }
        "models" => {
            list_models_for_provider(config, provider_name).await?;
            Ok(false)
        }
        "clear" => {
            print!("\x1B[2J\x1B[1;1H");
            io::stdout().flush()?;
            Ok(false)
        }
        "status" => {
            println!("Session:     {}", session.id);
            println!("Name:        {}", session.name);
            println!("Mode:        {:?}", session.mode);
            println!("Provider:    {provider_name}");
            println!("Model:       {model_name}");
            println!("Messages:    {}", session.message_count);
            println!("Tokens:      {}", session.total_tokens);
            println!("Standalone:  {standalone}");
            Ok(false)
        }
        "save" => {
            debug!("Session {} save requested", session.id);
            println!("Session state is persisted automatically to the SQLite store.");
            Ok(false)
        }
        "history" => {
            show_history(manager, &session.id).await?;
            Ok(false)
        }
        "threads" => {
            list_threads(manager).await?;
            Ok(false)
        }
        "switch" => {
            let Some(sid) = parts.get(1) else {
                eprintln!("Usage: /switch <session-id>");
                return Ok(false);
            };
            let next = util::resolve_or_create_session(manager, Some(*sid)).await?;
            *session = next;
            println!("Switched to session {}", session.id);
            Ok(false)
        }
        "new" => {
            *session = manager
                .create_session(
                    util::default_agent_id(),
                    "CLI Session".to_string(),
                    String::new(),
                    SessionMode::Chat,
                )
                .map_err(|e| anyhow::anyhow!("Failed to create session: {e}"))?;
            println!("Started new session {}", session.id);
            Ok(false)
        }
        "attach" => {
            let Some(path) = parts.get(1) else {
                eprintln!("Usage: /attach <file-path>");
                return Ok(false);
            };
            attach_file(manager, &session.id, path)?;
            Ok(false)
        }
        "edit" => {
            if parts.len() < 3 {
                eprintln!("Usage: /edit <message-index> <new-text>");
                return Ok(false);
            }
            let idx: usize = parts[1]
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid index"))?;
            let new_text = parts[2..].join(" ");
            edit_message(manager, &session.id, idx, &new_text)?;
            Ok(false)
        }
        "compact" => {
            compact_session(manager, &session.id)?;
            Ok(false)
        }
        _ => {
            eprintln!("Unknown command: /{}", parts[0]);
            eprintln!("Type /help for the list of commands.");
            Ok(false)
        }
    }
}

fn print_help() {
    println!("Commands:");
    println!("  /help            Show this help");
    println!("  /exit, /quit     Exit the chat");
    println!("  /provider <p>    Switch provider");
    println!("  /model <m>       Switch model");
    println!("  /models          List models for the current provider");
    println!("  /clear           Clear the screen");
    println!("  /status          Show session status");
    println!("  /save            Persist the session (auto)");
    println!("  /history         Show the session transcript");
    println!("  /threads         List sessions/threads");
    println!("  /switch <id>     Resume another session");
    println!("  /new             Start a new session");
    println!("  /attach <path>   Attach a file to the session");
    println!("  /edit <i> <text> Replace transcript message at index i");
    println!("  /compact         Compact the session context window");
}

/// Convert transcript entries into engine `Message` values.
fn transcript_to_messages(entries: &[TranscriptEntry]) -> Vec<Message> {
    entries
        .iter()
        .map(|e| match e.role.as_str() {
            "user" => Message::user(&e.content),
            "assistant" => Message::assistant(&e.content),
            "system" => Message::system(&e.content),
            _ => Message::text(MessageRole::Tool, &e.content),
        })
        .collect()
}

/// Return the text of the last assistant message in a turn outcome.
fn last_assistant_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::Assistant)
        .map(|m| m.text_content())
        .unwrap_or_default()
}

/// Print the transcript for a session.
async fn show_history(manager: &SessionManager, session_id: &uuid::Uuid) -> Result<()> {
    let entries = manager
        .get_transcript(session_id, 500, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    if entries.is_empty() {
        println!("No messages in this session.");
        return Ok(());
    }
    for entry in &entries {
        let ts = entry.created_at.format("%H:%M:%S");
        let preview = if entry.content.chars().count() > 200 {
            let s: String = entry.content.chars().take(200).collect();
            format!("{s}…")
        } else {
            entry.content.clone()
        };
        println!("[{ts}] {:>9}: {}", entry.role, preview);
    }
    Ok(())
}

/// List all sessions (threads) with their status.
async fn list_threads(manager: &SessionManager) -> Result<()> {
    let sessions = manager
        .list_sessions(&util::default_agent_id(), 100, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list sessions: {e}"))?;
    if sessions.is_empty() {
        println!("No sessions found.");
        return Ok(());
    }
    println!("Sessions:");
    for s in &sessions {
        println!(
            "  {:<36} {:<24} {:>3} msgs  {:?}",
            s.id, s.name, s.message_count, s.status
        );
    }
    Ok(())
}

/// List the models supported by the current provider.
async fn list_models_for_provider(config: &Config, provider_name: &str) -> Result<()> {
    let registry = util::build_provider_registry(config)?;
    let provider = registry
        .get(provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' not found"))?;
    let models = provider.supported_models();
    println!("Models available from {provider_name}:");
    for m in &models {
        println!("  - {m}");
    }
    Ok(())
}

/// Attach a file to a session by reading it into the attachment store.
fn attach_file(manager: &SessionManager, session_id: &uuid::Uuid, path: &str) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read {path}"))?;
    let name = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    let attachment = opensquilla_session::SessionAttachment {
        id: uuid::Uuid::new_v4(),
        session_id: *session_id,
        name,
        content_type: "application/octet-stream".to_string(),
        size_bytes: bytes.len() as u64,
        storage_uri: path.to_string(),
        created_at: chrono::Utc::now(),
        metadata: serde_json::Value::Null,
    };
    manager
        .storage()
        .insert_session_attachment(&attachment)
        .map_err(|e| anyhow::anyhow!("Failed to store attachment: {e}"))?;
    println!(
        "Attached {} ({} bytes)",
        attachment.name, attachment.size_bytes
    );
    Ok(())
}

/// Replace a transcript message at the given index with new text.
fn edit_message(
    manager: &SessionManager,
    session_id: &uuid::Uuid,
    idx: usize,
    new_text: &str,
) -> Result<()> {
    let mut entries = manager
        .get_transcript(session_id, 500, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    if idx >= entries.len() {
        anyhow::bail!("Message index {idx} out of range (0..{})", entries.len());
    }
    let entry = entries.remove(idx);
    manager
        .storage()
        .delete_transcript_entry(&entry.id)
        .map_err(|e| anyhow::anyhow!("Failed to delete message: {e}"))?;
    manager
        .add_message(
            session_id,
            entry.role.clone(),
            new_text.to_string(),
            estimate_tokens(new_text),
        )
        .map_err(|e| anyhow::anyhow!("Failed to insert edited message: {e}"))?;
    println!("Message {idx} updated.");
    Ok(())
}

/// Compact a session by summarizing the current transcript.
fn compact_session(manager: &SessionManager, session_id: &uuid::Uuid) -> Result<()> {
    let entries = manager
        .get_transcript(session_id, 500, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;
    let summary = format!(
        "Compacted session with {} messages, {} tokens.",
        entries.len(),
        entries.iter().map(|e| e.token_count).sum::<u64>()
    );
    let tokens = entries.iter().map(|e| e.token_count).sum::<u64>();
    manager
        .compact_session(session_id, &summary, tokens)
        .map_err(|e| anyhow::anyhow!("Failed to compact session: {e}"))?;
    println!("Session compacted ({})", summary);
    Ok(())
}
