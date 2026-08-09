//! Agent bridge: connects Tauri commands to the Rust agent engine.
//!
//! This is the core integration between the Vue 3 frontend and the Rust agent
//! runtime. Instead of WebSocket RPC to a Python gateway, the frontend calls
//! `invoke('send_message', { ... })` and receives streaming events via Tauri's
//! `listen()` API.
//!
//! ## Architecture
//!
//! ```text
//! Vue 3 Frontend                  Tauri (Rust)                     Engine
//! ──────────────                  ────────────                     ──────
//! invoke('send_message')
//!            ───────────────►  send_message()
//!                               ├─ create TurnRunner
//!                               ├─ create TurnGenerator
//!                               ├─ wire streaming_tx ──► mpsc channel
//!                               ├─ tokio::spawn turn task
//!                               │   ├─ emit TurnStart
//!                               │   ├─ AgentRuntime::execute_turn()
//!                               │   │   ├─ TurnRunner::run_turn()
//!                               │   │   │   ├─ stages (Harness, Bootstrap, ...)
//!                               │   │   │   └─ ProviderStage ──► generator.generate()
//!                               │   │   │                      └─ streaming_tx.send(StreamEvent)
//!                               │   │   └─ TurnOutcome
//!                               │   ├─ forward mpsc events ──► AppHandle::emit()
//!                               │   └─ emit TurnComplete
//!                               └─ return (invoke resolves)
//! ◄── listen('agent:stream:{id}')
//!     TurnStart, StreamEvent, ToolCallStart, TurnComplete
//! ```
//!
//! The `send_message` command returns immediately after spawning the turn
//! task. All streaming events are delivered via Tauri events on the channel
//! `agent:stream:{session_id}`. The frontend subscribes to this channel with
//! `listen()`.

use crate::error::{TauriError, TauriResult};
use crate::ipc::*;
use crate::state::AppState;
use async_trait::async_trait;
use opensquilla_core::error::Result as CoreResult;
use opensquilla_core::events::TurnEvent;
use opensquilla_core::types::{Message, MessageRole};
use opensquilla_engine::turn_runner::agent_bootstrap::AgentBootstrapStage;
use opensquilla_engine::turn_runner::compaction::CompactionStage;
use opensquilla_engine::turn_runner::finalizer::FinalizerStage;
use opensquilla_engine::turn_runner::harness::HarnessStage;
use opensquilla_engine::turn_runner::input::InputStage;
use opensquilla_engine::turn_runner::provider::ProviderStage;
use opensquilla_engine::turn_runner::stream_consumer::StreamConsumerStage;
use opensquilla_engine::{
    AgentHandle, AgentRuntime, AgentState, TurnGenerator, TurnOutcome, TurnRunnerBuilder,
};
use opensquilla_provider::{ChatConfig, Provider, ProviderResponse};
use opensquilla_session::ForkConfig;
use std::fmt;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::mpsc;
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

/// The event channel prefix for agent stream events.
/// The full channel name is `agent:stream:{session_id}`.
const AGENT_STREAM_EVENT_PREFIX: &str = "agent:stream:";

/// The event channel for session list updates.
pub const SESSION_LIST_CHANGED_EVENT: &str = "sessions:list-changed";

// ---------------------------------------------------------------------------
// TurnGenerator implementation (bridges Provider trait to engine)
// ---------------------------------------------------------------------------

/// A `TurnGenerator` that wraps an LLM `Provider` and `ChatConfig`.
///
/// The engine's `TurnRunner` calls `generate()` to get model responses. This
/// implementation delegates to the provider's `send_message()` (non-streaming)
/// or `stream_chat()` (streaming) method, forwarding streaming events through
/// an mpsc channel.
///
/// This is the critical adapter that connects the provider crate's `Provider`
/// trait to the engine crate's `TurnGenerator` trait.
pub struct ProviderTurnGenerator {
    provider: Arc<dyn Provider>,
    config: ChatConfig,
    streaming_tx: Option<mpsc::Sender<opensquilla_core::events::StreamEvent>>,
}

impl fmt::Debug for ProviderTurnGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderTurnGenerator")
            .field("config", &self.config)
            .field("streaming_tx", &self.streaming_tx.is_some())
            .finish()
    }
}

impl ProviderTurnGenerator {
    /// Create a new provider-backed turn generator.
    pub fn new(provider: Arc<dyn Provider>, config: ChatConfig) -> Self {
        Self {
            provider,
            config,
            streaming_tx: None,
        }
    }

    /// Attach a streaming event sender. When set, the generator will forward
    /// stream events from the provider to this channel.
    pub fn with_streaming_tx(
        mut self,
        tx: mpsc::Sender<opensquilla_core::events::StreamEvent>,
    ) -> Self {
        self.streaming_tx = Some(tx);
        self
    }
}

#[async_trait]
impl TurnGenerator for ProviderTurnGenerator {
    /// Generate a response by calling the provider.
    ///
    /// In non-streaming mode, calls `send_message` and returns the response
    /// messages. In streaming mode, calls `stream_chat` and forwards each
    /// `StreamEvent` to the streaming channel, then assembles the final
    /// messages from the accumulated deltas.
    #[instrument(skip(self, messages), fields(model = %self.config.model, provider = %self.provider.name()))]
    async fn generate(&self, messages: &[Message]) -> CoreResult<Vec<Message>> {
        let tools: &[opensquilla_core::types::ToolDefinition] = &[];

        if self.streaming_tx.is_some() && self.config.stream {
            // Streaming mode: call stream_chat and forward events.
            let stream = self
                .provider
                .stream_chat(&self.config, messages, tools)
                .await
                .map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))?;

            use futures::StreamExt;
            let mut stream = stream;
            let mut accumulated_text = String::new();
            let mut accumulated_reasoning = String::new();
            let mut tool_calls: Vec<opensquilla_core::types::ToolCall> = Vec::new();
            let mut final_usage: Option<opensquilla_core::types::Usage> = None;
            let mut stop_reason: Option<String> = None;

            while let Some(result) = stream.next().await {
                match result {
                    Ok(event) => {
                        use opensquilla_provider::StreamEvent as PStreamEvent;
                        match event {
                            PStreamEvent::Text { text } => {
                                accumulated_text.push_str(&text);
                                // Forward as a core StreamEvent.
                                if let Some(tx) = &self.streaming_tx {
                                    let _ = tx
                                        .send(
                                            opensquilla_core::events::StreamEvent::ContentBlockDelta {
                                                index: 0,
                                                delta: opensquilla_core::events::ContentBlockDelta::TextDelta {
                                                    text,
                                                },
                                            },
                                        )
                                        .await;
                                }
                            }
                            PStreamEvent::Reasoning { reasoning } => {
                                accumulated_reasoning.push_str(&reasoning);
                                if let Some(tx) = &self.streaming_tx {
                                    let _ = tx
                                        .send(
                                            opensquilla_core::events::StreamEvent::ContentBlockDelta {
                                                index: 0,
                                                delta: opensquilla_core::events::ContentBlockDelta::ReasoningDelta {
                                                    reasoning,
                                                },
                                            },
                                        )
                                        .await;
                                }
                            }
                            PStreamEvent::ToolCall {
                                id,
                                name,
                                arguments,
                            } => {
                                // Try to parse the accumulated arguments as JSON.
                                let input: serde_json::Value = serde_json::from_str(&arguments)
                                    .unwrap_or(serde_json::Value::Null);
                                let call = opensquilla_core::types::ToolCall::new(id, name, input);
                                if let Some(tx) = &self.streaming_tx {
                                    let _ = tx
                                        .send(opensquilla_core::events::StreamEvent::ContentBlockStart {
                                            index: tool_calls.len() + 1,
                                            block: opensquilla_core::types::ContentBlock::ToolUse(call.clone()),
                                        })
                                        .await;
                                }
                                tool_calls.push(call);
                            }
                            PStreamEvent::Done {
                                usage,
                                stop_reason: sr,
                                ..
                            } => {
                                final_usage = usage;
                                stop_reason = sr;
                            }
                            PStreamEvent::Error { message } => {
                                error!(error = %message, "Provider stream error");
                                if let Some(tx) = &self.streaming_tx {
                                    let _ = tx
                                        .send(opensquilla_core::events::StreamEvent::Error {
                                            message: message.clone(),
                                            code: Some("PROVIDER_STREAM_ERROR".to_string()),
                                        })
                                        .await;
                                }
                                return Err(opensquilla_core::error::Error::Provider(message));
                            }
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Provider stream item error");
                        return Err(opensquilla_core::error::Error::Provider(e.to_string()));
                    }
                }
            }

            // Send content block stop events.
            if let Some(tx) = &self.streaming_tx {
                if !accumulated_text.is_empty() {
                    let _ = tx
                        .send(opensquilla_core::events::StreamEvent::ContentBlockStop { index: 0 })
                        .await;
                }
                for (i, _) in tool_calls.iter().enumerate() {
                    let _ = tx
                        .send(opensquilla_core::events::StreamEvent::ContentBlockStop {
                            index: i + 1,
                        })
                        .await;
                }
                // Send message delta with stop reason.
                let _ = tx
                    .send(opensquilla_core::events::StreamEvent::MessageDelta {
                        delta: opensquilla_core::events::MessageDelta {
                            stop_reason: stop_reason.clone(),
                            stop_sequence: None,
                        },
                        usage: final_usage,
                    })
                    .await;
            }

            // Build the response message from accumulated content.
            let mut content = Vec::new();
            if !accumulated_reasoning.is_empty() {
                content.push(opensquilla_core::types::ContentBlock::Reasoning(
                    accumulated_reasoning,
                ));
            }
            if !accumulated_text.is_empty() {
                content.push(opensquilla_core::types::ContentBlock::Text(
                    accumulated_text,
                ));
            }
            for call in tool_calls {
                content.push(opensquilla_core::types::ContentBlock::ToolUse(call));
            }

            // Send message stop event.
            if let Some(tx) = &self.streaming_tx {
                let _ = tx
                    .send(opensquilla_core::events::StreamEvent::MessageStop {
                        content: content.clone(),
                        usage: final_usage,
                    })
                    .await;
            }

            let message = Message {
                role: MessageRole::Assistant,
                content,
                name: None,
                tool_call_id: None,
                tool_calls: None,
                tool_result: None,
            };

            Ok(vec![message])
        } else {
            // Non-streaming mode: call send_message directly.
            let response: ProviderResponse = self
                .provider
                .send_message(&self.config, messages, tools)
                .await
                .map_err(|e| opensquilla_core::error::Error::Provider(e.to_string()))?;

            // Forward the response as stream events if a channel is attached.
            if let Some(tx) = &self.streaming_tx {
                for (index, msg) in response.content.iter().enumerate() {
                    for block in &msg.content {
                        let _ = tx
                            .send(opensquilla_core::events::StreamEvent::ContentBlockStart {
                                index,
                                block: block.clone(),
                            })
                            .await;
                        let _ = tx
                            .send(opensquilla_core::events::StreamEvent::ContentBlockStop { index })
                            .await;
                    }
                }
                let _ = tx
                    .send(opensquilla_core::events::StreamEvent::MessageStop {
                        content: response
                            .content
                            .iter()
                            .flat_map(|m| m.content.clone())
                            .collect(),
                        usage: Some(response.usage),
                    })
                    .await;
            }

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

// ---------------------------------------------------------------------------
// Turn task orchestration
// ---------------------------------------------------------------------------

/// Configuration for building a TurnRunner for a message send.
#[derive(Debug, Clone)]
pub struct TurnConfig {
    pub max_tool_rounds: u32,
    pub max_messages_before_compaction: usize,
    pub streaming: bool,
}

impl Default for TurnConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 10,
            max_messages_before_compaction: 50,
            streaming: true,
        }
    }
}

/// Build a `TurnRunner` with the standard set of stages.
///
/// The stages execute in order:
/// 1. Harness — validates input
/// 2. Bootstrap — loads model/provider info from the generator
/// 3. Compaction — manages context window limits
/// 4. Input — processes input messages
/// 5. Provider — calls the LLM provider via the generator
/// 6. Stream — forwards streaming events
/// 7. Finalizer — post-processing and produces the output
pub fn build_turn_runner(config: &TurnConfig) -> opensquilla_engine::TurnRunner {
    let builder = TurnRunnerBuilder::new()
        .max_tool_rounds(config.max_tool_rounds)
        .streaming(config.streaming)
        .add_stage(Box::new(HarnessStage::new()))
        .add_stage(Box::new(AgentBootstrapStage::new("")))
        .add_stage(Box::new(CompactionStage::new(
            config.max_messages_before_compaction,
        )))
        .add_stage(Box::new(InputStage::new(
            config.max_messages_before_compaction,
        )))
        .add_stage(Box::new(ProviderStage::new(
            "default".to_string(),
            "default".to_string(),
            config.streaming,
        )))
        .add_stage(Box::new(StreamConsumerStage::new()))
        .add_stage(Box::new(FinalizerStage::new()));

    builder.build()
}

/// Spawn a turn execution task that streams events to the frontend.
///
/// This function:
/// 1. Creates an mpsc channel for streaming events.
/// 2. Wires the streaming sender into the TurnRunner's StageContext.
/// 3. Spawns a task that:
///    a. Emits `TurnStart` to the frontend.
///    b. Executes the turn via `AgentRuntime::execute_turn()`.
///    c. Forwards all engine events (TurnEvent, StreamEvent) to the frontend.
///    d. Emits `TurnComplete` or `TurnError` when done.
///
/// The task runs independently; the `send_message` command returns immediately
/// after spawning it.
pub fn spawn_turn_task(
    app: AppHandle,
    runtime: Arc<AgentRuntime>,
    session_id: String,
    messages: Vec<Message>,
    generator: Arc<dyn TurnGenerator>,
    _turn_config: TurnConfig,
) -> String {
    let turn_id = Uuid::new_v4().to_string();
    let event_channel = format!("{AGENT_STREAM_EVENT_PREFIX}{session_id}");

    // Channel for streaming events from the engine stages to the task.
    let (stream_tx, mut stream_rx) = mpsc::channel::<opensquilla_core::events::StreamEvent>(256);

    // Channel for turn events from the runtime.
    let (turn_event_tx, mut turn_event_rx) = mpsc::channel::<TurnEvent>(256);

    // Clone the app handle for the spawned task.
    let task_app = app.clone();
    let task_session_id = session_id.clone();
    let task_turn_id = turn_id.clone();
    let task_event_channel = event_channel.clone();
    let task_runtime = runtime.clone();
    let task_generator = generator.clone();

    // Spawn the turn execution task.
    tokio::spawn(async move {
        let timestamp = chrono::Utc::now().to_rfc3339();

        // Emit TurnStart.
        let turn_start = MessageStreamEvent::TurnStart {
            turn_id: task_turn_id.clone(),
            session_id: task_session_id.clone(),
            timestamp: timestamp.clone(),
        };
        if let Err(e) = task_app.emit(&task_event_channel, &turn_start) {
            warn!(error = %e, "Failed to emit TurnStart");
        }

        // Ensure the runtime is running.
        if !task_runtime.is_running().await {
            let error_event = MessageStreamEvent::TurnError {
                turn_id: task_turn_id.clone(),
                session_id: task_session_id.clone(),
                message: "Agent runtime is not running".to_string(),
                code: Some("RUNTIME_NOT_RUNNING".to_string()),
            };
            let _ = task_app.emit(&task_event_channel, &error_event);
            return;
        }

        // Register the agent if not already registered. We use the session ID
        // as the agent ID for simplicity.
        let agent_id = task_session_id.clone();
        if task_runtime.get_agent(&agent_id).is_none() {
            let handle = AgentHandle {
                agent_id: agent_id.clone(),
                state: AgentState::Idle,
                total_usage: opensquilla_core::types::Usage::default(),
                turn_count: 0,
            };
            task_runtime.register_agent(agent_id.clone(), handle);
        }

        // Spawn a sub-task to forward streaming events from the mpsc channel
        // to the frontend via Tauri events.
        let forward_app = task_app.clone();
        let forward_channel = task_event_channel.clone();
        let forward_session = task_session_id.clone();
        let forward_turn = task_turn_id.clone();
        let forward_task = tokio::spawn(async move {
            while let Some(stream_event) = stream_rx.recv().await {
                let payload = StreamEventPayload::from(stream_event);
                let event = MessageStreamEvent::StreamEvent { event: payload };
                if let Err(e) = forward_app.emit(&forward_channel, &event) {
                    warn!(error = %e, "Failed to emit stream event");
                    break;
                }
            }
            let _ = (forward_session, forward_turn);
        });

        // Spawn a sub-task to forward turn events.
        let turn_app = task_app.clone();
        let turn_channel = task_event_channel.clone();
        let turn_session = task_session_id.clone();
        let turn_turn_id = task_turn_id.clone();
        let turn_event_task = tokio::spawn(async move {
            while let Some(turn_event) = turn_event_rx.recv().await {
                let event = match &turn_event {
                    TurnEvent::GenerationStart { .. } => Some(MessageStreamEvent::from(turn_event)),
                    TurnEvent::Compaction { .. } => Some(MessageStreamEvent::from(turn_event)),
                    _ => None,
                };
                if let Some(event) = event {
                    let _ = turn_app.emit(&turn_channel, &event);
                }
            }
            let _ = (turn_session, turn_turn_id);
        });

        // Execute the turn.
        info!(turn_id = %task_turn_id, session_id = %task_session_id, "Executing turn");
        let outcome = task_runtime
            .execute_turn(&agent_id, messages, task_generator.as_ref())
            .await;

        // Drop the stream sender so the forward task can complete.
        drop(stream_tx);
        drop(turn_event_tx);

        // Wait for the forward tasks to finish draining.
        let _ = forward_task.await;
        let _ = turn_event_task.await;

        // Emit the final event based on the outcome.
        match outcome {
            Ok(turn_outcome) => {
                let (messages, usage, duration_ms) = match &turn_outcome {
                    TurnOutcome::Complete {
                        messages,
                        usage,
                        duration_ms,
                    } => (messages.clone(), *usage, *duration_ms),
                    TurnOutcome::Halted {
                        messages, usage, ..
                    } => (messages.clone(), *usage, 0),
                    TurnOutcome::Error {
                        messages, usage, ..
                    } => (messages.clone(), *usage, 0),
                };

                match turn_outcome {
                    TurnOutcome::Complete { .. } => {
                        let event = MessageStreamEvent::TurnComplete {
                            turn_id: task_turn_id.clone(),
                            session_id: task_session_id.clone(),
                            messages: messages.iter().map(MessageDto::from).collect(),
                            usage: UsagePayload::from(usage),
                            duration_ms,
                        };
                        let _ = task_app.emit(&task_event_channel, &event);
                        info!(turn_id = %task_turn_id, duration_ms = duration_ms, "Turn completed");
                    }
                    TurnOutcome::Halted { reason, .. } => {
                        let event = MessageStreamEvent::TurnError {
                            turn_id: task_turn_id.clone(),
                            session_id: task_session_id.clone(),
                            message: format!("Turn halted: {reason}"),
                            code: Some("TURN_HALTED".to_string()),
                        };
                        let _ = task_app.emit(&task_event_channel, &event);
                        warn!(turn_id = %task_turn_id, reason = %reason, "Turn halted");
                    }
                    TurnOutcome::Error { message, .. } => {
                        let event = MessageStreamEvent::TurnError {
                            turn_id: task_turn_id.clone(),
                            session_id: task_session_id.clone(),
                            message,
                            code: Some("TURN_ERROR".to_string()),
                        };
                        let _ = task_app.emit(&task_event_channel, &event);
                        error!(turn_id = %task_turn_id, "Turn error");
                    }
                }
            }
            Err(e) => {
                let event = MessageStreamEvent::TurnError {
                    turn_id: task_turn_id.clone(),
                    session_id: task_session_id.clone(),
                    message: e.to_string(),
                    code: Some("TURN_EXECUTION_ERROR".to_string()),
                };
                let _ = task_app.emit(&task_event_channel, &event);
                error!(turn_id = %task_turn_id, error = %e, "Turn execution failed");
            }
        }
    });

    turn_id
}

// ---------------------------------------------------------------------------
// Tauri Commands
// ---------------------------------------------------------------------------

/// Send a message to the agent and stream the response.
///
/// This is the primary command the Vue frontend calls to interact with the
/// agent. It:
/// 1. Resolves the provider and model from the request.
/// 2. Creates a `ProviderTurnGenerator` with the appropriate provider and config.
/// 3. Spawns a turn task that streams events to `agent:stream:{session_id}`.
/// 4. Returns the turn ID immediately (the response arrives via events).
///
/// The frontend should call `listen('agent:stream:{sessionId}')` before or
/// immediately after calling this command to receive the streaming events.
#[tauri::command]
pub async fn send_message(
    app: AppHandle,
    state: State<'_, AppState>,
    request: MessageSendRequest,
) -> TauriResult<String> {
    // Ensure the runtime is running.
    state.ensure_runtime_running().await?;

    // Store the user's message in the chat store.
    let session_id_str = request.session_id.clone();
    let _session_id = opensquilla_core::types::SessionId::from_string(&session_id_str)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id_str}")))?;

    let user_message = Message::user(&request.message);
    state.chat_store.add_message(
        &session_id_str,
        message_to_chat_response(&session_id_str, &user_message),
    );

    // Build the message list from history + current message.
    let mut messages: Vec<Message> = if request.history.is_empty() {
        // Use stored chat history.
        state
            .chat_store
            .get_history(&session_id_str, 1000, 0)
            .iter()
            .map(chat_response_to_message)
            .collect()
    } else {
        // Use provided history.
        request
            .history
            .iter()
            .map(dto_to_message)
            .collect::<Result<Vec<_>, _>>()?
    };

    // Ensure the latest user message is included.
    if messages.last().map(|m| &m.role) != Some(&MessageRole::User) {
        messages.push(user_message);
    }

    // Resolve the provider and model.
    let config = state.config().await;
    let provider_name = request.provider.as_deref().unwrap_or("default");
    let model = request.model.as_deref().unwrap_or("gpt-4").to_string();

    // Build the chat config.
    let chat_config = ChatConfig {
        model: model.clone(),
        stream: request.stream,
        ..Default::default()
    };

    // In a real implementation, we would resolve the provider from a
    // ProviderRegistry. For now, we need at least one provider registered.
    // The frontend should call `set_provider` or the config should specify one.
    // If no provider is available, we return an error.
    //
    // The provider registry is not part of AppState yet because the provider
    // crate's Provider trait requires concrete implementations (OpenAI, Anthropic,
    // etc.) that need API keys. The actual provider selection happens in the
    // `resolve_provider` helper.
    let provider = resolve_provider(&config, provider_name, &model)
        .map_err(|e| TauriError::internal(format!("Failed to resolve provider: {e}")))?;

    // Create the generator.
    let generator = Arc::new(ProviderTurnGenerator::new(provider, chat_config));

    // Build the turn config.
    let turn_config = TurnConfig {
        streaming: request.stream,
        ..Default::default()
    };

    // Spawn the turn task.
    let runtime = state.runtime();
    let turn_id = spawn_turn_task(
        app,
        runtime,
        session_id_str,
        messages,
        generator,
        turn_config,
    );

    Ok(turn_id)
}

/// Send a message synchronously (non-streaming) and return the full response.
///
/// Unlike `send_message`, this command waits for the turn to complete and
/// returns the full response. Useful for simple queries where streaming is
/// not needed.
#[tauri::command]
pub async fn send_message_sync(
    state: State<'_, AppState>,
    request: MessageSendRequest,
) -> TauriResult<MessageSendResponse> {
    state.ensure_runtime_running().await?;

    let session_id_str = request.session_id.clone();
    let _session_id = opensquilla_core::types::SessionId::from_string(&session_id_str)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id_str}")))?;

    let user_message = Message::user(&request.message);
    state.chat_store.add_message(
        &session_id_str,
        message_to_chat_response(&session_id_str, &user_message),
    );

    let mut messages: Vec<Message> = if request.history.is_empty() {
        state
            .chat_store
            .get_history(&session_id_str, 1000, 0)
            .iter()
            .map(chat_response_to_message)
            .collect()
    } else {
        request
            .history
            .iter()
            .map(dto_to_message)
            .collect::<Result<Vec<_>, _>>()?
    };

    if messages.last().map(|m| &m.role) != Some(&MessageRole::User) {
        messages.push(user_message);
    }

    let config = state.config().await;
    let provider_name = request.provider.as_deref().unwrap_or("default");
    let model = request.model.as_deref().unwrap_or("gpt-4").to_string();

    let chat_config = ChatConfig {
        model: model.clone(),
        stream: false,
        ..Default::default()
    };

    let provider = resolve_provider(&config, provider_name, &model)
        .map_err(|e| TauriError::internal(format!("Failed to resolve provider: {e}")))?;

    let generator = ProviderTurnGenerator::new(provider, chat_config);

    let runtime = state.runtime();
    let turn_id = Uuid::new_v4().to_string();
    let agent_id = session_id_str.clone();

    // Register the agent if needed.
    if runtime.get_agent(&agent_id).is_none() {
        runtime.register_agent(
            agent_id.clone(),
            AgentHandle {
                agent_id: agent_id.clone(),
                state: AgentState::Idle,
                total_usage: opensquilla_core::types::Usage::default(),
                turn_count: 0,
            },
        );
    }

    let start = std::time::Instant::now();
    let outcome = runtime
        .execute_turn(&agent_id, messages, &generator)
        .await
        .map_err(TauriError::from)?;

    let duration_ms = start.elapsed().as_millis() as u64;

    let (response_messages, usage) = match outcome {
        TurnOutcome::Complete {
            messages, usage, ..
        } => (messages, usage),
        TurnOutcome::Halted {
            messages, usage, ..
        } => (messages, usage),
        TurnOutcome::Error {
            messages, usage, ..
        } => (messages, usage),
    };

    // Store assistant messages in the chat store.
    for msg in &response_messages {
        if msg.role == MessageRole::Assistant {
            state.chat_store.add_message(
                &session_id_str,
                message_to_chat_response(&session_id_str, msg),
            );
        }
    }

    Ok(MessageSendResponse {
        turn_id,
        session_id: session_id_str,
        messages: response_messages.iter().map(MessageDto::from).collect(),
        usage: UsagePayload::from(usage),
        duration_ms,
    })
}

/// Cancel an in-progress turn.
///
/// Best-effort: the engine runtime does not retain a handle to the spawned
/// turn task (the `JoinHandle` is discarded after `tokio::spawn`), so this
/// marks the registered agent as stopped. It does not abort the underlying
/// provider request or the spawned tokio task.
#[tauri::command]
pub async fn cancel_turn(state: State<'_, AppState>, session_id: String) -> TauriResult<bool> {
    let runtime = state.runtime();
    if let Some(mut handle) = runtime.get_agent(&session_id) {
        let active = matches!(
            handle.state,
            AgentState::Processing
                | AgentState::Thinking
                | AgentState::WaitingForTool
                | AgentState::Compacting
        );
        handle.state = AgentState::Stopped;
        runtime.register_agent(session_id.clone(), handle);
        if active {
            info!(
                session_id = %session_id,
                "Cancel turn: marked agent stopped (best-effort; engine has no turn abort handle)"
            );
            Ok(true)
        } else {
            info!(
                session_id = %session_id,
                "Cancel turn requested but agent is not actively running a turn"
            );
            Ok(false)
        }
    } else {
        info!(
            session_id = %session_id,
            "Cancel turn requested but no registered agent found"
        );
        Ok(false)
    }
}

/// Abort a session's running turn. Same best-effort semantics as [`cancel_turn`].
#[tauri::command]
pub async fn abort_session(state: State<'_, AppState>, session_id: String) -> TauriResult<bool> {
    cancel_turn(state, session_id).await
}

/// Get the chat history for a session.
#[tauri::command]
pub async fn get_chat_history(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<Vec<MessageDto>> {
    let messages = state.chat_store.get_history(&session_id, 1000, 0);
    let messages: Vec<Message> = messages.iter().map(chat_response_to_message).collect();
    Ok(messages.iter().map(MessageDto::from).collect())
}

/// Clear the chat history for a session.
#[tauri::command]
pub async fn clear_chat_history(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<bool> {
    state.chat_store.clear(&session_id);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Session commands
// ---------------------------------------------------------------------------

/// Create a new session.
#[tauri::command]
pub async fn create_session(
    app: AppHandle,
    state: State<'_, AppState>,
    request: SessionCreateRequest,
) -> TauriResult<SessionResponse> {
    let title = request.title.unwrap_or_else(|| "New Session".to_string());
    let model = request.model.unwrap_or_else(|| "default".to_string());

    let entry = state.session_store.create(title, model);

    // Emit a session list changed event.
    let _ = app.emit(SESSION_LIST_CHANGED_EVENT, ());

    let info = SessionInfo {
        id: entry.id.to_string(),
        title: entry.title,
        model: entry.model,
        agent_id: request.agent_id.unwrap_or_default(),
        created_at: entry.created_at.to_rfc3339(),
        updated_at: entry.updated_at.to_rfc3339(),
        state: entry.state,
        mode: request.mode,
        message_count: 0,
        system_prompt: request.system_prompt,
        total_tokens: 0,
    };

    Ok(SessionResponse { session: info })
}

/// List all sessions.
#[tauri::command]
pub async fn list_sessions(state: State<'_, AppState>) -> TauriResult<SessionListResponse> {
    let entries = state.session_store.list();
    let sessions: Vec<SessionInfo> = entries
        .iter()
        .map(|entry| SessionInfo {
            id: entry.id.to_string(),
            title: entry.title.clone(),
            model: entry.model.clone(),
            agent_id: String::new(),
            created_at: entry.created_at.to_rfc3339(),
            updated_at: entry.updated_at.to_rfc3339(),
            state: entry.state.clone(),
            mode: "chat".to_string(),
            message_count: 0,
            system_prompt: None,
            total_tokens: 0,
        })
        .collect();
    let count = sessions.len();
    Ok(SessionListResponse { sessions, count })
}

/// Get a session by ID.
#[tauri::command]
pub async fn get_session(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<SessionResponse> {
    let sid = opensquilla_core::types::SessionId::from_string(&session_id)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id}")))?;

    let entry = state
        .session_store
        .get(&sid)
        .ok_or_else(|| TauriError::not_found(format!("Session {session_id} not found")))?;

    let info = SessionInfo {
        id: entry.id.to_string(),
        title: entry.title,
        model: entry.model,
        agent_id: String::new(),
        created_at: entry.created_at.to_rfc3339(),
        updated_at: entry.updated_at.to_rfc3339(),
        state: entry.state,
        mode: "chat".to_string(),
        message_count: 0,
        system_prompt: None,
        total_tokens: 0,
    };

    Ok(SessionResponse { session: info })
}

/// Delete a session.
#[tauri::command]
pub async fn delete_session(
    app: AppHandle,
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<bool> {
    let sid = opensquilla_core::types::SessionId::from_string(&session_id)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id}")))?;

    match state.session_store.delete(&sid) {
        Ok(_entry) => {
            let _ = app.emit(SESSION_LIST_CHANGED_EVENT, ());
            Ok(true)
        }
        Err(e) if e.status == 404 => Ok(false),
        Err(e) => Err(TauriError::from(e)),
    }
}

/// Archive a session.
#[tauri::command]
pub async fn archive_session(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<SessionResponse> {
    let sid = opensquilla_core::types::SessionId::from_string(&session_id)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id}")))?;

    let entry = state.session_store.archive(&sid)?;

    let info = SessionInfo {
        id: entry.id.to_string(),
        title: entry.title,
        model: entry.model,
        agent_id: String::new(),
        created_at: entry.created_at.to_rfc3339(),
        updated_at: entry.updated_at.to_rfc3339(),
        state: entry.state,
        mode: "chat".to_string(),
        message_count: 0,
        system_prompt: None,
        total_tokens: 0,
    };

    Ok(SessionResponse { session: info })
}

// ---------------------------------------------------------------------------
// Provider / Model / Skill commands
// ---------------------------------------------------------------------------

/// List configured providers.
#[tauri::command]
pub async fn list_providers(state: State<'_, AppState>) -> TauriResult<ProviderListResponse> {
    let config = state.config().await;
    let providers: Vec<ProviderInfo> = config.providers.iter().map(ProviderInfo::from).collect();
    let default_provider = providers.first().map(|p| p.name.clone());
    let count = providers.len();
    Ok(ProviderListResponse {
        providers,
        default_provider,
        count,
    })
}

/// List available models.
#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> TauriResult<ModelListResponse> {
    let config = state.config().await;
    let mut models: Vec<ModelInfoDto> = Vec::new();

    // Collect models from all providers.
    for provider in &config.providers {
        for model_id in &provider.models {
            let model_info = ModelInfoDto {
                id: model_id.clone(),
                name: model_id.clone(),
                provider: provider.name.clone(),
                context_window: 128_000,
                max_output_tokens: 4096,
                capabilities: opensquilla_core::model::ModelCapabilities::default(),
                display_name: None,
            };
            models.push(model_info);
        }
    }

    let default_model = config
        .models
        .as_ref()
        .and_then(|m| m.default_model.clone())
        .or_else(|| models.first().map(|m| m.id.clone()));

    let count = models.len();
    Ok(ModelListResponse {
        models,
        default_model,
        count,
    })
}

/// List available skills.
#[tauri::command]
pub async fn list_skills(state: State<'_, AppState>) -> TauriResult<SkillListResponse> {
    let config = state.config().await;

    // If skills are configured, scan the directories.
    let skills: Vec<SkillInfo> = if let Some(skills_config) = &config.skills {
        if !skills_config.enabled || skills_config.skill_dirs.is_empty() {
            Vec::new()
        } else {
            // Use the opensquilla-skills crate to scan.
            let loader = opensquilla_skills::SkillLoader::new();
            for dir in &skills_config.skill_dirs {
                let path = std::path::PathBuf::from(dir);
                if path.exists() {
                    loader.register_layer_dir(opensquilla_skills::SkillLayer::Bundled, path);
                }
            }
            let _ = loader.scan_all().await;
            let specs = loader.get_skills(None).await;
            specs
                .iter()
                .map(|spec| SkillInfo {
                    id: spec.id.clone(),
                    name: spec.name.clone(),
                    kind: format!("{:?}", spec.kind).to_lowercase(),
                    description: spec.description.clone(),
                    layer: spec.layer.to_string(),
                    version: spec.version.clone(),
                    author: spec.author.clone(),
                    tags: spec.tags.clone(),
                    is_meta: spec.is_meta(),
                    disabled: false,
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    let count = skills.len();
    Ok(SkillListResponse { skills, count })
}

// ---------------------------------------------------------------------------
// Health / Config commands
// ---------------------------------------------------------------------------

/// Run a health check and return a health report.
#[tauri::command]
pub async fn health_check(state: State<'_, AppState>) -> TauriResult<HealthReport> {
    let config = state.config().await;
    let health = opensquilla_recovery::health::HealthCheck::new(&config);
    let result = health.run_full_check().await;

    let gateway_running = state.is_gateway_running().await;
    let gateway_url = state.gateway_url().await;

    let status = match result.status {
        opensquilla_recovery::health::HealthStatus::Healthy => "healthy",
        opensquilla_recovery::health::HealthStatus::Degraded => "degraded",
        opensquilla_recovery::health::HealthStatus::Unhealthy => "unhealthy",
    };

    let components: Vec<HealthComponent> = result
        .components
        .into_iter()
        .map(|c| HealthComponent {
            name: c.name,
            status: match c.status {
                opensquilla_recovery::health::HealthStatus::Healthy => "healthy",
                opensquilla_recovery::health::HealthStatus::Degraded => "degraded",
                opensquilla_recovery::health::HealthStatus::Unhealthy => "unhealthy",
            }
            .to_string(),
            description: c.description,
            latency_ms: c.latency_ms,
            details: c.details,
        })
        .collect();

    let issues: Vec<HealthIssue> = result
        .issues
        .into_iter()
        .map(|i| HealthIssue {
            component: i.component,
            severity: format!("{:?}", i.severity).to_lowercase(),
            message: i.message,
            suggestion: i.suggestion,
        })
        .collect();

    Ok(HealthReport {
        status: status.to_string(),
        uptime_seconds: result.uptime_seconds,
        timestamp: result.timestamp.to_rfc3339(),
        components,
        issues,
        gateway_running,
        gateway_url,
    })
}

/// Get the full configuration.
#[tauri::command]
pub async fn get_config(state: State<'_, AppState>) -> TauriResult<ConfigGetResponse> {
    let config = state.config().await;
    let json = serde_json::to_value(config.deref()).map_err(TauriError::from)?;
    Ok(ConfigGetResponse { config: json })
}

/// Set a configuration value.
#[tauri::command]
pub async fn set_config(
    state: State<'_, AppState>,
    request: ConfigSetRequest,
) -> TauriResult<ConfigSetResponse> {
    // Determine the type of the value and set it appropriately.
    {
        let mut config = state.config_mut().await;
        // Try to set the value using the config's flat-path setter.
        let value_str = match &request.value {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            other => other.to_string(),
        };
        config
            .set(&request.key, &value_str)
            .map_err(|e| TauriError::bad_request(format!("Failed to set config: {e}")))?;
    }

    // Persist the config to disk.
    {
        let config = state.config().await;
        if let Err(e) = config.save() {
            warn!(error = %e, "Failed to persist config to disk");
        }
    }

    Ok(ConfigSetResponse {
        key: request.key,
        value: request.value,
        status: "set".to_string(),
    })
}

/// Get a single configuration value by key.
#[tauri::command]
pub async fn get_config_value(
    state: State<'_, AppState>,
    key: String,
) -> TauriResult<Option<serde_json::Value>> {
    let config = state.config().await;
    Ok(config.get(&key).map(serde_json::Value::String))
}

/// List all configuration values.
#[tauri::command]
pub async fn list_config(state: State<'_, AppState>) -> TauriResult<serde_json::Value> {
    let config = state.config().await;
    let flat = config.list();
    let mut map = serde_json::Map::new();
    for (k, v) in flat {
        // Try to parse as JSON value, falling back to string.
        let value =
            serde_json::from_str::<serde_json::Value>(&v).unwrap_or(serde_json::Value::String(v));
        map.insert(k, value);
    }
    Ok(serde_json::Value::Object(map))
}

// ---------------------------------------------------------------------------
// Session fork / compaction commands
// ---------------------------------------------------------------------------

/// Fork an existing session into a new one via the SQLite session manager.
#[tauri::command]
pub async fn fork_session(
    state: State<'_, AppState>,
    session_id: String,
    fork_event: Option<String>,
    title: Option<String>,
) -> TauriResult<SessionResponse> {
    let source_id = opensquilla_core::types::SessionId::from_string(&session_id)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id}")))?;

    let config = ForkConfig {
        name: title,
        fork_event: fork_event.unwrap_or_else(|| "fork".to_string()),
        ..Default::default()
    };

    let manager = state.session_manager().await;
    let session = manager.fork(&source_id.0, config)?;

    let info = SessionInfo {
        id: session.id.to_string(),
        title: session.name,
        model: String::new(),
        agent_id: session.agent_id.to_string(),
        created_at: session.created_at.to_rfc3339(),
        updated_at: session.updated_at.to_rfc3339(),
        state: "active".to_string(),
        mode: "chat".to_string(),
        message_count: session.message_count,
        system_prompt: if session.system_prompt.is_empty() {
            None
        } else {
            Some(session.system_prompt)
        },
        total_tokens: session.total_tokens,
    };

    Ok(SessionResponse { session: info })
}

/// Trigger context compaction for a session via the SQLite session manager.
#[tauri::command]
pub async fn compact_session(
    state: State<'_, AppState>,
    session_id: String,
) -> TauriResult<OperationResult> {
    let sid = opensquilla_core::types::SessionId::from_string(&session_id)
        .ok_or_else(|| TauriError::bad_request(format!("Invalid session_id: {session_id}")))?;

    let manager = state.session_manager().await;
    let report = manager.compact(&sid.0)?;

    Ok(OperationResult {
        ok: true,
        message: Some(format!(
            "Compacted session {} (strategy {}, {} entries, {} -> {} tokens)",
            report.session_id,
            report.strategy.label(),
            report.entries_compacted,
            report.tokens_before,
            report.tokens_after,
        )),
    })
}

// ---------------------------------------------------------------------------
// Provider status commands
// ---------------------------------------------------------------------------

/// Build a status object for a single provider configuration.
fn provider_status_json(provider: &opensquilla_core::config::ProviderConfig) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "name".to_string(),
        serde_json::Value::String(provider.name.clone()),
    );
    map.insert(
        "providerType".to_string(),
        serde_json::Value::String(provider.provider_type.clone()),
    );
    map.insert(
        "configured".to_string(),
        serde_json::Value::Bool(provider.api_key.is_some()),
    );
    map.insert(
        "defaultModel".to_string(),
        match &provider.default_model {
            Some(m) => serde_json::Value::String(m.clone()),
            None => serde_json::Value::Null,
        },
    );
    map.insert(
        "models".to_string(),
        serde_json::Value::Array(
            provider
                .models
                .iter()
                .map(|m| serde_json::Value::String(m.clone()))
                .collect(),
        ),
    );
    map.insert(
        "baseUrl".to_string(),
        match &provider.base_url {
            Some(b) => serde_json::Value::String(b.clone()),
            None => serde_json::Value::Null,
        },
    );
    serde_json::Value::Object(map)
}

/// Get the status of a single provider (by id, or the first/default provider).
#[tauri::command]
pub async fn get_provider_status(
    state: State<'_, AppState>,
    provider_id: Option<String>,
) -> TauriResult<serde_json::Value> {
    let config = state.config().await;
    let provider = match provider_id {
        Some(id) => config
            .find_provider(&id)
            .ok_or_else(|| TauriError::not_found(format!("Provider '{id}' not found in config")))?,
        None => config
            .providers
            .first()
            .ok_or_else(|| TauriError::not_found("No providers configured".to_string()))?,
    };
    Ok(provider_status_json(provider))
}

/// Get the status of all configured providers.
#[tauri::command]
pub async fn get_all_provider_statuses(
    state: State<'_, AppState>,
) -> TauriResult<serde_json::Value> {
    let config = state.config().await;
    let statuses: Vec<serde_json::Value> =
        config.providers.iter().map(provider_status_json).collect();
    let default_provider = config.providers.first().map(|p| p.name.clone());
    Ok(serde_json::json!({
        "providers": statuses,
        "defaultProvider": default_provider,
        "count": statuses.len(),
    }))
}

// ---------------------------------------------------------------------------
// Config patch / reset / effective commands
// ---------------------------------------------------------------------------

/// Get the effective configuration (same single source as [`get_config`]).
#[tauri::command]
pub async fn get_config_effective(state: State<'_, AppState>) -> TauriResult<ConfigGetResponse> {
    let config = state.config().await;
    let json = serde_json::to_value(config.deref()).map_err(TauriError::from)?;
    Ok(ConfigGetResponse { config: json })
}

/// Apply a batch of key/value patches to the configuration.
///
/// Patches are applied to the single source-of-truth `Config` held in
/// `AppState`, then persisted to disk via `Config::save`, so a successful patch
/// survives a restart.
#[tauri::command]
pub async fn patch_config(
    state: State<'_, AppState>,
    patches: Vec<ConfigPatch>,
    safe: Option<bool>,
) -> TauriResult<ConfigSetResponse> {
    if patches.is_empty() {
        return Ok(ConfigSetResponse {
            key: String::new(),
            value: serde_json::json!({ "applied": 0 }),
            status: "patched".to_string(),
        });
    }

    // For `safe` patches, validate the patched state against a staged copy
    // before mutating the live config so a rejected patch is a no-op.
    if safe.unwrap_or(false) {
        let mut staged = (*state.config().await).clone();
        for patch in &patches {
            staged
                .set_value(&patch.key, &patch.value)
                .map_err(|e| {
                    TauriError::bad_request(format!(
                        "Failed to apply patch '{}': {e}",
                        patch.key
                    ))
                })?;
        }
        let store = opensquilla_gateway::ConfigStore::from_config(staged);
        let issues = store.validate();
        if !issues.is_empty() {
            return Err(TauriError::bad_request(format!(
                "Patched configuration is invalid: {}",
                issues.join("; ")
            )));
        }
    }

    // Apply to the single source of truth, then persist to disk. A save failure
    // is surfaced to the caller rather than silently dropped.
    {
        let mut config = state.config_mut().await;
        for patch in &patches {
            config
                .set_value(&patch.key, &patch.value)
                .map_err(|e| {
                    TauriError::bad_request(format!(
                        "Failed to apply patch '{}': {e}",
                        patch.key
                    ))
                })?;
        }
    }
    {
        let config = state.config().await;
        config.save().map_err(|e| {
            TauriError::internal(format!("Failed to persist config: {e}"))
        })?;
    }

    let first_key = patches.first().map(|p| p.key.clone()).unwrap_or_default();

    // Keep channel system messages in sync with the UI language preference.
    if let Some(locale) = patches.iter().find_map(|p| {
        if p.key == "control_ui.default_locale" {
            p.value.as_str()
        } else {
            None
        }
    }) {
        if let Some(gateway) = state.get_gateway().await {
            gateway.set_channel_locale(locale);
        }
    }

    Ok(ConfigSetResponse {
        key: first_key,
        value: serde_json::json!({ "applied": patches.len() }),
        status: "patched".to_string(),
    })
}

/// Reset a configuration key (or the whole configuration) to defaults.
#[tauri::command]
pub async fn reset_config(
    state: State<'_, AppState>,
    key: Option<String>,
) -> TauriResult<OperationResult> {
    match key {
        Some(k) => {
            let removed = state.config_store.delete(&k);
            Ok(OperationResult {
                ok: true,
                message: Some(format!("Reset config key '{k}' (removed: {removed})")),
            })
        }
        None => {
            state.config_store.reset();
            Ok(OperationResult {
                ok: true,
                message: Some("Configuration reset to defaults".to_string()),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use std::ops::Deref;

/// Convert a `MessageDto` back to a `Message`.
fn dto_to_message(dto: &MessageDto) -> Result<Message, TauriError> {
    let role = match dto.role.as_str() {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        other => {
            return Err(TauriError::bad_request(format!(
                "Invalid message role: {other}"
            )));
        }
    };

    let mut content = Vec::new();
    for block in &dto.content {
        match block {
            ContentBlockDto::Text { text } => {
                content.push(opensquilla_core::types::ContentBlock::Text(text.clone()));
            }
            ContentBlockDto::ToolUse { id, name, input } => {
                content.push(opensquilla_core::types::ContentBlock::ToolUse(
                    opensquilla_core::types::ToolCall::new(id, name, input.clone()),
                ));
            }
            ContentBlockDto::ToolResult {
                tool_use_id,
                content: result_content,
                is_error,
            } => {
                let result = if *is_error {
                    opensquilla_core::types::ToolResult::error(tool_use_id, result_content)
                } else {
                    opensquilla_core::types::ToolResult::success(tool_use_id, result_content)
                };
                content.push(opensquilla_core::types::ContentBlock::ToolResult(result));
            }
            ContentBlockDto::Reasoning { reasoning } => {
                content.push(opensquilla_core::types::ContentBlock::Reasoning(
                    reasoning.clone(),
                ));
            }
        }
    }

    Ok(Message {
        role,
        content,
        name: dto.name.clone(),
        tool_call_id: dto.tool_call_id.clone(),
        tool_calls: None,
        tool_result: None,
    })
}

/// Convert a `Message` to a `ChatMessageResponse` for the chat store.
fn message_to_chat_response(
    session_id: &str,
    msg: &Message,
) -> opensquilla_gateway::chat::ChatMessageResponse {
    opensquilla_gateway::chat::ChatMessageResponse {
        id: Uuid::new_v4().to_string(),
        session_id: session_id.to_string(),
        role: format!("{:?}", msg.role).to_lowercase(),
        content: msg.text_content(),
        timestamp: chrono::Utc::now(),
        model: None,
    }
}

/// Convert a `ChatMessageResponse` to a `Message`.
fn chat_response_to_message(resp: &opensquilla_gateway::chat::ChatMessageResponse) -> Message {
    let role = match resp.role.as_str() {
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "system" => MessageRole::System,
        _ => MessageRole::User,
    };
    Message {
        role,
        content: vec![opensquilla_core::types::ContentBlock::Text(
            resp.content.clone(),
        )],
        name: None,
        tool_call_id: None,
        tool_calls: None,
        tool_result: None,
    }
}

/// Resolve a provider from the config.
///
/// This function looks up the provider configuration and creates a concrete
/// provider implementation. Currently, it creates an `OpenAIProvider` for
/// "openai"-compatible providers and an `AnthropicProvider` for "anthropic".
///
/// In a full implementation, this would use a `ProviderRegistry` to look up
/// pre-registered providers.
fn resolve_provider(
    config: &opensquilla_core::config::Config,
    name: &str,
    _model: &str,
) -> Result<Arc<dyn Provider>, String> {
    // Find the provider config by name, or use the first one.
    let provider_config = if name == "default" {
        config.providers.first().ok_or("No providers configured")?
    } else {
        config
            .find_provider(name)
            .ok_or_else(|| format!("Provider '{name}' not found in config"))?
    };

    let api_key = provider_config.api_key.clone().unwrap_or_default();
    let base_url = provider_config.base_url.clone();

    // Create the appropriate provider based on the type.
    let provider_type = &provider_config.provider_type;
    let provider_name = &provider_config.name;
    let default_base_url = match provider_type.as_str() {
        "openai" | "openai_compat" => "https://api.openai.com/v1",
        "deepseek" => "https://api.deepseek.com/v1",
        "dashscope" | "qwen" => "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "moonshot" => "https://api.moonshot.cn/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "zhipu" => "https://open.bigmodel.cn/api/paas/v4",
        "siliconflow" => "https://api.siliconflow.cn/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "azure" => "https://api.openai.azure.com",
        "mistral" => "https://api.mistral.ai/v1",
        "anthropic" => "https://api.anthropic.com",
        "ollama" => "http://localhost:11434",
        _ => "https://api.openai.com/v1",
    };
    let base_url = base_url.unwrap_or_else(|| default_base_url.to_string());

    let provider: Arc<dyn Provider> = match provider_type.as_str() {
        "openai" | "openai_compat" | "deepseek" | "dashscope" | "qwen" | "moonshot" | "groq"
        | "zhipu" | "siliconflow" | "openrouter" | "azure" | "mistral" => {
            Arc::new(opensquilla_provider::OpenAiCompatProvider::new(
                provider_name.clone(),
                base_url,
                api_key,
            ))
        }
        "anthropic" => Arc::new(opensquilla_provider::AnthropicProvider::new(
            provider_name.clone(),
            base_url,
            api_key,
        )),
        "ollama" => Arc::new(opensquilla_provider::OllamaProvider::new(
            provider_name.clone(),
            base_url,
        )),
        other => {
            return Err(format!(
                "Unknown provider type: '{other}'. Supported: openai, anthropic, ollama"
            ));
        }
    };

    Ok(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_config_default() {
        let config = TurnConfig::default();
        assert_eq!(config.max_tool_rounds, 10);
        assert_eq!(config.max_messages_before_compaction, 50);
        assert!(config.streaming);
    }

    #[test]
    fn test_build_turn_runner() {
        let config = TurnConfig::default();
        let _runner = build_turn_runner(&config);
        // The runner should be created without panicking.
    }

    #[test]
    fn test_dto_to_message() {
        let dto = MessageDto {
            role: "user".to_string(),
            content: vec![ContentBlockDto::Text {
                text: "Hello".to_string(),
            }],
            name: None,
            tool_call_id: None,
        };
        let msg = dto_to_message(&dto).unwrap();
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            opensquilla_core::types::ContentBlock::Text(text) => {
                assert_eq!(text, "Hello");
            }
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_dto_to_message_invalid_role() {
        let dto = MessageDto {
            role: "invalid".to_string(),
            content: vec![],
            name: None,
            tool_call_id: None,
        };
        assert!(dto_to_message(&dto).is_err());
    }
}
