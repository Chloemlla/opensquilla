//! Sub-agent management.
//!
//! Sub-agents are process-internal `Agent` instances spawned from a parent
//! agent. They share the provider configuration (via `Arc<dyn Provider>` when
//! the `provider` feature is enabled, or via a shared `TurnGenerator`
//! otherwise) and run their own turns concurrently on the tokio runtime.
//! Results are collected back into the parent turn.
//!
//! Concurrency is bounded by a [`tokio::sync::Semaphore`]; the manager can
//! share a token budget across the batch and, when a [`crate::runtime::TurnRunner`]
//! is attached, sub-agents run the full 8-stage pipeline instead of the plain
//! `respond` path.

use crate::agent::{Agent, AgentConfig, TurnGenerator, TurnOutcome, UsageEvent};
use opensquilla_core::error::{Error, Result};
use opensquilla_core::types::Message;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, instrument};

/// A spawned sub-agent handle.
///
/// The handle holds the sub-agent's state and can await its turn outcome.
/// The sub-agent runs in the background until `join` is awaited, or until it
/// is explicitly cancelled.
#[derive(Debug)]
pub struct SubAgentHandle {
    /// The unique identifier of the sub-agent.
    pub id: String,
    /// The prompt sent to the sub-agent.
    pub prompt: Message,
    /// The task that runs the sub-agent's turn.
    task: tokio::task::JoinHandle<Result<TurnOutcome>>,
    /// When the sub-agent was spawned.
    started_at: Instant,
}

impl SubAgentHandle {
    /// Await the sub-agent's turn outcome.
    pub async fn join(self) -> Result<TurnOutcome> {
        match self.task.await {
            Ok(outcome) => outcome,
            Err(e) => Err(Error::Internal(format!(
                "sub-agent task {} panicked: {e}",
                self.id
            ))),
        }
    }

    /// Cancel the sub-agent task if it is still running.
    pub fn abort(&self) {
        self.task.abort();
    }

    /// Wall-clock time elapsed since the sub-agent was spawned.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Whether the sub-agent task has completed.
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }
}

/// Configuration for spawning a sub-agent.
#[derive(Debug, Clone)]
pub struct SubAgentSpec {
    /// The sub-agent's unique identifier.
    pub id: String,
    /// The prompt to send as the sub-agent's first user message.
    pub prompt: String,
    /// The maximum number of tool rounds the sub-agent may run.
    pub max_tool_rounds: u32,
    /// A per-sub-agent token budget, shared out of the parent's allocation.
    pub token_budget_tokens: Option<u64>,
    /// The workspace directory sub-agent commands run in.
    pub workspace_dir: Option<PathBuf>,
    /// An optional system prompt overriding the default identity.
    pub system_prompt: Option<String>,
}

impl SubAgentSpec {
    /// Create a new sub-agent specification.
    pub fn new(id: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            max_tool_rounds: 10,
            token_budget_tokens: None,
            workspace_dir: None,
            system_prompt: None,
        }
    }

    /// Set the maximum number of tool rounds.
    pub fn with_max_tool_rounds(mut self, rounds: u32) -> Self {
        self.max_tool_rounds = rounds.max(1);
        self
    }

    /// Set a per-sub-agent token budget.
    pub fn with_token_budget(mut self, tokens: u64) -> Self {
        self.token_budget_tokens = Some(tokens);
        self
    }

    /// Set the workspace directory commands run in.
    pub fn with_workspace(mut self, dir: PathBuf) -> Self {
        self.workspace_dir = Some(dir);
        self
    }

    /// Set an override system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }
}

/// A manager for spawning and collecting process-internal sub-agents.
///
/// The manager holds a shared, cloneable generator so every sub-agent uses
/// the same model/provider configuration. It tracks live sub-agents and can
/// join them in completion order or wait for all of them.
#[derive(Debug, Clone)]
pub struct SubAgentManager {
    /// The shared generator used to drive each sub-agent's turns.
    generator: Arc<dyn TurnGenerator>,
    /// The maximum number of concurrent sub-agents.
    max_concurrent: usize,
    /// Semaphore bounding concurrent sub-agent tasks.
    semaphore: Arc<tokio::sync::Semaphore>,
    /// Optional total token budget shared across a batch.
    token_budget: Option<u64>,
    /// Optional runner used to drive full pipeline turns.
    runner: Option<Arc<crate::runtime::TurnRunner>>,
}

impl SubAgentManager {
    /// Create a new sub-agent manager sharing the given generator.
    pub fn new(generator: Arc<dyn TurnGenerator>) -> Self {
        Self {
            generator,
            max_concurrent: 8,
            semaphore: Arc::new(tokio::sync::Semaphore::new(8)),
            token_budget: None,
            runner: None,
        }
    }

    /// Set the maximum number of concurrent sub-agents.
    pub fn max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self.semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));
        self
    }

    /// Share a total token budget across a batch of sub-agents.
    pub fn with_token_budget(mut self, tokens: u64) -> Self {
        self.token_budget = Some(tokens);
        self
    }

    /// Attach a runner so sub-agents execute the full 8-stage pipeline.
    pub fn with_runner(mut self, runner: Arc<crate::runtime::TurnRunner>) -> Self {
        self.runner = Some(runner);
        self
    }

    /// The shared generator.
    pub fn generator(&self) -> &Arc<dyn TurnGenerator> {
        &self.generator
    }

    /// Build an `Agent` from a sub-agent spec.
    fn build_agent(&self, spec: &SubAgentSpec) -> Agent {
        let mut config = AgentConfig {
            max_turns: spec.max_tool_rounds.max(1),
            max_tool_calls_per_turn: 8,
            timeout_seconds: 60,
            allow_tool_execution: true,
            allow_subprocess: true,
            workspace_dir: spec.workspace_dir.clone(),
            default_model: self.generator.model_name().to_string(),
            default_provider: self.generator.provider_name().to_string(),
            system_prompt: spec.system_prompt.clone().unwrap_or_default(),
            context_window_tokens: 128_000,
        };
        if let Some(budget) = spec.token_budget_tokens {
            // The token budget caps the per-sub-agent context by limiting the
            // number of tool rounds it can burn through.
            let max_rounds = (budget / 4_000).max(1) as u32;
            config.max_turns = config.max_turns.min(max_rounds);
        }
        Agent::with_config(
            spec.id.clone(),
            Box::new(SharedGenerator(self.generator.clone())),
            config,
        )
        .named(spec.id.clone())
    }

    /// Spawn a sub-agent that runs its own turn in the background.
    ///
    /// The sub-agent receives the given prompt as its first user message and
    /// executes a turn (through the attached runner when available, otherwise
    /// the plain `respond` path) using the shared generator. A semaphore
    /// permit bounds the number of concurrently executing sub-agents.
    #[instrument(skip(self), fields(subagent = %spec.id))]
    pub async fn spawn(&self, spec: SubAgentSpec) -> Result<SubAgentHandle> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Error::Internal("sub-agent semaphore closed".to_string()))?;

        let mut agent = self.build_agent(&spec);
        agent.initialize();

        debug!(subagent = %spec.id, "Spawning sub-agent");

        let prompt = Message::user(&spec.prompt);
        let task_prompt = prompt.clone();
        let runner = self.runner.clone();
        let provider_name = agent.provider_name().to_string();
        let model_name = agent.model_name().to_string();

        let task = tokio::spawn(async move {
            let _permit = permit;
            let outcome = if let Some(runner) = runner {
                agent
                    .run_turn_with_runner(vec![task_prompt], runner.as_ref())
                    .await
            } else {
                match agent.respond(&[task_prompt]).await {
                    Ok(response) => {
                        // The plain path does not record usage; approximate it
                        // from the response text and record it against the
                        // sub-agent so parent accounting stays complete.
                        let output_tokens = response
                            .iter()
                            .map(|m| m.text_content().chars().count() as u64 / 4)
                            .sum();
                        agent.track_usage(UsageEvent::new(
                            model_name,
                            provider_name,
                            0,
                            output_tokens,
                        ));
                        Ok(TurnOutcome::Complete {
                            messages: response,
                            usage: agent.get_usage_stats().to_usage(),
                            duration_ms: 0,
                        })
                    }
                    Err(e) => Err(e),
                }
            };
            outcome
        });

        Ok(SubAgentHandle {
            id: spec.id,
            prompt,
            task,
            started_at: Instant::now(),
        })
    }

    /// Spawn a sub-agent and immediately await its outcome.
    ///
    /// This is a convenience wrapper around [`SubAgentManager::spawn`] and
    /// [`SubAgentHandle::join`].
    pub async fn run_subagent(&self, spec: SubAgentSpec) -> Result<TurnOutcome> {
        let handle = self.spawn(spec).await?;
        handle.join().await
    }

    /// Allocate the manager's total token budget across a batch of specs.
    ///
    /// Specs that already carry a per-sub-agent budget are left unchanged and
    /// their budgets are subtracted from the total first; the remaining budget
    /// is split evenly across the rest. Returns the specs with budgets applied.
    pub fn allocate_budget(&self, mut specs: Vec<SubAgentSpec>) -> Vec<SubAgentSpec> {
        let Some(total) = self.token_budget else {
            return specs;
        };
        let explicit_sum: u64 = specs.iter().filter_map(|s| s.token_budget_tokens).sum();
        let unallocated = specs
            .iter()
            .filter(|s| s.token_budget_tokens.is_none())
            .count();
        if unallocated == 0 {
            return specs;
        }
        let remaining = total.saturating_sub(explicit_sum);
        let per_sub = remaining / unallocated as u64;
        for spec in &mut specs {
            if spec.token_budget_tokens.is_none() {
                spec.token_budget_tokens = Some(per_sub);
            }
        }
        specs
    }

    /// Collect results from multiple sub-agents, waiting for all of them.
    ///
    /// Sub-agents are spawned with the shared semaphore limiting concurrency;
    /// each is joined and its outcome recorded. A failed sub-agent is recorded
    /// as an `Err` in the map rather than aborting the whole batch.
    pub async fn collect_all(
        &self,
        specs: Vec<SubAgentSpec>,
    ) -> std::collections::HashMap<String, Result<TurnOutcome>> {
        let mut results = std::collections::HashMap::new();
        let specs = self.allocate_budget(specs);

        let mut spawned = Vec::new();
        for spec in specs {
            match self.spawn(spec).await {
                Ok(handle) => spawned.push(handle),
                Err(e) => {
                    error!(error = %e, "Failed to spawn sub-agent");
                }
            }
        }
        for handle in spawned {
            let id = handle.id.clone();
            match handle.join().await {
                Ok(outcome) => {
                    results.insert(id, Ok(outcome));
                }
                Err(e) => {
                    error!(subagent = %id, error = %e, "Sub-agent failed");
                    results.insert(id, Err(e));
                }
            }
        }
        results
    }

    /// Spawn up to `max_concurrent` sub-agents concurrently.
    ///
    /// This spawns a single wave of at most [`SubAgentManager::max_concurrent`]
    /// sub-agents and returns their live handles without waiting for them to
    /// complete. Use [`SubAgentHandle::join`] to await each result.
    pub async fn spawn_many(&self, specs: Vec<SubAgentSpec>) -> Vec<SubAgentHandle> {
        let specs = self.allocate_budget(specs);
        let limit = self.max_concurrent.min(specs.len());
        let mut handles = Vec::with_capacity(limit);
        for spec in specs.into_iter().take(limit) {
            match self.spawn(spec).await {
                Ok(handle) => handles.push(handle),
                Err(e) => {
                    error!(error = %e, "Failed to spawn sub-agent");
                }
            }
        }
        handles
    }

    /// Get the number of sub-agents that can run concurrently.
    pub fn concurrency_limit(&self) -> usize {
        self.max_concurrent
    }
}

/// Wrapper that lets an `Arc<dyn TurnGenerator>` be re-boxed as
/// `Box<dyn TurnGenerator>` for each sub-agent's `Agent`.
#[derive(Debug)]
struct SharedGenerator(Arc<dyn TurnGenerator>);

#[async_trait::async_trait]
impl TurnGenerator for SharedGenerator {
    async fn generate(&self, messages: &[Message]) -> Result<Vec<Message>> {
        self.0.generate(messages).await
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    fn provider_name(&self) -> &str {
        self.0.provider_name()
    }
}

/// A provider-cloneable sub-agent when the `provider` feature is enabled.
///
/// When the engine is built with the provider crate, sub-agents can share an
/// `Arc<dyn Provider>` directly. This type wraps a provider and exposes it as
/// a `TurnGenerator` so it can drive an `Agent`.
#[cfg(feature = "provider")]
pub struct ProviderSubAgent {
    /// The shared provider instance.
    provider: Arc<opensquilla_provider::Provider>,
    /// The chat configuration to use for generation.
    config: opensquilla_provider::ChatConfig,
    /// The tools made available to the sub-agent.
    tools: Vec<opensquilla_core::types::ToolDefinition>,
    /// A per-sub-agent token budget.
    token_budget_tokens: Option<u64>,
}

#[cfg(feature = "provider")]
impl std::fmt::Debug for ProviderSubAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSubAgent")
            .field("provider", &self.provider.name())
            .field("model", &self.config.model)
            .field("tools", &self.tools.len())
            .field("token_budget_tokens", &self.token_budget_tokens)
            .finish()
    }
}

#[cfg(feature = "provider")]
impl ProviderSubAgent {
    /// Create a sub-agent wrapper around a shared provider.
    pub fn new(
        provider: Arc<opensquilla_provider::Provider>,
        config: opensquilla_provider::ChatConfig,
    ) -> Self {
        Self {
            provider,
            config,
            tools: Vec::new(),
            token_budget_tokens: None,
        }
    }

    /// Provide the tool definitions exposed to this sub-agent.
    pub fn with_tools(mut self, tools: Vec<opensquilla_core::types::ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    /// Set a per-sub-agent token budget.
    pub fn with_token_budget(mut self, tokens: u64) -> Self {
        self.token_budget_tokens = Some(tokens);
        self
    }

    /// Spawn a sub-agent driven by this provider wrapper.
    ///
    /// Returns a handle that resolves to the sub-agent's turn outcome.
    pub async fn spawn(
        self,
        id: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<SubAgentHandle> {
        let mut spec = SubAgentSpec::new(id, prompt);
        spec.token_budget_tokens = self.token_budget_tokens;
        let agent = Agent::with_config(spec.id.clone(), Box::new(self), AgentConfig::default())
            .named(spec.id.clone());
        let mut agent = agent;
        agent.initialize();

        let task_prompt = Message::user(&spec.prompt);
        let prompt = task_prompt.clone();
        let task = tokio::spawn(async move {
            agent
                .respond(&[task_prompt])
                .await
                .map(|response| TurnOutcome::Complete {
                    messages: response,
                    usage: agent.get_usage_stats().to_usage(),
                    duration_ms: 0,
                })
        });
        Ok(SubAgentHandle {
            id: spec.id,
            prompt,
            task,
            started_at: Instant::now(),
        })
    }
}

#[cfg(feature = "provider")]
#[async_trait::async_trait]
impl TurnGenerator for ProviderSubAgent {
    async fn generate(&self, messages: &[Message]) -> Result<Vec<Message>> {
        let response = self
            .provider
            .send_message(&self.config, messages, &self.tools)
            .await
            .map_err(|e| Error::Provider(e.to_string()))?;
        Ok(response.content)
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct MockGenerator;

    #[async_trait::async_trait]
    impl TurnGenerator for MockGenerator {
        async fn generate(&self, _m: &[Message]) -> Result<Vec<Message>> {
            Ok(vec![Message::assistant("sub-agent reply")])
        }
        fn model_name(&self) -> &str {
            "mock"
        }
        fn provider_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_spec_builder() {
        let spec = SubAgentSpec::new("s1", "do it")
            .with_max_tool_rounds(5)
            .with_token_budget(10_000);
        assert_eq!(spec.max_tool_rounds, 5);
        assert_eq!(spec.token_budget_tokens, Some(10_000));
    }

    #[test]
    fn test_allocate_budget_splits_evenly() {
        let manager = SubAgentManager::new(Arc::new(MockGenerator)).with_token_budget(9_000);
        let specs = vec![
            SubAgentSpec::new("a", "1"),
            SubAgentSpec::new("b", "2"),
            SubAgentSpec::new("c", "3"),
        ];
        let allocated = manager.allocate_budget(specs);
        assert_eq!(allocated.len(), 3);
        for spec in &allocated {
            assert_eq!(spec.token_budget_tokens, Some(3_000));
        }
    }

    #[test]
    fn test_allocate_budget_respects_explicit() {
        let manager = SubAgentManager::new(Arc::new(MockGenerator)).with_token_budget(10_000);
        let specs = vec![
            SubAgentSpec::new("a", "1").with_token_budget(7_000),
            SubAgentSpec::new("b", "2"),
        ];
        let allocated = manager.allocate_budget(specs);
        assert_eq!(allocated[0].token_budget_tokens, Some(7_000));
        assert_eq!(allocated[1].token_budget_tokens, Some(3_000));
    }

    #[test]
    fn test_build_agent_config() {
        let manager = SubAgentManager::new(Arc::new(MockGenerator));
        let spec = SubAgentSpec::new("s1", "hi").with_max_tool_rounds(4);
        let agent = manager.build_agent(&spec);
        assert_eq!(agent.id(), "s1");
        assert_eq!(agent.config().max_turns, 4);
    }

    #[tokio::test]
    async fn test_run_subagent() {
        let manager = SubAgentManager::new(Arc::new(MockGenerator));
        let outcome = manager
            .run_subagent(SubAgentSpec::new("s1", "hello"))
            .await
            .unwrap();
        assert!(outcome.is_success());
    }

    #[tokio::test]
    async fn test_collect_all() {
        let manager = SubAgentManager::new(Arc::new(MockGenerator)).max_concurrent(2);
        let specs = vec![
            SubAgentSpec::new("a", "1"),
            SubAgentSpec::new("b", "2"),
            SubAgentSpec::new("c", "3"),
        ];
        let results = manager.collect_all(specs).await;
        assert_eq!(results.len(), 3);
        for outcome in results.values() {
            assert!(outcome.as_ref().unwrap().is_success());
        }
    }
}
