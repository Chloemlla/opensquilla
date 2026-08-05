//! # Meta-skill orchestration
//!
//! A *meta-skill* is a skill whose `kind` is `meta` or `meta_sop` and which
//! declares a DAG of [`SkillStep`]s. The [`MetaOrchestrator`] executes that
//! DAG: it topologically orders the steps, runs ready steps concurrently
//! (bounded by a parallelism cap), evaluates each step's `when` condition
//! against the running context, retries and times out individual steps, routes
//! outputs, and streams progress as [`MetaEvent`]s.
//!
//! Six step executors are built in:
//!
//! 1. `agent`        — runs a one-shot sub-agent turn.
//! 2. `llm_classify` — single constrained LLM call against a closed label set.
//! 3. `llm_chat`     — single unconstrained LLM call.
//! 4. `tool_call`    — direct tool invocation, bypassing the LLM.
//! 5. `skill_exec`   — executes a sub-skill.
//! 6. `user_input`   — requests user input.
//!
//! The orchestrator depends only on minimal injected protocols
//! ([`SubAgentRunner`], [`LlmChat`], [`ToolInvoker`], [`SkillResolver`],
//! [`UserInputHandler`]); it never constructs a sub-agent, an LLM call, or a
//! tool dispatch itself. Callers (the gateway / turn runner) inject concrete
//! implementations that capture provider, tool definitions, and usage tracking
//! from the parent turn.

use crate::types::{SkillSpec, SkillStep, StepOutput, StepType};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Injected dependency protocols
// ---------------------------------------------------------------------------
//
// The orchestrator depends only on these minimal protocols — it never owns
// the sub-Agent construction, the LLM call, or the tool dispatch. Callers
// (the gateway / turn runner) inject concrete implementations that capture
// provider / tool_defs / usage tracking from the parent turn.

/// Runs a one-shot sub-Agent turn and returns its final plain-text output.
#[async_trait]
pub trait SubAgentRunner: Send + Sync {
    async fn run_agent(&self, system_prompt: &str, user_message: &str) -> Result<String, String>;
}

/// Lightweight LLM-only call (no tool loop). Returns the model's reply text.
#[async_trait]
pub trait LlmChat: Send + Sync {
    async fn chat(&self, system_prompt: &str, user_message: &str) -> Result<String, String>;
}

/// Direct tool invoker — bypasses the LLM. Returns the tool result as text.
#[async_trait]
pub trait ToolInvoker: Send + Sync {
    async fn invoke(&self, tool_name: &str, args: &Value) -> Result<String, String>;
}

/// Resolves a sub-skill by ID (used by `skill_exec` steps).
#[async_trait]
pub trait SkillResolver: Send + Sync {
    async fn resolve(&self, id: &str) -> Result<Option<SkillSpec>, String>;
}

/// Gathers user input for `user_input` steps. Returns a JSON value with the
/// collected field values.
#[async_trait]
pub trait UserInputHandler: Send + Sync {
    async fn request_input(&self, prompt: &str, fields: &[Value]) -> Result<Value, String>;
}

/// Shared, runtime-mutable set of injected dependencies.
#[derive(Default)]
pub struct MetaDependencies {
    agent_runner: Option<Arc<dyn SubAgentRunner>>,
    llm_chat: Option<Arc<dyn LlmChat>>,
    tool_invoker: Option<Arc<dyn ToolInvoker>>,
    skill_resolver: Option<Arc<dyn SkillResolver>>,
    user_input_handler: Option<Arc<dyn UserInputHandler>>,
}

// ---------------------------------------------------------------------------
// Step executors
// ---------------------------------------------------------------------------

/// The execution context handed to a [`StepExecutor`].
#[derive(Clone)]
pub struct ExecutionContext {
    /// The step being executed.
    pub step: SkillStep,
    /// Rendered variables: `inputs.<name>`, `outputs.<step_id>`, plus the
    /// flattened context for legacy templates.
    pub variables: HashMap<String, Value>,
    /// Injected dependencies (runner / chat / invoker / resolver / input).
    pub deps: Arc<RwLock<MetaDependencies>>,
    /// Shared filesystem root for cross-skill artifacts.
    pub workspace_dir: Option<PathBuf>,
}

/// Per-kind step body. The orchestrator dispatches every step to the
/// executor registered for its [`StepType`].
#[async_trait]
pub trait StepExecutor: Send + Sync {
    /// The step type this executor handles.
    fn step_type(&self) -> StepType;
    /// Execute the step and return its JSON result.
    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String>;
}

/// Build a Tera instance with the custom helper functions used by templates in
/// the skill system.
fn build_tera() -> tera::Tera {
    let mut tera = tera::Tera::default();
    tera.register_function(
        "uuid",
        |_: &tera::Value, _: &HashMap<String, tera::Value>| {
            Ok(tera::Value::String(uuid::Uuid::new_v4().to_string()))
        },
    );
    tera.register_function(
        "now_iso",
        |_: &tera::Value, _: &HashMap<String, tera::Value>| {
            Ok(tera::Value::String(chrono::Utc::now().to_rfc3339()))
        },
    );
    tera.register_function(
        "ts",
        |_: &tera::Value, _: &HashMap<String, tera::Value>| {
            Ok(tera::Value::Number(chrono::Utc::now().timestamp().into()))
        },
    );
    tera
}

/// Renders a Jinja-ish `{{ var }}` template against the current variables.
/// Falls back to the raw template on any parse error so malformed templates
/// never crash a run.
pub fn render_template(template: &str, variables: &HashMap<String, Value>) -> String {
    if template.is_empty() {
        return String::new();
    }
    let mut ctx = tera::Context::new();
    for (k, v) in variables {
        ctx.insert(k, v);
    }
    build_tera()
        .render_str(template, &ctx)
        .unwrap_or_else(|_| template.to_string())
}

/// Renders `tool_args` / `with_args` maps, treating string values as templates.
pub fn render_args(args: &HashMap<String, Value>, variables: &HashMap<String, Value>) -> Value {
    let mut out = serde_json::Map::new();
    for (k, v) in args {
        if let Value::String(s) = v {
            out.insert(k.clone(), Value::String(render_template(s, variables)));
        } else if let Value::Object(_) = v {
            out.insert(k.clone(), render_value(v, variables));
        } else {
            out.insert(k.clone(), v.clone());
        }
    }
    Value::Object(out)
}

/// Recursively render templates inside a JSON value.
fn render_value(v: &Value, variables: &HashMap<String, Value>) -> Value {
    match v {
        Value::String(s) => Value::String(render_template(s, variables)),
        Value::Array(items) => {
            Value::Array(items.iter().map(|i| render_value(i, variables)).collect())
        }
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, val)| (k.clone(), render_value(val, variables)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Coerce a raw classifier reply into one of the closed choices.
pub fn coerce_to_choice(raw: &str, choices: &[String]) -> Option<String> {
    if choices.is_empty() {
        return None;
    }
    let text = raw.trim();
    if choices.iter().any(|c| c == text) {
        return Some(text.to_string());
    }
    let stripped = text.trim_matches(['\'', '"', '`', '.', ',', '!', '?', ' ', '\t', '\r', '\n']);
    if choices.iter().any(|c| c == stripped) {
        return Some(stripped.to_string());
    }
    let upper = stripped.to_uppercase();
    choices.iter().find(|c| c.to_uppercase() == upper).cloned()
}

// ---------------------------------------------------------------------------
// Executor 1/6 — agent
// ---------------------------------------------------------------------------

/// Executor 1/6 — `agent`: runs a sub-agent turn.
#[derive(Clone)]
pub struct AgentExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl AgentExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for AgentExecutor {
    fn step_type(&self) -> StepType {
        StepType::Agent
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let system_prompt =
            render_template(ctx.step.prompt.as_deref().unwrap_or(""), &ctx.variables);
        let user_message = ctx
            .variables
            .get("user_message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let runner = ctx.deps.read().unwrap().agent_runner.clone();
        if let Some(runner) = runner {
            let output = runner.run_agent(&system_prompt, &user_message).await?;
            Ok(json!({
                "status": "executed",
                "step_type": "agent",
                "output": output,
            }))
        } else {
            Ok(json!({
                "status": "executed",
                "step_type": "agent",
                "prompt": system_prompt,
                "degraded": true,
                "output": "",
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor 2/6 — llm_classify
// ---------------------------------------------------------------------------

/// Executor 2/6 — `llm_classify`: single constrained LLM call.
#[derive(Clone)]
pub struct LlmClassifyExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl LlmClassifyExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for LlmClassifyExecutor {
    fn step_type(&self) -> StepType {
        StepType::LlmClassify
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let choices = ctx.step.output_choices.clone();
        if choices.is_empty() {
            return Err(format!(
                "Step '{}' (kind=llm_classify) has no output_choices",
                ctx.step.id
            ));
        }
        let user_message = render_template(ctx.step.prompt.as_deref().unwrap_or(""), &ctx.variables);

        let chat = ctx.deps.read().unwrap().llm_chat.clone();
        if let Some(chat) = chat {
            let choices_str = choices.join(" | ");
            let system_prompt = format!(
                "You are a deterministic classifier. Read the user's input and decide \
                 which single label applies. Reply with EXACTLY ONE of: {choices_str}\n\
                 Do not add quotes, punctuation, prefixes, or explanations — emit only the label."
            );
            let raw = chat.chat(&system_prompt, &user_message).await?;
            let classification = coerce_to_choice(&raw, &choices)
                .unwrap_or_else(|| raw.trim().to_string());
            Ok(json!({
                "status": "classified",
                "step_type": "llm_classify",
                "classification": classification,
                "raw": raw,
            }))
        } else {
            Ok(json!({
                "status": "classified",
                "step_type": "llm_classify",
                "classification": "pending",
                "degraded": true,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor 3/6 — llm_chat
// ---------------------------------------------------------------------------

/// Executor 3/6 — `llm_chat`: single unconstrained LLM call.
#[derive(Clone)]
pub struct LlmChatExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl LlmChatExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for LlmChatExecutor {
    fn step_type(&self) -> StepType {
        StepType::LlmChat
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let args = &ctx.step.with_args;
        let system_raw = args
            .get("system")
            .and_then(|v| v.as_str())
            .unwrap_or("You are a precise workflow step. Reply only with the requested deliverable.");
        let system_prompt = render_template(system_raw, &ctx.variables);

        let user_raw = args
            .get("task")
            .or_else(|| args.get("prompt"))
            .or_else(|| args.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or(ctx.step.prompt.as_deref().unwrap_or(""));
        let user_message = render_template(user_raw, &ctx.variables);
        if user_message.trim().is_empty() {
            return Err(format!(
                "Step '{}' (kind=llm_chat) has no task/prompt/text",
                ctx.step.id
            ));
        }

        let chat = ctx.deps.read().unwrap().llm_chat.clone();
        if let Some(chat) = chat {
            let response = chat.chat(&system_prompt, &user_message).await?;
            Ok(json!({
                "status": "completed",
                "step_type": "llm_chat",
                "response": response,
            }))
        } else {
            Ok(json!({
                "status": "completed",
                "step_type": "llm_chat",
                "response": "pending",
                "degraded": true,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor 4/6 — tool_call
// ---------------------------------------------------------------------------

/// Executor 4/6 — `tool_call`: direct tool invocation.
#[derive(Clone)]
pub struct ToolCallExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl ToolCallExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for ToolCallExecutor {
    fn step_type(&self) -> StepType {
        StepType::ToolCall
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let tool = ctx.step.tool.clone().unwrap_or_default();
        if tool.is_empty() {
            return Err(format!("Step '{}' (kind=tool_call) has no tool", ctx.step.id));
        }
        let args = render_args(&ctx.step.tool_args, &ctx.variables);

        let invoker = ctx.deps.read().unwrap().tool_invoker.clone();
        if let Some(invoker) = invoker {
            let output = invoker.invoke(&tool, &args).await?;
            Ok(json!({
                "status": "called",
                "step_type": "tool_call",
                "tool": tool,
                "args": args,
                "output": output,
            }))
        } else {
            Ok(json!({
                "status": "called",
                "step_type": "tool_call",
                "tool": tool,
                "output": "",
                "degraded": true,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor 5/6 — skill_exec
// ---------------------------------------------------------------------------

/// Executor 5/6 — `skill_exec`: nested skill execution.
#[derive(Clone)]
pub struct SkillExecExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl SkillExecExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for SkillExecExecutor {
    fn step_type(&self) -> StepType {
        StepType::SkillExec
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let skill_id = ctx.step.skill.clone().unwrap_or_default();
        if skill_id.is_empty() {
            return Err(format!("Step '{}' (kind=skill_exec) has no skill", ctx.step.id));
        }

        let resolver = ctx.deps.read().unwrap().skill_resolver.clone();
        if let Some(resolver) = resolver {
            let sub = resolver
                .resolve(&skill_id)
                .await?
                .ok_or_else(|| format!("Sub-skill '{}' not found", skill_id))?;

            let body = format!(
                "You are executing the sub-skill '{}'.\n\n{}",
                sub.name, sub.description
            );
            let user_message =
                render_template(ctx.step.prompt.as_deref().unwrap_or(""), &ctx.variables);

            let runner = ctx.deps.read().unwrap().agent_runner.clone();
            if let Some(runner) = runner {
                let output = runner.run_agent(&body, &user_message).await?;
                Ok(json!({
                    "status": "executed",
                    "step_type": "skill_exec",
                    "skill": skill_id,
                    "output": output,
                }))
            } else {
                Ok(json!({
                    "status": "executed",
                    "step_type": "skill_exec",
                    "skill": skill_id,
                    "loaded": sub.name,
                    "degraded": true,
                    "output": "",
                }))
            }
        } else {
            Ok(json!({
                "status": "dispatched",
                "step_type": "skill_exec",
                "skill": skill_id,
                "degraded": true,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Executor 6/6 — user_input
// ---------------------------------------------------------------------------

/// Executor 6/6 — `user_input`: gather user input.
#[derive(Clone)]
pub struct UserInputExecutor {
    deps: Arc<RwLock<MetaDependencies>>,
}

impl UserInputExecutor {
    pub fn new(deps: Arc<RwLock<MetaDependencies>>) -> Self {
        Self { deps }
    }
}

#[async_trait]
impl StepExecutor for UserInputExecutor {
    fn step_type(&self) -> StepType {
        StepType::UserInput
    }

    async fn execute(&self, ctx: &ExecutionContext) -> Result<Value, String> {
        let prompt = render_template(ctx.step.prompt.as_deref().unwrap_or(""), &ctx.variables);
        let fields: Vec<Value> = ctx
            .step
            .with_args
            .get("fields")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let handler = ctx.deps.read().unwrap().user_input_handler.clone();
        if let Some(handler) = handler {
            let input = handler.request_input(&prompt, &fields).await?;
            Ok(json!({
                "status": "received",
                "step_type": "user_input",
                "prompt": prompt,
                "input": input,
            }))
        } else {
            Ok(json!({
                "status": "awaiting_input",
                "step_type": "user_input",
                "prompt": prompt,
                "degraded": true,
            }))
        }
    }
}

// ---------------------------------------------------------------------------
// Event streaming
// ---------------------------------------------------------------------------

/// Progress events emitted while a meta-skill DAG runs. Subscribe via
/// [`MetaOrchestrator::subscribe`] to stream progress for UI reporting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetaEvent {
    /// The run started and the DAG was validated.
    RunStarted {
        run_id: String,
        skill_id: String,
        total_steps: usize,
    },
    /// A step began executing.
    StepStarted {
        run_id: String,
        step_id: String,
        step_name: String,
        step_type: String,
    },
    /// A step produced an output value.
    StepOutput {
        run_id: String,
        step_id: String,
        output: Value,
    },
    /// A step was skipped because its `when` condition evaluated false.
    StepSkipped {
        run_id: String,
        step_id: String,
        reason: String,
    },
    /// A failed step is being retried.
    StepRetrying {
        run_id: String,
        step_id: String,
        attempt: u32,
        error: String,
    },
    /// A step completed successfully.
    StepCompleted {
        run_id: String,
        step_id: String,
    },
    /// A step failed permanently (no retries left).
    StepFailed {
        run_id: String,
        step_id: String,
        error: String,
    },
    /// A step's output was routed to another step via `route_to`.
    StepRouted {
        run_id: String,
        from_step: String,
        to_step: String,
        on_error: bool,
    },
    /// The run was cancelled by a caller.
    RunCancelled { run_id: String },
    /// The run finished, success or failure.
    RunCompleted {
        run_id: String,
        success: bool,
        outputs: HashMap<String, Value>,
    },
}

// ---------------------------------------------------------------------------
// DAG utilities
// ---------------------------------------------------------------------------

/// A dependency edge between steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagEdge {
    /// The step that must complete first.
    pub from: String,
    /// The step that depends on `from`.
    pub to: String,
}

/// A validated DAG of steps: topological order plus dependency metadata.
#[derive(Debug, Clone)]
pub struct Dag {
    /// Steps keyed by id, in declaration order.
    pub steps: Vec<SkillStep>,
    /// A valid topological ordering of the step ids.
    pub order: Vec<String>,
    /// Dependency edges.
    pub edges: Vec<DagEdge>,
}

impl Dag {
    /// Build and validate a DAG from a list of steps.
    ///
    /// Errors if any `depends_on` refers to an unknown step or if the
    /// dependency graph contains a cycle.
    pub fn new(steps: &[SkillStep]) -> Result<Self, String> {
        let ids: HashSet<&str> = steps.iter().map(|s| s.id.as_str()).collect();
        for step in steps {
            if step.id.is_empty() {
                return Err("step id must not be empty".to_string());
            }
            if let Some(deps) = &step.depends_on {
                for dep in deps {
                    if !ids.contains(dep.as_str()) {
                        return Err(format!(
                            "Step '{}' depends on unknown step '{}'",
                            step.id, dep
                        ));
                    }
                }
            }
        }

        let edges = build_edges(steps);
        let order = topological_order(steps, &edges)?;
        Ok(Self {
            steps: steps.to_vec(),
            order,
            edges,
        })
    }

    /// The steps that have no dependencies (the DAG's roots).
    pub fn roots(&self) -> Vec<String> {
        let depended: HashSet<&str> = self.edges.iter().map(|e| e.to.as_str()).collect();
        self.steps
            .iter()
            .filter(|s| !depended.contains(s.id.as_str()))
            .map(|s| s.id.clone())
            .collect()
    }

    /// The direct dependents of a step id.
    pub fn dependents_of(&self, id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.from == id)
            .map(|e| e.to.clone())
            .collect()
    }

    /// The direct dependencies of a step id.
    pub fn dependencies_of(&self, id: &str) -> Vec<String> {
        self.edges
            .iter()
            .filter(|e| e.to == id)
            .map(|e| e.from.clone())
            .collect()
    }

    /// The longest path (in edges) from any root to this step.
    pub fn depth_of(&self, id: &str) -> usize {
        let mut depth: HashMap<String, usize> = HashMap::new();
        for step_id in &self.order {
            let deps = self.dependencies_of(step_id);
            let d = deps
                .iter()
                .map(|d| depth.get(d).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            depth.insert(step_id.clone(), d + 1);
        }
        depth.get(id).copied().unwrap_or(0)
    }

    /// Validate that the DAG is fully connected (every step reachable from a
    /// root, directly or transitively).
    pub fn is_fully_reachable(&self) -> bool {
        self.order.len() == self.steps.len()
    }
}

fn build_edges(steps: &[SkillStep]) -> Vec<DagEdge> {
    let mut edges = Vec::new();
    for step in steps {
        if let Some(deps) = &step.depends_on {
            for dep in deps {
                edges.push(DagEdge {
                    from: dep.clone(),
                    to: step.id.clone(),
                });
            }
        }
    }
    edges
}

/// Kahn's algorithm. Returns the topological order or an error if there is a
/// cycle.
fn topological_order(steps: &[SkillStep], edges: &[DagEdge]) -> Result<Vec<String>, String> {
    let mut in_degree: HashMap<String, usize> =
        steps.iter().map(|s| (s.id.clone(), 0usize)).collect();
    let mut adjacency: HashMap<String, Vec<String>> =
        steps.iter().map(|s| (s.id.clone(), Vec::new())).collect();

    for edge in edges {
        *in_degree.get_mut(&edge.to).unwrap() += 1;
        adjacency.get_mut(&edge.from).unwrap().push(edge.to.clone());
    }

    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| id.clone())
        .collect();

    let mut order = Vec::with_capacity(steps.len());
    while let Some(id) = queue.pop_front() {
        order.push(id.clone());
        if let Some(next) = adjacency.get(&id) {
            for n in next {
                let degree = in_degree.get_mut(n).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(n.clone());
                }
            }
        }
    }

    if order.len() != steps.len() {
        let sorted: HashSet<&str> = order.iter().map(|s| s.as_str()).collect();
        let unsorted: Vec<String> = steps
            .iter()
            .map(|s| s.id.clone())
            .filter(|id| !sorted.contains(id.as_str()))
            .collect();
        return Err(format!("Cycle detected in DAG. Steps in cycle: {:?}", unsorted));
    }
    Ok(order)
}

// ---------------------------------------------------------------------------
// Run management
// ---------------------------------------------------------------------------

/// A handle to a running (or finished) meta-skill run.
///
/// Obtained from [`MetaOrchestrator::start_run`]. The run continues in the
/// background; the handle can be used to cancel it or subscribe to events.
#[derive(Clone)]
pub struct MetaRun {
    /// The unique run id.
    pub run_id: String,
    /// Cancellation flag shared with the orchestrator task.
    cancel_flag: Arc<AtomicBool>,
    /// The orchestrator that owns this run.
    orchestrator: Arc<MetaOrchestrator>,
}

impl MetaRun {
    /// Request cancellation of this run. The orchestrator checks the flag
    /// between steps and aborts as soon as possible.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// Subscribe to this run's progress events.
    pub fn subscribe(&self) -> broadcast::Receiver<MetaEvent> {
        self.orchestrator.subscribe()
    }

    /// The run id as a string.
    pub fn id(&self) -> &str {
        &self.run_id
    }
}

/// Per-run bookkeeping kept by the orchestrator.
struct RunState {
    /// Whether cancellation was requested for this run.
    cancelled: Arc<AtomicBool>,
}

// ---------------------------------------------------------------------------
// MetaOrchestrator
// ---------------------------------------------------------------------------

/// Orchestrates a meta-skill DAG: topologically orders the steps, runs
/// ready steps concurrently (capped by `max_parallelism`), evaluates `when`
/// conditions, and streams [`MetaEvent`]s for progress reporting.
pub struct MetaOrchestrator {
    /// The current execution context (inputs + routed outputs).
    context: Arc<Mutex<HashMap<String, Value>>>,
    /// Step execution results, keyed by step ID.
    results: Arc<Mutex<HashMap<String, Value>>>,
    /// Registered step executors keyed by step type.
    executors: Arc<RwLock<HashMap<StepType, Arc<dyn StepExecutor>>>>,
    /// Injected dependencies shared with the built-in executors.
    deps: Arc<RwLock<MetaDependencies>>,
    /// Event channel for progress reporting.
    event_tx: broadcast::Sender<MetaEvent>,
    /// Concurrency cap. `None` = unbounded.
    max_parallelism: Option<usize>,
    /// Shared filesystem root for cross-skill artifacts.
    workspace_dir: Option<PathBuf>,
    /// Active run states keyed by run id.
    runs: Arc<RwLock<HashMap<String, RunState>>>,
}

impl MetaOrchestrator {
    /// Create an orchestrator with the six built-in executors and a default
    /// concurrency cap of 4.
    pub fn new() -> Self {
        let deps = Arc::new(RwLock::new(MetaDependencies::default()));
        let executors = Arc::new(RwLock::new(HashMap::<StepType, Arc<dyn StepExecutor>>::new()));
        let (event_tx, _) = broadcast::channel(256);

        let orch = Self {
            context: Arc::new(Mutex::new(HashMap::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
            executors,
            deps,
            event_tx,
            max_parallelism: Some(4),
            workspace_dir: None,
            runs: Arc::new(RwLock::new(HashMap::new())),
        };

        orch.register_executor(Arc::new(AgentExecutor::new(orch.deps.clone())));
        orch.register_executor(Arc::new(LlmClassifyExecutor::new(orch.deps.clone())));
        orch.register_executor(Arc::new(LlmChatExecutor::new(orch.deps.clone())));
        orch.register_executor(Arc::new(ToolCallExecutor::new(orch.deps.clone())));
        orch.register_executor(Arc::new(SkillExecExecutor::new(orch.deps.clone())));
        orch.register_executor(Arc::new(UserInputExecutor::new(orch.deps.clone())));
        orch
    }

    /// Register a custom executor. Overrides the built-in executor for the
    /// same step type.
    pub fn register_executor(&self, executor: Arc<dyn StepExecutor>) {
        if let Ok(mut map) = self.executors.write() {
            map.insert(executor.step_type(), executor);
        }
    }

    /// Inject the sub-agent runner used by `agent` / `skill_exec` steps.
    pub fn set_agent_runner<R: SubAgentRunner + 'static>(&self, runner: R) -> &Self {
        if let Ok(mut deps) = self.deps.write() {
            deps.agent_runner = Some(Arc::new(runner));
        }
        self
    }

    /// Inject the LLM chat callback used by `llm_chat` / `llm_classify`.
    pub fn set_llm_chat<C: LlmChat + 'static>(&self, chat: C) -> &Self {
        if let Ok(mut deps) = self.deps.write() {
            deps.llm_chat = Some(Arc::new(chat));
        }
        self
    }

    /// Inject the direct tool invoker used by `tool_call` steps.
    pub fn set_tool_invoker<I: ToolInvoker + 'static>(&self, invoker: I) -> &Self {
        if let Ok(mut deps) = self.deps.write() {
            deps.tool_invoker = Some(Arc::new(invoker));
        }
        self
    }

    /// Inject the sub-skill resolver used by `skill_exec` steps.
    pub fn set_skill_resolver<R: SkillResolver + 'static>(&self, resolver: R) -> &Self {
        if let Ok(mut deps) = self.deps.write() {
            deps.skill_resolver = Some(Arc::new(resolver));
        }
        self
    }

    /// Inject the user-input handler used by `user_input` steps.
    pub fn set_user_input_handler<H: UserInputHandler + 'static>(&self, handler: H) -> &Self {
        if let Ok(mut deps) = self.deps.write() {
            deps.user_input_handler = Some(Arc::new(handler));
        }
        self
    }

    /// Set the shared workspace root for cross-skill artifacts.
    pub fn set_workspace_dir(&self, dir: PathBuf) -> &Self {
        self.workspace_dir = Some(dir);
        self
    }

    /// Set the concurrency cap. `None` = unbounded.
    pub fn set_max_parallelism(&self, max: Option<usize>) -> &Self {
        self.max_parallelism = max;
        self
    }

    /// Subscribe to progress events emitted while a run executes.
    pub fn subscribe(&self) -> broadcast::Receiver<MetaEvent> {
        self.event_tx.subscribe()
    }

    /// Build a validated [`Dag`] for a skill's steps.
    pub fn build_dag(&self, steps: &[SkillStep]) -> Result<Dag, String> {
        Dag::new(steps)
    }

    /// Start a meta-skill run in the background and return a [`MetaRun`]
    /// handle. The handle can cancel the run or subscribe to its events.
    pub fn start_run(
        &self,
        skill: SkillSpec,
        initial_context: HashMap<String, Value>,
    ) -> Result<MetaRun, String> {
        if !skill.is_meta() {
            return Err(format!("Skill '{}' is not a meta-skill", skill.id));
        }
        // Validate the DAG eagerly so failures surface before spawning.
        Dag::new(&skill.steps)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        if let Ok(mut runs) = self.runs.write() {
            runs.insert(
                run_id.clone(),
                RunState {
                    cancelled: cancel_flag.clone(),
                },
            );
        }

        let orchestrator = Arc::new(self.clone_handle());
        tokio::spawn(async move {
            let result = orchestrator
                .execute_with_cancel(&skill, initial_context, cancel_flag.clone())
                .await;
            orchestrator.finish_run(&run_id);
            let _ = result;
        });

        Ok(MetaRun {
            run_id,
            cancel_flag,
            orchestrator: Arc::new(self.clone_handle()),
        })
    }

    /// Cancel a running meta-skill by run id.
    pub fn cancel_run(&self, run_id: &str) -> bool {
        if let Ok(runs) = self.runs.read() {
            if let Some(state) = runs.get(run_id) {
                state.cancelled.store(true, Ordering::SeqCst);
                return true;
            }
        }
        false
    }

    /// Whether a run is still active (not yet finished).
    pub fn is_run_active(&self, run_id: &str) -> bool {
        self.runs.read().map(|r| r.contains_key(run_id)).unwrap_or(false)
    }

    fn finish_run(&self, run_id: &str) {
        if let Ok(mut runs) = self.runs.write() {
            runs.remove(run_id);
        }
    }

    /// Internal clone of the orchestrator's shared state, used to hand an
    /// independent `Arc` to a spawned run task.
    fn clone_handle(&self) -> MetaOrchestrator {
        MetaOrchestrator {
            context: self.context.clone(),
            results: self.results.clone(),
            executors: self.executors.clone(),
            deps: self.deps.clone(),
            event_tx: self.event_tx.clone(),
            max_parallelism: self.max_parallelism,
            workspace_dir: self.workspace_dir.clone(),
            runs: self.runs.clone(),
        }
    }

    /// Execute a meta-skill's DAG workflow.
    ///
    /// Steps whose `depends_on` is satisfied run concurrently (up to
    /// `max_parallelism`). `when` conditions are evaluated against the
    /// current context + prior step outputs. Returns the final outputs
    /// mapped through `skill.outputs`.
    pub async fn execute(
        &self,
        skill: &SkillSpec,
        initial_context: HashMap<String, Value>,
    ) -> Result<HashMap<String, Value>, String> {
        self.execute_with_cancel(skill, initial_context, Arc::new(AtomicBool::new(false)))
            .await
    }

    /// Execute with an explicit cancellation flag.
    async fn execute_with_cancel(
        &self,
        skill: &SkillSpec,
        initial_context: HashMap<String, Value>,
        cancel_flag: Arc<AtomicBool>,
    ) -> Result<HashMap<String, Value>, String> {
        if !skill.is_meta() {
            return Err(format!("Skill '{}' is not a meta-skill", skill.id));
        }

        {
            let mut ctx = self.context.lock().map_err(|e| e.to_string())?;
            ctx.extend(initial_context);
        }

        // Validate the DAG (cycle / missing-dependency detection).
        self.build_dag(&skill.steps)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        self.emit(MetaEvent::RunStarted {
            run_id: run_id.clone(),
            skill_id: skill.id.clone(),
            total_steps: skill.steps.len(),
        });
        info!("Executing meta-skill '{}' with {} steps", skill.name, skill.steps.len());

        let step_map: HashMap<String, SkillStep> = skill
            .steps
            .iter()
            .map(|s| (s.id.clone(), s.clone()))
            .collect();

        let mut remaining: HashMap<String, usize> = HashMap::new();
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
        for step in &skill.steps {
            remaining.insert(step.id.clone(), 0);
            dependents.insert(step.id.clone(), Vec::new());
        }
        for step in &skill.steps {
            if let Some(deps) = &step.depends_on {
                for dep in deps {
                    if let Some(deg) = remaining.get_mut(&step.id) {
                        *deg += 1;
                    }
                    if let Some(list) = dependents.get_mut(dep) {
                        if !list.contains(&step.id) {
                            list.push(step.id.clone());
                        }
                    }
                }
            }
        }

        let mut ready: Vec<String> = remaining
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| id.clone())
            .collect();

        let mut running: tokio::task::JoinSet<(String, Result<Value, String>)> =
            tokio::task::JoinSet::new();
        let mut outputs: HashMap<String, Value> = HashMap::new();
        let mut finished: HashSet<String> = HashSet::new();
        // Set when a step failed but routed to a fallback (`route_on_error`).
        // When set, the strict "every step finished" check is relaxed because
        // the failed branch's downstream steps are intentionally stranded.
        let mut routed_on_error = false;

        while !ready.is_empty() || !running.is_empty() {
            if cancel_flag.load(Ordering::SeqCst) {
                running.abort_all();
                while running.join_next().await.is_some() {}
                self.emit(MetaEvent::RunCancelled { run_id: run_id.clone() });
                return Err("Meta-skill run cancelled".to_string());
            }

            // Sort ready by priority (higher first) so higher-priority steps
            // among the ready set run first.
            ready.sort_by(|a, b| {
                let pa = step_map.get(a).and_then(|s| s.priority).unwrap_or(0);
                let pb = step_map.get(b).and_then(|s| s.priority).unwrap_or(0);
                pb.cmp(&pa)
            });

            // Spawn all currently ready steps, subject to the parallelism cap.
            while !ready.is_empty() && self.parallelism_available(&running) {
                let step_id = ready.remove(0);

                // Evaluate `when` before spawning; skipped steps still emit
                // an empty output so downstream `depends_on` links unblock.
                let ctx_snapshot = self.context.lock().map_err(|e| e.to_string())?.clone();
                let should_run = evaluate_when(
                    step_map
                        .get(&step_id)
                        .map(|s| s.when.as_deref().unwrap_or(""))
                        .unwrap_or(""),
                    &ctx_snapshot,
                    &outputs,
                )?;
                if !should_run {
                    outputs.insert(step_id.clone(), Value::Null);
                    finished.insert(step_id.clone());
                    self.emit(MetaEvent::StepSkipped {
                        run_id: run_id.clone(),
                        step_id: step_id.clone(),
                        reason: "when condition evaluated false".to_string(),
                    });
                    info!(
                        "Step '{}' skipped (when condition false)",
                        step_map.get(&step_id).map(|s| s.name.as_str()).unwrap_or(&step_id)
                    );
                    self.release_step(&step_id, &mut remaining, &dependents, &mut ready);
                    continue;
                }

                let step = step_map
                    .get(&step_id)
                    .cloned()
                    .ok_or_else(|| format!("Unknown step '{}'", step_id))?;

                self.emit(MetaEvent::StepStarted {
                    run_id: run_id.clone(),
                    step_id: step_id.clone(),
                    step_name: step.name.clone(),
                    step_type: step.step_type.to_string(),
                });

                let variables = self.run_variables(&outputs)?;
                let deps = self.deps.clone();
                let executors = self.executors.clone();
                let workspace_dir = self.workspace_dir.clone();
                let event_tx = self.event_tx.clone();
                let run_id_for_task = run_id.clone();
                running.spawn(async move {
                    let result = run_step(
                        &step,
                        variables,
                        deps,
                        executors,
                        workspace_dir,
                        event_tx,
                        &run_id_for_task,
                    )
                    .await;
                    (step.id.clone(), result)
                });
            }

            if running.is_empty() {
                break;
            }

            match running.join_next().await {
                Some(Ok((step_id, Ok(output)))) => {
                    finished.insert(step_id.clone());
                    // Apply `pick` extraction from the step's output config.
                    let output = if let Some(step) = step_map.get(&step_id) {
                        apply_output_pick(&output, step.output.as_ref())
                    } else {
                        output
                    };
                    outputs.insert(step_id.clone(), output.clone());
                    {
                        let mut res = self.results.lock().map_err(|e| e.to_string())?;
                        res.insert(step_id.clone(), output.clone());
                    }

                    // Route output into the shared context if configured.
                    if let Some(step) = step_map.get(&step_id) {
                        if let Some(ref out) = step.output {
                            if let Some(ref var) = out.var {
                                let mut ctx = self.context.lock().map_err(|e| e.to_string())?;
                                ctx.insert(var.clone(), output.clone());
                            }
                        }
                    }

                    self.emit(MetaEvent::StepOutput {
                        run_id: run_id.clone(),
                        step_id: step_id.clone(),
                        output: output.clone(),
                    });
                    self.emit(MetaEvent::StepCompleted {
                        run_id: run_id.clone(),
                        step_id: step_id.clone(),
                    });
                    self.release_step(&step_id, &mut remaining, &dependents, &mut ready);

                    // Handle `route_to`: additionally schedule the target step.
                    if let Some(step) = step_map.get(&step_id) {
                        if let Some(ref out) = step.output {
                            if let Some(target) = &out.route_to {
                                let target_ready = step_map.contains_key(target)
                                    && !finished.contains(target)
                                    && !ready.contains(target)
                                    && remaining.get(target).copied().unwrap_or(0) == 0
                                    && running.len() < self.max_parallelism.unwrap_or(usize::MAX);
                                if target_ready {
                                    ready.push(target.clone());
                                    self.emit(MetaEvent::StepRouted {
                                        run_id: run_id.clone(),
                                        from_step: step_id.clone(),
                                        to_step: target.clone(),
                                        on_error: false,
                                    });
                                }
                            }
                        }
                    }
                }
                Some(Ok((step_id, Err(error)))) => {
                    // If the step routes errors to a fallback, follow it;
                    // otherwise abort the run.
                    let route = step_map
                        .get(&step_id)
                        .and_then(|s| s.output.as_ref())
                        .and_then(|o| o.route_on_error.clone());
                    self.emit(MetaEvent::StepFailed {
                        run_id: run_id.clone(),
                        step_id: step_id.clone(),
                        error: error.clone(),
                    });
                    match route {
                        Some(target)
                            if step_map.contains_key(&target)
                                && !dependents
                                    .get(&step_id)
                                    .is_some_and(|d| d.contains(&target))
                                && remaining.get(&target).copied().unwrap_or(0) == 0 =>
                        {
                            routed_on_error = true;
                            finished.insert(step_id.clone());
                            outputs.insert(step_id.clone(), json!({ "error": error.clone() }));
                            // Strand any direct dependents of the failed step:
                            // they cannot run without it. Mark them finished
                            // with a skip so the run can complete.
                            if let Some(list) = dependents.get_mut(&step_id) {
                                let stranded: Vec<String> = list.clone();
                                list.clear();
                                for dep in stranded {
                                    finished.insert(dep.clone());
                                    outputs.insert(
                                        dep.clone(),
                                        json!({ "skipped": "upstream step failed", "from": step_id }),
                                    );
                                }
                            }
                            self.emit(MetaEvent::StepRouted {
                                run_id: run_id.clone(),
                                from_step: step_id.clone(),
                                to_step: target.clone(),
                                on_error: true,
                            });
                            ready.push(target);
                            self.release_step(&step_id, &mut remaining, &dependents, &mut ready);
                        }
                        _ => {
                            running.abort_all();
                            while running.join_next().await.is_some() {}
                            self.emit(MetaEvent::RunCompleted {
                                run_id,
                                success: false,
                                outputs: outputs.clone(),
                            });
                            return Err(error);
                        }
                    }
                }
                Some(Err(join_err)) => {
                    return Err(format!("Step task panicked: {}", join_err));
                }
                None => break,
            }
        }

        // Every step must have been scheduled (completed or skipped), unless a
        // failure routed to a fallback (which intentionally strands a branch).
        if !routed_on_error && finished.len() != skill.steps.len() {
            let missing: Vec<String> = skill
                .steps
                .iter()
                .filter(|s| !finished.contains(&s.id))
                .map(|s| s.id.clone())
                .collect();
            self.emit(MetaEvent::RunCompleted {
                run_id,
                success: false,
                outputs: outputs.clone(),
            });
            return Err(format!(
                "DAG did not fully execute; steps never scheduled: {:?}",
                missing
            ));
        }

        // Collect the final outputs through `skill.outputs`.
        let final_outputs = skill.outputs.clone();
        let results = self.results.lock().map_err(|e| e.to_string())?;
        let mut result = HashMap::new();
        for (output_key, step_id) in &final_outputs {
            if let Some(value) = results.get(step_id) {
                result.insert(output_key.clone(), value.clone());
            }
        }

        self.emit(MetaEvent::RunCompleted {
            run_id,
            success: true,
            outputs: result.clone(),
        });
        info!("Meta-skill '{}' execution complete", skill.name);
        Ok(result)
    }

    fn emit(&self, event: MetaEvent) {
        let _ = self.event_tx.send(event);
    }

    fn parallelism_available(
        &self,
        running: &tokio::task::JoinSet<(String, Result<Value, String>)>,
    ) -> bool {
        match self.max_parallelism {
            Some(max) if max > 0 => running.len() < max,
            _ => true,
        }
    }

    fn release_step(
        &self,
        step_id: &str,
        remaining: &mut HashMap<String, usize>,
        dependents: &HashMap<String, Vec<String>>,
        ready: &mut Vec<String>,
    ) {
        if let Some(list) = dependents.get(step_id) {
            for dep in list {
                if let Some(deg) = remaining.get_mut(dep) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 && !ready.contains(dep) {
                        ready.push(dep.clone());
                    }
                }
            }
        }
    }

    fn run_variables(
        &self,
        outputs: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Value>, String> {
        let ctx = self.context.lock().map_err(|e| e.to_string())?;
        let mut variables = ctx.clone();
        variables.insert(
            "inputs".to_string(),
            Value::Object(ctx.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        );
        variables.insert(
            "outputs".to_string(),
            Value::Object(outputs.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        );
        for (k, v) in outputs {
            variables.entry(k.clone()).or_insert_with(|| v.clone());
        }
        Ok(variables)
    }

    /// Get the execution result for a specific step.
    pub fn get_step_result(&self, step_id: &str) -> Option<Value> {
        self.results.lock().ok().and_then(|r| r.get(step_id).cloned())
    }

    /// Get all step results of the last run.
    pub fn get_step_results(&self) -> HashMap<String, Value> {
        self.results.lock().ok().map(|r| r.clone()).unwrap_or_default()
    }

    /// The full current context (inputs + routed outputs).
    pub fn current_context(&self) -> HashMap<String, Value> {
        self.context.lock().ok().map(|c| c.clone()).unwrap_or_default()
    }

    /// Clear all execution state.
    pub fn reset(&self) {
        if let Ok(mut ctx) = self.context.lock() {
            ctx.clear();
        }
        if let Ok(mut results) = self.results.lock() {
            results.clear();
        }
        if let Ok(mut runs) = self.runs.write() {
            runs.clear();
        }
    }
}

impl Default for MetaOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a meta-skill DAG in the background, returning a handle to the final
/// outputs. Call [`MetaOrchestrator::subscribe`] first to receive progress
/// events.
pub fn spawn_orchestrator(
    orchestrator: Arc<MetaOrchestrator>,
    skill: SkillSpec,
    initial_context: HashMap<String, Value>,
) -> JoinHandle<Result<HashMap<String, Value>, String>> {
    tokio::spawn(async move { orchestrator.execute(&skill, initial_context).await })
}

// ---------------------------------------------------------------------------
// Step runner (used inside spawned tasks)
// ---------------------------------------------------------------------------

/// Runs a single step body with retry + timeout, dispatching to the executor
/// registered for the step's type. Emits `StepRetrying` events on transient
/// failures.
async fn run_step(
    step: &SkillStep,
    variables: HashMap<String, Value>,
    deps: Arc<RwLock<MetaDependencies>>,
    executors: Arc<RwLock<HashMap<StepType, Arc<dyn StepExecutor>>>>,
    workspace_dir: Option<PathBuf>,
    event_tx: broadcast::Sender<MetaEvent>,
    run_id: &str,
) -> Result<Value, String> {
    let max_retries = step.max_retries.unwrap_or(0);
    let timeout_secs = step.timeout_secs;
    let backoff_ms = step.retry_backoff_ms.unwrap_or(200);
    let mut attempt = 0u32;

    loop {
        let ctx = ExecutionContext {
            step: step.clone(),
            variables: variables.clone(),
            deps: deps.clone(),
            workspace_dir: workspace_dir.clone(),
        };

        let executor = executors
            .read()
            .unwrap()
            .get(&ctx.step.step_type)
            .cloned()
            .ok_or_else(|| format!("No executor registered for step type {:?}", ctx.step.step_type))?;

        let fut = executor.execute(&ctx);
        let result = match timeout_secs {
            Some(secs) => match timeout(Duration::from_secs(secs), fut).await {
                Ok(r) => r,
                Err(_) => Err(format!("Step '{}' timed out after {}s", step.id, secs)),
            },
            None => fut.await,
        };

        match result {
            Ok(v) => return Ok(v),
            Err(e) if attempt < max_retries => {
                attempt += 1;
                let delay = backoff_ms.saturating_mul(1u64 << (attempt - 1).min(6));
                warn!(
                    "Step '{}' failed (attempt {}/{}): {}. Retrying in {}ms",
                    step.id, attempt, max_retries, e, delay
                );
                let _ = event_tx.send(MetaEvent::StepRetrying {
                    run_id: run_id.to_string(),
                    step_id: step.id.clone(),
                    attempt,
                    error: e,
                });
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Apply a step's `output.pick` list to extract sub-fields from a JSON output.
fn apply_output_pick(output: &Value, step_output: Option<&StepOutput>) -> Value {
    let Some(pick) = step_output.and_then(|o| o.pick.as_ref()) else {
        return output.clone();
    };
    if pick.is_empty() {
        return output.clone();
    }
    match output {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for key in pick {
                if let Some(v) = map.get(key) {
                    out.insert(key.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        Value::String(s) => {
            // A string may be the JSON serialization of an object.
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                apply_output_pick(&parsed, step_output)
            } else {
                output.clone()
            }
        }
        _ => output.clone(),
    }
}

// ---------------------------------------------------------------------------
// `when` expression evaluation
// ---------------------------------------------------------------------------

/// Evaluates a `when` expression against the shared context and prior step
/// outputs.
///
/// Supported syntax:
/// - `var`, `!var` — truthiness / negation
/// - `var == "value"`, `var != "value"`, `var > 3`, `var <= 10`
/// - `var in ["a", "b"]` or `var in ('a', 'b')`
/// - `var not in ["a", "b"]`
/// - `var.contains("x")`, `var.startswith("x")`, `var.endswith("x")`
/// - `len(var) > 3`
/// - `var ~ "regex"` — regular-expression match
/// - `a && b`, `a || b`
/// - dotted paths: `inputs.foo`, `outputs.step_id`
pub fn evaluate_when(
    expr: &str,
    ctx: &HashMap<String, Value>,
    outputs: &HashMap<String, Value>,
) -> Result<bool, String> {
    let e = expr.trim();
    if e.is_empty() {
        return Ok(true);
    }

    if let Some(parts) = split_top_level(e, "||") {
        for part in parts {
            if evaluate_when(part.trim(), ctx, outputs)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }

    if let Some(parts) = split_top_level(e, "&&") {
        for part in parts {
            if !evaluate_when(part.trim(), ctx, outputs)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Some(rest) = e.strip_prefix('!') {
        return Ok(!evaluate_when(rest.trim(), ctx, outputs)?);
    }

    // `not in` — must be checked before the ` in ` operator.
    if let Some(idx) = find_operator(e, " not in ") {
        let left = e[..idx].trim();
        let right = e[idx + " not in ".len()..].trim();
        let (open, close) = bracket_pair(right)?;
        let inner = &right[1..right.len() - 1];
        let items = split_list(inner, open, close);
        let left_val = resolve_var(left, ctx, outputs);
        let is_in = items.iter().any(|item| value_matches(&left_val, item));
        return Ok(!is_in);
    }

    // Function-call style predicates.
    if let Some(result) = eval_function_predicate(e, ctx, outputs)? {
        return Ok(result);
    }

    // Membership: `x in [a, b]`
    if let Some(idx) = find_operator(e, " in ") {
        let left = e[..idx].trim();
        let right = e[idx + 4..].trim();
        let (open, close) = bracket_pair(right)?;
        let inner = &right[1..right.len() - 1];
        let items = split_list(inner, open, close);
        let left_val = resolve_var(left, ctx, outputs);
        return Ok(items.iter().any(|item| value_matches(&left_val, item)));
    }

    // Regex match: `var ~ "pattern"`
    if let Some(idx) = find_operator(e, "~") {
        let left = e[..idx].trim();
        let right = e[idx + 1..].trim();
        let pattern = unquote(right).ok_or_else(|| format!("Invalid regex literal: {right}"))?;
        let val = resolve_var(left, ctx, outputs);
        let text = match val {
            Value::String(s) => s,
            other => other.to_string(),
        };
        let re = regex::Regex::new(&pattern)
            .map_err(|err| format!("Invalid regex '{pattern}': {err}"))?;
        return Ok(re.is_match(&text));
    }

    for op in ["==", "!=", ">=", "<=", ">", "<"] {
        if let Some(idx) = find_operator(e, op) {
            let left = e[..idx].trim();
            let right = e[idx + op.len()..].trim();
            let lval = resolve_var(left, ctx, outputs);
            let rval = parse_literal(right);
            return compare_values(&lval, &rval, op);
        }
    }

    Ok(is_truthy(&resolve_var(e, ctx, outputs)))
}

/// Evaluate function-call-style predicates: `x.contains(...)`,
/// `x.startswith(...)`, `x.endswith(...)`, `len(x) op N`.
///
/// Returns `Ok(None)` when the expression is not a known function predicate.
fn eval_function_predicate(
    e: &str,
    ctx: &HashMap<String, Value>,
    outputs: &HashMap<String, Value>,
) -> Result<Option<bool>, String> {
    // `len(x) op N`
    if let Some(rest) = e.strip_prefix("len(") {
        if let Some(end) = rest.find(')') {
            let arg = &rest[..end];
            let after = rest[end + 1..].trim();
            if after.is_empty() {
                let val = resolve_var(arg, ctx, outputs);
                return Ok(Some(is_truthy(&val)));
            }
            for op in ["==", "!=", ">=", "<=", ">", "<"] {
                if let Some(after_rest) = after.strip_prefix(op) {
                    let rhs = parse_literal(after_rest.trim());
                    let len = length_of(resolve_var(arg, ctx, outputs));
                    let l = Value::Number(len.into());
                    return Ok(Some(compare_values(&l, &rhs, op)?));
                }
            }
        }
    }

    // `x.contains(...)`, `x.startswith(...)`, `x.endswith(...)`
    for (fn_name, kind) in [
        ("contains", StrPredicate::Contains),
        ("startswith", StrPredicate::StartsWith),
        ("endswith", StrPredicate::EndsWith),
    ] {
        let marker = format!(".{fn_name}(");
        if let Some(idx) = e.find(&marker) {
            let var_part = &e[..idx];
            let after = &e[idx + marker.len()..];
            let Some(end) = after.find(')') else {
                return Ok(None);
            };
            let arg_literal = &after[..end];
            let Some(arg) = unquote(arg_literal.trim()) else {
                return Err(format!("Invalid string literal in {fn_name}: {arg_literal}"));
            };
            let val = resolve_var(var_part.trim(), ctx, outputs);
            let text = match val {
                Value::String(s) => s,
                Value::Null => String::new(),
                other => other.to_string(),
            };
            let result = match kind {
                StrPredicate::Contains => text.contains(&arg),
                StrPredicate::StartsWith => text.starts_with(&arg),
                StrPredicate::EndsWith => text.ends_with(&arg),
            };
            return Ok(Some(result));
        }
    }

    Ok(None)
}

enum StrPredicate {
    Contains,
    StartsWith,
    EndsWith,
}

/// The bracket pair surrounding a list literal.
fn bracket_pair(right: &str) -> Result<(char, char), String> {
    let right = right.trim();
    if right.starts_with('[') && right.ends_with(']') {
        Ok(('[', ']'))
    } else if right.starts_with('(') && right.ends_with(')') {
        Ok(('(', ')'))
    } else {
        Err(format!("Invalid list literal: {right}"))
    }
}

/// The length of a JSON value (string chars, array len, object keys, number
/// digits).
fn length_of(v: Value) -> usize {
    match v {
        Value::String(s) => s.chars().count(),
        Value::Array(a) => a.len(),
        Value::Object(o) => o.len(),
        Value::Number(n) => n.to_string().len(),
        Value::Bool(_) => 1,
        Value::Null => 0,
    }
}

/// Strip matching quotes from a string literal.
fn unquote(s: &str) -> Option<String> {
    let s = s.trim();
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\'')))
    {
        Some(s[1..s.len() - 1].to_string())
    } else {
        None
    }
}

/// Splits `s` on `op` at the top level (outside quotes/brackets). Returns
/// `None` if the operator never appears at top level.
fn split_top_level<'a>(s: &'a str, op: &str) -> Option<Vec<&'a str>> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut quote: Option<char> = None;
    let mut depth = 0i32;
    let mut found = false;
    let mut i = 0usize;

    while i < s.len() {
        let c = s[i..].chars().next().unwrap();
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
        } else {
            match c {
                '"' | '\'' => quote = Some(c),
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                _ => {}
            }
            if depth == 0 && s[i..].starts_with(op) {
                parts.push(&s[start..i]);
                i += op.len();
                start = i;
                found = true;
                continue;
            }
        }
        i += c.len_utf8();
    }
    parts.push(&s[start..]);
    if found {
        Some(parts)
    } else {
        None
    }
}

/// Finds the first top-level occurrence of `op` in `s`.
fn find_operator(s: &str, op: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    let mut depth = 0i32;
    let mut i = 0usize;
    while i < s.len() {
        let c = s[i..].chars().next().unwrap();
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
        } else {
            match c {
                '"' | '\'' => quote = Some(c),
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                _ => {}
            }
            if depth == 0 && s[i..].starts_with(op) {
                return Some(i);
            }
        }
        i += c.len_utf8();
    }
    None
}

/// Splits a comma-separated list, trimming quotes from each item.
fn split_list(s: &str, open: char, close: char) -> Vec<String> {
    let mut items = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut depth = 0i32;

    for c in s.chars() {
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            current.push(c);
            continue;
        }
        match c {
            '"' | '\'' => {
                quote = Some(c);
                current.push(c);
            }
            c if c == open => {
                depth += 1;
                current.push(c);
            }
            c if c == close => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                items.push(current.trim().trim_matches('"').trim_matches('\'').to_string());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    if !current.trim().is_empty() {
        items.push(current.trim().trim_matches('"').trim_matches('\'').to_string());
    }
    items
}

/// Resolves a variable path against the context and prior step outputs.
fn resolve_var(path: &str, ctx: &HashMap<String, Value>, outputs: &HashMap<String, Value>) -> Value {
    let path = path.trim();
    if path.is_empty() {
        return Value::Null;
    }
    if let Some(rest) = path.strip_prefix("inputs.") {
        return ctx.get(rest).cloned().unwrap_or(Value::Null);
    }
    if let Some(rest) = path.strip_prefix("outputs.") {
        return outputs.get(rest).cloned().unwrap_or(Value::Null);
    }
    if let Some(v) = outputs.get(path) {
        return v.clone();
    }
    if let Some(v) = ctx.get(path) {
        return v.clone();
    }
    // Nested path traversal, e.g. `user.name` or `config.timeout`.
    if path.contains('.') {
        let mut parts = path.split('.');
        if let Some(first) = parts.next() {
            if let Some(mut current) = ctx.get(first).cloned().or_else(|| outputs.get(first).cloned()) {
                for part in parts {
                    current = match &current {
                        Value::Object(map) => map.get(part).cloned().unwrap_or(Value::Null),
                        Value::Array(items) => {
                            if let Ok(idx) = part.parse::<usize>() {
                                items.get(idx).cloned().unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            }
                        }
                        _ => Value::Null,
                    };
                }
                return current;
            }
        }
    }
    Value::Null
}

/// Parses a right-hand literal (string, number, bool, or null).
fn parse_literal(s: &str) -> Value {
    let s = s.trim();
    if s.is_empty() {
        return Value::Null;
    }
    if s == "true" {
        return json!(true);
    }
    if s == "false" {
        return json!(false);
    }
    if s == "null" || s == "none" {
        return Value::Null;
    }
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return json!(s[1..s.len() - 1]);
    }
    if let Ok(n) = s.parse::<i64>() {
        return json!(n);
    }
    if let Ok(f) = s.parse::<f64>() {
        return json!(f);
    }
    json!(s)
}

/// Compares two values with the given operator.
fn compare_values(l: &Value, r: &Value, op: &str) -> Result<bool, String> {
    match op {
        "==" => Ok(values_equal(l, r)),
        "!=" => Ok(!values_equal(l, r)),
        ">" | ">=" | "<" | "<=" => {
            let lf = l.as_f64().ok_or_else(|| format!("Left operand not numeric: {}", l))?;
            let rf = r.as_f64().ok_or_else(|| format!("Right operand not numeric: {}", r))?;
            Ok(match op {
                ">" => lf > rf,
                ">=" => lf >= rf,
                "<" => lf < rf,
                _ => lf <= rf,
            })
        }
        _ => Err(format!("Unsupported operator: {}", op)),
    }
}

fn values_equal(l: &Value, r: &Value) -> bool {
    match (l, r) {
        (Value::String(a), Value::String(b)) => a == b,
        (Value::Number(a), Value::Number(b)) => a == b,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        // Allow string <-> number cross-comparison for convenience.
        (Value::String(a), Value::Number(b)) => a.parse::<f64>().ok() == b.as_f64(),
        (Value::Number(a), Value::String(b)) => a.as_f64() == b.parse::<f64>().ok(),
        _ => false,
    }
}

/// Whether a left-hand value matches a list item.
fn value_matches(val: &Value, item: &str) -> bool {
    let item = item.trim();
    match val {
        Value::String(s) => s == item,
        Value::Number(n) => {
            n.to_string() == item || item.parse::<f64>().ok() == n.as_f64()
        }
        Value::Bool(b) => (item == "true") == *b || (item == "false") == !*b,
        Value::Null => item.eq_ignore_ascii_case("null") || item.eq_ignore_ascii_case("none"),
        _ => false,
    }
}

/// Truthiness for JSON values (null/false/empty are falsy).
pub fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> HashMap<String, Value> {
        let mut m = HashMap::new();
        m.insert("language".to_string(), json!("zh"));
        m.insert("score".to_string(), json!(42));
        m.insert("confirmed".to_string(), json!(true));
        m.insert("username".to_string(), json!("alice"));
        m.insert("config".to_string(), json!({ "timeout": 30, "nested": { "flag": true } }));
        m.insert("items".to_string(), json!(["a", "b", "c"]));
        m
    }

    #[test]
    fn when_equality_and_truthiness() {
        let c = ctx();
        let o = HashMap::new();
        assert!(evaluate_when("language == \"zh\"", &c, &o).unwrap());
        assert!(!evaluate_when("language != \"zh\"", &c, &o).unwrap());
        assert!(evaluate_when("confirmed", &c, &o).unwrap());
        assert!(!evaluate_when("!confirmed", &c, &o).unwrap());
        assert!(!evaluate_when("missing", &c, &o).unwrap());
    }

    #[test]
    fn when_comparisons_and_membership() {
        let c = ctx();
        let o = HashMap::new();
        assert!(evaluate_when("score > 40", &c, &o).unwrap());
        assert!(evaluate_when("score >= 42", &c, &o).unwrap());
        assert!(evaluate_when("language in ['en', 'zh']", &c, &o).unwrap());
        assert!(!evaluate_when("language in ['en']", &c, &o).unwrap());
        assert!(evaluate_when("language not in ['en', 'fr']", &c, &o).unwrap());
        assert!(!evaluate_when("language not in ['en', 'zh']", &c, &o).unwrap());
    }

    #[test]
    fn when_string_predicates() {
        let c = ctx();
        let o = HashMap::new();
        assert!(evaluate_when("username.contains(\"ali\")", &c, &o).unwrap());
        assert!(!evaluate_when("username.contains(\"bob\")", &c, &o).unwrap());
        assert!(evaluate_when("username.startswith(\"al\")", &c, &o).unwrap());
        assert!(evaluate_when("username.endswith(\"ce\")", &c, &o).unwrap());
        assert!(evaluate_when("len(username) == 5", &c, &o).unwrap());
        assert!(evaluate_when("len(items) >= 3", &c, &o).unwrap());
    }

    #[test]
    fn when_regex_and_nested_paths() {
        let c = ctx();
        let o = HashMap::new();
        assert!(evaluate_when("username ~ \"^al\"", &c, &o).unwrap());
        assert!(!evaluate_when("username ~ \"^bo\"", &c, &o).unwrap());
        assert!(evaluate_when("config.timeout > 10", &c, &o).unwrap());
        assert!(evaluate_when("config.nested.flag == true", &c, &o).unwrap());
        assert!(evaluate_when("items[0] == \"a\"", &c, &o).unwrap());
    }

    #[test]
    fn when_logic_and_outputs() {
        let c = ctx();
        let mut o = HashMap::new();
        o.insert("summary".to_string(), json!("done"));
        assert!(evaluate_when("confirmed && score > 10", &c, &o).unwrap());
        assert!(evaluate_when("missing || outputs.summary == \"done\"", &c, &o).unwrap());
    }

    #[test]
    fn dag_topological_sort() {
        let steps = vec![
            SkillStep::new("a", StepType::Agent),
            {
                let mut s = SkillStep::new("b", StepType::LlmChat);
                s.depends_on = Some(vec!["a".to_string()]);
                s
            },
            {
                let mut s = SkillStep::new("c", StepType::ToolCall);
                s.depends_on = Some(vec!["b".to_string()]);
                s
            },
        ];
        let dag = Dag::new(&steps).unwrap();
        assert_eq!(dag.order, vec!["a", "b", "c"]);
        assert_eq!(dag.roots(), vec!["a"]);
        assert_eq!(dag.dependents_of("a"), vec!["b"]);
        assert_eq!(dag.depth_of("c"), 3);
    }

    #[test]
    fn dag_cycle_detection() {
        let steps = vec![
            {
                let mut s = SkillStep::new("a", StepType::Agent);
                s.depends_on = Some(vec!["c".to_string()]);
                s
            },
            {
                let mut s = SkillStep::new("b", StepType::Agent);
                s.depends_on = Some(vec!["a".to_string()]);
                s
            },
            {
                let mut s = SkillStep::new("c", StepType::Agent);
                s.depends_on = Some(vec!["b".to_string()]);
                s
            },
        ];
        assert!(Dag::new(&steps).is_err());
    }

    #[test]
    fn dag_unknown_dependency() {
        let steps = vec![{
            let mut s = SkillStep::new("a", StepType::Agent);
            s.depends_on = Some(vec!["ghost".to_string()]);
            s
        }];
        assert!(Dag::new(&steps).is_err());
    }

    #[test]
    fn output_pick_extraction() {
        let out = json!({ "result": 42, "reason": "ok", "junk": 1 });
        let step_out = StepOutput {
            var: None,
            route_to: None,
            route_on_error: None,
            export: false,
            pick: Some(vec!["result".to_string(), "reason".to_string()]),
        };
        let picked = apply_output_pick(&out, Some(&step_out));
        assert_eq!(picked, json!({ "result": 42, "reason": "ok" }));
    }

    #[test]
    fn render_template_custom_functions() {
        let vars: HashMap<String, Value> = HashMap::new();
        let uuid = render_template("{{ uuid() }}", &vars);
        assert_eq!(uuid.len(), 36);
        let ts = render_template("{{ ts() }}", &vars);
        assert!(ts.parse::<i64>().is_ok());
    }

    #[test]
    fn render_args_recursive() {
        let mut args = HashMap::new();
        args.insert(
            "message".to_string(),
            json!({"text": "hello {{ inputs.name }}", "tags": ["{{ inputs.tag }}"]}),
        );
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), json!("World"));
        vars.insert("tag".to_string(), json!("x"));
        let rendered = render_args(&args, &vars);
        assert_eq!(rendered["message"]["text"], "hello World");
        assert_eq!(rendered["message"]["tags"][0], "x");
    }

    #[test]
    fn coerce_classifier_choices() {
        assert_eq!(coerce_to_choice("  Yes ", &["yes".to_string(), "no".to_string()]), Some("yes".to_string()));
        assert_eq!(coerce_to_choice("YES!", &["yes".to_string()]), Some("yes".to_string()));
        assert_eq!(coerce_to_choice("maybe", &["yes".to_string(), "no".to_string()]), None);
    }
}
