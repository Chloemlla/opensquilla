use crate::types::CronJob;
use async_trait::async_trait;
use opensquilla_memory::{DreamEngine, DreamSummary};
use std::sync::Arc;

/// Result of a handler execution.
#[derive(Debug, Clone)]
pub struct HandlerResult {
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
}

impl HandlerResult {
    pub fn success(result: impl Into<String>) -> Self {
        Self {
            success: true,
            result: Some(result.into()),
            error: None,
        }
    }
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            result: None,
            error: Some(error.into()),
        }
    }
}

/// Trait for handling cron job executions.
#[async_trait]
pub trait CronJobHandler: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, job: &CronJob) -> HandlerResult;
}

/// Handler that sends a heartbeat signal.
pub struct HeartbeatHandler {
    name: String,
}
impl HeartbeatHandler {
    pub fn new() -> Self {
        Self {
            name: "heartbeat".to_string(),
        }
    }
}

#[async_trait]
impl CronJobHandler for HeartbeatHandler {
    fn name(&self) -> &str {
        &self.name
    }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        tracing::debug!("Heartbeat handler for job '{}' (id: {})", job.name, job.id);
        HandlerResult::success(format!("Heartbeat at {}", chrono::Utc::now().to_rfc3339()))
    }
}

/// Handler that automatically proposes actions.
pub struct AutoProposeHandler {
    name: String,
}
impl AutoProposeHandler {
    pub fn new() -> Self {
        Self {
            name: "auto_propose".to_string(),
        }
    }
}

#[async_trait]
impl CronJobHandler for AutoProposeHandler {
    fn name(&self) -> &str {
        &self.name
    }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        tracing::info!("Auto-propose handler for job '{}'", job.name);
        HandlerResult::success("Auto-propose generated".to_string())
    }
}

/// Handler that triggers dream consolidation.
///
/// When a [`DreamEngine`] is injected (via [`DreamHandler::new_with_engine`]
/// or [`HandlerRegistry::with_dream_engine`]), `execute` asks the engine to
/// run a dream cycle for the job's agent only if one is due. Without an
/// injected engine the handler degrades to a no-op success so that default
/// registries (which cannot construct a `MemoryStore`-backed engine) keep
/// working.
pub struct DreamHandler {
    name: String,
    engine: Option<Arc<DreamEngine>>,
}
impl DreamHandler {
    pub fn new() -> Self {
        Self {
            name: "dream".to_string(),
            engine: None,
        }
    }

    /// Create a handler wired to a concrete [`DreamEngine`].
    pub fn new_with_engine(engine: Arc<DreamEngine>) -> Self {
        Self {
            name: "dream".to_string(),
            engine: Some(engine),
        }
    }

    /// The injected engine, if any.
    pub fn engine(&self) -> Option<&Arc<DreamEngine>> {
        self.engine.as_ref()
    }
}

#[async_trait]
impl CronJobHandler for DreamHandler {
    fn name(&self) -> &str {
        &self.name
    }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        let Some(agent_id) = job.agent_id else {
            return HandlerResult::success(format!(
                "Dream skipped: job '{}' (id: {}) is not bound to an agent",
                job.name, job.id
            ));
        };
        let Some(engine) = &self.engine else {
            return HandlerResult::success(format!(
                "Dream skipped: no DreamEngine injected for job '{}' (id: {})",
                job.name, job.id
            ));
        };
        match engine.run_dream_if_due(&agent_id).await {
            Ok(Some(summary)) => {
                let summary = describe_summary(&summary);
                tracing::info!("Dream cycle for agent {} completed: {}", agent_id, summary);
                HandlerResult::success(format!(
                    "Dream cycle complete for agent {}: {}",
                    agent_id, summary
                ))
            }
            Ok(None) => {
                tracing::debug!("Dream not due for agent {} (job '{}')", agent_id, job.name);
                HandlerResult::success(format!(
                    "Dream not due for agent {} (job '{}')",
                    agent_id, job.name
                ))
            }
            Err(err) => {
                tracing::error!("Dream cycle failed for agent {}: {}", agent_id, err);
                HandlerResult::failure(format!(
                    "Dream cycle failed for agent {}: {}",
                    agent_id, err
                ))
            }
        }
    }
}

/// Render a [`DreamSummary`] as a short human-readable line.
fn describe_summary(summary: &DreamSummary) -> String {
    format!(
        "{} memories, {} consolidated, {} patterns, {} abstractions",
        summary.total_memories,
        summary.consolidated,
        summary.patterns_found,
        summary.abstractions_created
    )
}

/// A registry of named cron job handlers.
pub struct HandlerRegistry {
    handlers: Vec<Box<dyn CronJobHandler>>,
}

impl HandlerRegistry {
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Box::new(HeartbeatHandler::new()));
        r.register(Box::new(AutoProposeHandler::new()));
        r.register(Box::new(DreamHandler::new()));
        r
    }

    /// Builder: replace the default no-op `dream` handler with one wired to a
    /// concrete [`DreamEngine`].
    ///
    /// Usage: `HandlerRegistry::with_defaults().with_dream_engine(engine)`.
    /// Keeps all existing `with_defaults()` callers source-compatible.
    pub fn with_dream_engine(mut self, engine: Arc<DreamEngine>) -> Self {
        self.handlers.retain(|h| h.name() != "dream");
        self.register(Box::new(DreamHandler::new_with_engine(engine)));
        self
    }

    pub fn register(&mut self, handler: Box<dyn CronJobHandler>) {
        tracing::info!("Registered cron job handler: {}", handler.name());
        self.handlers.push(handler);
    }

    pub fn get(&self, name: &str) -> Option<&dyn CronJobHandler> {
        self.handlers
            .iter()
            .find(|h| h.name() == name)
            .map(|h| h.as_ref())
    }

    pub fn has_handler(&self, name: &str) -> bool {
        self.handlers.iter().any(|h| h.name() == name)
    }

    pub fn handler_names(&self) -> Vec<String> {
        self.handlers.iter().map(|h| h.name().to_string()).collect()
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_heartbeat_handler() {
        let handler = HeartbeatHandler::new();
        assert_eq!(handler.name(), "heartbeat");
        let job = CronJob::new("test", crate::types::ScheduleKind::Every(60), "heartbeat");
        let result = handler.execute(&job).await;
        assert!(result.success);
    }

    #[test]
    fn test_handler_registry() {
        let mut registry = HandlerRegistry::new();
        assert!(registry.is_empty());
        registry.register(Box::new(HeartbeatHandler::new()));
        assert_eq!(registry.len(), 1);
        assert!(registry.has_handler("heartbeat"));
    }

    #[test]
    fn test_default_registry() {
        let registry = HandlerRegistry::with_defaults();
        assert_eq!(registry.len(), 3);
        assert!(registry.has_handler("heartbeat"));
        assert!(registry.has_handler("auto_propose"));
        assert!(registry.has_handler("dream"));
    }

    #[tokio::test]
    async fn test_dream_handler_without_engine_is_noop_success() {
        let handler = DreamHandler::new();
        let mut job = CronJob::new("dream", crate::types::ScheduleKind::Every(3600), "dream");
        job.agent_id = Some(uuid::Uuid::new_v4());
        let result = handler.execute(&job).await;
        assert!(result.success);
        assert!(result.result.unwrap().contains("no DreamEngine injected"));
    }

    #[tokio::test]
    async fn test_dream_handler_requires_agent_id() {
        let handler = DreamHandler::new();
        let job = CronJob::new("dream", crate::types::ScheduleKind::Every(3600), "dream");
        let result = handler.execute(&job).await;
        assert!(result.success);
        assert!(result.result.unwrap().contains("not bound to an agent"));
    }

    #[tokio::test]
    async fn test_with_dream_engine_replaces_dream_handler() {
        let engine = Arc::new(opensquilla_memory::DreamEngine::new(
            opensquilla_memory::MemoryStore::in_memory().unwrap(),
        ));
        let registry = HandlerRegistry::with_defaults().with_dream_engine(engine.clone());
        assert_eq!(registry.len(), 3);
        assert!(registry.has_handler("dream"));
        let dream = registry.get("dream").unwrap();
        assert_eq!(dream.name(), "dream");
    }

    #[tokio::test]
    async fn test_dream_handler_not_due_when_engine_injected() {
        let store = opensquilla_memory::MemoryStore::in_memory().unwrap();
        let engine = Arc::new(opensquilla_memory::DreamEngine::new(store));
        let handler = DreamHandler::new_with_engine(engine.clone());
        let agent = uuid::Uuid::new_v4();
        // A fresh engine has no last-consolidation record, so the first
        // invocation of the handler would be due. Run a cycle directly first
        // so the engine records "consolidated just now", then the handler
        // should report "not due".
        engine.run_dream_cycle(&agent).await.unwrap();
        let mut job = CronJob::new("dream", crate::types::ScheduleKind::Every(3600), "dream");
        job.agent_id = Some(agent);
        let result = handler.execute(&job).await;
        assert!(result.success);
        assert!(result.result.unwrap().contains("not due"));
    }
}
