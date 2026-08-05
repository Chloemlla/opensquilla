use async_trait::async_trait;
use crate::types::CronJob;

/// Result of a handler execution.
#[derive(Debug, Clone)]
pub struct HandlerResult {
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
}

impl HandlerResult {
    pub fn success(result: impl Into<String>) -> Self {
        Self { success: true, result: Some(result.into()), error: None }
    }
    pub fn failure(error: impl Into<String>) -> Self {
        Self { success: false, result: None, error: Some(error.into()) }
    }
}

/// Trait for handling cron job executions.
#[async_trait]
pub trait CronJobHandler: Send + Sync {
    fn name(&self) -> &str;
    async fn execute(&self, job: &CronJob) -> HandlerResult;
}

/// Handler that sends a heartbeat signal.
pub struct HeartbeatHandler { name: String }
impl HeartbeatHandler {
    pub fn new() -> Self { Self { name: "heartbeat".to_string() } }
}

#[async_trait]
impl CronJobHandler for HeartbeatHandler {
    fn name(&self) -> &str { &self.name }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        tracing::debug!("Heartbeat handler for job '{}' (id: {})", job.name, job.id);
        HandlerResult::success(format!("Heartbeat at {}", chrono::Utc::now().to_rfc3339()))
    }
}

/// Handler that automatically proposes actions.
pub struct AutoProposeHandler { name: String }
impl AutoProposeHandler {
    pub fn new() -> Self { Self { name: "auto_propose".to_string() } }
}

#[async_trait]
impl CronJobHandler for AutoProposeHandler {
    fn name(&self) -> &str { &self.name }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        tracing::info!("Auto-propose handler for job '{}'", job.name);
        HandlerResult::success("Auto-propose generated".to_string())
    }
}

/// Handler that triggers dream consolidation.
pub struct DreamHandler { name: String }
impl DreamHandler {
    pub fn new() -> Self { Self { name: "dream".to_string() } }
}

#[async_trait]
impl CronJobHandler for DreamHandler {
    fn name(&self) -> &str { &self.name }
    async fn execute(&self, job: &CronJob) -> HandlerResult {
        tracing::info!("Dream consolidation triggered for agent_id={:?} by job '{}'", job.agent_id, job.name);
        HandlerResult::success("Dream consolidation triggered".to_string())
    }
}

/// A registry of named cron job handlers.
pub struct HandlerRegistry {
    handlers: Vec<Box<dyn CronJobHandler>>,
}

impl HandlerRegistry {
    pub fn new() -> Self { Self { handlers: Vec::new() } }

    pub fn with_defaults() -> Self {
        let mut r = Self::new();
        r.register(Box::new(HeartbeatHandler::new()));
        r.register(Box::new(AutoProposeHandler::new()));
        r.register(Box::new(DreamHandler::new()));
        r
    }

    pub fn register(&mut self, handler: Box<dyn CronJobHandler>) {
        tracing::info!("Registered cron job handler: {}", handler.name());
        self.handlers.push(handler);
    }

    pub fn get(&self, name: &str) -> Option<&dyn CronJobHandler> {
        self.handlers.iter().find(|h| h.name() == name).map(|h| h.as_ref())
    }

    pub fn has_handler(&self, name: &str) -> bool {
        self.handlers.iter().any(|h| h.name() == name)
    }

    pub fn handler_names(&self) -> Vec<String> {
        self.handlers.iter().map(|h| h.name().to_string()).collect()
    }

    pub fn len(&self) -> usize { self.handlers.len() }
    pub fn is_empty(&self) -> bool { self.handlers.is_empty() }
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
}
