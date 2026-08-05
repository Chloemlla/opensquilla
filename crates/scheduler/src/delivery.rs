use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{info, warn};
use crate::types::JobExecution;

/// Trait for delivering job execution results.
#[async_trait]
pub trait DeliveryChannel: Send + Sync {
    async fn deliver(&self, execution: &JobExecution) -> DeliveryResult;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct DeliveryResult {
    pub success: bool,
    pub error: Option<String>,
}

impl DeliveryResult {
    pub fn success() -> Self { Self { success: true, error: None } }
    pub fn failure(error: impl Into<String>) -> Self {
        Self { success: false, error: Some(error.into()) }
    }
}

/// In-process channel delivery using tokio's mpsc.
pub struct ChannelDelivery {
    name: String,
    sender: mpsc::UnboundedSender<JobExecution>,
}

impl ChannelDelivery {
    pub fn new(name: impl Into<String>) -> (Self, mpsc::UnboundedReceiver<JobExecution>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { name: name.into(), sender: tx }, rx)
    }
}

#[async_trait]
impl DeliveryChannel for ChannelDelivery {
    fn name(&self) -> &str { &self.name }
    async fn deliver(&self, execution: &JobExecution) -> DeliveryResult {
        match self.sender.send(execution.clone()) {
            Ok(_) => DeliveryResult::success(),
            Err(e) => DeliveryResult::failure(e.to_string()),
        }
    }
}

/// Delivery chain that tries multiple delivery channels.
pub struct DeliveryChain {
    name: String,
    channels: Vec<Box<dyn DeliveryChannel>>,
}

impl DeliveryChain {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), channels: Vec::new() }
    }
    pub fn add(&mut self, channel: Box<dyn DeliveryChannel>) { self.channels.push(channel); }

    /// Get the chain name.
    pub fn name(&self) -> &str { &self.name }

    /// Get the number of channels in the chain.
    pub fn len(&self) -> usize { self.channels.len() }

    /// Check whether the chain is empty.
    pub fn is_empty(&self) -> bool { self.channels.is_empty() }

    pub async fn deliver_all(&self, execution: &JobExecution) -> Vec<DeliveryResult> {
        let mut results = Vec::with_capacity(self.channels.len());
        for channel in &self.channels {
            let result = channel.deliver(execution).await;
            if !result.success {
                warn!("Delivery channel '{}' failed for execution {}: {:?}", channel.name(), execution.id, result.error);
            }
            results.push(result);
        }
        results
    }

    pub async fn deliver_first(&self, execution: &JobExecution) -> DeliveryResult {
        for channel in &self.channels {
            let result = channel.deliver(execution).await;
            if result.success { return result; }
        }
        DeliveryResult::failure("All delivery channels failed".to_string())
    }
}

/// A delivery channel that logs execution results.
pub struct LogDelivery { name: String }
impl LogDelivery {
    pub fn new(name: impl Into<String>) -> Self { Self { name: name.into() } }
}

#[async_trait]
impl DeliveryChannel for LogDelivery {
    fn name(&self) -> &str { &self.name }
    async fn deliver(&self, execution: &JobExecution) -> DeliveryResult {
        let status = if execution.success { "SUCCESS" } else { "FAILED" };
        info!("[{}] Job {} execution {}: attempt={}, duration={}ms, error={:?}",
              status, execution.job_id, execution.id, execution.attempt,
              execution.duration_ms.unwrap_or(0), execution.error);
        DeliveryResult::success()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_channel_delivery() {
        let (delivery, mut rx) = ChannelDelivery::new("test_channel");
        let exec = JobExecution::new(uuid::Uuid::new_v4(), 0);
        let result = delivery.deliver(&exec).await;
        assert!(result.success);
        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, exec.id);
    }

    #[tokio::test]
    async fn test_log_delivery() {
        let delivery = LogDelivery::new("test_log");
        let exec = JobExecution::new(uuid::Uuid::new_v4(), 0);
        assert!(delivery.deliver(&exec).await.success);
    }
}
