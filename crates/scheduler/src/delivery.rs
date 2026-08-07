use crate::types::JobExecution;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::{info, warn};

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
    pub fn success() -> Self {
        Self {
            success: true,
            error: None,
        }
    }
    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            error: Some(error.into()),
        }
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
        (
            Self {
                name: name.into(),
                sender: tx,
            },
            rx,
        )
    }
}

#[async_trait]
impl DeliveryChannel for ChannelDelivery {
    fn name(&self) -> &str {
        &self.name
    }
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
        Self {
            name: name.into(),
            channels: Vec::new(),
        }
    }
    pub fn add(&mut self, channel: Box<dyn DeliveryChannel>) {
        self.channels.push(channel);
    }

    /// Get the chain name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of channels in the chain.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Check whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }

    pub async fn deliver_all(&self, execution: &JobExecution) -> Vec<DeliveryResult> {
        let mut results = Vec::with_capacity(self.channels.len());
        for channel in &self.channels {
            let result = channel.deliver(execution).await;
            if !result.success {
                warn!(
                    "Delivery channel '{}' failed for execution {}: {:?}",
                    channel.name(),
                    execution.id,
                    result.error
                );
            }
            results.push(result);
        }
        results
    }

    pub async fn deliver_first(&self, execution: &JobExecution) -> DeliveryResult {
        for channel in &self.channels {
            let result = channel.deliver(execution).await;
            if result.success {
                return result;
            }
        }
        DeliveryResult::failure("All delivery channels failed".to_string())
    }
}

/// A delivery channel that logs execution results.
pub struct LogDelivery {
    name: String,
}
impl LogDelivery {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl DeliveryChannel for LogDelivery {
    fn name(&self) -> &str {
        &self.name
    }
    async fn deliver(&self, execution: &JobExecution) -> DeliveryResult {
        let status = if execution.success {
            "SUCCESS"
        } else {
            "FAILED"
        };
        info!(
            "[{}] Job {} execution {}: attempt={}, duration={}ms, error={:?}",
            status,
            execution.job_id,
            execution.id,
            execution.attempt,
            execution.duration_ms.unwrap_or(0),
            execution.error
        );
        DeliveryResult::success()
    }
}

/// Validate that a webhook URL uses http(s) and has a hostname.
///
/// Ports `validate_webhook_url` from `src/opensquilla/scheduler/delivery.py`.
pub fn validate_webhook_url(url: &str) -> Result<(), String> {
    if url.is_empty() {
        return Err("webhook URL is required".to_string());
    }

    let scheme_end = match url.find("://") {
        Some(idx) => idx,
        None => return Err(format!("invalid webhook URL: {:?}", url)),
    };
    let scheme = url[..scheme_end].to_lowercase();

    if scheme != "http" && scheme != "https" {
        return Err(format!(
            "webhook URL must use http or https scheme, got {:?}",
            scheme
        ));
    }

    let rest = &url[scheme_end + 3..];
    let hostname_end = rest
        .find(['/', '?', '#', ':'])
        .unwrap_or(rest.len());
    let hostname = &rest[..hostname_end];

    if hostname.is_empty() {
        return Err(format!("webhook URL is missing a hostname: {:?}", url));
    }

    Ok(())
}

/// Webhook delivery channel — POSTs `JobExecution` as JSON to a URL.
///
/// Ports `_post_to_webhook` from `src/opensquilla/scheduler/delivery.py`.
/// The URL is validated at construction via [`validate_webhook_url`].
pub struct WebhookDelivery {
    url: String,
    token: Option<String>,
}

impl WebhookDelivery {
    /// Create a new webhook delivery channel with an optional bearer token.
    pub fn new(url: impl Into<String>, token: Option<String>) -> Result<Self, String> {
        let url = url.into();
        validate_webhook_url(&url)?;
        Ok(Self { url, token })
    }

    /// The configured webhook URL.
    pub fn url(&self) -> &str {
        &self.url
    }
}

#[async_trait]
impl DeliveryChannel for WebhookDelivery {
    fn name(&self) -> &str {
        "webhook"
    }

    async fn deliver(&self, execution: &JobExecution) -> DeliveryResult {
        let payload = match serde_json::to_value(execution) {
            Ok(v) => v,
            Err(e) => {
                return DeliveryResult::failure(format!("failed to serialize execution: {}", e));
            }
        };
        // TODO(parity): POST `payload` to `self.url` with
        // Content-Type: application/json and optional Authorization: Bearer
        // <self.token>. Python uses httpx with a 10s timeout
        // (src/opensquilla/scheduler/delivery.py:_post_to_webhook, line 323).
        // reqwest is not in the scheduler Cargo.toml — add it when wiring real HTTP.
        let _ = (payload, &self.token);
        DeliveryResult::failure("webhook delivery not compiled — reqwest not in Cargo.toml")
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

    #[test]
    fn test_validate_webhook_url_valid() {
        assert!(validate_webhook_url("https://example.com/webhook").is_ok());
        assert!(validate_webhook_url("http://localhost:8080/hook").is_ok());
        assert!(validate_webhook_url("https://example.com:443/path?query=1").is_ok());
    }

    #[test]
    fn test_validate_webhook_url_invalid_scheme() {
        assert!(validate_webhook_url("ftp://example.com/webhook").is_err());
        assert!(validate_webhook_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_webhook_url_missing_hostname() {
        assert!(validate_webhook_url("https:///webhook").is_err());
        assert!(validate_webhook_url("https://").is_err());
    }

    #[test]
    fn test_validate_webhook_url_empty() {
        assert!(validate_webhook_url("").is_err());
    }

    #[tokio::test]
    async fn test_webhook_delivery_url_validated_at_construction() {
        let result = WebhookDelivery::new("not-a-url", None);
        assert!(result.is_err());
        let ok = WebhookDelivery::new("https://example.com/hook", None).unwrap();
        assert_eq!(ok.url(), "https://example.com/hook");
    }

    #[tokio::test]
    async fn test_webhook_delivery_deliver_returns_failure_without_reqwest() {
        let delivery = WebhookDelivery::new("https://example.com/hook", None).unwrap();
        let exec = JobExecution::new(uuid::Uuid::new_v4(), 0);
        let result = delivery.deliver(&exec).await;
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not compiled"));
    }
}
