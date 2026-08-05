use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info};

use opensquilla_core::config::Config;

/// A telemetry event recording usage data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Unique event ID.
    pub id: String,
    /// Event type (e.g., "session_start", "message_sent", "error").
    pub event_type: String,
    /// Timestamp of the event.
    pub timestamp: DateTime<Utc>,
    /// Session ID if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Provider used, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model used, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Duration of the operation in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Token usage information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    /// Custom attributes.
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

/// Token usage information.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Aggregated telemetry metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelemetryMetrics {
    /// Total number of events recorded.
    pub total_events: u64,
    /// Total number of messages processed.
    pub total_messages: u64,
    /// Total tokens used.
    pub total_tokens: u64,
    /// Average response time in milliseconds.
    pub avg_response_time_ms: f64,
    /// Number of errors recorded.
    pub total_errors: u64,
    /// Events by type.
    pub events_by_type: HashMap<String, u64>,
    /// Token usage by provider.
    pub tokens_by_provider: HashMap<String, u64>,
}

/// Usage telemetry collector.
#[derive(Debug, Clone)]
pub struct Telemetry {
    events: Arc<RwLock<Vec<TelemetryEvent>>>,
    enabled: bool,
    metrics: Arc<RwLock<TelemetryMetrics>>,
}

impl Telemetry {
    /// Create a new telemetry collector.
    pub fn new(config: &Config) -> Self {
        let enabled = config
            .get("telemetry.enabled")
            .unwrap_or_else(|| "true".to_string())
            == "true";

        if enabled {
            info!("Telemetry collector initialized");
        } else {
            debug!("Telemetry is disabled");
        }

        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            enabled,
            metrics: Arc::new(RwLock::new(TelemetryMetrics::default())),
        }
    }

    /// Record a telemetry event.
    pub async fn record_event(&self, event: TelemetryEvent) {
        if !self.enabled {
            return;
        }

        let mut metrics = self.metrics.write().await;
        metrics.total_events += 1;

        *metrics
            .events_by_type
            .entry(event.event_type.clone())
            .or_insert(0) += 1;

        if let Some(usage) = event.token_usage {
            metrics.total_tokens += usage.total_tokens;
            if let Some(ref provider) = event.provider {
                *metrics
                    .tokens_by_provider
                    .entry(provider.clone())
                    .or_insert(0) += usage.total_tokens;
            }
        }

        if event.event_type == "error" {
            metrics.total_errors += 1;
        }

        if event.event_type == "message" {
            metrics.total_messages += 1;
        }

        if let Some(duration) = event.duration_ms {
            let total = metrics.avg_response_time_ms * (metrics.total_messages as f64);
            metrics.avg_response_time_ms =
                (total + duration as f64) / (metrics.total_messages as f64);
        }

        let mut events = self.events.write().await;
        events.push(event);

        // Trim events if too many
        if events.len() > 10_000 {
            let len = events.len();
            events.drain(0..len - 10_000);
        }
    }

    /// Record a session start event.
    pub async fn record_session_start(&self, session_id: &str, provider: &str, model: &str) {
        self.record_event(TelemetryEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: "session_start".to_string(),
            timestamp: Utc::now(),
            session_id: Some(session_id.to_string()),
            provider: Some(provider.to_string()),
            model: Some(model.to_string()),
            duration_ms: None,
            token_usage: None,
            attributes: HashMap::new(),
        })
        .await;
    }

    /// Record a message event.
    pub async fn record_message(
        &self,
        session_id: &str,
        provider: &str,
        model: &str,
        duration_ms: u64,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) {
        self.record_event(TelemetryEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: "message".to_string(),
            timestamp: Utc::now(),
            session_id: Some(session_id.to_string()),
            provider: Some(provider.to_string()),
            model: Some(model.to_string()),
            duration_ms: Some(duration_ms),
            token_usage: Some(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }),
            attributes: HashMap::new(),
        })
        .await;
    }

    /// Record an error event.
    pub async fn record_error(&self, error: &str, session_id: Option<&str>) {
        let mut attributes = HashMap::new();
        attributes.insert("error".to_string(), error.to_string());

        self.record_event(TelemetryEvent {
            id: uuid::Uuid::new_v4().to_string(),
            event_type: "error".to_string(),
            timestamp: Utc::now(),
            session_id: session_id.map(|s| s.to_string()),
            provider: None,
            model: None,
            duration_ms: None,
            token_usage: None,
            attributes,
        })
        .await;
    }

    /// Get the current aggregated metrics.
    pub async fn get_metrics(&self) -> TelemetryMetrics {
        self.metrics.read().await.clone()
    }

    /// Get all recorded events.
    pub async fn get_events(&self) -> Vec<TelemetryEvent> {
        self.events.read().await.clone()
    }

    /// Get events filtered by type.
    pub async fn get_events_by_type(&self, event_type: &str) -> Vec<TelemetryEvent> {
        let events = self.events.read().await;
        events
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Clear all events but keep metrics.
    pub async fn clear_events(&self) {
        let mut events = self.events.write().await;
        events.clear();
        debug!("Telemetry events cleared");
    }

    /// Reset all metrics and events.
    pub async fn reset(&self) {
        let mut events = self.events.write().await;
        events.clear();
        let mut metrics = self.metrics.write().await;
        *metrics = TelemetryMetrics::default();
        debug!("Telemetry fully reset");
    }

    /// Check if telemetry is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Export events as JSON.
    pub async fn export_json(&self) -> String {
        let events = self.events.read().await;
        serde_json::to_string_pretty(&*events).unwrap_or_default()
    }
}
