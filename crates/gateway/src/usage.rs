//! Usage tracking and cost RPC handlers.
//!
//! Provides `rpc_usage` for usage tracking and cost aggregation. Usage events
//! are recorded in an in-memory ledger keyed by session and model.

use chrono::{DateTime, Utc};
use opensquilla_core::error::AppError;
use opensquilla_core::types::Usage;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A single usage event record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEvent {
    pub id: Uuid,
    pub session_id: String,
    pub model: String,
    pub provider: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
    pub timestamp: DateTime<Utc>,
}

impl UsageEvent {
    /// Create a new usage event from a [`Usage`] snapshot.
    pub fn from_usage(
        session_id: &str,
        model: &str,
        provider: &str,
        usage: Usage,
        cost_usd: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id: session_id.to_string(),
            model: model.to_string(),
            provider: provider.to_string(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: usage.total_tokens,
            cost_usd,
            timestamp: Utc::now(),
        }
    }
}

/// Aggregated usage totals.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_tokens: u64,
    pub total_cost_usd: f64,
    pub event_count: u64,
    pub by_model: HashMap<String, ModelUsage>,
    pub by_session: HashMap<String, SessionUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
    pub calls: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: f64,
    pub calls: u64,
}

/// In-memory usage ledger.
#[derive(Clone, Default)]
pub struct UsageStore {
    events: Arc<Mutex<Vec<UsageEvent>>>,
}

impl UsageStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a usage event.
    pub fn record(&self, event: UsageEvent) {
        self.events.lock().push(event);
    }

    /// List events, optionally filtered by session id.
    pub fn list(&self, session_id: Option<&str>, limit: usize) -> Vec<UsageEvent> {
        let events = self.events.lock();
        let filtered: Vec<UsageEvent> = events
            .iter()
            .filter(|e| session_id.is_none_or(|sid| e.session_id == sid))
            .cloned()
            .collect();
        filtered.into_iter().rev().take(limit).collect()
    }

    /// Compute aggregate usage.
    pub fn summary(&self) -> UsageSummary {
        let events = self.events.lock();
        let mut summary = UsageSummary {
            event_count: events.len() as u64,
            ..Default::default()
        };
        for event in events.iter() {
            summary.total_input_tokens += event.input_tokens;
            summary.total_output_tokens += event.output_tokens;
            summary.total_tokens += event.total_tokens;
            summary.total_cost_usd += event.cost_usd;

            let model = summary.by_model.entry(event.model.clone()).or_default();
            model.input_tokens += event.input_tokens;
            model.output_tokens += event.output_tokens;
            model.total_tokens += event.total_tokens;
            model.cost_usd += event.cost_usd;
            model.calls += 1;

            let session = summary
                .by_session
                .entry(event.session_id.clone())
                .or_default();
            session.input_tokens += event.input_tokens;
            session.output_tokens += event.output_tokens;
            session.total_tokens += event.total_tokens;
            session.cost_usd += event.cost_usd;
            session.calls += 1;
        }
        summary
    }

    /// Clear all usage events.
    pub fn clear(&self) -> u64 {
        let mut events = self.events.lock();
        let count = events.len() as u64;
        events.clear();
        count
    }
}

/// Register usage RPC handlers on the given registry.
pub fn register_usage_handlers(registry: &mut RpcRegistry, store: UsageStore) {
    let store = Arc::new(store);

    // usage.record — record a usage event
    registry.register(rpc_handler("usage.record", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'session_id' parameter"))?;
                let model = params
                    .get("model")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AppError::bad_request("Missing 'model' parameter"))?;
                let provider = params
                    .get("provider")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let input_tokens = params
                    .get("input_tokens")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| AppError::bad_request("Missing 'input_tokens' parameter"))?;
                let output_tokens = params
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| AppError::bad_request("Missing 'output_tokens' parameter"))?;
                let cost_usd = params
                    .get("cost_usd")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let usage = Usage::new(input_tokens, output_tokens);
                let event = UsageEvent::from_usage(session_id, model, &provider, usage, cost_usd);
                store.record(event.clone());
                serde_json::to_value(event).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // usage.summary — aggregated usage totals
    registry.register(rpc_handler("usage.summary", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let summary = store.summary();
                serde_json::to_value(summary).map_err(|e| AppError::internal(e.to_string()))
            }
        }
    }));

    // usage.list — list usage events, optionally filtered by session
    registry.register(rpc_handler("usage.list", {
        let store = store.clone();
        move |params| {
            let store = store.clone();
            async move {
                let session_id = params.get("session_id").and_then(|v| v.as_str());
                let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
                let events = store.list(session_id, limit);
                Ok(serde_json::json!({
                    "events": events,
                    "count": events.len(),
                }))
            }
        }
    }));

    // usage.by_model — usage broken down by model
    registry.register(rpc_handler("usage.by_model", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let summary = store.summary();
                Ok(serde_json::json!({
                    "by_model": summary.by_model,
                }))
            }
        }
    }));

    // usage.by_session — usage broken down by session
    registry.register(rpc_handler("usage.by_session", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let summary = store.summary();
                Ok(serde_json::json!({
                    "by_session": summary.by_session,
                }))
            }
        }
    }));

    // usage.clear — clear all recorded usage
    registry.register(rpc_handler("usage.clear", {
        let store = store.clone();
        move |_params| {
            let store = store.clone();
            async move {
                let cleared = store.clear();
                Ok(serde_json::json!({"cleared": cleared}))
            }
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_usage_record_and_summary() {
        let store = UsageStore::new();
        let mut registry = RpcRegistry::new();
        register_usage_handlers(&mut registry, store);

        let params = serde_json::json!({
            "session_id": "s1",
            "model": "gpt-4o",
            "provider": "openai",
            "input_tokens": 100u64,
            "output_tokens": 50u64,
            "cost_usd": 0.002,
        });
        let r = registry.dispatch("usage.record", params).await;
        assert!(r.unwrap().is_ok());

        let r = registry
            .dispatch("usage.summary", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["total_tokens"], 150);
        assert_eq!(resp["event_count"], 1);
        assert_eq!(resp["total_cost_usd"], 0.002);
    }

    #[tokio::test]
    async fn test_usage_by_model_and_session() {
        let store = UsageStore::new();
        let mut registry = RpcRegistry::new();
        register_usage_handlers(&mut registry, store);

        for (model, session) in [("gpt-4o", "s1"), ("claude", "s1"), ("gpt-4o", "s2")] {
            let _ = registry
                .dispatch(
                    "usage.record",
                    serde_json::json!({
                        "session_id": session,
                        "model": model,
                        "input_tokens": 10u64,
                        "output_tokens": 5u64,
                    }),
                )
                .await
                .unwrap();
        }

        let r = registry
            .dispatch("usage.by_model", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["by_model"]["gpt-4o"].is_object());

        let r = registry
            .dispatch("usage.by_session", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert!(resp["by_session"]["s1"].is_object());
    }

    #[tokio::test]
    async fn test_usage_clear() {
        let store = UsageStore::new();
        let mut registry = RpcRegistry::new();
        register_usage_handlers(&mut registry, store);

        let _ = registry
            .dispatch(
                "usage.record",
                serde_json::json!({
                    "session_id": "s1",
                    "model": "m",
                    "input_tokens": 1u64,
                    "output_tokens": 1u64,
                }),
            )
            .await
            .unwrap();

        let r = registry
            .dispatch("usage.clear", serde_json::Value::Null)
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["cleared"], 1);
    }
}
