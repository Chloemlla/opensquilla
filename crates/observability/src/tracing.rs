use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use opentelemetry::trace::{SpanContext, SpanId, TraceFlags, TraceId, TraceState};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use opensquilla_core::config::Config;

/// A span in a distributed trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub attributes: HashMap<String, String>,
    pub status: SpanStatus,
}

impl TraceSpan {
    fn new(name: &str, trace_id: &str, parent_span_id: Option<String>) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            span_id: SpanId::from_u64(rand_span_id()).to_string(),
            parent_span_id,
            name: name.to_string(),
            start_time: Utc::now(),
            end_time: None,
            attributes: HashMap::new(),
            status: SpanStatus::Unset,
        }
    }

    fn finish(&mut self) {
        self.end_time = Some(Utc::now());
    }

    fn set_attribute(&mut self, key: &str, value: &str) {
        self.attributes.insert(key.to_string(), value.to_string());
    }

    fn set_status(&mut self, status: SpanStatus) {
        self.status = status;
    }
}

/// Status of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error,
}

/// A distributed tracer for tracking requests across services.
#[derive(Debug, Clone)]
pub struct Tracer {
    spans: Arc<RwLock<Vec<TraceSpan>>>,
    next_id: Arc<AtomicU64>,
    enabled: bool,
    sample_rate: f64,
}

impl Tracer {
    /// Create a new tracer.
    pub fn new(config: &Config) -> Self {
        let enabled = config
            .get("tracing.enabled")
            .unwrap_or_else(|| "true".to_string())
            == "true";
        let sample_rate = config
            .get("tracing.sample_rate")
            .unwrap_or_else(|| "1.0".to_string())
            .parse::<f64>()
            .unwrap_or(1.0);

        if enabled {
            info!("Tracer initialized with sample rate {sample_rate}");
        }

        Self {
            spans: Arc::new(RwLock::new(Vec::new())),
            next_id: Arc::new(AtomicU64::new(1)),
            enabled,
            sample_rate,
        }
    }

    /// Generate a new trace ID.
    pub fn generate_trace_id(&self) -> String {
        let high = self.next_id.fetch_add(1, Ordering::SeqCst);
        let low = self.next_id.fetch_add(1, Ordering::SeqCst);
        format!("{:016x}{:016x}", high, low)
    }

    /// Start a new span within a trace.
    pub async fn start_span(
        &self,
        name: &str,
        trace_id: &str,
        parent_span_id: Option<String>,
    ) -> Option<TraceSpan> {
        if !self.enabled {
            return None;
        }

        // Apply sampling
        if self.sample_rate < 1.0 && fastrand::f64() > self.sample_rate {
            return None;
        }

        let span = TraceSpan::new(name, trace_id, parent_span_id);
        debug!("Starting span: {} ({})", span.name, span.span_id);
        Some(span)
    }

    /// End a span and record it.
    pub async fn end_span(&self, mut span: TraceSpan) {
        span.finish();
        let mut spans = self.spans.write().await;
        spans.push(span);
    }

    /// Set an attribute on a span.
    pub fn set_attribute(span: &mut TraceSpan, key: &str, value: &str) {
        span.set_attribute(key, value);
    }

    /// Set the status of a span.
    pub fn set_status(span: &mut TraceSpan, status: SpanStatus) {
        span.set_status(status);
    }

    /// Record an error on a span.
    pub fn record_error(span: &mut TraceSpan, error: &dyn std::error::Error) {
        span.set_status(SpanStatus::Error);
        span.set_attribute("error.message", &error.to_string());
    }

    /// Get all recorded spans.
    pub async fn get_spans(&self) -> Vec<TraceSpan> {
        let spans = self.spans.read().await;
        spans.clone()
    }

    /// Clear all recorded spans.
    pub async fn clear_spans(&self) {
        let mut spans = self.spans.write().await;
        spans.clear();
    }

    /// Export spans to OpenTelemetry-compatible format.
    pub async fn export_spans(&self) -> Vec<opentelemetry::trace::SpanData> {
        let spans = self.spans.read().await;
        let mut otel_spans = Vec::new();

        for span in spans.iter() {
            let trace_id = TraceId::from_hex(&span.trace_id).unwrap_or(TraceId::INVALID);
            let span_id = SpanId::from_hex(&span.span_id).unwrap_or(SpanId::INVALID);
            let parent_span_id = span
                .parent_span_id
                .as_ref()
                .and_then(|id| SpanId::from_hex(id))
                .unwrap_or(SpanId::INVALID);

            let span_context = SpanContext::new(
                trace_id,
                span_id,
                TraceFlags::default(),
                false,
                TraceState::default(),
            );

            // Build attributes
            let mut attributes = Vec::new();
            for (key, value) in &span.attributes {
                attributes.push(opentelemetry::KeyValue::new(
                    key.clone(),
                    value.clone(),
                ));
            }

            let status = match span.status {
                SpanStatus::Ok => opentelemetry::trace::Status::Ok,
                SpanStatus::Error => opentelemetry::trace::Status::Error {
                    description: span
                        .attributes
                        .get("error.message")
                        .cloned()
                        .unwrap_or_default()
                        .into(),
                },
                SpanStatus::Unset => opentelemetry::trace::Status::Unset,
            };

            // Create a minimal SpanData representation
            let span_data = opentelemetry::trace::SpanData::new(
                span_context,
                parent_span_id,
                0, // span kind
                span.name.clone(),
                span.start_time.into(),
                span.end_time.unwrap_or_else(Utc::now).into(),
                attributes,
                Vec::new(), // events
                Vec::new(), // links
                status,
            );
            otel_spans.push(span_data);
        }

        otel_spans
    }

    /// Check if tracing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

fn rand_span_id() -> u64 {
    fastrand::u64(1..u64::MAX)
}

// A simple fast random number generator for IDs.
// We use a basic LCG since we don't want to pull in a full rand crate.
mod fastrand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static STATE: AtomicU64 = AtomicU64::new(0x853c49e6748fea9b);

    pub fn u64(range: std::ops::Range<u64>) -> u64 {
        let old = STATE.fetch_add(0x9e3779b97f4a7c15, Ordering::SeqCst);
        let x = old.wrapping_mul(0x9e3779b97f4a7c15);
        let x = x ^ (x >> 30);
        let x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        let x = x ^ (x >> 27);
        let x = x.wrapping_mul(0x94d049bb133111eb);
        let x = x ^ (x >> 31);
        range.start + (x % (range.end - range.start))
    }

    pub fn f64() -> f64 {
        (u64(0..u64::MAX) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}