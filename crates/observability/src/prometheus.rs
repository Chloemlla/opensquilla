//! Prometheus metrics export.
//!
//! Provides a small, dependency-light metric registry with counters, gauges,
//! and histograms, plus a Prometheus text exposition renderer and an axum
//! `/metrics` scrape endpoint.
//!
//! # Usage
//!
//! ```
//! use opensquilla_observability::prometheus::{MetricsRegistry, Counter};
//!
//! let registry = MetricsRegistry::new();
//! let http_requests: Counter = registry.counter(
//!     "http_requests_total",
//!     "Total number of HTTP requests handled.",
//! );
//! http_requests.inc();
//! let body = registry.render();
//! ```
//!
//! A process-global registry is available via [`global_registry`] for
//! ad-hoc recording, and [`metrics_router`] mounts the scrape endpoint on an
//! axum router.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tracing::debug;

/// Default histogram buckets (seconds), matching the Prometheus client
/// defaults.
pub const DEFAULT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// A monotonic counter. Cheap to clone and share; the value is shared.
#[derive(Debug, Clone)]
pub struct Counter {
    value: Arc<AtomicU64>,
    name: &'static str,
    help: &'static str,
}

impl Counter {
    /// Increment the counter by one.
    pub fn inc(&self) {
        self.inc_by(1);
    }

    /// Increment the counter by `v`.
    pub fn inc_by(&self, v: u64) {
        self.value.fetch_add(v, Ordering::Relaxed);
    }

    /// Read the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// A gauge that can go up and down.
#[derive(Debug, Clone)]
pub struct Gauge {
    value: Arc<AtomicU64>,
    name: &'static str,
    help: &'static str,
}

impl Gauge {
    /// Set the gauge to an exact value.
    pub fn set(&self, v: f64) {
        self.value.store(f64_to_bits(v), Ordering::Relaxed);
    }

    /// Add `v` to the current value.
    pub fn add(&self, v: f64) {
        let mut current = self.get();
        loop {
            let next = f64_to_bits(current + v);
            match self.value.compare_exchange_weak(
                f64_to_bits(current),
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = f64::from_bits(actual),
            }
        }
    }

    /// Read the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }
}

/// Convert an `f64` into its `u64` bit pattern for lock-free storage.
fn f64_to_bits(v: f64) -> u64 {
    v.to_bits()
}

/// A cumulative histogram with configurable buckets.
#[derive(Debug, Clone)]
pub struct Histogram {
    name: &'static str,
    help: &'static str,
    buckets: Arc<Vec<f64>>,
    state: Arc<Mutex<HistogramState>>,
}

#[derive(Debug, Default)]
struct HistogramState {
    /// Cumulative observation counts per bucket.
    counts: Vec<u64>,
    /// Total number of observations.
    count: u64,
    /// Sum of observed values.
    sum: f64,
}

impl Histogram {
    fn new(name: &'static str, help: &'static str, buckets: Vec<f64>) -> Self {
        let bucket_count = buckets.len();
        Self {
            name,
            help,
            buckets: Arc::new(buckets),
            state: Arc::new(Mutex::new(HistogramState {
                counts: vec![0; bucket_count],
                ..Default::default()
            })),
        }
    }

    /// Record a single observation.
    pub fn observe(&self, value: f64) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.count += 1;
        state.sum += value;
        for (i, bucket) in self.buckets.iter().enumerate() {
            if value <= *bucket {
                state.counts[i] += 1;
            }
        }
    }

    /// Record an observation in milliseconds (converts to seconds).
    pub fn observe_millis(&self, millis: f64) {
        self.observe(millis / 1000.0);
    }

    /// The configured bucket boundaries (upper bounds).
    pub fn buckets(&self) -> &[f64] {
        &self.buckets
    }

    /// Total number of observations.
    pub fn count(&self) -> u64 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).count
    }
}

/// A registry of named Prometheus metrics.
///
/// Metrics are created on first use and shared thereafter; calling
/// [`MetricsRegistry::counter`] twice with the same name returns the same
/// underlying counter.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    counters: Mutex<HashMap<String, Counter>>,
    gauges: Mutex<HashMap<String, Gauge>>,
    histograms: Mutex<HashMap<String, Histogram>>,
}

impl MetricsRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get (or create) a counter with the given name and help text.
    pub fn counter(&self, name: &'static str, help: &'static str) -> Counter {
        let mut counters = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        counters
            .entry(name.to_string())
            .or_insert_with(|| Counter {
                value: Arc::new(AtomicU64::new(0)),
                name,
                help,
            })
            .clone()
    }

    /// Get (or create) a gauge with the given name and help text.
    pub fn gauge(&self, name: &'static str, help: &'static str) -> Gauge {
        let mut gauges = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
        gauges
            .entry(name.to_string())
            .or_insert_with(|| Gauge {
                value: Arc::new(AtomicU64::new(0)),
                name,
                help,
            })
            .clone()
    }

    /// Get (or create) a histogram with the given name, help text, and bucket
    /// boundaries. When `buckets` is empty, [`DEFAULT_BUCKETS`] are used.
    pub fn histogram(
        &self,
        name: &'static str,
        help: &'static str,
        buckets: Vec<f64>,
    ) -> Histogram {
        let mut histograms = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
        histograms
            .entry(name.to_string())
            .or_insert_with(|| {
                let buckets = if buckets.is_empty() {
                    DEFAULT_BUCKETS.to_vec()
                } else {
                    buckets
                };
                Histogram::new(name, help, buckets)
            })
            .clone()
    }

    /// Render all metrics in the Prometheus text exposition format.
    pub fn render(&self) -> String {
        let mut out = String::new();

        {
            let counters = self.counters.lock().unwrap_or_else(|e| e.into_inner());
            for counter in counters.values() {
                out.push_str(&format!("# HELP {} {}\n", counter.name, counter.help));
                out.push_str(&format!("# TYPE {} counter\n", counter.name));
                out.push_str(&format!("{} {}\n", counter.name, counter.get()));
            }
        }

        {
            let gauges = self.gauges.lock().unwrap_or_else(|e| e.into_inner());
            for gauge in gauges.values() {
                out.push_str(&format!("# HELP {} {}\n", gauge.name, gauge.help));
                out.push_str(&format!("# TYPE {} gauge\n", gauge.name));
                out.push_str(&format!("{} {}\n", gauge.name, gauge.get()));
            }
        }

        {
            let histograms = self.histograms.lock().unwrap_or_else(|e| e.into_inner());
            for histogram in histograms.values() {
                out.push_str(&format!("# HELP {} {}\n", histogram.name, histogram.help));
                out.push_str(&format!("# TYPE {} histogram\n", histogram.name));
                let state = histogram.state.lock().unwrap_or_else(|e| e.into_inner());
                for (i, bucket) in histogram.buckets.iter().enumerate() {
                    out.push_str(&format!(
                        "{}_bucket{{le=\"{}\"}} {}\n",
                        histogram.name, bucket, state.counts[i]
                    ));
                }
                out.push_str(&format!(
                    "{}_bucket{{le=\"+Inf\"}} {}\n",
                    histogram.name, state.count
                ));
                out.push_str(&format!("{}_sum {}\n", histogram.name, state.sum));
                out.push_str(&format!("{}_count {}\n", histogram.name, state.count));
            }
        }

        out
    }

    /// The number of registered counters.
    pub fn counter_count(&self) -> usize {
        self.counters
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
}

static GLOBAL_REGISTRY: OnceLock<Arc<MetricsRegistry>> = OnceLock::new();

/// The process-global metrics registry, shared across the application.
pub fn global_registry() -> Arc<MetricsRegistry> {
    GLOBAL_REGISTRY
        .get_or_init(|| {
            debug!("Prometheus global registry initialized");
            Arc::new(MetricsRegistry::new())
        })
        .clone()
}

/// Increment a global counter by one, creating it if needed.
pub fn increment_counter(name: &'static str, help: &'static str) {
    global_registry().counter(name, help).inc();
}

/// Record an observation on a global histogram, creating it if needed.
pub fn observe_histogram(name: &'static str, help: &'static str, value: f64) {
    global_registry()
        .histogram(name, help, Vec::new())
        .observe(value);
}

/// Build an axum router serving the Prometheus scrape endpoint at `/metrics`.
///
/// The returned router carries no state (`Router<()>`), so it can be merged
/// into a larger router with `Router::merge` or nested with `Router::nest`.
pub fn metrics_router(registry: Arc<MetricsRegistry>) -> axum::Router {
    use axum::routing::get;

    axum::Router::new().route(
        "/metrics",
        get(move || {
            let registry = registry.clone();
            async move { registry.render() }
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_increments() {
        let registry = MetricsRegistry::new();
        let c = registry.counter("requests_total", "Requests.");
        c.inc();
        c.inc_by(4);
        assert_eq!(c.get(), 5);
        // Same name returns the same underlying counter.
        let c2 = registry.counter("requests_total", "Requests.");
        c2.inc();
        assert_eq!(c.get(), 6);
    }

    #[test]
    fn gauge_set_and_add() {
        let registry = MetricsRegistry::new();
        let g = registry.gauge("temperature", "Temp.");
        g.set(20.5);
        assert!((g.get() - 20.5).abs() < 1e-9);
        g.add(1.5);
        assert!((g.get() - 22.0).abs() < 1e-9);
        g.set(-3.0);
        assert!((g.get() + 3.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_observes_and_counts() {
        let registry = MetricsRegistry::new();
        let h = registry.histogram("latency_seconds", "Latency.", vec![0.1, 0.5, 1.0]);
        h.observe(0.05);
        h.observe(0.3);
        h.observe(0.8);
        h.observe(2.0);
        assert_eq!(h.count(), 4);
        let state = h.state.lock().unwrap();
        assert_eq!(state.counts[0], 1); // <= 0.1
        assert_eq!(state.counts[1], 2); // <= 0.5
        assert_eq!(state.counts[2], 3); // <= 1.0
        assert!((state.sum - 3.15).abs() < 1e-9);
    }

    #[test]
    fn render_produces_prometheus_text() {
        let registry = MetricsRegistry::new();
        let c = registry.counter("requests_total", "Total requests.");
        c.inc_by(3);
        let h = registry.histogram("latency_seconds", "Latency.", Vec::new());
        h.observe(0.05);

        let text = registry.render();
        assert!(text.contains("# TYPE requests_total counter"));
        assert!(text.contains("requests_total 3"));
        assert!(text.contains("# TYPE latency_seconds histogram"));
        assert!(text.contains("latency_seconds_bucket{le=\"+Inf\"} 1"));
        assert!(text.contains("latency_seconds_count 1"));
    }
}
