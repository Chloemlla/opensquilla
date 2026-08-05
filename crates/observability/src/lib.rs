//! # OpenSquilla Observability
//!
//! Observability infrastructure: structured logging, distributed tracing,
//! telemetry/metrics collection, Prometheus metrics export, subsystem health
//! checks, and audit logging.

pub mod logging;
pub mod tracing;
pub mod telemetry;
pub mod audit;
pub mod prometheus;
pub mod health;

pub use logging::Logger;
pub use tracing::Tracer;
pub use telemetry::Telemetry;
pub use audit::AuditLog;
pub use prometheus::{global_registry, metrics_router, Counter, Gauge, Histogram, MetricsRegistry};
pub use health::{
    health_router, ComponentHealth, HealthCheck, HealthIssue, HealthRegistry, HealthReport,
    HealthStatus, IssueSeverity,
};