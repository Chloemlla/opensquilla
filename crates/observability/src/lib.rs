//! # OpenSquilla Observability
//!
//! Observability infrastructure: structured logging, distributed tracing,
//! telemetry/metrics collection, Prometheus metrics export, subsystem health
//! checks, and audit logging.

pub mod audit;
pub mod health;
pub mod logging;
pub mod prometheus;
pub mod telemetry;
pub mod tracing;

pub use audit::AuditLog;
pub use health::{
    ComponentHealth, HealthCheck, HealthIssue, HealthRegistry, HealthReport, HealthStatus,
    IssueSeverity, health_router,
};
pub use logging::Logger;
pub use prometheus::{Counter, Gauge, Histogram, MetricsRegistry, global_registry, metrics_router};
pub use telemetry::Telemetry;
pub use tracing::Tracer;
