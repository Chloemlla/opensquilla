use async_trait::async_trait;

use crate::report::SubsystemHealth;

/// A health check for a single subsystem.
///
/// Implementations must be `Send + Sync` so they can be shared across async
/// tasks and combined into a single report.
#[async_trait]
pub trait HealthCheck: Send + Sync {
    /// The name of the subsystem (e.g. `config`, `database`).
    fn name(&self) -> &str;

    /// Run the health check and produce a subsystem report.
    async fn check(&self) -> SubsystemHealth;
}
