use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single schema migration.
///
/// Migrations are applied in ascending `version` order. Each migration carries
/// forward SQL plus an optional reverse (rollback) SQL statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaMigration {
    /// Unique version number. Applied in ascending order.
    pub version: i64,
    /// Human-readable name.
    pub name: String,
    /// SQL statements to apply.
    pub sql: String,
    /// SQL statements to reverse the migration (for rollback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollback_sql: Option<String>,
    /// Optional human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl SchemaMigration {
    /// Create a new migration with forward SQL only.
    pub fn new(version: i64, name: impl Into<String>, sql: impl Into<String>) -> Self {
        Self {
            version,
            name: name.into(),
            sql: sql.into(),
            rollback_sql: None,
            description: None,
        }
    }

    /// Attach rollback SQL to this migration.
    pub fn with_rollback(mut self, rollback_sql: impl Into<String>) -> Self {
        self.rollback_sql = Some(rollback_sql.into());
        self
    }

    /// Attach a description to this migration.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A record of a migration that has been applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedMigration {
    /// The applied version number.
    pub version: i64,
    /// The migration name.
    pub name: String,
    /// When the migration was applied.
    pub applied_at: DateTime<Utc>,
}
