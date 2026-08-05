use std::collections::HashMap;
use std::path::Path;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension};

use crate::migration::{AppliedMigration, SchemaMigration};

/// The table that tracks applied schema versions.
const SCHEMA_VERSION_TABLE: &str = "CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL
)";

/// Runs schema migrations against a SQLite database.
pub struct MigrationRunner {
    connection: Connection,
    migrations: Vec<SchemaMigration>,
}

impl MigrationRunner {
    /// Create a runner over an existing open connection.
    pub fn new(connection: Connection) -> crate::Result<Self> {
        connection.execute_batch(SCHEMA_VERSION_TABLE)?;
        Ok(Self {
            connection,
            migrations: Vec::new(),
        })
    }

    /// Open a SQLite database at the given path and prepare the version table.
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let connection = Connection::open(path.as_ref())?;
        Self::new(connection)
    }

    /// Open an in-memory SQLite database (mainly for tests).
    pub fn in_memory() -> crate::Result<Self> {
        Self::new(Connection::open_in_memory()?)
    }

    /// Register a migration, keeping the list sorted by version.
    pub fn add_migration(&mut self, migration: SchemaMigration) -> &mut Self {
        self.migrations.push(migration);
        self.migrations.sort_by_key(|m| m.version);
        self
    }

    /// Register several migrations at once.
    pub fn add_migrations(&mut self, migrations: Vec<SchemaMigration>) -> &mut Self {
        self.migrations.extend(migrations);
        self.migrations.sort_by_key(|m| m.version);
        self
    }

    /// The full ordered list of registered migrations.
    pub fn migrations(&self) -> &[SchemaMigration] {
        &self.migrations
    }

    /// The current schema version (0 when nothing has been applied).
    pub fn current_version(&self) -> crate::Result<i64> {
        Ok(self.connection.query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )?)
    }

    /// Apply a single migration in a transaction. Errors if the version has
    /// already been applied.
    pub fn apply_migration(
        &mut self,
        migration: &SchemaMigration,
    ) -> crate::Result<AppliedMigration> {
        let already: Option<i64> = self
            .connection
            .query_row(
                "SELECT version FROM schema_version WHERE version = ?1",
                [migration.version],
                |row| row.get(0),
            )
            .optional()?;

        if already.is_some() {
            return Err(crate::Error::AlreadyApplied(migration.version));
        }

        let tx = self.connection.transaction()?;
        tx.execute_batch(&migration.sql)?;
        let applied_at = opensquilla_core::time::now();
        tx.execute(
            "INSERT INTO schema_version (version, name, applied_at) VALUES (?1, ?2, ?3)",
            rusqlite::params![migration.version, migration.name, applied_at.to_rfc3339()],
        )?;
        tx.commit()?;

        tracing::debug!(version = migration.version, name = %migration.name, "applied migration");
        Ok(AppliedMigration {
            version: migration.version,
            name: migration.name.clone(),
            applied_at,
        })
    }

    /// Apply every registered migration that has not yet been applied.
    /// Returns the list of newly applied migrations.
    pub fn apply_all(&mut self) -> crate::Result<Vec<AppliedMigration>> {
        let pending: Vec<SchemaMigration> = self
            .migrations
            .iter()
            .filter(|m| !self.is_applied(m.version).unwrap_or(false))
            .cloned()
            .collect();

        let mut applied = Vec::new();
        for migration in &pending {
            applied.push(self.apply_migration(migration)?);
        }
        Ok(applied)
    }

    /// List all applied migrations, oldest first.
    pub fn list_migrations(&self) -> crate::Result<Vec<AppliedMigration>> {
        let mut stmt = self
            .connection
            .prepare("SELECT version, name, applied_at FROM schema_version ORDER BY version")?;
        let rows = stmt.query_map([], |row| {
            let applied_at_str: String = row.get(2)?;
            let applied_at = chrono::DateTime::parse_from_rfc3339(&applied_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());
            Ok(AppliedMigration {
                version: row.get(0)?,
                name: row.get(1)?,
                applied_at,
            })
        })?;

        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Whether a given version has been applied.
    pub fn is_applied(&self, version: i64) -> crate::Result<bool> {
        let existing: Option<i64> = self
            .connection
            .query_row(
                "SELECT version FROM schema_version WHERE version = ?1",
                [version],
                |row| row.get(0),
            )
            .optional()?;
        Ok(existing.is_some())
    }

    /// Roll back all migrations above `target_version`, newest first, using
    /// each migration's rollback SQL. Returns the versions rolled back.
    pub fn rollback(&mut self, target_version: i64) -> crate::Result<Vec<i64>> {
        let applied = self.list_migrations()?;
        let mut versions = applied
            .iter()
            .map(|a| a.version)
            .filter(|v| *v > target_version)
            .collect::<Vec<_>>();
        versions.sort_unstable_by(|a, b| b.cmp(a)); // newest first

        let rollback_plans = versions
            .iter()
            .map(|v| {
                let migration = self.migrations.iter().find(|m| m.version == *v);
                match migration.and_then(|m| m.rollback_sql.clone()) {
                    Some(sql) => Ok((*v, sql)),
                    None => Err(crate::Error::RollbackFailed(format!(
                        "No rollback SQL for version {v}"
                    ))),
                }
            })
            .collect::<crate::Result<Vec<_>>>()?;

        let mut rolled_back = Vec::new();
        for (version, sql) in rollback_plans {
            let tx = self.connection.transaction()?;
            tx.execute_batch(&sql)?;
            tx.execute("DELETE FROM schema_version WHERE version = ?1", [version])?;
            tx.commit()?;
            rolled_back.push(version);
        }
        Ok(rolled_back)
    }

    /// Migrations registered by version, for lookups.
    pub fn migration_map(&self) -> HashMap<i64, &SchemaMigration> {
        self.migrations.iter().map(|m| (m.version, m)).collect()
    }
}
