use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// A pre-migration snapshot backup of a SQLite database file.
#[derive(Debug, Clone)]
pub struct Backup {
    /// Path to the backup copy.
    pub path: PathBuf,
    /// SHA-256 checksum of the backup contents.
    pub checksum: String,
}

/// Create a backup copy of the database file at `db_path` inside `backup_dir`.
///
/// The backup file is named `<db_name>.<timestamp>.bak`.
pub fn snapshot(db_path: impl AsRef<Path>, backup_dir: impl AsRef<Path>) -> crate::Result<Backup> {
    let db_path = db_path.as_ref();
    let backup_dir = backup_dir.as_ref();

    if !db_path.exists() {
        return Err(crate::Error::InvalidState(format!(
            "Database {} does not exist; nothing to back up",
            db_path.display()
        )));
    }

    std::fs::create_dir_all(backup_dir)?;

    let file_name = db_path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "opensquilla.db".into());
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let backup_path = backup_dir.join(format!("{file_name}.{timestamp}.bak"));

    std::fs::copy(db_path, &backup_path)?;
    let checksum = checksum_of(&backup_path)?;

    tracing::info!(
        path = %backup_path.display(),
        checksum = %checksum,
        "created pre-migration database backup"
    );

    Ok(Backup {
        path: backup_path,
        checksum,
    })
}

/// Compute the SHA-256 checksum of a file.
pub fn checksum_of(path: impl AsRef<Path>) -> crate::Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(hex::encode(hasher.finalize()))
}
