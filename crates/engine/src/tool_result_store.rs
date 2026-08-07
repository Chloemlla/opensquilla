//! Persistent raw tool-result storage for provider-context projections.
//!
//! Ports the Python `ToolResultStore` (`src/opensquilla/engine/tool_result_store.py`):
//! a disk-backed store for full tool-result snapshots that are omitted from
//! provider context. Records are content-addressed (truncated sha256 handle),
//! deduplicated on identical content, and pruned to fit per-result and
//! whole-disk byte budgets with retention enforcement.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Per-result byte budget: an individual snapshot at or above this size is
/// considered for compression (when available) and rejected outright if it
/// still exceeds the budget after compression.
pub const DEFAULT_TOOL_RESULT_MAX_BYTES: usize = 8 * 1024 * 1024;

/// Whole-disk byte budget: the store is pruned to fit this total before a new
/// record is written.
pub const DEFAULT_TOOL_RESULT_DISK_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// Retention window in seconds: records older than this are evicted on the
/// next cleanup scan.
pub const DEFAULT_TOOL_RESULT_RETENTION_SECONDS: u64 = 7 * 24 * 60 * 60;

/// Top-level session bucket directory name under the store root.
pub const TOOL_RESULT_STORE_SESSION_BUCKET: &str = "s";

/// Uncompressed content file name within a record directory.
pub const TOOL_RESULT_CONTENT_NAME: &str = "content.txt";

/// Compressed content file name within a record directory.
pub const TOOL_RESULT_COMPRESSED_CONTENT_NAME: &str = "content.txt.gz";

/// Metadata file name within a record directory.
pub const TOOL_RESULT_META_NAME: &str = "meta.json";

/// Hex chars of the content sha256 used to derive a deterministic
/// (content-addressed) handle. 32 hex chars = 128 bits, which satisfies the
/// `tr-<32 hex>` handle format and makes truncated-digest collisions between
/// distinct payloads negligible.
const CONTENT_HANDLE_HEX: usize = 32;

/// Raised when a raw tool-result snapshot exceeds store budgets.
#[derive(Debug, Clone, Error)]
#[error("{0}")]
pub struct ToolResultStoreBudgetError(pub String);

/// A full raw tool result stored on disk.
///
/// Mirrors the Python `ToolResultRecord` dataclass. `content` holds the raw
/// (decompressed) text; `stored_size_bytes` is `None` when the record was
/// written uncompressed and equals the on-disk byte count otherwise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultRecord {
    pub handle: String,
    pub tool_use_id: String,
    pub tool_name: String,
    pub session_id: String,
    pub session_key: String,
    pub agent_id: String,
    pub sha256: String,
    pub chars: usize,
    pub size_bytes: usize,
    pub created_at: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_size_bytes: Option<u64>,
    #[serde(default = "default_storage_encoding")]
    pub storage_encoding: String,
}

fn default_storage_encoding() -> String {
    "utf-8".to_string()
}

/// On-disk metadata persisted alongside the content file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMeta {
    handle: String,
    tool_use_id: String,
    tool_name: String,
    session_id: String,
    session_key: String,
    agent_id: String,
    sha256: String,
    chars: usize,
    size_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    stored_size_bytes: Option<u64>,
    #[serde(default = "default_storage_encoding")]
    storage_encoding: String,
    content_file: String,
    created_at: String,
}

/// Stat-only record metadata collected during cleanup scans.
#[derive(Debug, Clone)]
struct StoredStat {
    /// Content-file mtime as seconds since the UNIX epoch (used for age-based
    /// retention and oldest-first pruning).
    created_at_secs: i64,
    /// On-disk content-file size in bytes.
    size_bytes: u64,
    /// Absolute path to the record directory.
    record_dir: PathBuf,
}

/// Disk-backed store for full raw tool results omitted from provider context.
///
/// The store is organized as `<root>/s/<session>/<handle[3..5]>/<handle>/`,
/// where each leaf directory holds a `meta.json` and a content file
/// (`content.txt` or, when compression is available, `content.txt.gz`).
/// Writes are content-addressed: identical content that survived retention is
/// reused (with its access time refreshed) instead of rewritten, so the store
/// stops re-growing on repeats.
pub struct ToolResultStore {
    root: PathBuf,
}

impl ToolResultStore {
    /// Create a store rooted at `root`. The directory is created lazily on the
    /// first write.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The store's disk root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Store a full raw tool result and return its record.
    ///
    /// If a record holding identical content (full sha256 match) already
    /// exists and survived retention, it is reused and its access time is
    /// refreshed; no new bytes are written. Otherwise a new record is written
    /// after retention and budget pruning make room for it.
    ///
    /// Pass `None` for any budget/retention knob to disable that check.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        &self,
        content: &str,
        tool_use_id: &str,
        tool_name: &str,
        session_id: &str,
        session_key: &str,
        agent_id: &str,
        max_bytes: Option<usize>,
        disk_budget_bytes: Option<u64>,
        retention_seconds: Option<u64>,
    ) -> Result<ToolResultRecord, ToolResultStoreBudgetError> {
        let session_id = validate_non_empty("session_id", session_id)?;
        let session_key = validate_non_empty("session_key", session_key)?;
        let agent_id = validate_non_empty("agent_id", agent_id)?;

        let payload = content.as_bytes();
        let raw_size_bytes = payload.len();
        if raw_size_bytes == 0 {
            return Err(ToolResultStoreBudgetError(
                "tool result snapshot is empty".to_string(),
            ));
        }

        // TODO(parity): Python compresses payloads exceeding `max_bytes` with
        // gzip (compresslevel=6) and stores `content.txt.gz` with
        // `storage_encoding = "gzip+utf-8"` when the compressed form is
        // smaller. `flate2` is not in the engine's Cargo.toml and no
        // compression crate is available in this workspace, so compression is
        // skipped here and records are always stored uncompressed. `read`
        // still understands the `gzip+utf-8` encoding for forward-compatibility
        // with records written by the Python store.
        let stored_payload = payload;
        let content_name = TOOL_RESULT_CONTENT_NAME;
        let storage_encoding = "utf-8";
        let stored_size_bytes: u64 = raw_size_bytes as u64;

        if let Some(max) = max_bytes {
            if stored_size_bytes as usize > max {
                return Err(ToolResultStoreBudgetError(format!(
                    "tool result snapshot exceeds per-result budget (stored={stored_size_bytes}, raw={raw_size_bytes}, max={max})"
                )));
            }
        }

        let sha = hex::encode(Sha256::digest(payload));
        let primary_handle = format!("tr-{}", &sha[..CONTENT_HANDLE_HEX]);

        // One cleanup scan feeds both retention and the budget prune below, so
        // a new write pays a single store walk instead of re-scanning the whole
        // store once per cleanup pass (issue #305). Retention runs first — a
        // deduped write must never bypass cleanup nor reuse a record retention
        // is about to evict — and the surviving records prune to fit.
        let stats = self.iter_record_stats();
        let survivors = self.remove_expired(stats, retention_seconds);

        // Content-addressed snapshots: identical content that survived
        // retention is reused instead of rewritten — refreshing its access
        // time so a frequently re-projected record stays hot — and only
        // genuinely new content pays the budget prune and the write below.
        if let Some(reused) = self.existing_record(
            &primary_handle,
            &sha,
            content,
            tool_use_id,
            tool_name,
            &session_id,
            &session_key,
            &agent_id,
            raw_size_bytes,
        )? {
            self.touch(&primary_handle, &session_id);
            return Ok(reused);
        }

        if let Some(disk_budget) = disk_budget_bytes {
            self.prune_to_fit(survivors, stored_size_bytes, disk_budget)?;
        }

        let created_at = Utc::now()
            .to_rfc3339_opts(chrono::SecondsFormat::Nanos, true)
            .replace("+00:00", "Z");

        // The deterministic handle is tried first; a random handle is only
        // needed for the negligible chance of a truncated-digest collision
        // with *different* content already occupying that directory.
        let mut candidate_handles: Vec<String> = vec![primary_handle.clone()];
        for _ in 0..4 {
            candidate_handles.push(format!("tr-{}", random_hex_16()));
        }

        for handle in &candidate_handles {
            let record_dir = self.record_dir(handle, &session_id);
            let plain = record_dir.join(TOOL_RESULT_CONTENT_NAME);
            let gz = record_dir.join(TOOL_RESULT_COMPRESSED_CONTENT_NAME);
            if plain.exists() || gz.exists() {
                // A concurrent writer may have just stored the same content
                // here; reuse it. Otherwise it is a genuine collision and we
                // try a random handle.
                if let Some(reused) = self.existing_record(
                    handle,
                    &sha,
                    content,
                    tool_use_id,
                    tool_name,
                    &session_id,
                    &session_key,
                    &agent_id,
                    raw_size_bytes,
                )? {
                    self.touch(handle, &session_id);
                    return Ok(reused);
                }
                continue;
            }

            let record = ToolResultRecord {
                handle: handle.clone(),
                tool_use_id: tool_use_id.to_string(),
                tool_name: tool_name.to_string(),
                session_id: session_id.clone(),
                session_key: session_key.clone(),
                agent_id: agent_id.clone(),
                sha256: sha.clone(),
                chars: content.chars().count(),
                size_bytes: raw_size_bytes,
                created_at: created_at.clone(),
                content: content.to_string(),
                stored_size_bytes: Some(stored_size_bytes),
                storage_encoding: storage_encoding.to_string(),
            };

            let meta = StoredMeta {
                handle: record.handle.clone(),
                tool_use_id: record.tool_use_id.clone(),
                tool_name: record.tool_name.clone(),
                session_id: record.session_id.clone(),
                session_key: record.session_key.clone(),
                agent_id: record.agent_id.clone(),
                sha256: record.sha256.clone(),
                chars: record.chars,
                size_bytes: record.size_bytes,
                stored_size_bytes: record.stored_size_bytes,
                storage_encoding: record.storage_encoding.clone(),
                content_file: content_name.to_string(),
                created_at: record.created_at.clone(),
            };

            // Write meta first, then content last so its presence marks a
            // complete record for `existing_record` (dedup) and
            // `iter_record_stats` (cleanup). A concurrent cleanup may still
            // delete the meta first; both readers treat a missing meta as
            // "not a usable record", so that race is harmless.
            if let Err(e) = self.write_record(&record_dir, &meta, stored_payload) {
                remove_record_dir(&record_dir);
                return Err(ToolResultStoreBudgetError(format!(
                    "failed to persist tool result record: {e}"
                )));
            }
            return Ok(record);
        }

        Err(ToolResultStoreBudgetError(
            "could not allocate unique tool result handle".to_string(),
        ))
    }

    /// Read a previously stored tool result by handle and session.
    ///
    /// Verifies the session id, sha256, and byte size recorded in the
    /// metadata against the on-disk content, returning a [`ToolResultRecord`]
    /// with the decompressed text.
    pub fn read(&self, handle: &str, session_id: &str) -> io::Result<ToolResultRecord> {
        let session_id = validate_non_empty_io("session_id", session_id)?;
        let normalized = validate_handle_io(handle)?;
        let record_dir = self.record_dir(&normalized, &session_id);
        let meta_path = record_dir.join(TOOL_RESULT_META_NAME);
        let meta_raw = fs::read_to_string(&meta_path)?;
        let meta: StoredMeta = serde_json::from_str(&meta_raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        let content_name = if meta.content_file.is_empty() {
            TOOL_RESULT_CONTENT_NAME
        } else {
            meta.content_file.as_str()
        };
        let storage_encoding = if meta.storage_encoding.is_empty() {
            "utf-8"
        } else {
            meta.storage_encoding.as_str()
        };
        let content_path = record_dir.join(content_name);

        let content = if storage_encoding == "gzip+utf-8" {
            // TODO(parity): decompression requires `flate2` which is not in the
            // engine's Cargo.toml. Records written by this store are always
            // uncompressed; this branch only triggers for records written by
            // the Python store during a mixed-language migration.
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "gzip-compressed tool result records are not readable by the Rust store (flate2 not available)",
            ));
        } else {
            fs::read_to_string(&content_path)?
        };

        let payload = content.as_bytes();
        let sha = hex::encode(Sha256::digest(payload));
        if meta.session_id != session_id {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "tool result session mismatch",
            ));
        }
        if sha != meta.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tool result hash mismatch",
            ));
        }
        let size_bytes = meta.size_bytes;
        if size_bytes != payload.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tool result size mismatch",
            ));
        }
        let stored_size_bytes = match meta.stored_size_bytes {
            Some(v) => Some(v),
            None => Some(content_path.metadata().map(|m| m.len()).unwrap_or(0)),
        };

        Ok(ToolResultRecord {
            handle: normalized,
            tool_use_id: meta.tool_use_id,
            tool_name: meta.tool_name,
            session_id: meta.session_id,
            session_key: meta.session_key,
            agent_id: meta.agent_id,
            sha256: sha,
            chars: content.chars().count(),
            size_bytes: payload.len(),
            created_at: meta.created_at,
            content,
            stored_size_bytes,
            storage_encoding: storage_encoding.to_string(),
        })
    }

    /// Return the already-stored record for `handle` iff it holds this exact
    /// content (full sha256 match). Makes repeated writes idempotent and
    /// detects the negligible truncated-digest collision. Costs one existence
    /// check plus one small meta read; never scans the store.
    #[allow(clippy::too_many_arguments)]
    fn existing_record(
        &self,
        handle: &str,
        sha: &str,
        content: &str,
        tool_use_id: &str,
        tool_name: &str,
        session_id: &str,
        session_key: &str,
        agent_id: &str,
        size_bytes: usize,
    ) -> Result<Option<ToolResultRecord>, ToolResultStoreBudgetError> {
        let record_dir = self.record_dir(handle, session_id);
        let plain = record_dir.join(TOOL_RESULT_CONTENT_NAME);
        let gz = record_dir.join(TOOL_RESULT_COMPRESSED_CONTENT_NAME);
        if !plain.exists() && !gz.exists() {
            return Ok(None);
        }
        let meta = match self.read_meta(&record_dir) {
            Some(m) => m,
            None => return Ok(None),
        };
        if meta.sha256 != sha {
            return Ok(None);
        }
        Ok(Some(ToolResultRecord {
            handle: handle.to_string(),
            tool_use_id: tool_use_id.to_string(),
            tool_name: tool_name.to_string(),
            session_id: session_id.to_string(),
            session_key: session_key.to_string(),
            agent_id: agent_id.to_string(),
            sha256: sha.to_string(),
            chars: content.chars().count(),
            size_bytes,
            created_at: meta.created_at,
            content: content.to_string(),
            stored_size_bytes: meta.stored_size_bytes,
            storage_encoding: if meta.storage_encoding.is_empty() {
                "utf-8".to_string()
            } else {
                meta.storage_encoding
            },
        }))
    }

    fn read_meta(&self, record_dir: &Path) -> Option<StoredMeta> {
        let meta_path = record_dir.join(TOOL_RESULT_META_NAME);
        let raw = fs::read_to_string(&meta_path).ok()?;
        serde_json::from_str(&raw).ok()
    }

    /// Refresh a record's last-access time so a frequently reused snapshot is
    /// not evicted by retention while it is still being projected.
    ///
    /// Uses `std::fs::File::set_times` (stable since Rust 1.75) to update the
    /// modification time without rewriting the file.
    fn touch(&self, handle: &str, session_id: &str) {
        let record_dir = self.record_dir(handle, session_id);
        for content_name in [
            TOOL_RESULT_CONTENT_NAME,
            TOOL_RESULT_COMPRESSED_CONTENT_NAME,
        ] {
            let content_path = record_dir.join(content_name);
            if !content_path.exists() {
                continue;
            }
            let Ok(file) = fs::File::open(&content_path) else {
                continue;
            };
            let times = fs::FileTimes::new().set_modified(SystemTime::now());
            let _ = file.set_times(times);
        }
    }

    fn record_dir(&self, handle: &str, session_id: &str) -> PathBuf {
        let normalized = validate_handle(handle).expect("validated by caller");
        self.root
            .join(TOOL_RESULT_STORE_SESSION_BUCKET)
            .join(safe_token(session_id))
            .join(&normalized[3..5])
            .join(normalized)
    }

    /// Write a record's meta and content atomically (write-to-tmp + rename).
    fn write_record(
        &self,
        record_dir: &Path,
        meta: &StoredMeta,
        content_payload: &[u8],
    ) -> io::Result<()> {
        fs::create_dir_all(record_dir)?;
        let meta_bytes = serde_json::to_vec(meta)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        atomic_write_bytes(&record_dir.join(TOOL_RESULT_META_NAME), &meta_bytes)?;
        atomic_write_bytes(
            &record_dir.join(meta.content_file.as_str()),
            content_payload,
        )?;
        Ok(())
    }

    /// Enumerate stored records for cleanup using only filesystem stat — size
    /// from the content file and age from its mtime — instead of parsing every
    /// `meta.json`. Cleanup runs only when genuinely new content is stored, and
    /// even then this keeps the scan to cheap stat calls rather than O(records)
    /// JSON reads.
    fn iter_record_stats(&self) -> Vec<StoredStat> {
        let root = self.root.join(TOOL_RESULT_STORE_SESSION_BUCKET);
        if !root.exists() {
            return Vec::new();
        }
        let mut records = Vec::new();
        for content_name in [
            TOOL_RESULT_CONTENT_NAME,
            TOOL_RESULT_COMPRESSED_CONTENT_NAME,
        ] {
            walk_content_files(&root, content_name, &mut records);
        }
        records
    }

    /// Delete records older than the retention window and return the
    /// survivors, so the caller can reuse this single scan for the budget
    /// prune instead of walking the store again.
    fn remove_expired(
        &self,
        records: Vec<StoredStat>,
        retention_seconds: Option<u64>,
    ) -> Vec<StoredStat> {
        let Some(retention) = retention_seconds else {
            return records;
        };
        let now_secs = system_time_to_secs(SystemTime::now());
        let cutoff = now_secs - (retention as i64).max(0);
        let mut survivors = Vec::with_capacity(records.len());
        for record in records {
            if record.created_at_secs < cutoff {
                remove_record_dir(&record.record_dir);
            } else {
                survivors.push(record);
            }
        }
        survivors
    }

    /// Prune oldest records to fit the disk budget before writing `incoming_bytes`.
    /// Raises a budget error if the incoming snapshot alone exceeds the budget.
    fn prune_to_fit(
        &self,
        records: Vec<StoredStat>,
        incoming_bytes: u64,
        disk_budget_bytes: u64,
    ) -> Result<(), ToolResultStoreBudgetError> {
        let budget = (disk_budget_bytes as i64).max(0) as u64;
        let mut sorted = records;
        sorted.sort_by_key(|r| r.created_at_secs);
        let mut current: u64 = sorted.iter().map(|r| r.size_bytes).sum();
        if current.saturating_add(incoming_bytes) <= budget {
            return Ok(());
        }
        for record in sorted {
            remove_record_dir(&record.record_dir);
            current = current.saturating_sub(record.size_bytes);
            if current.saturating_add(incoming_bytes) <= budget {
                return Ok(());
            }
        }
        if incoming_bytes > budget {
            return Err(ToolResultStoreBudgetError(format!(
                "tool result snapshot exceeds disk budget ({incoming_bytes} > {budget})"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers (mirror the Python module-level functions).
// ---------------------------------------------------------------------------

/// Walk `root` recursively for files named `content_name` and append a
/// [`StoredStat`] for each well-formed `tr-<32hex>` record directory.
fn walk_content_files(root: &Path, content_name: &str, out: &mut Vec<StoredStat>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_content_files(&path, content_name, out);
            continue;
        }
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name != content_name {
            continue;
        }
        let record_dir = match path.parent() {
            Some(d) => d,
            None => continue,
        };
        // Only ever consider (and later delete) well-formed tr-<32hex> record
        // dirs, so cleanup can never touch a stray or foreign file that happens
        // to live under the shared media root.
        let dir_name = record_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if validate_handle(dir_name).is_err() {
            continue;
        }
        let stat = match fs::metadata(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mtime = stat.modified().map(system_time_to_secs).unwrap_or(0);
        out.push(StoredStat {
            created_at_secs: mtime,
            size_bytes: stat.len(),
            record_dir: record_dir.to_path_buf(),
        });
    }
}

/// Atomic write: write to a sibling `<name>.tmp` then rename over the target.
/// Mirrors the Python `_atomic_write_bytes` helper.
fn atomic_write_bytes(path: &Path, data: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, data)?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Remove every file in `record_dir` then the directory itself, ignoring
/// errors (best-effort cleanup).
fn remove_record_dir(record_dir: &Path) {
    if let Ok(entries) = fs::read_dir(record_dir) {
        for entry in entries.flatten() {
            let _ = fs::remove_file(entry.path());
        }
    }
    let _ = fs::remove_dir(record_dir);
}

/// Validate a `tr-<32 hex>` handle, returning it unchanged on success.
fn validate_handle(value: &str) -> Result<String, ToolResultStoreBudgetError> {
    if !value.starts_with("tr-") {
        return Err(ToolResultStoreBudgetError(
            "tool result handle is invalid".to_string(),
        ));
    }
    let suffix = &value[3..];
    if suffix.len() != 32 || !suffix.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ToolResultStoreBudgetError(
            "tool result handle is invalid".to_string(),
        ));
    }
    Ok(value.to_string())
}

fn validate_handle_io(value: &str) -> io::Result<String> {
    validate_handle(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.0))
}

/// Validate that a string is non-empty after trimming; return the trimmed
/// value.
fn validate_non_empty(name: &str, value: &str) -> Result<String, ToolResultStoreBudgetError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(ToolResultStoreBudgetError(format!("{name} is required")));
    }
    Ok(trimmed.to_string())
}

fn validate_non_empty_io(name: &str, value: &str) -> io::Result<String> {
    validate_non_empty(name, value).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.0))
}

/// Sanitize a session token for use as a path component: replace runs of
/// non-safe characters with `-`, strip leading/trailing dots and dashes,
/// truncate to 80 chars, falling back to `"session"` when empty.
fn safe_token(value: &str) -> String {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"[^A-Za-z0-9._-]+").expect("valid regex"));
    let token = re.replace_all(value.trim(), "-");
    let token = token.trim_matches(|c: char| c == '.' || c == '-');
    let truncated = &token[..token.len().min(80)];
    if truncated.is_empty() {
        "session".to_string()
    } else {
        truncated.to_string()
    }
}

/// Convert a `SystemTime` to seconds since the UNIX epoch (negative if before).
fn system_time_to_secs(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => -(e.duration().as_secs() as i64),
    }
}

/// Generate 16 pseudo-random bytes as a 32-char hex string.
///
/// TODO(parity): Python uses `secrets.token_hex(16)` for CSPRNG-quality
/// handles. The engine's Cargo.toml does not include `rand` or `getrandom`,
/// so this falls back to a `SystemTime` nanosecond seed + simple LCG.
/// Collisions on the fallback are still negligible (32 hex chars from a
/// time-seeded mix), but this is not cryptographically random.
fn random_hex_16() -> String {
    let mut buf = [0u8; 16];
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    // Simple LCG over the seed to fill 16 bytes — not cryptographic, but
    // sufficient for handle uniqueness within a single store.
    let mut state = seed;
    for byte in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (state >> 56) as u8;
    }
    hex::encode(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_store() -> (TempdirCleanup, ToolResultStore) {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!(
            "opensquilla_trs_test_{}_{}",
            std::process::id(),
            id
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let store = ToolResultStore::new(&dir);
        (TempdirCleanup(dir), store)
    }

    struct TempdirCleanup(PathBuf);

    impl Drop for TempdirCleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn write_and_read_roundtrip() {
        let (_dir, store) = temp_store();
        let record = store
            .write(
                "hello world",
                "tu_1",
                "Read",
                "sess_1",
                "sk_1",
                "agent_1",
                None,
                None,
                None,
            )
            .unwrap();
        assert_eq!(record.content, "hello world");
        assert_eq!(record.session_id, "sess_1");
        assert!(record.handle.starts_with("tr-"));
        assert_eq!(record.sha256, hex::encode(Sha256::digest(b"hello world")));

        let read_back = store.read(&record.handle, "sess_1").unwrap();
        assert_eq!(read_back.content, "hello world");
        assert_eq!(read_back.handle, record.handle);
        assert_eq!(read_back.sha256, record.sha256);
    }

    #[test]
    fn write_reuses_identical_content() {
        let (_dir, store) = temp_store();
        let first = store
            .write(
                "same", "tu_1", "Read", "sess_1", "sk_1", "agent_1", None, None, None,
            )
            .unwrap();
        let second = store
            .write(
                "same", "tu_1", "Read", "sess_1", "sk_1", "agent_1", None, None, None,
            )
            .unwrap();
        assert_eq!(first.handle, second.handle);
    }

    #[test]
    fn empty_content_rejected() {
        let (_dir, store) = temp_store();
        let err = store
            .write(
                "", "tu_1", "Read", "sess_1", "sk_1", "agent_1", None, None, None,
            )
            .unwrap_err();
        assert!(err.0.contains("empty"));
    }

    #[test]
    fn per_result_budget_enforced() {
        let (_dir, store) = temp_store();
        let err = store
            .write(
                &"x".repeat(100),
                "tu_1",
                "Read",
                "sess_1",
                "sk_1",
                "agent_1",
                Some(10),
                None,
                None,
            )
            .unwrap_err();
        assert!(err.0.contains("per-result budget"));
    }

    #[test]
    fn read_rejects_wrong_session() {
        let (_dir, store) = temp_store();
        let record = store
            .write(
                "data", "tu_1", "Read", "sess_1", "sk_1", "agent_1", None, None, None,
            )
            .unwrap();
        let err = store.read(&record.handle, "sess_2").unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
    }

    #[test]
    fn handle_validation() {
        assert!(validate_handle("tr-").is_err());
        assert!(validate_handle("tr-short").is_err());
        assert!(validate_handle("tr-0123456789abcdef0123456789abcdef").is_ok());
        assert!(validate_handle("xx-0123456789abcdef0123456789abcdef").is_err());
    }

    #[test]
    fn safe_token_sanitizes() {
        assert_eq!(safe_token("session_1"), "session_1");
        assert_eq!(safe_token("../etc/passwd"), "etc-passwd");
        assert_eq!(safe_token("   "), "session");
    }
}
