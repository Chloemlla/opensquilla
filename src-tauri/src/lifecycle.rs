//! App lifecycle: single-instance enforcement, gateway ownership verification
//! (PID lock file), startup orchestration, graceful shutdown, and temp cleanup.
//!
//! This module replaces five Electron modules:
//! - `desktop-gateway-ownership.ts` + `desktop-gateway-ownership-verification.ts`
//!   → [`OwnershipRecord`] and the verification coordinator. In the
//!   single-process Rust model the "gateway" *is* this process, so the
//!   ownership record is the cross-instance handoff barrier: a second launch
//!   reads it to decide whether the running owner is provably the same
//!   instance (HMAC challenge) before deferring.
//! - `desktop-profile-context.ts` → [`ProfilePaths`].
//! - `gateway-lifecycle.ts` → [`lifecycle_allows_process_spawn`] and
//!   [`stop_and_join_lifecycle_processes`].
//! - `desktop-cleanup.ts` → [`DesktopCleanupMode`] and the scope-containment
//!   checks.
//!
//! The single-instance lock itself is enforced by `tauri-plugin-single-instance`
//! in `main.rs`; this module provides the higher-level ownership + drain logic
//! that runs once the lock is held (or while deciding whether to defer).

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::storage;

// ---------------------------------------------------------------------------
// Constants — must stay byte-identical to the Python/Electron runtime.
// ---------------------------------------------------------------------------

/// Schema version of the ownership record. Bumped only on a wire-format break.
pub const OWNERSHIP_SCHEMA_VERSION: u32 = 1;
/// The protocol identifier written into every ownership record.
pub const OWNERSHIP_PROTOCOL: &str = "opensquilla-desktop-gateway-ownership-v1";
/// The on-disk filename for the ownership record.
pub const OWNERSHIP_FILENAME: &str = "desktop-gateway.json";
/// Reject any record larger than this to bound a malicious/linked file.
const RECORD_MAX_BYTES: u64 = 16 * 1024;

/// A base64url owner token (32–128 chars of `[A-Za-z0-9_-]`).
pub fn owner_token_re() -> &'static Regex {
    static RE: once_cell::sync::OnceCell<Regex> = once_cell::sync::OnceCell::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z0-9_-]{32,128}$").unwrap())
}

/// A 64-hex-char SHA-256 profile fingerprint.
pub fn profile_fingerprint_re() -> &'static Regex {
    static RE: once_cell::sync::OnceCell<Regex> = once_cell::sync::OnceCell::new();
    RE.get_or_init(|| Regex::new(r"^[0-9a-f]{64}$").unwrap())
}

// ---------------------------------------------------------------------------
// Profile paths (desktop-profile-context.ts)
// ---------------------------------------------------------------------------

/// The kind of profile a path tuple describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProfileKind {
    Primary,
    Recovery,
}

/// The authoritative profile paths the desktop runtime operates on.
///
/// Historical recovery profiles are enumerated by [`all_profile_contexts`] for
/// one-time consolidation but can never become active.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePaths {
    pub kind: ProfileKind,
    pub recovery_id: Option<String>,
    pub home: PathBuf,
    pub credential_path: PathBuf,
    pub logs_dir: PathBuf,
}

/// The canonical primary profile rooted under `userData/opensquilla`.
pub fn primary_profile_paths(user_data: &Path) -> ProfilePaths {
    let root = user_data.to_path_buf();
    ProfilePaths {
        kind: ProfileKind::Primary,
        recovery_id: None,
        home: root.join("opensquilla"),
        credential_path: root.join("desktop-credential.json"),
        logs_dir: root.join("logs"),
    }
}

/// Validate a recovery profile id (a UUIDv4) without trusting the caller.
pub fn is_recovery_profile_id(value: &str) -> bool {
    uuid::Uuid::parse_str(value)
        .map(|u| u.get_version() == Some(uuid::Version::Random))
        .unwrap_or(false)
}

/// A recovery profile rooted under `userData/recovery-profiles/{id}`.
pub fn recovery_profile_paths(user_data: &Path, recovery_id: &str) -> io::Result<ProfilePaths> {
    if !is_recovery_profile_id(recovery_id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Invalid recovery profile id.",
        ));
    }
    let root = user_data.join("recovery-profiles").join(recovery_id);
    Ok(ProfilePaths {
        kind: ProfileKind::Recovery,
        recovery_id: Some(recovery_id.to_string()),
        home: root.join("opensquilla"),
        credential_path: root.join("desktop-credential.json"),
        logs_dir: root.join("logs"),
    })
}

/// Safely enumerate the primary profile plus any valid recovery profiles.
///
/// Symlinks, junctions, missing homes, or malformed entries are ignored and
/// never traversed — exactly the no-follow contract of the Electron shell.
pub fn all_profile_contexts(user_data: &Path) -> Vec<ProfilePaths> {
    let mut profiles = vec![primary_profile_paths(user_data)];
    let recovery_root = user_data.join("recovery-profiles");
    let entries = match std::fs::read_dir(&recovery_root) {
        Ok(rd) => rd,
        Err(_) => return profiles,
    };
    // A symlinked recovery root is unsafe; bail to the primary only.
    if std::fs::symlink_metadata(&recovery_root)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false)
    {
        return profiles;
    }
    let mut ids: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
        .filter(|s| is_recovery_profile_id(s))
        .collect();
    ids.sort();
    for id in ids {
        if let Ok(profile) = recovery_profile_paths(user_data, &id) {
            if recovery_profile_status(user_data, &profile) == DirStatus::Valid {
                profiles.push(profile);
            }
        }
    }
    profiles
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DirStatus {
    Valid,
    Missing,
    Unsafe,
}

fn real_directory_status(path: &Path) -> DirStatus {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() && !m.file_type().is_symlink() => DirStatus::Valid,
        Ok(_) => DirStatus::Unsafe,
        Err(e) if e.kind() == io::ErrorKind::NotFound => DirStatus::Missing,
        Err(_) => DirStatus::Unsafe,
    }
}

fn recovery_profile_status(user_data: &Path, profile: &ProfilePaths) -> DirStatus {
    let recovery_root = user_data.join("recovery-profiles");
    let profile_root = profile
        .home
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    for path in [
        recovery_root.as_path(),
        profile_root.as_path(),
        profile.home.as_path(),
    ] {
        match real_directory_status(path) {
            DirStatus::Valid => {}
            other => return other,
        }
    }
    // Confirm each path resolves inside its parent (no symlink escape).
    let resolved_recovery_root = std::fs::canonicalize(&recovery_root).unwrap_or(recovery_root);
    let resolved_profile_root = std::fs::canonicalize(&profile_root).unwrap_or(profile_root);
    let resolved_home =
        std::fs::canonicalize(&profile.home).unwrap_or_else(|_| profile.home.clone());
    if resolved_profile_root.parent() != Some(resolved_recovery_root.as_path()) {
        return DirStatus::Unsafe;
    }
    if resolved_home.parent() != Some(resolved_profile_root.as_path()) {
        return DirStatus::Unsafe;
    }
    DirStatus::Valid
}

/// SHA-256 fingerprint of the canonical profile home path, hiding the real
/// path from the ownership record. Mirrors `recovery.locking.profile_lock_key`.
pub fn profile_fingerprint(profile_home: &Path) -> String {
    let canonical =
        std::fs::canonicalize(profile_home).unwrap_or_else(|_| profile_home.to_path_buf());
    let mut display = canonical.to_string_lossy().to_string();
    // Normalize separators and, on Windows, case-fold (NTFS is case-insensitive).
    if cfg!(windows) {
        display = display.to_lowercase();
    }
    let mut hasher = Sha256::new();
    hasher.update(display.as_bytes());
    hex::encode(hasher.finalize())
}

// ---------------------------------------------------------------------------
// Ownership record (desktop-gateway-ownership.ts)
// ---------------------------------------------------------------------------

/// The persisted ownership record. Written by the live owner; read by any
/// second launch to verify the running gateway is provably ours.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnershipRecord {
    pub schema_version: u32,
    pub protocol: String,
    pub profile_fingerprint: String,
    pub pid: u32,
    pub start_identity: String,
    pub port: u16,
    pub version: String,
    pub instance_nonce: String,
}

/// The outcome of loading an ownership record from disk.
#[derive(Debug, Clone)]
pub enum OwnershipRecordLoad {
    /// No record present — the cross-process handoff barrier is open.
    Missing,
    /// A record exists but is malformed, linked, oversized, or future-schema.
    /// Never trusted to stop a process.
    Invalid,
    /// A well-formed record was read.
    Valid(OwnershipRecord),
}

/// The launch authority bound to the current process: the instance nonce,
/// profile fingerprint, and bound port. The record is matched against this,
/// not the immediate PID, so a `uv run` wrapper and its child share authority.
#[derive(Debug, Clone)]
pub struct LaunchAuthority {
    pub instance_nonce: String,
    pub profile_fingerprint: String,
    pub port: u16,
}

/// True when a record was produced by this exact launch authority.
pub fn ownership_matches_launch(record: &OwnershipRecord, authority: &LaunchAuthority) -> bool {
    record.instance_nonce == authority.instance_nonce
        && record.profile_fingerprint == authority.profile_fingerprint
        && record.port == authority.port
}

/// Parse an untrusted JSON value into an [`OwnershipRecord`].
///
/// Every field is validated; a single bad field makes the whole record invalid
/// rather than partially trusted.
pub fn parse_ownership_record(value: &serde_json::Value) -> Option<OwnershipRecord> {
    let obj = value.as_object()?;
    if obj.get("schema_version").and_then(|v| v.as_u64()) != Some(OWNERSHIP_SCHEMA_VERSION as u64) {
        return None;
    }
    if obj.get("protocol").and_then(|v| v.as_str()) != Some(OWNERSHIP_PROTOCOL) {
        return None;
    }
    let profile_fingerprint = obj.get("profile_fingerprint")?.as_str()?;
    if !profile_fingerprint_re().is_match(profile_fingerprint) {
        return None;
    }
    let pid = obj.get("pid")?.as_u64()?;
    if pid == 0 || pid > u32::MAX as u64 {
        return None;
    }
    let start_identity = obj.get("start_identity")?.as_str()?;
    if start_identity.is_empty() || start_identity.len() > 256 {
        return None;
    }
    let port = obj.get("port")?.as_u64()?;
    if !(1..=65535).contains(&port) {
        return None;
    }
    let version = obj.get("version")?.as_str()?;
    if version.is_empty() || version.len() > 128 {
        return None;
    }
    let instance_nonce = obj.get("instance_nonce")?.as_str()?;
    if !owner_token_re().is_match(instance_nonce) {
        return None;
    }
    Some(OwnershipRecord {
        schema_version: OWNERSHIP_SCHEMA_VERSION,
        protocol: OWNERSHIP_PROTOCOL.to_string(),
        profile_fingerprint: profile_fingerprint.to_string(),
        pid: pid as u32,
        start_identity: start_identity.to_string(),
        port: port as u16,
        version: version.to_string(),
        instance_nonce: instance_nonce.to_string(),
    })
}

/// The path to the ownership record inside the state directory.
pub fn ownership_record_path(state_dir: &Path) -> PathBuf {
    state_dir.join(OWNERSHIP_FILENAME)
}

/// Read, but never repair or delete, the runtime-owned record.
///
/// A malformed, linked, oversized, future-schema, or concurrently-replaced
/// record is untrusted (returns [`OwnershipRecordLoad::Invalid`]). A missing
/// file returns [`OwnershipRecordLoad::Missing`] — the handoff barrier.
pub fn load_ownership_record(state_dir: &Path) -> OwnershipRecordLoad {
    let path = ownership_record_path(state_dir);
    let before = match std::fs::symlink_metadata(&path) {
        Ok(m) => {
            if !m.is_file() || m.file_type().is_symlink() {
                return OwnershipRecordLoad::Invalid;
            }
            if m.len() > RECORD_MAX_BYTES {
                return OwnershipRecordLoad::Invalid;
            }
            m
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => return OwnershipRecordLoad::Missing,
        Err(_) => return OwnershipRecordLoad::Invalid,
    };
    let raw = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return OwnershipRecordLoad::Missing,
        Err(_) => return OwnershipRecordLoad::Invalid,
    };
    // Re-stat after the read to detect a concurrent replacement.
    let after = match std::fs::symlink_metadata(&path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return OwnershipRecordLoad::Missing,
        Err(_) => return OwnershipRecordLoad::Invalid,
    };
    let before_mtime = before
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    let after_mtime = after
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok());
    if before.len() != after.len() || before_mtime != after_mtime {
        return OwnershipRecordLoad::Invalid;
    }
    let parsed: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return OwnershipRecordLoad::Invalid,
    };
    match parse_ownership_record(&parsed) {
        Some(record) => OwnershipRecordLoad::Valid(record),
        None => OwnershipRecordLoad::Invalid,
    }
}

/// Generate a fresh 32-byte base64url instance nonce.
pub fn create_instance_nonce() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    base64_url_encode(&bytes)
}

fn base64_url_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Canonical sorted-ASCII JSON for the ownership HMAC, matching the Python
/// `json.dumps(..., sort_keys=True, separators=(',', ':'), ensure_ascii=True)`.
fn canonical_sorted_json(value: &serde_json::Value) -> String {
    canonical_value(value)
}

fn canonical_value(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = String::from("{");
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap());
                out.push(':');
                out.push_str(&canonical_value(&map[*k]));
            }
            out.push('}');
            out
        }
        serde_json::Value::Array(items) => {
            let mut out = String::from("[");
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_value(v));
            }
            out.push(']');
            out
        }
        // serde_json already emits ASCII strings compactly.
        other => serde_json::to_string(other).unwrap(),
    }
}

/// Compute the identity HMAC over a canonical payload, keyed by the instance
/// nonce. Returns a 64-hex-char digest.
pub fn identity_proof(nonce: &str, payload: &serde_json::Value) -> String {
    use hmac::{Hmac, Mac};
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(nonce.as_bytes())
        .expect("HMAC accepts any key length");
    mac.update(canonical_sorted_json(payload).as_bytes());
    hex::encode(mac.finalize().into_bytes())
}

/// Constant-time comparison of two 64-hex digests. Returns false if either
/// input is not a 64-hex string.
pub fn safe_hex_equal(left: &str, right: &str) -> bool {
    let re = profile_fingerprint_re();
    if !re.is_match(left) || !re.is_match(right) {
        return false;
    }
    let l = match hex::decode(left) {
        Ok(b) => b,
        Err(_) => return false,
    };
    let r = match hex::decode(right) {
        Ok(b) => b,
        Err(_) => return false,
    };
    // Constant-time compare.
    let mut diff: u8 = 0;
    for (a, b) in l.iter().zip(r.iter()) {
        diff |= a ^ b;
    }
    diff == 0
}

/// True only when two records describe the exact same instance.
pub fn same_ownership_instance(left: &OwnershipRecord, right: &OwnershipRecord) -> bool {
    left.protocol == right.protocol
        && left.profile_fingerprint == right.profile_fingerprint
        && left.pid == right.pid
        && left.start_identity == right.start_identity
        && left.port == right.port
        && left.instance_nonce == right.instance_nonce
}

/// A best-effort record key for budgeting verification attempts per directory.
pub fn ownership_record_key(state_dir: &Path, record: &OwnershipRecord) -> String {
    let dir_key = ownership_directory_key(state_dir);
    format!(
        "[{dir_key},{},{},{},{},{},{},{},{}]",
        record.schema_version,
        record.protocol,
        record.profile_fingerprint,
        record.pid,
        record.start_identity,
        record.port,
        record.version,
        record.instance_nonce,
    )
}

fn ownership_directory_key(dir: &Path) -> String {
    let resolved = dir.to_string_lossy().to_string();
    if cfg!(windows) {
        resolved.to_lowercase()
    } else {
        resolved
    }
}

/// Process-start identity prefixes the Gateway records about itself.
const PROCESS_START_IDENTITY_SCHEMES: &[&str] = &[
    "linux-proc-start-ticks:",
    "windows-creation-filetime:",
    "posix-ps-lstart:",
];

/// True only when the live process at the recorded PID provably started at a
/// different time than the recorded owner (i.e. the OS recycled the PID).
///
/// A null or cross-scheme identity never conflicts; this may only shortcut
/// waiting, never grant authority over a process.
pub fn start_identity_conflicts(recorded: &str, live: Option<&str>) -> bool {
    let Some(live) = live else {
        return false;
    };
    let scheme = PROCESS_START_IDENTITY_SCHEMES
        .iter()
        .copied()
        .find(|prefix| live.starts_with(prefix));
    let Some(scheme) = scheme else {
        return false;
    };
    if !recorded.starts_with(scheme) {
        return false;
    }
    recorded != live
}

/// Best-effort start identity of the live process occupying `pid`, in the same
/// format the Gateway records about itself. Returns `None` on platforms that
/// cannot answer; callers must fail open on `None`.
pub fn process_start_identity(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    #[cfg(target_os = "linux")]
    {
        return linux_proc_stat_start_identity(pid);
    }
    #[cfg(target_os = "windows")]
    {
        return windows_process_start_identity(pid);
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        return posix_process_start_identity(pid);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", unix)))]
    {
        let _ = pid;
        None
    }
}

#[cfg(target_os = "linux")]
fn linux_proc_stat_start_identity(pid: u32) -> Option<String> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // `comm` is parenthesized and may contain spaces or `)`. Fields after the
    // final close-paren begin at field 3; starttime is field 22 → index 19.
    let close_paren = stat.rfind(')')?;
    let fields: Vec<&str> = stat[close_paren + 1..].split_ascii_whitespace().collect();
    let start_ticks = fields.get(19)?;
    if !start_ticks.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("linux-proc-start-ticks:{start_ticks}"))
}

#[cfg(target_os = "windows")]
fn windows_process_start_identity(pid: u32) -> Option<String> {
    // GetProcessTimes creation time as a .NET filetime. We shell out to
    // PowerShell for limited-information access; a denied/missing process
    // yields None and the caller stays on the conservative path.
    let output = std::process::Command::new("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &format!(
                "$ErrorActionPreference='Stop'; (Get-Process -Id {pid}).StartTime.ToFileTime()"
            ),
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !value.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(format!("windows-creation-filetime:{value}"))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn posix_process_start_identity(pid: u32) -> Option<String> {
    let ps = if std::path::Path::new("/bin/ps").exists() {
        "/bin/ps"
    } else {
        "ps"
    };
    let output = std::process::Command::new(ps)
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .env("LC_ALL", "C")
        .env("LANG", "C")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    posix_ps_lstart_identity(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn posix_ps_lstart_identity(stdout: &str) -> Option<String> {
    let value: String = stdout
        .split_ascii_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if value.is_empty() {
        return None;
    }
    Some(format!("posix-ps-lstart:{value}"))
}

/// Whether a PID still occupies a process slot (ESRCH is the only reliable
/// negative across platforms; EPERM still proves occupancy).
pub fn process_may_still_be_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(unix)]
    {
        // SAFETY: kill(pid, 0) is signal 0 — it never delivers a signal, only
        // checks for process existence / permission. This is the POSIX-standard
        // liveness probe and is safe to call.
        let rc = unsafe { libc_kill(pid, 0) };
        if rc == 0 {
            return true;
        }
        // errno == ESRCH (3) means no such process; anything else (EPERM, etc.)
        // means a process occupies the PID.
        let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        err != 3 // ESRCH
    }
    #[cfg(windows)]
    {
        // OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION. A failure other
        // than "invalid parameter" still implies the PID may be live.
        use windows::Win32::Foundation::{CloseHandle, HANDLE};
        use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
        unsafe {
            let handle: HANDLE = match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
                Ok(h) => h,
                Err(_) => {
                    // ERROR_INVALID_PARAMETER (87) on Windows means the PID is
                    // definitely free; anything else is ambiguous → fail open.
                    let err = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    return err != 87;
                }
            };
            let _ = CloseHandle(handle);
            true
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        true
    }
}

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

#[cfg(unix)]
#[allow(non_upper_case_globals)]
unsafe fn libc_kill(pid: u32, sig: i32) -> i32 {
    unsafe { kill(pid as i32, sig) }
}

// ---------------------------------------------------------------------------
// Verification coordinator (desktop-gateway-ownership-verification.ts)
// ---------------------------------------------------------------------------

/// Options for the ownership verification coordinator.
#[derive(Debug, Clone)]
pub struct VerificationOptions {
    pub identity_ready_timeout: Duration,
    pub poll_interval: Duration,
    pub challenge_timeout: Duration,
    pub max_record_budgets_per_directory: usize,
}

impl Default for VerificationOptions {
    fn default() -> Self {
        Self {
            identity_ready_timeout: Duration::from_secs(45),
            poll_interval: Duration::from_millis(250),
            challenge_timeout: Duration::from_millis(750),
            max_record_budgets_per_directory: 8,
        }
    }
}

/// Coordinates readiness verification for ownership records.
///
/// Every exact record instance receives one process-local readiness budget.
/// Concurrent callers share one poll, while later sequential callers still
/// perform a fresh identity challenge and liveness check without resetting the
/// deadline. Verification results never grant or cache shutdown authority.
pub struct OwnershipVerificationCoordinator {
    states: Mutex<HashMap<String, VerificationState>>,
    budget_keys_by_directory: Mutex<HashMap<String, Vec<String>>>,
    options: VerificationOptions,
}

#[derive(Debug, Clone)]
struct VerificationState {
    deadline: Instant,
    in_flight: Option<Arc<tokio::sync::Mutex<()>>>,
}

impl OwnershipVerificationCoordinator {
    /// Create a new coordinator with the given options.
    pub fn new(options: VerificationOptions) -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            budget_keys_by_directory: Mutex::new(HashMap::new()),
            options,
        }
    }

    /// Verify a record is ready, polling until the per-record budget expires.
    ///
    /// Always challenges first, including after the budget expires, so a later
    /// startup phase can recover an orphan that became ready after the original
    /// poll — without granting authority from a cached result.
    pub async fn verify_when_ready(&self, state_dir: &Path, record: &OwnershipRecord) -> bool {
        let key = ownership_record_key(state_dir, record);
        let deadline = self.state_deadline(state_dir, &key);
        let mut start_identity_checked = false;
        loop {
            if challenge_identity(record, self.options.challenge_timeout).await {
                return true;
            }
            match load_ownership_record(state_dir) {
                OwnershipRecordLoad::Valid(current)
                    if ownership_record_key(state_dir, &current) == key
                        && process_may_still_be_alive(record.pid) =>
                {
                    if !start_identity_checked {
                        start_identity_checked = true;
                        let live = process_start_identity(record.pid);
                        if start_identity_conflicts(&record.start_identity, live.as_deref()) {
                            return false;
                        }
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        return false;
                    }
                    let sleep = self.options.poll_interval.min(deadline.duration_since(now));
                    tokio::time::sleep(sleep).await;
                }
                _ => return false,
            }
        }
    }

    fn state_deadline(&self, state_dir: &Path, key: &str) -> Instant {
        let dir_key = ownership_directory_key(state_dir);
        let mut states = self.states.lock();
        if let Some(state) = states.get(key) {
            return state.deadline;
        }
        let mut budgets = self.budget_keys_by_directory.lock();
        let entry = budgets.entry(dir_key).or_default();
        let deadline = if entry.len() >= self.options.max_record_budgets_per_directory {
            // Abnormal churn must not create unbounded memory or a chain of
            // fresh 45-second waits. Give additional unknown records an
            // already-expired budget.
            Instant::now()
        } else {
            Instant::now() + self.options.identity_ready_timeout
        };
        entry.push(key.to_string());
        states.insert(
            key.to_string(),
            VerificationState {
                deadline,
                in_flight: None,
            },
        );
        deadline
    }
}

/// Probe the loopback identity endpoint to prove the listener is the exact
/// process named by the record. A 200 health check, PID match, or record alone
/// is never enough — the listener must HMAC-sign a random challenge with the
/// instance nonce.
pub async fn challenge_identity(record: &OwnershipRecord, timeout: Duration) -> bool {
    let challenge = create_instance_nonce();
    if !owner_token_re().is_match(&challenge) {
        return false;
    }
    let url = format!("http://127.0.0.1:{}/api/desktop/identity", record.port);
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .unwrap_or_default();
    let body = serde_json::json!({ "challenge": challenge });
    let response = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(_) => return false,
    };
    if !response.status().is_success() {
        return false;
    }
    let identity: serde_json::Value = match response.json().await {
        Ok(v) => v,
        Err(_) => return false,
    };
    identity_matches_record(&identity, record, &challenge)
}

fn identity_matches_record(
    identity: &serde_json::Value,
    record: &OwnershipRecord,
    challenge: &str,
) -> bool {
    let obj = match identity.as_object() {
        Some(o) => o,
        None => return false,
    };
    // The identity carries every record field plus challenge + proof.
    let id_challenge = match obj.get("challenge").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => return false,
    };
    if id_challenge != challenge {
        return false;
    }
    let proof = match obj.get("proof").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return false,
    };
    // Every shared field must agree.
    let fields = [
        (
            "schema_version",
            serde_json::Value::from(record.schema_version),
        ),
        (
            "protocol",
            serde_json::Value::from(record.protocol.as_str()),
        ),
        (
            "profile_fingerprint",
            serde_json::Value::from(record.profile_fingerprint.as_str()),
        ),
        ("pid", serde_json::Value::from(record.pid)),
        (
            "start_identity",
            serde_json::Value::from(record.start_identity.as_str()),
        ),
        ("port", serde_json::Value::from(record.port)),
        ("version", serde_json::Value::from(record.version.as_str())),
    ];
    for (k, expected) in fields {
        if obj.get(k) != Some(&expected) {
            return false;
        }
    }
    // Reconstruct the unsigned payload and verify the HMAC.
    let mut unsigned = identity.clone();
    if let Some(map) = unsigned.as_object_mut() {
        map.remove("proof");
    }
    let expected = identity_proof(&record.instance_nonce, &unsigned);
    safe_hex_equal(proof, &expected)
}

/// Wait for the ownership record to disappear — the cross-process handoff
/// barrier. A different valid owner is never treated as ours to replace.
pub async fn wait_for_ownership_release(
    state_dir: &Path,
    record: &OwnershipRecord,
    timeout: Duration,
    poll_interval: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match load_ownership_record(state_dir) {
            OwnershipRecordLoad::Missing => return true,
            OwnershipRecordLoad::Valid(current) if !same_ownership_instance(&current, record) => {
                return false;
            }
            _ => {}
        }
        let now = Instant::now();
        if now >= deadline {
            return false;
        }
        let sleep = poll_interval.min(deadline.duration_since(now));
        tokio::time::sleep(sleep).await;
    }
}

// ---------------------------------------------------------------------------
// Process drain (gateway-lifecycle.ts)
// ---------------------------------------------------------------------------

/// Whether the lifecycle currently allows spawning a new owned process.
pub fn lifecycle_allows_process_spawn(
    lifecycle_closing: bool,
    profile_writer_admission_closed: bool,
    live_owned_process_count: usize,
) -> bool {
    !lifecycle_closing && !profile_writer_admission_closed && live_owned_process_count == 0
}

/// Stop the current process and join every process that remains owned by the
/// lifecycle, including children whose stop was initiated by an earlier flow.
///
/// A bounded retry closes the small race where a previously-started async flow
/// publishes its child while an earlier snapshot is being awaited. Exhaustion
/// fails closed: callers must not continue while any owned process is live.
pub async fn stop_and_join_lifecycle_processes<T, F, G, H, I>(
    options: LifecycleProcessDrainOptions<T, F, G, H, I>,
) -> bool
where
    T: PartialEq + Eq + Send + Clone + 'static,
    F: Fn() -> Option<T> + Send + 'static,
    G: Fn(&T) + Send + 'static,
    H: Fn() -> Vec<T> + Send + 'static,
    I: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>> + Send + 'static,
{
    let max_rounds = options.max_rounds.unwrap_or(8);
    for _ in 0..max_rounds {
        if let Some(current) = (options.current_process)() {
            (options.stop_current_process)(&current);
        }
        let processes: Vec<T> = (options.live_processes)();
        // De-duplicate owned processes by value equality.
        let mut unique: Vec<T> = Vec::with_capacity(processes.len());
        for p in processes {
            if !unique.contains(&p) {
                unique.push(p);
            }
        }
        if unique.is_empty() {
            return (options.current_process)().is_none();
        }
        let futures: Vec<_> = unique
            .into_iter()
            .map(|p| (options.wait_for_exit)(p))
            .collect();
        let results = futures::future::join_all(futures).await;
        if !results.into_iter().all(|exited| exited) {
            return false;
        }
    }
    (options.current_process)().is_none() && (options.live_processes)().is_empty()
}

/// Options for [`stop_and_join_lifecycle_processes`].
pub struct LifecycleProcessDrainOptions<T, F, G, H, I>
where
    T: PartialEq + Eq + Send + Clone + 'static,
    F: Fn() -> Option<T> + Send + 'static,
    G: Fn(&T) + Send + 'static,
    H: Fn() -> Vec<T> + Send + 'static,
    I: Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>> + Send + 'static,
{
    pub current_process: F,
    pub stop_current_process: G,
    pub live_processes: H,
    pub wait_for_exit: I,
    pub max_rounds: Option<usize>,
    pub _marker: std::marker::PhantomData<T>,
}

// ---------------------------------------------------------------------------
// Cleanup (desktop-cleanup.ts)
// ---------------------------------------------------------------------------

/// The scope of a cleanup operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DesktopCleanupMode {
    ResetCurrentSettings,
    DeleteCurrentProfile,
    DeleteAllUserData,
}

impl DesktopCleanupMode {
    /// Parse a cleanup mode from an untrusted string.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "reset-current-settings" => Some(Self::ResetCurrentSettings),
            "delete-current-profile" => Some(Self::DeleteCurrentProfile),
            "delete-all-user-data" => Some(Self::DeleteAllUserData),
            _ => None,
        }
    }

    /// The kebab-case string form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ResetCurrentSettings => "reset-current-settings",
            Self::DeleteCurrentProfile => "delete-current-profile",
            Self::DeleteAllUserData => "delete-all-user-data",
        }
    }
}

/// The outcome of a cleanup operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DesktopCleanupOutcome {
    Ready,
    Blocked,
    Complete,
    Partial,
}

/// A single item in a cleanup inventory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopCleanupItem {
    pub kind: String,
    pub path: String,
    pub exists: bool,
    pub identity: Option<String>,
}

/// A cleanup report returned by the cleanup command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopCleanupReport {
    pub schema_version: u32,
    pub outcome: DesktopCleanupOutcome,
    pub stable_code: String,
    pub mode: DesktopCleanupMode,
    pub items: Vec<DesktopCleanupItem>,
    pub transaction_id: String,
    pub revision: u64,
    pub scope_fingerprint: String,
}

/// True when every item in the report is contained under `user_data` (no
/// path escape via `..`, symlinks, or absolute paths outside the root).
pub fn cleanup_scope_is_contained(report: &DesktopCleanupReport, user_data: &Path) -> bool {
    let root = user_data.to_path_buf();
    report.items.iter().all(|item| {
        let item_path = Path::new(&item.path);
        let resolved = std::fs::canonicalize(item_path).unwrap_or_else(|_| item_path.to_path_buf());
        match resolved.strip_prefix(&root) {
            Ok(rel) => !rel.starts_with(".."),
            Err(_) => {
                // Not under root at all. Containment is decided by whether the
                // (unresolved) path textually starts with the root and has no
                // `..` escape segment — a last-resort signal mirroring the
                // Electron `relative()` check.
                let rel = item_path
                    .strip_prefix(&root)
                    .unwrap_or(item_path)
                    .to_string_lossy()
                    .to_string();
                !rel.starts_with("..") && item_path.starts_with(&root)
            }
        }
    })
}

/// True when two reports describe the same cleanup scope (same mode, same
/// contained item set).
pub fn same_cleanup_scope(
    displayed: &DesktopCleanupReport,
    refreshed: &DesktopCleanupReport,
    user_data: &Path,
) -> bool {
    if displayed.mode != refreshed.mode {
        return false;
    }
    if !cleanup_scope_is_contained(displayed, user_data)
        || !cleanup_scope_is_contained(refreshed, user_data)
    {
        return false;
    }
    let signature = |report: &DesktopCleanupReport| -> Vec<String> {
        let mut sig: Vec<String> = report
            .items
            .iter()
            .map(|item| format!("{}\u{0}{}", item.kind, item.path))
            .collect();
        sig.sort();
        sig
    };
    signature(displayed) == signature(refreshed)
}

// ---------------------------------------------------------------------------
// Temp cleanup on quit
// ---------------------------------------------------------------------------

/// Best-effort removal of temp files the desktop shell created during this run.
///
/// Walks the OS temp dir for entries matching the app's prefix and removes
/// them. Errors are logged and never propagated — a leftover temp file must
/// not block shutdown.
pub fn cleanup_temp_files() {
    let Some(tmp) = std::env::temp_dir().canonicalize().ok() else {
        return;
    };
    let entries = match std::fs::read_dir(&tmp) {
        Ok(rd) => rd,
        Err(e) => {
            tracing::debug!(error = %e, "could not read temp dir for cleanup");
            return;
        }
    };
    let prefixes = ["opensquilla-", "osq-update-", "osq-secret-"];
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let name = match entry.file_name().to_str() {
            Some(n) => n.to_string(),
            None => continue,
        };
        if !prefixes.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        let path = entry.path();
        let result = if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        };
        if result.is_ok() {
            removed += 1;
        } else if let Err(e) = result {
            tracing::debug!(path = %path.display(), error = %e, "could not remove temp entry");
        }
    }
    if removed > 0 {
        tracing::info!(removed, "cleaned up desktop temp files");
    }
}

/// Save the persistent secret-storage key so secrets survive restarts.
///
/// This is the lifecycle-facing hook that flushes the in-memory store key to
/// disk during graceful shutdown; see [`storage::SecretStore`].
pub fn persist_secret_store_key(store: &storage::SecretStore, config_dir: &Path) {
    if let Err(e) = store.flush_key(config_dir) {
        tracing::warn!(error = %e, "could not persist secret-store key");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_record() {
        let json = serde_json::json!({
            "schema_version": 1,
            "protocol": OWNERSHIP_PROTOCOL,
            "profile_fingerprint": "a".repeat(64),
            "pid": 1234,
            "start_identity": "linux-proc-start-ticks:999",
            "port": 8080,
            "version": "0.5.2",
            "instance_nonce": "n".repeat(32),
        });
        let record = parse_ownership_record(&json).expect("valid record");
        assert_eq!(record.pid, 1234);
        assert_eq!(record.port, 8080);
    }

    #[test]
    fn rejects_bad_schema() {
        let mut json = serde_json::json!({
            "schema_version": 99,
            "protocol": OWNERSHIP_PROTOCOL,
            "profile_fingerprint": "a".repeat(64),
            "pid": 1,
            "start_identity": "x",
            "port": 1,
            "version": "0.1.0",
            "instance_nonce": "n".repeat(32),
        });
        assert!(parse_ownership_record(&json).is_none());
        // Fix schema, break protocol.
        json["schema_version"] = serde_json::json!(1);
        json["protocol"] = serde_json::json!("wrong");
        assert!(parse_ownership_record(&json).is_none());
    }

    #[test]
    fn rejects_bad_port_and_pid() {
        let mut json = serde_json::json!({
            "schema_version": 1,
            "protocol": OWNERSHIP_PROTOCOL,
            "profile_fingerprint": "a".repeat(64),
            "pid": 1,
            "start_identity": "x",
            "port": 1,
            "version": "0.1.0",
            "instance_nonce": "n".repeat(32),
        });
        json["port"] = serde_json::json!(0);
        assert!(parse_ownership_record(&json).is_none());
        json["port"] = serde_json::json!(70000);
        assert!(parse_ownership_record(&json).is_none());
        json["port"] = serde_json::json!(8080);
        json["pid"] = serde_json::json!(0);
        assert!(parse_ownership_record(&json).is_none());
    }

    #[test]
    fn safe_hex_equal_is_constant_time_correct() {
        let a = "a".repeat(64);
        let b = "a".repeat(64);
        assert!(safe_hex_equal(&a, &b));
        let mut c = "a".repeat(64);
        c.replace_range(0..1, "b");
        assert!(!safe_hex_equal(&a, &c));
        assert!(!safe_hex_equal("nothex", &a));
    }

    #[test]
    fn identity_proof_is_deterministic() {
        let nonce = "k".repeat(32);
        let payload = serde_json::json!({"b": 1, "a": "x"});
        let p1 = identity_proof(&nonce, &payload);
        let p2 = identity_proof(&nonce, &payload);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 64);
    }

    #[test]
    fn canonical_sorted_json_sorts_keys() {
        let v = serde_json::json!({"b": 1, "a": 2, "c": [3, 1]});
        assert_eq!(canonical_sorted_json(&v), r#"{"a":2,"b":1,"c":[3,1]}"#);
    }

    #[test]
    fn cleanup_mode_roundtrips() {
        for m in [
            DesktopCleanupMode::ResetCurrentSettings,
            DesktopCleanupMode::DeleteCurrentProfile,
            DesktopCleanupMode::DeleteAllUserData,
        ] {
            assert_eq!(DesktopCleanupMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(DesktopCleanupMode::parse("bogus"), None);
    }

    #[test]
    fn start_identity_conflict_logic() {
        // Same scheme, different value → conflict.
        assert!(start_identity_conflicts(
            "linux-proc-start-ticks:100",
            Some("linux-proc-start-ticks:200"),
        ));
        // Same value → no conflict.
        assert!(!start_identity_conflicts(
            "linux-proc-start-ticks:100",
            Some("linux-proc-start-ticks:100"),
        ));
        // Cross-scheme → never conflict.
        assert!(!start_identity_conflicts(
            "linux-proc-start-ticks:100",
            Some("windows-creation-filetime:999"),
        ));
        // Null live → never conflict.
        assert!(!start_identity_conflicts(
            "linux-proc-start-ticks:100",
            None,
        ));
    }

    #[test]
    fn primary_profile_paths_are_stable() {
        let ud = Path::new("/tmp/userdata");
        let p = primary_profile_paths(ud);
        assert_eq!(p.kind, ProfileKind::Primary);
        assert!(p.recovery_id.is_none());
        assert!(p.home.ends_with("opensquilla"));
        assert!(p.credential_path.ends_with("desktop-credential.json"));
    }

    #[test]
    fn recovery_id_validation() {
        assert!(is_recovery_profile_id(
            "550e8400-e29b-41d4-a716-446655440000"
        ));
        assert!(!is_recovery_profile_id("not-a-uuid"));
        assert!(!is_recovery_profile_id(
            "550e8400-e29b-31d4-a716-446655440000" // v3, not v4
        ));
    }
}
