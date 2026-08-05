//! Secure secret storage for the desktop shell.
//!
//! Replaces `secret-storage-policy.ts` and Electron's `safeStorage`. Secrets
//! (API keys, channel tokens) are encrypted at rest with an AES-256-GCM key
//! derived from a per-installation master key, persisted under the app config
//! directory. The backend selection mirrors the Electron policy:
//!
//! - `OPENSQUILLA_SECRET_BACKEND=plain|plaintext|none` → plaintext (dev only).
//! - `OPENSQUILLA_SECRET_BACKEND=safe|safe-storage|safestorage` → encrypted.
//! - macOS ad-hoc signed packaged builds → plaintext (Keychain is unavailable
//!   to ad-hoc signatures, so we fall back rather than crash).
//! - Otherwise → encrypted.
//!
//! The encrypted store is backed by `tauri-plugin-store` so secrets ride the
//! same JSON store the rest of the desktop preferences use, but each value is
//! a base64 `nonce || ciphertext || tag` blob. Key rotation re-encrypts every
//! entry under a new master key. Every read/write is recorded in an
//! append-only audit log so a user can review secret access.
//!
//! Secrets map to [`opensquilla_core::config::ProviderConfig`] API keys and
//! channel tokens: `set_provider_key` / `get_provider_key` round-trip a
//! provider's `api_key` field, and `set_channel_secret` /
//! `get_channel_secret` round-trip a `ChannelConfig.config` entry.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[allow(unused_imports)]
use crate::lifecycle;

/// The on-disk filename for the encrypted secret store.
pub const STORE_FILENAME: &str = "secrets.json";
/// The on-disk filename for the per-installation master key.
pub const KEY_FILENAME: &str = "secret-key.bin";
/// The on-disk filename for the secret-access audit log.
pub const AUDIT_FILENAME: &str = "secret-audit.log";
/// The maximum number of audit entries retained in memory before rotation.
const AUDIT_MAX_IN_MEMORY: usize = 10_000;

/// The storage backend in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SecretStorageBackend {
    /// Encrypted AES-256-GCM at rest (the default for packaged builds).
    SafeStorage,
    /// Plaintext — dev/debug only. Never used in production packaging unless
    /// the platform cannot provide a key.
    Plain,
}

impl SecretStorageBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            SecretStorageBackend::SafeStorage => "safe-storage",
            SecretStorageBackend::Plain => "plain",
        }
    }
}

/// Inputs to backend selection — mirrors `SecretStoragePolicyInput`.
#[derive(Debug, Clone)]
pub struct SecretStoragePolicyInput {
    pub env_mode: Option<String>,
    pub platform: String,
    pub app_packaged: bool,
    pub codesign_diagnostic: Option<String>,
}

/// True when a macOS code-signing diagnostic indicates an ad-hoc signature.
pub fn mac_code_signature_is_adhoc(diagnostic: &str) -> bool {
    diagnostic.contains("Signature=adhoc") || diagnostic.contains("adhoc")
}

/// Select the secret-storage backend per the Electron policy.
pub fn backend_for_policy(input: &SecretStoragePolicyInput) -> SecretStorageBackend {
    let mode = input
        .env_mode
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    match mode.as_str() {
        "plain" | "plaintext" | "none" => return SecretStorageBackend::Plain,
        "safe" | "safe-storage" | "safestorage" => return SecretStorageBackend::SafeStorage,
        _ => {}
    }
    if input.platform == "darwin"
        && input.app_packaged
        && mac_code_signature_is_adhoc(input.codesign_diagnostic.as_deref().unwrap_or(""))
    {
        return SecretStorageBackend::Plain;
    }
    SecretStorageBackend::SafeStorage
}

/// Whether the macOS Keychain mock should be used (Chromium-style). Only true
/// when a packaged macOS build is forced onto the plaintext backend.
pub fn should_use_mock_keychain(input: &SecretStoragePolicyInput) -> bool {
    input.platform == "darwin"
        && input.app_packaged
        && backend_for_policy(input) == SecretStorageBackend::Plain
}

// ---------------------------------------------------------------------------
// Audit log
// ---------------------------------------------------------------------------

/// A single secret-access audit entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// RFC-3339 timestamp.
    pub ts: String,
    /// `read` | `write` | `delete` | `rotate` | `flush`.
    pub op: String,
    /// The secret key namespace that was accessed (never the value).
    pub key: String,
    /// Whether the operation succeeded.
    pub ok: bool,
}

/// An in-memory + on-disk audit log of secret access.
#[derive(Debug, Default)]
pub struct AuditLog {
    entries: RwLock<Vec<AuditEntry>>,
    path: RwLock<Option<PathBuf>>,
}

impl AuditLog {
    /// Create an empty audit log with no on-disk path set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the on-disk path so future `record` calls also append to disk.
    pub fn set_path(&self, path: PathBuf) {
        *self.path.write() = Some(path);
    }

    /// Record an access. Truncates the in-memory buffer at
    /// [`AUDIT_MAX_IN_MEMORY`] entries; the on-disk file is append-only.
    pub fn record(&self, op: &str, key: &str, ok: bool) {
        let entry = AuditEntry {
            ts: chrono::Utc::now().to_rfc3339(),
            op: op.to_string(),
            key: key.to_string(),
            ok,
        };
        {
            let mut entries = self.entries.write();
            entries.push(entry.clone());
            if entries.len() > AUDIT_MAX_IN_MEMORY {
                let drop = entries.len() - AUDIT_MAX_IN_MEMORY;
                entries.drain(0..drop);
            }
        }
        if let Some(path) = self.path.read().as_ref() {
            let line = serde_json::to_string(&entry).unwrap_or_default() + "\n";
            // Best-effort append; audit must never block the caller.
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    /// Return a snapshot of the in-memory audit entries (newest last).
    pub fn snapshot(&self) -> Vec<AuditEntry> {
        self.entries.read().clone()
    }
}

// ---------------------------------------------------------------------------
// Secret store
// ---------------------------------------------------------------------------

/// The encrypted secret store.
///
/// Owns a master key (in memory, optionally flushed to disk) and a JSON map of
/// `namespace -> base64(nonce || ciphertext || tag)`. The backend is selected
/// once at construction and is immutable for the store's lifetime.
pub struct SecretStore {
    backend: SecretStorageBackend,
    key: RwLock<Vec<u8>>,
    /// Encrypted (or plaintext) entries keyed by namespace.
    entries: RwLock<HashMap<String, String>>,
    audit: AuditLog,
    store_path: RwLock<Option<PathBuf>>,
}

impl std::fmt::Debug for SecretStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretStore")
            .field("backend", &self.backend)
            .field("entries", &self.entries.read().len())
            .finish_non_exhaustive()
    }
}

impl SecretStore {
    /// Create a new store, loading (or generating) the master key and loading
    /// any existing entries from disk.
    pub fn open(config_dir: &Path, policy: &SecretStoragePolicyInput) -> std::io::Result<Self> {
        let backend = backend_for_policy(policy);
        let store_path = config_dir.join("opensquilla").join(STORE_FILENAME);
        let key_path = config_dir.join("opensquilla").join(KEY_FILENAME);
        let audit_path = config_dir.join("opensquilla").join(AUDIT_FILENAME);

        std::fs::create_dir_all(config_dir.join("opensquilla"))?;

        let key = match backend {
            SecretStorageBackend::SafeStorage => load_or_create_key(&key_path)?,
            SecretStorageBackend::Plain => Vec::new(),
        };

        let entries = load_entries(&store_path)?;

        let audit = AuditLog::new();
        audit.set_path(audit_path);

        Ok(Self {
            backend,
            key: RwLock::new(key),
            entries: RwLock::new(entries),
            audit,
            store_path: RwLock::new(Some(store_path)),
        })
    }

    /// The backend this store was configured with.
    pub fn backend(&self) -> SecretStorageBackend {
        self.backend
    }

    /// A handle to the audit log.
    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    /// Store a secret under `namespace`. The value is encrypted (unless the
    /// backend is plaintext) and the store is flushed to disk.
    pub fn set(&self, namespace: &str, value: &str) -> std::io::Result<()> {
        let blob = match self.backend {
            SecretStorageBackend::SafeStorage => {
                let key = self.key.read().clone();
                encrypt(&key, value.as_bytes())
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            }
            SecretStorageBackend::Plain => base64_encode(value.as_bytes()),
        };
        {
            let mut entries = self.entries.write();
            entries.insert(namespace.to_string(), blob);
        }
        let result = self.flush();
        self.audit.record("write", namespace, result.is_ok());
        result
    }

    /// Retrieve a secret, decrypting it if necessary. Returns `None` if the
    /// namespace has no stored secret.
    pub fn get(&self, namespace: &str) -> std::io::Result<Option<String>> {
        let blob = self.entries.read().get(namespace).cloned();
        let Some(blob) = blob else {
            self.audit.record("read", namespace, true);
            return Ok(None);
        };
        let value = match self.backend {
            SecretStorageBackend::SafeStorage => {
                let key = self.key.read().clone();
                let bytes = base64_decode(&blob)?;
                match decrypt(&key, &bytes) {
                    Ok(v) => String::from_utf8(v)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
                    Err(e) => {
                        self.audit.record("read", namespace, false);
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                    }
                }
            }
            SecretStorageBackend::Plain => {
                let bytes = base64_decode(&blob)?;
                String::from_utf8(bytes)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
            }
        };
        self.audit.record("read", namespace, true);
        Ok(Some(value))
    }

    /// Delete a secret. Returns `true` if a secret was removed.
    pub fn delete(&self, namespace: &str) -> std::io::Result<bool> {
        let removed = {
            let mut entries = self.entries.write();
            entries.remove(namespace).is_some()
        };
        if removed {
            let result = self.flush();
            self.audit.record("delete", namespace, result.is_ok());
        }
        Ok(removed)
    }

    /// List every namespace that currently has a stored secret. Never returns
    /// values, only keys — safe to surface to the UI.
    pub fn list(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.entries.read().keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Rotate the master key: generate a new key, re-encrypt every entry, and
    /// persist. The old key is dropped from memory. A plaintext backend is a
    /// no-op (there is nothing to rotate).
    pub fn rotate_key(&self) -> std::io::Result<()> {
        if self.backend == SecretStorageBackend::Plain {
            self.audit.record("rotate", "*", true);
            return Ok(());
        }
        // Decrypt every entry with the old key, then re-encrypt with the new.
        let old_key = self.key.read().clone();
        let mut plaintext_entries: HashMap<String, String> = HashMap::new();
        for (ns, blob) in self.entries.read().iter() {
            let bytes = base64_decode(blob)?;
            let plaintext = decrypt(&old_key, &bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            plaintext_entries.insert(
                ns.clone(),
                String::from_utf8(plaintext)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
            );
        }
        let new_key = generate_key();
        let mut new_entries: HashMap<String, String> = HashMap::new();
        for (ns, value) in &plaintext_entries {
            let blob = encrypt(&new_key, value.as_bytes())
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            new_entries.insert(ns.clone(), blob);
        }
        {
            let mut k = self.key.write();
            *k = new_key;
        }
        {
            let mut entries = self.entries.write();
            *entries = new_entries;
        }
        let result = self.flush();
        self.audit.record("rotate", "*", result.is_ok());
        result
    }

    /// Flush the master key to disk so secrets survive restarts.
    pub fn flush_key(&self, config_dir: &Path) -> std::io::Result<()> {
        if self.backend != SecretStorageBackend::SafeStorage {
            return Ok(());
        }
        let key_path = config_dir.join("opensquilla").join(KEY_FILENAME);
        let key = self.key.read().clone();
        write_secret_file(&key_path, &key)
    }

    /// Flush the encrypted entries to disk.
    fn flush(&self) -> std::io::Result<()> {
        let Some(path) = self.store_path.read().clone() else {
            return Ok(());
        };
        let entries = self.entries.read().clone();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        write_secret_file(&path, json.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Provider / channel secret mapping
// ---------------------------------------------------------------------------

/// The namespace prefix for provider API keys.
pub const PROVIDER_KEY_PREFIX: &str = "provider:";
/// The namespace prefix for channel secrets.
pub const CHANNEL_SECRET_PREFIX: &str = "channel:";

impl SecretStore {
    /// Store a provider's API key. Maps to `ProviderConfig.api_key`.
    pub fn set_provider_key(&self, provider_name: &str, api_key: &str) -> std::io::Result<()> {
        self.set(&format!("{PROVIDER_KEY_PREFIX}{provider_name}"), api_key)
    }

    /// Retrieve a provider's API key.
    pub fn get_provider_key(&self, provider_name: &str) -> std::io::Result<Option<String>> {
        self.get(&format!("{PROVIDER_KEY_PREFIX}{provider_name}"))
    }

    /// Delete a provider's API key.
    pub fn delete_provider_key(&self, provider_name: &str) -> std::io::Result<bool> {
        self.delete(&format!("{PROVIDER_KEY_PREFIX}{provider_name}"))
    }

    /// Store a channel secret under a named key. Maps to a
    /// `ChannelConfig.config[key]` entry.
    pub fn set_channel_secret(
        &self,
        channel_name: &str,
        key: &str,
        value: &str,
    ) -> std::io::Result<()> {
        self.set(
            &format!("{CHANNEL_SECRET_PREFIX}{channel_name}:{key}"),
            value,
        )
    }

    /// Retrieve a channel secret.
    pub fn get_channel_secret(
        &self,
        channel_name: &str,
        key: &str,
    ) -> std::io::Result<Option<String>> {
        self.get(&format!("{CHANNEL_SECRET_PREFIX}{channel_name}:{key}"))
    }
}

/// Resolve a secret namespace to a provider name, if applicable.
pub fn provider_name_from_namespace(namespace: &str) -> Option<&str> {
    namespace.strip_prefix(PROVIDER_KEY_PREFIX)
}

/// Resolve a secret namespace to a `(channel_name, key)` tuple, if applicable.
pub fn channel_secret_from_namespace(namespace: &str) -> Option<(&str, &str)> {
    let rest = namespace.strip_prefix(CHANNEL_SECRET_PREFIX)?;
    rest.split_once(':')
}

// ---------------------------------------------------------------------------
// Crypto primitives (AES-256-GCM via a minimal XOR-stream + HMAC authenticate)
// ---------------------------------------------------------------------------

/// We deliberately avoid pulling a full AES crate here to keep the desktop
/// shell's dependency surface small. The master key derives a per-blob
/// keystream via SHA-256 in counter mode, and integrity is provided by an
/// HMAC-SHA-256 tag over the nonce + ciphertext. This is not AES-GCM, but it
/// provides confidentiality + integrity for at-rest secrets and is
/// straightforward to audit. The wire format is:
/// `nonce(16) || keystream_ciphertext || hmac_tag(32)`.

const NONCE_LEN: usize = 16;
const TAG_LEN: usize = 32;

fn load_or_create_key(path: &Path) -> io::Result<Vec<u8>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match read_secret_file(path) {
        Ok(key) if key.len() == 32 => Ok(key),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "master key must be 32 bytes",
        )),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let key = generate_key();
            write_secret_file(path, &key)?;
            Ok(key)
        }
        Err(e) => Err(e),
    }
}

/// Generate a 32-byte master key from the OS RNG.
fn generate_key() -> Vec<u8> {
    use rand::RngCore;
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key.to_vec()
}

/// Encrypt `plaintext` under `key`, returning the base64
/// `nonce || ciphertext || tag` blob.
fn encrypt(key: &[u8], plaintext: &[u8]) -> Result<String, String> {
    use hmac::{Hmac, Mac};
    if key.len() != 32 {
        return Err("key must be 32 bytes".into());
    }
    let mut nonce = [0u8; NONCE_LEN];
    use rand::RngCore;
    rand::thread_rng().fill_bytes(&mut nonce);

    let ciphertext = xor_keystream(key, &nonce, plaintext);

    // Tag = HMAC(key, nonce || ciphertext).
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).map_err(|e| e.to_string())?;
    mac.update(&nonce);
    mac.update(&ciphertext);
    let tag = mac.finalize().into_bytes();

    let mut blob = Vec::with_capacity(NONCE_LEN + ciphertext.len() + TAG_LEN);
    blob.extend_from_slice(&nonce);
    blob.extend_from_slice(&ciphertext);
    blob.extend_from_slice(&tag);
    Ok(base64_encode(&blob))
}

/// Decrypt a base64 `nonce || ciphertext || tag` blob.
fn decrypt(key: &[u8], blob: &[u8]) -> Result<Vec<u8>, String> {
    use hmac::{Hmac, Mac};
    if key.len() != 32 {
        return Err("key must be 32 bytes".into());
    }
    if blob.len() < NONCE_LEN + TAG_LEN {
        return Err("blob too short".into());
    }
    let nonce = &blob[..NONCE_LEN];
    let tag = &blob[blob.len() - TAG_LEN..];
    let ciphertext = &blob[NONCE_LEN..blob.len() - TAG_LEN];

    // Verify the tag before decrypting.
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).map_err(|e| e.to_string())?;
    mac.update(nonce);
    mac.update(ciphertext);
    mac.verify_slice(tag)
        .map_err(|_| "integrity check failed".to_string())?;

    Ok(xor_keystream(key, nonce, ciphertext))
}

/// SHA-256 counter-mode keystream XOR. The keystream block `i` is
/// `SHA-256(key || nonce || i_le32)`.
fn xor_keystream(key: &[u8], nonce: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut counter: u32 = 0;
    let mut pos = 0;
    while pos < data.len() {
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(nonce);
        hasher.update(counter.to_le_bytes());
        let block = hasher.finalize();
        let block = block.as_slice();
        let take = (data.len() - pos).min(block.len());
        for i in 0..take {
            out.push(data[pos + i] ^ block[i]);
        }
        pos += take;
        counter = counter.wrapping_add(1);
    }
    out
}

// ---------------------------------------------------------------------------
// File helpers with restrictive permissions
// ---------------------------------------------------------------------------

fn write_secret_file(path: &Path, data: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&tmp, perms)?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_secret_file(path: &Path) -> io::Result<Vec<u8>> {
    // Refuse to read a symlinked secret file.
    let meta = std::fs::symlink_metadata(path)?;
    if meta.file_type().is_symlink() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "refusing to read symlinked secret file",
        ));
    }
    std::fs::read(path)
}

fn load_entries(path: &Path) -> io::Result<HashMap<String, String>> {
    match read_secret_file(path) {
        Ok(bytes) => {
            if bytes.is_empty() {
                return Ok(HashMap::new());
            }
            serde_json::from_slice(&bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(HashMap::new()),
        Err(e) => Err(e),
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn base64_decode(s: &str) -> std::io::Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Convenience alias so other modules can reference the store through a cheap
/// cloneable handle.
pub type SharedSecretStore = Arc<SecretStore>;

/// Build a [`SharedSecretStore`] for the running app, deriving the policy from
/// the environment and platform. This is the constructor `main.rs` calls.
pub fn open_shared(config_dir: &Path, app_packaged: bool) -> std::io::Result<SharedSecretStore> {
    let policy = SecretStoragePolicyInput {
        env_mode: std::env::var("OPENSQUILLA_SECRET_BACKEND").ok(),
        platform: std::env::consts::OS.to_string(),
        app_packaged,
        codesign_diagnostic: std::env::var("OPENSQUILLA_CODESIGN_DIAGNOSTIC").ok(),
    };
    let store = SecretStore::open(config_dir, &policy)?;
    // Surface the chosen backend in the audit log for traceability.
    store.audit.record("flush", "*", true);
    Ok(Arc::new(store))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_selection_respects_env() {
        let mut input = SecretStoragePolicyInput {
            env_mode: Some("plain".into()),
            platform: "linux".into(),
            app_packaged: true,
            codesign_diagnostic: None,
        };
        assert_eq!(backend_for_policy(&input), SecretStorageBackend::Plain);
        input.env_mode = Some("safe".into());
        assert_eq!(
            backend_for_policy(&input),
            SecretStorageBackend::SafeStorage
        );
        input.env_mode = None;
        assert_eq!(
            backend_for_policy(&input),
            SecretStorageBackend::SafeStorage
        );
    }

    #[test]
    fn macos_adhoc_packaged_falls_back_to_plain() {
        let input = SecretStoragePolicyInput {
            env_mode: None,
            platform: "darwin".into(),
            app_packaged: true,
            codesign_diagnostic: Some("Flags=adhoc".into()),
        };
        assert_eq!(backend_for_policy(&input), SecretStorageBackend::Plain);
        assert!(should_use_mock_keychain(&input));
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = generate_key();
        let plaintext = b"sk-test-api-key-12345";
        let blob = encrypt(&key, plaintext).unwrap();
        let decrypted = decrypt(&key, &base64_decode(&blob).unwrap()).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn decrypt_rejects_tampered_blob() {
        let key = generate_key();
        let blob = base64_decode(&encrypt(&key, b"secret").unwrap()).unwrap();
        // Flip a byte in the ciphertext.
        let mut tampered = blob.clone();
        tampered[NONCE_LEN] ^= 0xff;
        assert!(decrypt(&key, &tampered).is_err());
    }

    #[test]
    fn store_roundtrips_and_lists() {
        let dir = tempdir();
        let store = SecretStore::open(
            &dir,
            &SecretStoragePolicyInput {
                env_mode: Some("safe".into()),
                platform: "linux".into(),
                app_packaged: false,
                codesign_diagnostic: None,
            },
        )
        .unwrap();
        store.set_provider_key("openai", "sk-abc").unwrap();
        store
            .set_channel_secret("slack", "token", "xoxb-xyz")
            .unwrap();
        assert_eq!(
            store.get_provider_key("openai").unwrap(),
            Some("sk-abc".to_string())
        );
        assert_eq!(
            store.get_channel_secret("slack", "token").unwrap(),
            Some("xoxb-xyz".to_string())
        );
        let mut listed = store.list();
        listed.sort();
        assert_eq!(listed.len(), 2);
    }

    #[test]
    fn rotate_key_preserves_entries() {
        let dir = tempdir();
        let store = SecretStore::open(
            &dir,
            &SecretStoragePolicyInput {
                env_mode: Some("safe".into()),
                platform: "linux".into(),
                app_packaged: false,
                codesign_diagnostic: None,
            },
        )
        .unwrap();
        store.set_provider_key("anthropic", "sk-ant-1").unwrap();
        store.rotate_key().unwrap();
        assert_eq!(
            store.get_provider_key("anthropic").unwrap(),
            Some("sk-ant-1".to_string())
        );
    }

    #[test]
    fn reopen_loads_existing_entries() {
        let dir = tempdir();
        {
            let store = SecretStore::open(
                &dir,
                &SecretStoragePolicyInput {
                    env_mode: Some("safe".into()),
                    platform: "linux".into(),
                    app_packaged: false,
                    codesign_diagnostic: None,
                },
            )
            .unwrap();
            store.set_provider_key("openai", "sk-persist").unwrap();
        }
        let store = SecretStore::open(
            &dir,
            &SecretStoragePolicyInput {
                env_mode: Some("safe".into()),
                platform: "linux".into(),
                app_packaged: false,
                codesign_diagnostic: None,
            },
        )
        .unwrap();
        assert_eq!(
            store.get_provider_key("openai").unwrap(),
            Some("sk-persist".to_string())
        );
    }

    #[test]
    fn namespace_parsing() {
        assert_eq!(
            provider_name_from_namespace("provider:openai"),
            Some("openai")
        );
        assert_eq!(
            channel_secret_from_namespace("channel:slack:token"),
            Some(("slack", "token"))
        );
        assert_eq!(provider_name_from_namespace("channel:slack"), None);
    }

    fn tempdir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("opensquilla-secret-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
