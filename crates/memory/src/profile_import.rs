//! External configuration import to memory.
//!
//! Parses external config files (JSON, YAML, TOML) and uses an LLM to extract
//! memory-worthy information, which is then stored as structured memories. The
//! [`ProfileImporter`] supports importing a single file or a whole directory,
//! auto-detecting the config family (Claude Code, OpenClaw, Hermes, or generic)
//! and offering a dry-run [`ImportPlan`] before anything is persisted.
//!
//! A [`ProfileDetector`] scans well-known config locations so a user's existing
//! profiles can be discovered and imported in one call.

use chrono::Utc;
use opensquilla_core::result::CoreResult;
use opensquilla_core::types::MemoryId;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::MemoryStore;
use crate::types::MemoryEntry;

/// A parsed external config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSource {
    /// Absolute or relative path to the file.
    pub path: String,
    /// Detected file kind (`json`, `yaml`, `toml`, `unknown`).
    pub kind: String,
    /// The parsed, normalized JSON value of the config.
    pub parsed: serde_json::Value,
}

/// An extracted memory-worthy fact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedMemory {
    /// A summary sentence for the memory content.
    pub content: String,
    /// Semantic type of the memory (`preference`, `fact`, `pattern`, ...).
    pub memory_type: String,
    /// Importance in `[0,1]`.
    pub importance: f64,
    /// Optional tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional structured payload (e.g. the raw config subset).
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// A trait for LLM-backed extraction of memory-worthy information from config.
///
/// The memory crate deliberately does not depend on the provider crate; any
/// caller (gateway, engine) can implement this trait and inject it into the
/// [`ProfileImporter`]. A deterministic heuristic fallback is provided so the
/// importer still works without an LLM.
#[async_trait::async_trait]
pub trait MemoryExtractor: Send + Sync {
    /// Extract memories from a parsed config file.
    async fn extract(&self, source: &ImportSource) -> CoreResult<Vec<ExtractedMemory>>;

    /// A short label used in logs.
    fn name(&self) -> &str {
        "extractor"
    }
}

/// Deterministic, heuristic extractor used when no LLM is configured.
///
/// Walks the parsed JSON tree and emits a memory for each scalar value whose
/// key looks like a preference or notable fact (`prefer`, `favorite`, `like`,
/// `theme`, `model`, `provider`, ...). It is intentionally simple; prefer an
/// LLM-backed extractor for quality.
pub struct HeuristicExtractor;

#[async_trait::async_trait]
impl MemoryExtractor for HeuristicExtractor {
    async fn extract(&self, source: &ImportSource) -> CoreResult<Vec<ExtractedMemory>> {
        let mut memories = Vec::new();
        collect_scalars(&source.parsed, &source.path, "", &mut memories);
        Ok(memories)
    }
}

/// Configuration for the [`ProfileImporter`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileImporterConfig {
    /// Default importance for imported memories.
    pub default_importance: f64,
    /// Whether to tag imported memories with the source file name.
    pub tag_with_source: bool,
}

impl Default for ProfileImporterConfig {
    fn default() -> Self {
        Self {
            default_importance: 0.4,
            tag_with_source: true,
        }
    }
}

/// Summary of an import operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSummary {
    pub agent_id: Uuid,
    pub files_processed: u64,
    pub memories_created: u64,
    pub failed_files: Vec<String>,
}

// ---------------------------------------------------------------------------
// Config-family detection
// ---------------------------------------------------------------------------

/// The recognized config families.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfigType {
    /// Claude Code (`.claude/settings.json`, `claude_desktop_config.json`, ...).
    Claude,
    /// OpenClaw (`.openclaw/config.*`, ...).
    OpenClaw,
    /// Hermes (`.hermes/config.*`, ...).
    Hermes,
    /// A supported generic config file (JSON/YAML/TOML).
    Generic,
    /// Not a recognized or supported config file.
    Unknown,
}

impl ConfigType {
    pub fn label(&self) -> &'static str {
        match self {
            ConfigType::Claude => "claude",
            ConfigType::OpenClaw => "openclaw",
            ConfigType::Hermes => "hermes",
            ConfigType::Generic => "generic",
            ConfigType::Unknown => "unknown",
        }
    }
}

/// Auto-detect the config family from a file path.
pub fn detect_config_type(path: &str) -> ConfigType {
    let lower = path.to_lowercase();
    let file_name = std::path::Path::new(path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if lower.contains(".claude")
        || lower.contains("claude_desktop")
        || lower.contains("claude-code")
    {
        return ConfigType::Claude;
    }
    if lower.contains(".openclaw") || lower.contains("openclaw") {
        return ConfigType::OpenClaw;
    }
    if lower.contains(".hermes") || lower.contains("hermes") {
        return ConfigType::Hermes;
    }
    if file_name.contains("claude") {
        return ConfigType::Claude;
    }
    if file_name.contains("openclaw") {
        return ConfigType::OpenClaw;
    }
    if file_name.contains("hermes") {
        return ConfigType::Hermes;
    }
    if is_supported(path) {
        return ConfigType::Generic;
    }
    ConfigType::Unknown
}

/// A config file discovered on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedProfile {
    pub path: String,
    pub config_type: ConfigType,
}

/// Scans well-known config locations for importable profiles.
#[derive(Debug, Clone, Default)]
pub struct ProfileDetector {
    pub search_paths: Vec<String>,
}

impl ProfileDetector {
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        let appdata = std::env::var("APPDATA").unwrap_or_default();
        let xdg = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| format!("{}/.config", home));

        let mut paths = Vec::new();
        // Claude Code
        paths.push(format!("{}/.claude/settings.json", home));
        paths.push(format!("{}/.claude.json", home));
        paths.push(format!("{}/claude/claude_desktop_config.json", appdata));
        paths.push(format!("{}/claude-code/settings.json", appdata));
        // OpenClaw
        paths.push(format!("{}/.openclaw/config.json", home));
        paths.push(format!("{}/.openclaw/config.yaml", home));
        paths.push(format!("{}/openclaw/config.toml", xdg));
        // Hermes
        paths.push(format!("{}/.hermes/config.yaml", home));
        paths.push(format!("{}/.hermes/config.toml", home));
        // Generic
        paths.push(format!("{}/opensus/config.toml", xdg));
        Self {
            search_paths: paths,
        }
    }

    /// Override the candidate search paths.
    pub fn with_search_paths(paths: Vec<String>) -> Self {
        Self {
            search_paths: paths,
        }
    }

    /// Return the config files that actually exist on disk.
    pub fn scan(&self) -> Vec<DetectedProfile> {
        self.search_paths
            .iter()
            .filter(|p| std::path::Path::new(p).exists())
            .map(|p| DetectedProfile {
                path: p.clone(),
                config_type: detect_config_type(p),
            })
            .collect()
    }

    /// Scan the default locations plus extra candidate paths.
    pub fn scan_paths(&self, extra: &[String]) -> Vec<DetectedProfile> {
        let mut all = self.search_paths.clone();
        all.extend(extra.iter().cloned());
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for p in all {
            if !seen.insert(p.clone()) {
                continue;
            }
            if std::path::Path::new(&p).exists() {
                let config_type = detect_config_type(&p);
                out.push(DetectedProfile {
                    path: p,
                    config_type,
                });
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Import result / plan types
// ---------------------------------------------------------------------------

/// The result of importing a single config path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub path: String,
    pub config_type: ConfigType,
    pub memories_created: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// A preview item describing what a planned import would extract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPlanItem {
    pub path: String,
    pub config_type: ConfigType,
    pub estimated_memories: usize,
}

/// A dry-run preview of an import operation (nothing is persisted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPlan {
    pub agent_id: Uuid,
    pub items: Vec<ImportPlanItem>,
}

impl ImportPlan {
    /// The total number of memories that a real import would create.
    pub fn total_estimated_memories(&self) -> usize {
        self.items.iter().map(|i| i.estimated_memories).sum()
    }
}

// ---------------------------------------------------------------------------
// ProfileImporter
// ---------------------------------------------------------------------------

/// Imports external config files into the memory store.
#[derive(Clone)]
pub struct ProfileImporter {
    store: MemoryStore,
    extractor: Option<std::sync::Arc<dyn MemoryExtractor>>,
    config: ProfileImporterConfig,
}

impl ProfileImporter {
    /// Create an importer with the heuristic (non-LLM) extractor.
    pub fn new(store: MemoryStore) -> Self {
        Self {
            store,
            extractor: None,
            config: ProfileImporterConfig::default(),
        }
    }

    /// Create an importer with a custom extractor (e.g. an LLM-backed one).
    pub fn with_extractor(
        store: MemoryStore,
        extractor: std::sync::Arc<dyn MemoryExtractor>,
    ) -> Self {
        Self {
            store,
            extractor: Some(extractor),
            config: ProfileImporterConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ProfileImporterConfig) -> Self {
        self.config = config;
        self
    }

    pub fn store(&self) -> &MemoryStore {
        &self.store
    }

    /// Parse a config file into an [`ImportSource`].
    pub fn parse_file(&self, path: &str) -> CoreResult<ImportSource> {
        let content =
            std::fs::read_to_string(path).map_err(|e| opensquilla_core::error::CoreError::Io(e))?;
        let kind = detect_kind(path);
        let parsed = match kind.as_str() {
            "json" => serde_json::from_str(&content).map_err(|e| {
                opensquilla_core::error::CoreError::Config(format!(
                    "Invalid JSON in {}: {}",
                    path, e
                ))
            })?,
            "yaml" => {
                let value: serde_yaml::Value = serde_yaml::from_str(&content).map_err(|e| {
                    opensquilla_core::error::CoreError::Config(format!(
                        "Invalid YAML in {}: {}",
                        path, e
                    ))
                })?;
                serde_json::to_value(value).map_err(|e| {
                    opensquilla_core::error::CoreError::Config(format!(
                        "YAML not serializable to JSON in {}: {}",
                        path, e
                    ))
                })?
            }
            "toml" => {
                let value: toml::Value = toml::from_str(&content).map_err(|e| {
                    opensquilla_core::error::CoreError::Config(format!(
                        "Invalid TOML in {}: {}",
                        path, e
                    ))
                })?;
                serde_json::to_value(value).map_err(|e| {
                    opensquilla_core::error::CoreError::Config(format!(
                        "TOML not serializable to JSON in {}: {}",
                        path, e
                    ))
                })?
            }
            other => {
                return Err(opensquilla_core::error::CoreError::InvalidInput(format!(
                    "Unsupported config file kind '{}' for {}",
                    other, path
                )));
            }
        };

        Ok(ImportSource {
            path: path.to_string(),
            kind,
            parsed,
        })
    }

    /// Import a single config file into memory. Returns the created memory ids.
    pub async fn import_from_file(&self, agent_id: Uuid, path: &str) -> CoreResult<Vec<MemoryId>> {
        let source = self.parse_file(path)?;
        let extracted = self.run_extraction(&source).await?;
        let mut ids = Vec::new();
        for mem in extracted {
            ids.push(self.store_extracted(agent_id, &source, mem)?);
        }
        info!("Imported {} memories from {}", ids.len(), path);
        Ok(ids)
    }

    /// Import all supported config files in a directory (non-recursive).
    pub async fn import_from_directory(
        &self,
        agent_id: Uuid,
        dir: &str,
    ) -> CoreResult<ImportSummary> {
        let mut summary = ImportSummary {
            agent_id,
            files_processed: 0,
            memories_created: 0,
            failed_files: Vec::new(),
        };

        let entries =
            std::fs::read_dir(dir).map_err(|e| opensquilla_core::error::CoreError::Io(e))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            if !is_supported(&path_str) {
                continue;
            }

            summary.files_processed += 1;
            match self.import_from_file(agent_id, &path_str).await {
                Ok(ids) => summary.memories_created += ids.len() as u64,
                Err(e) => {
                    warn!("Failed to import {}: {}", path_str, e);
                    summary.failed_files.push(path_str);
                }
            }
        }

        debug!(
            "Directory import complete: {} files, {} memories, {} failures",
            summary.files_processed,
            summary.memories_created,
            summary.failed_files.len()
        );
        Ok(summary)
    }

    /// Import a single path, which may be a file or a directory.
    ///
    /// Files are routed through the config-family-aware importers when their
    /// type is detected; directories are scanned non-recursively.
    pub async fn import_from_path(
        &self,
        agent_id: Uuid,
        path: &str,
    ) -> CoreResult<Vec<ImportResult>> {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            let summary = self.import_from_directory(agent_id, path).await?;
            return Ok(vec![ImportResult {
                path: path.to_string(),
                config_type: ConfigType::Unknown,
                memories_created: summary.memories_created,
                success: summary.failed_files.is_empty(),
                error: if summary.failed_files.is_empty() {
                    None
                } else {
                    Some(format!(
                        "{} files failed to import",
                        summary.failed_files.len()
                    ))
                },
            }]);
        }
        if p.is_file() {
            let config_type = detect_config_type(path);
            let result = match config_type {
                ConfigType::Claude => self.import_from_claude(agent_id, path).await?,
                ConfigType::OpenClaw => self.import_from_openclaw(agent_id, path).await?,
                ConfigType::Hermes => self.import_from_hermes(agent_id, path).await?,
                _ => match self.import_from_file(agent_id, path).await {
                    Ok(ids) => ImportResult {
                        path: path.to_string(),
                        config_type,
                        memories_created: ids.len() as u64,
                        success: true,
                        error: None,
                    },
                    Err(e) => ImportResult {
                        path: path.to_string(),
                        config_type,
                        memories_created: 0,
                        success: false,
                        error: Some(e.to_string()),
                    },
                },
            };
            return Ok(vec![result]);
        }
        Ok(vec![ImportResult {
            path: path.to_string(),
            config_type: ConfigType::Unknown,
            memories_created: 0,
            success: false,
            error: Some("Path does not exist".to_string()),
        }])
    }

    /// Import a Claude Code config (`.claude/settings.json`, ...).
    pub async fn import_from_claude(&self, agent_id: Uuid, path: &str) -> CoreResult<ImportResult> {
        self.import_specialized(agent_id, path, ConfigType::Claude, "claude")
            .await
    }

    /// Import an OpenClaw config (`.openclaw/config.*`, ...).
    pub async fn import_from_openclaw(
        &self,
        agent_id: Uuid,
        path: &str,
    ) -> CoreResult<ImportResult> {
        self.import_specialized(agent_id, path, ConfigType::OpenClaw, "openclaw")
            .await
    }

    /// Import a Hermes config (`.hermes/config.*`, ...).
    pub async fn import_from_hermes(&self, agent_id: Uuid, path: &str) -> CoreResult<ImportResult> {
        self.import_specialized(agent_id, path, ConfigType::Hermes, "hermes")
            .await
    }

    /// Produce a dry-run import plan without persisting anything.
    pub async fn plan_import(&self, agent_id: Uuid, path: &str) -> CoreResult<ImportPlan> {
        let p = std::path::Path::new(path);
        let mut paths: Vec<String> = Vec::new();
        if p.is_dir() {
            let entries =
                std::fs::read_dir(p).map_err(|e| opensquilla_core::error::CoreError::Io(e))?;
            for entry in entries.flatten() {
                let p2 = entry.path();
                if p2.is_file() && is_supported(&p2.to_string_lossy()) {
                    paths.push(p2.to_string_lossy().to_string());
                }
            }
        } else if p.is_file() {
            paths.push(path.to_string());
        }

        let mut items = Vec::new();
        for candidate in paths {
            let config_type = detect_config_type(&candidate);
            let estimated = match self.parse_file(&candidate) {
                Ok(source) => self.run_extraction(&source).await.unwrap_or_default().len(),
                Err(_) => 0,
            };
            items.push(ImportPlanItem {
                path: candidate,
                config_type,
                estimated_memories: estimated,
            });
        }
        Ok(ImportPlan { agent_id, items })
    }

    async fn import_specialized(
        &self,
        agent_id: Uuid,
        path: &str,
        config_type: ConfigType,
        memory_type: &str,
    ) -> CoreResult<ImportResult> {
        match self.parse_file(path) {
            Ok(source) => {
                let extracted = match config_type {
                    ConfigType::Claude => extract_claude_memories(&source.parsed),
                    ConfigType::OpenClaw => extract_openclaw_memories(&source.parsed),
                    ConfigType::Hermes => extract_hermes_memories(&source.parsed),
                    _ => {
                        let mut memories = Vec::new();
                        collect_scalars(&source.parsed, &source.path, "", &mut memories);
                        memories
                    }
                };
                let mut created = 0u64;
                for mut mem in extracted {
                    // Re-tag generic facts/preferences with the family name.
                    if mem.memory_type == "preference" || mem.memory_type == "fact" {
                        mem.memory_type = memory_type.to_string();
                    }
                    self.store_extracted(agent_id, &source, mem)?;
                    created += 1;
                }
                Ok(ImportResult {
                    path: path.to_string(),
                    config_type,
                    memories_created: created,
                    success: true,
                    error: None,
                })
            }
            Err(e) => Ok(ImportResult {
                path: path.to_string(),
                config_type,
                memories_created: 0,
                success: false,
                error: Some(e.to_string()),
            }),
        }
    }

    async fn run_extraction(&self, source: &ImportSource) -> CoreResult<Vec<ExtractedMemory>> {
        if let Some(extractor) = &self.extractor {
            return extractor.extract(source).await;
        }
        // Fall back to the deterministic heuristic extractor.
        HeuristicExtractor.extract(source).await
    }

    fn store_extracted(
        &self,
        agent_id: Uuid,
        source: &ImportSource,
        mem: ExtractedMemory,
    ) -> CoreResult<MemoryId> {
        let now = Utc::now();
        let id = MemoryId(Uuid::new_v4());
        let mut tags = mem.tags;
        if self.config.tag_with_source {
            if let Some(name) = std::path::Path::new(&source.path)
                .file_stem()
                .and_then(|s| s.to_str())
            {
                tags.push(format!("source:{}", name));
            }
            tags.push(format!("config:{}", source.kind));
        }

        let entry = MemoryEntry {
            id,
            agent_id,
            content: mem.content,
            tags,
            embedding: None,
            created_at: now,
            updated_at: now,
            accessed_at: None,
            source: "profile_import".to_string(),
            memory_type: mem.memory_type,
            importance: mem.importance.clamp(0.0, 1.0),
            importance_score: mem.importance.clamp(0.0, 1.0),
            access_count: 0,
            metadata: serde_json::json!({
                "source_file": source.path,
                "source_kind": source.kind,
                "payload": mem.payload,
            }),
        };

        self.store.insert_memory(&entry)?;
        Ok(id)
    }
}

fn detect_kind(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .filter(|e| matches!(e.as_str(), "json" | "yaml" | "yml" | "toml"))
        .map(|e| match e.as_str() {
            "yml" => "yaml".to_string(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn is_supported(path: &str) -> bool {
    matches!(detect_kind(path).as_str(), "json" | "yaml" | "toml")
}

// ---------------------------------------------------------------------------
// Config-family-specific extraction heuristics
// ---------------------------------------------------------------------------

/// Extract memory-worthy facts from a Claude Code config.
fn extract_claude_memories(parsed: &serde_json::Value) -> Vec<ExtractedMemory> {
    let mut out = Vec::new();
    let Some(obj) = parsed.as_object() else {
        return out;
    };
    for (key, value) in obj {
        match key.to_lowercase().as_str() {
            "model" | "theme" | "appearance" | "include" | "exclude" => {
                push_specialized_scalar(&mut out, key, value, "claude", 0.5);
            }
            "permissions" => {
                if let Some(perms) = value.as_array() {
                    let names: Vec<&str> = perms.iter().filter_map(|p| p.as_str()).collect();
                    if !names.is_empty() {
                        out.push(ExtractedMemory {
                            content: format!("Claude Code permissions: {}", names.join(", ")),
                            memory_type: "claude".to_string(),
                            importance: 0.5,
                            tags: vec!["claude".to_string(), "permissions".to_string()],
                            payload: value.clone(),
                        });
                    }
                }
            }
            "env" => {
                if let Some(env) = value.as_object() {
                    let vars: Vec<&String> = env.keys().collect();
                    if !vars.is_empty() {
                        out.push(ExtractedMemory {
                            content: format!(
                                "Claude Code environment variables: {}",
                                vars.iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            memory_type: "claude".to_string(),
                            importance: 0.4,
                            tags: vec!["claude".to_string(), "env".to_string()],
                            payload: value.clone(),
                        });
                    }
                }
            }
            "hooks" => {
                if let Some(hooks) = value.as_object() {
                    let names: Vec<&String> = hooks.keys().collect();
                    if !names.is_empty() {
                        out.push(ExtractedMemory {
                            content: format!(
                                "Claude Code hooks configured: {}",
                                names
                                    .iter()
                                    .map(|s| s.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                            memory_type: "claude".to_string(),
                            importance: 0.5,
                            tags: vec!["claude".to_string(), "hooks".to_string()],
                            payload: value.clone(),
                        });
                    }
                }
            }
            _ => push_specialized_scalar(&mut out, key, value, "claude", 0.4),
        }
    }
    out
}

/// Extract memory-worthy facts from an OpenClaw config.
fn extract_openclaw_memories(parsed: &serde_json::Value) -> Vec<ExtractedMemory> {
    let mut out = Vec::new();
    let Some(obj) = parsed.as_object() else {
        return out;
    };
    for (key, value) in obj {
        match key.to_lowercase().as_str() {
            "model" | "provider" | "voice" | "name" | "personality" | "persona" => {
                if is_nested(value) {
                    collect_nested_strings(&mut out, key, value, "openclaw");
                } else {
                    push_specialized_scalar(&mut out, key, value, "openclaw", 0.5);
                }
            }
            _ => push_specialized_scalar(&mut out, key, value, "openclaw", 0.4),
        }
    }
    out
}

/// Extract memory-worthy facts from a Hermes config.
fn extract_hermes_memories(parsed: &serde_json::Value) -> Vec<ExtractedMemory> {
    let mut out = Vec::new();
    let Some(obj) = parsed.as_object() else {
        return out;
    };
    for (key, value) in obj {
        match key.to_lowercase().as_str() {
            "model" | "temperature" | "provider" | "system_prompt" | "systemprompt"
            | "max_tokens" | "top_p" | "context_window" => {
                push_specialized_scalar(&mut out, key, value, "hermes", 0.5);
            }
            "personality" => collect_nested_strings(&mut out, key, value, "hermes"),
            _ => push_specialized_scalar(&mut out, key, value, "hermes", 0.4),
        }
    }
    out
}

fn is_nested(value: &serde_json::Value) -> bool {
    value.is_object() || value.is_array()
}

/// Emit a scalar leaf as a specialized memory.
fn push_specialized_scalar(
    out: &mut Vec<ExtractedMemory>,
    key: &str,
    value: &serde_json::Value,
    memory_type: &str,
    importance: f64,
) {
    if let Some(content) = scalar_to_content(key, value) {
        out.push(ExtractedMemory {
            content,
            memory_type: memory_type.to_string(),
            importance,
            tags: vec!["import".to_string(), memory_type.to_string()],
            payload: value.clone(),
        });
    }
}

/// Recursively collect string leaves from a nested value (persona objects, etc.).
fn collect_nested_strings(
    out: &mut Vec<ExtractedMemory>,
    prefix: &str,
    value: &serde_json::Value,
    memory_type: &str,
) {
    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let key = format!("{}.{}", prefix, k);
                if is_nested(v) {
                    collect_nested_strings(out, &key, v, memory_type);
                } else if let Some(content) = scalar_to_content(&key, v) {
                    out.push(ExtractedMemory {
                        content,
                        memory_type: memory_type.to_string(),
                        importance: 0.4,
                        tags: vec!["import".to_string(), memory_type.to_string()],
                        payload: v.clone(),
                    });
                }
            }
        }
        serde_json::Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let key = format!("{}[{}]", prefix, i);
                if is_nested(item) {
                    collect_nested_strings(out, &key, item, memory_type);
                } else if let Some(content) = scalar_to_content(&key, item) {
                    out.push(ExtractedMemory {
                        content,
                        memory_type: memory_type.to_string(),
                        importance: 0.4,
                        tags: vec!["import".to_string(), memory_type.to_string()],
                        payload: item.clone(),
                    });
                }
            }
        }
        _ => {}
    }
}

/// Recursively collect scalar values from a parsed config into extracted memories.
fn collect_scalars(
    value: &serde_json::Value,
    path: &str,
    prefix: &str,
    out: &mut Vec<ExtractedMemory>,
) {
    const KEY_MARKERS: &[&str] = &[
        "prefer",
        "favorite",
        "favourite",
        "like",
        "theme",
        "model",
        "provider",
        "language",
        "timezone",
        "editor",
        "shell",
    ];

    match value {
        serde_json::Value::Object(map) => {
            for (k, v) in map {
                let key = k.to_lowercase();
                let child_prefix = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{}.{}", prefix, k)
                };
                let is_leaf_marker = KEY_MARKERS.iter().any(|m| key.contains(m));
                let is_scalar = matches!(
                    v,
                    serde_json::Value::String(_)
                        | serde_json::Value::Bool(_)
                        | serde_json::Value::Number(_)
                );
                if is_leaf_marker && is_scalar {
                    // Marker + scalar leaf: emit a preference and do not
                    // recurse (avoids duplicate facts).
                    if let Some(content) = scalar_to_content(k, v) {
                        out.push(ExtractedMemory {
                            content,
                            memory_type: "preference".to_string(),
                            importance: 0.5,
                            tags: vec!["import".to_string(), "config".to_string()],
                            payload: v.clone(),
                        });
                    }
                    continue;
                }
                // Recurse into nested objects / arrays.
                collect_scalars(v, path, &child_prefix, out);
            }
        }
        serde_json::Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                let child_prefix = format!("{}[{}]", prefix, i);
                collect_scalars(item, path, &child_prefix, out);
            }
        }
        _ => {
            // Scalar reached through a non-marker path: emit a generic fact.
            if !prefix.is_empty() {
                if let Some(content) = scalar_to_content(prefix, value) {
                    out.push(ExtractedMemory {
                        content,
                        memory_type: "fact".to_string(),
                        importance: 0.3,
                        tags: vec!["import".to_string(), "config".to_string()],
                        payload: value.clone(),
                    });
                }
            }
        }
    }
}

/// Convert a scalar value into a sentence describing the key-value pair.
fn scalar_to_content(key: &str, value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(format!("{}: {}", key, s)),
        serde_json::Value::Bool(b) => Some(format!("{}: {}", key, b)),
        serde_json::Value::Number(n) => Some(format!("{}: {}", key, n)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "osq_memory_profile_import_{}_{}",
            name,
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &std::path::Path, content: &str) {
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn test_detect_kind() {
        assert_eq!(detect_kind("a.json"), "json");
        assert_eq!(detect_kind("a.yml"), "yaml");
        assert_eq!(detect_kind("a.yaml"), "yaml");
        assert_eq!(detect_kind("a.toml"), "toml");
        assert_eq!(detect_kind("a.txt"), "unknown");
    }

    #[test]
    fn test_detect_config_type() {
        assert_eq!(
            detect_config_type("/home/user/.claude/settings.json"),
            ConfigType::Claude
        );
        assert_eq!(
            detect_config_type("/home/user/.claude.json"),
            ConfigType::Claude
        );
        assert_eq!(
            detect_config_type("/home/user/.openclaw/config.yaml"),
            ConfigType::OpenClaw
        );
        assert_eq!(
            detect_config_type("/home/user/.hermes/config.toml"),
            ConfigType::Hermes
        );
        assert_eq!(
            detect_config_type("/tmp/settings.json"),
            ConfigType::Generic
        );
        assert_eq!(detect_config_type("/tmp/notes.txt"), ConfigType::Unknown);
    }

    #[test]
    fn test_profile_detector_scan() {
        let detector = ProfileDetector::new();
        let found = detector.scan();
        // Should not panic; may be empty depending on the machine.
        for p in &found {
            assert!(std::path::Path::new(&p.path).exists());
        }
    }

    #[test]
    fn test_parse_json() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store);
        let dir = temp_dir("json");
        let path = dir.join("config.json");
        write(&path, r#"{"preferred_language": "rust", "theme": "dark"}"#);
        let source = importer.parse_file(path.to_str().unwrap()).unwrap();
        assert_eq!(source.kind, "json");
        assert!(source.parsed.is_object());
    }

    #[tokio::test]
    async fn test_import_from_file_json() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());
        let dir = temp_dir("file");
        let path = dir.join("settings.json");
        write(&path, r#"{"preferred_language": "rust", "editor": "vim"}"#);
        let agent = Uuid::new_v4();
        let ids = importer
            .import_from_file(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert!(!ids.is_empty());

        let all = store.list_memories(&agent, None, 100, 0).unwrap();
        assert_eq!(all.len(), ids.len());
        assert!(all.iter().all(|m| m.source == "profile_import"));
    }

    #[tokio::test]
    async fn test_import_from_directory() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());

        let dir = temp_dir("dir");
        write(&dir.join("a.json"), r#"{"preferred_model": "gpt-4o"}"#);
        write(
            &dir.join("b.toml"),
            "theme = \"dark\"\nprovider = \"openai\"\n",
        );
        std::fs::File::create(dir.join("notes.txt")).unwrap(); // ignored

        let agent = Uuid::new_v4();
        let summary = importer
            .import_from_directory(agent, dir.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(summary.files_processed, 2);
        assert!(summary.memories_created >= 2);
        assert!(summary.failed_files.is_empty());
    }

    #[test]
    fn test_parse_toml() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store);
        let dir = temp_dir("toml");
        let path = dir.join("c.toml");
        write(&path, "theme = \"dark\"\n[llm]\nmodel = \"gpt-4o\"\n");
        let source = importer.parse_file(path.to_str().unwrap()).unwrap();
        assert_eq!(source.kind, "toml");
        assert_eq!(source.parsed["theme"], serde_json::json!("dark"));
    }

    #[tokio::test]
    async fn test_import_from_claude() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());
        let dir = temp_dir("claude");
        let path = dir.join("settings.json");
        write(
            &path,
            r#"{"model": "claude-opus-4-6", "theme": "dark", "permissions": ["read", "write"]}"#,
        );
        let agent = Uuid::new_v4();
        let result = importer
            .import_from_claude(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.memories_created >= 2);

        let all = store.list_memories(&agent, None, 100, 0).unwrap();
        assert!(all.iter().all(|m| m.source == "profile_import"));
        assert!(all.iter().any(|m| m.memory_type == "claude"));
    }

    #[tokio::test]
    async fn test_import_from_openclaw() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());
        let dir = temp_dir("openclaw");
        let path = dir.join("config.yaml");
        write(
            &path,
            "model: claude-opus-4-6\nprovider: anthropic\npersona:\n  name: Ada\n  tone: friendly\n",
        );
        let agent = Uuid::new_v4();
        let result = importer
            .import_from_openclaw(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.memories_created >= 2);
    }

    #[tokio::test]
    async fn test_import_from_hermes() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());
        let dir = temp_dir("hermes");
        let path = dir.join("config.toml");
        write(
            &path,
            "model = \"hermes-3\"\ntemperature = 0.7\nsystem_prompt = \"You are helpful\"\n",
        );
        let agent = Uuid::new_v4();
        let result = importer
            .import_from_hermes(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert!(result.success);
        assert!(result.memories_created >= 2);
    }

    #[tokio::test]
    async fn test_import_from_path_file_and_dir() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store.clone());
        let dir = temp_dir("path");
        let path = dir.join("settings.json");
        write(&path, r#"{"preferred_language": "rust"}"#);
        let agent = Uuid::new_v4();

        let file_results = importer
            .import_from_path(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(file_results.len(), 1);
        assert!(file_results[0].success);

        let dir_results = importer
            .import_from_path(agent, dir.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(dir_results.len(), 1);
        assert!(dir_results[0].success);
    }

    #[tokio::test]
    async fn test_import_from_path_missing() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store);
        let agent = Uuid::new_v4();
        let results = importer
            .import_from_path(agent, "/nonexistent/definitely/missing.json")
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].success);
        assert!(results[0].error.is_some());
    }

    #[tokio::test]
    async fn test_plan_import() {
        let store = MemoryStore::in_memory().unwrap();
        let importer = ProfileImporter::new(store);
        let dir = temp_dir("plan");
        let path = dir.join("settings.json");
        write(&path, r#"{"model": "gpt-4o", "theme": "dark"}"#);
        let agent = Uuid::new_v4();
        let plan = importer
            .plan_import(agent, path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(plan.agent_id, agent);
        assert_eq!(plan.items.len(), 1);
        assert!(plan.items[0].estimated_memories >= 2);
        assert!(plan.total_estimated_memories() >= 2);
    }

    #[test]
    fn test_config_type_label() {
        assert_eq!(ConfigType::Claude.label(), "claude");
        assert_eq!(ConfigType::OpenClaw.label(), "openclaw");
        assert_eq!(ConfigType::Hermes.label(), "hermes");
        assert_eq!(ConfigType::Generic.label(), "generic");
        assert_eq!(ConfigType::Unknown.label(), "unknown");
    }
}
