//! # Skill loader
//!
//! The [`SkillLoader`] discovers, parses, validates, and caches skills from the
//! six-layer directory coverage system:
//!
//! 1. `EXTRA`    — external extra directories
//! 2. `BUNDLED`  — built-in skills compiled into the binary
//! 3. `MANAGED`  — community-installed skills
//! 4. `PERSONAL` — user-installed skills
//! 5. `PROJECT`  — workspace project skills
//! 6. `WORKSPACE`— workspace root skills
//!
//! Each layer is scanned for `SKILL.md` files. The YAML frontmatter of each
//! file is deserialized into a [`crate::types::SkillManifest`] and normalized
//! into a [`crate::types::SkillSpec`]. When two layers define the same skill
//! `id`, the higher-priority layer wins.
//!
//! The loader maintains a directory cache keyed by file path with the last-seen
//! modification time, so re-scans only re-parse files that actually changed. An
//! optional file-system watcher (via the `notify` crate) triggers hot reloads
//! without an explicit scan.

use crate::eligibility::EligibilityChecker;
use crate::types::{
    rank_skills, SkillFilter, SkillKind, SkillLayer, SkillManifest, SkillMatch, SkillScope,
    SkillSpec, SkillVisibility,
};
use dashmap::DashMap;
use notify::{Event, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tracing::{debug, info, warn};
use walkdir::WalkDir;

/// Errors produced while loading, parsing, or validating skills.
#[derive(Debug, Error)]
pub enum SkillLoadError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("SKILL.md at {path} does not start with a '---' frontmatter delimiter")]
    MissingFrontmatter { path: PathBuf },
    #[error("SKILL.md at {path} is missing the closing '---' frontmatter delimiter")]
    UnclosedFrontmatter { path: PathBuf },
    #[error("YAML frontmatter parse error at {path}: {message}")]
    Yaml { path: PathBuf, message: String },
    #[error("invalid skill at {path}: {issues:?}")]
    Validation { path: PathBuf, issues: Vec<String> },
    #[error("watcher error: {0}")]
    Watcher(String),
    #[error("internal error: {0}")]
    Internal(String),
}

impl SkillLoadError {
    /// Path this error relates to, if any.
    pub fn path(&self) -> Option<&Path> {
        match self {
            SkillLoadError::Io { path, .. }
            | SkillLoadError::MissingFrontmatter { path }
            | SkillLoadError::UnclosedFrontmatter { path }
            | SkillLoadError::Yaml { path, .. }
            | SkillLoadError::Validation { path, .. } => Some(path),
            _ => None,
        }
    }
}

/// One non-fatal problem found while loading a skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadWarning {
    /// Path of the skill the warning refers to.
    pub path: PathBuf,
    /// Human-readable warning text.
    pub message: String,
}

/// Summary of a scan operation.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    /// Number of layer directories scanned.
    pub scanned_dirs: usize,
    /// Total number of `SKILL.md` files discovered.
    pub found_skills: usize,
    /// Number of skills (re)loaded into the registry.
    pub loaded_skills: usize,
    /// Number of skills skipped (e.g. duplicate lower-priority).
    pub skipped: usize,
    /// Fatal errors encountered (per-file, not fatal to the whole scan).
    pub errors: Vec<SkillLoadError>,
    /// Non-fatal warnings.
    pub warnings: Vec<LoadWarning>,
    /// Duration of the scan.
    pub elapsed: Duration,
}

impl LoadReport {
    /// Whether every file scanned cleanly.
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Tuning knobs for the loader.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Maximum directory depth below a layer root to search for `SKILL.md`.
    pub max_depth: usize,
    /// Follow directory symlinks during traversal.
    pub follow_links: bool,
    /// Whether to re-validate cached skills on rescan.
    pub revalidate: bool,
    /// Whether to skip hidden files and directories (leading `.`).
    pub ignore_hidden: bool,
    /// Minimum interval between hot-reload scans, for debouncing.
    pub hot_reload_debounce: Duration,
    /// Whether a failed skill file aborts the whole scan (`false` = fail-soft).
    pub fail_fast: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            follow_links: true,
            revalidate: true,
            ignore_hidden: true,
            hot_reload_debounce: Duration::from_millis(250),
            fail_fast: false,
        }
    }
}

/// A cache entry tracking a scanned `SKILL.md`.
#[derive(Debug, Clone)]
struct DirCacheEntry {
    /// Modification time of the file at last scan (nanoseconds since epoch).
    mtime_ns: u64,
    /// The parsed skill, if it parsed successfully.
    spec: Option<SkillSpec>,
}

/// The skill loader.
///
/// Internally a shared registry of specs plus layer indices and a directory
/// cache. All maps are `Arc<DashMap>` so the hot-reload task and concurrent
/// readers can share them cheaply.
pub struct SkillLoader {
    /// All loaded skills, indexed by ID.
    skills: Arc<DashMap<String, SkillSpec>>,
    /// Skill IDs organized by layer.
    layers: Arc<DashMap<SkillLayer, Vec<String>>>,
    /// Registered directories, organized by layer.
    layer_dirs: Arc<DashMap<SkillLayer, Vec<PathBuf>>>,
    /// File cache keyed by `SKILL.md` path.
    cache: Arc<DashMap<PathBuf, DirCacheEntry>>,
    /// Loader configuration.
    config: LoaderConfig,
    /// Time of the last completed scan.
    last_scan: std::sync::RwLock<Option<Instant>>,
    /// Optional file-system watcher for hot-reload.
    watcher: Option<notify::RecommendedWatcher>,
    /// Shared flag set while a hot-reload scan is in flight.
    reloading: Arc<std::sync::atomic::AtomicBool>,
}

impl Default for SkillLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillLoader {
    /// Create a new skill loader with default configuration.
    pub fn new() -> Self {
        Self {
            skills: Arc::new(DashMap::new()),
            layers: Arc::new(DashMap::new()),
            layer_dirs: Arc::new(DashMap::new()),
            cache: Arc::new(DashMap::new()),
            config: LoaderConfig::default(),
            last_scan: std::sync::RwLock::new(None),
            watcher: None,
            reloading: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Build a loader with custom configuration.
    pub fn with_config(config: LoaderConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// The loader configuration.
    pub fn config(&self) -> &LoaderConfig {
        &self.config
    }

    /// Register a directory for a given skill layer.
    ///
    /// Idempotent: registering the same directory twice is a no-op.
    pub fn register_layer_dir(&self, layer: SkillLayer, dir: PathBuf) {
        let mut entries = self.layer_dirs.entry(layer).or_default();
        if !entries.contains(&dir) {
            info!("Registered {} layer directory: {:?}", layer, dir);
            entries.push(dir);
        }
    }

    /// Register multiple directories for a layer.
    pub fn register_layer_dirs<I>(&self, layer: SkillLayer, dirs: I)
    where
        I: IntoIterator<Item = PathBuf>,
    {
        for dir in dirs {
            self.register_layer_dir(layer, dir);
        }
    }

    /// Deregister a directory. Returns `true` if it was registered.
    pub fn unregister_layer_dir(&self, layer: SkillLayer, dir: &Path) -> bool {
        let mut removed = false;
        if let Some(mut entries) = self.layer_dirs.get_mut(&layer) {
            let before = entries.len();
            entries.retain(|d| d != dir);
            removed = entries.len() != before;
        }
        removed
    }

    /// The registered directories for a layer.
    pub fn layer_dirs(&self, layer: SkillLayer) -> Vec<PathBuf> {
        self.layer_dirs
            .get(&layer)
            .map(|d| d.value().clone())
            .unwrap_or_default()
    }

    /// All registered layer directories.
    pub fn all_layer_dirs(&self) -> Vec<(SkillLayer, PathBuf)> {
        let mut out = Vec::new();
        for entry in self.layer_dirs.iter() {
            for dir in entry.value() {
                out.push((*entry.key(), dir.clone()));
            }
        }
        out
    }

    // -----------------------------------------------------------------------
    // Scanning
    // -----------------------------------------------------------------------

    /// Scan all registered directories for `SKILL.md` files and return the
    /// number of skills loaded.
    pub async fn scan_all(&self) -> Result<usize, String> {
        self.scan_all_report().await.map(|r| r.loaded_skills)
    }

    /// Scan all registered directories, returning a detailed report.
    pub async fn scan_all_report(&self) -> Result<LoadReport, String> {
        let start = Instant::now();
        let mut report = LoadReport::default();

        // Snapshot the registered dirs first (avoid holding a guard across awaits).
        let dirs: Vec<(SkillLayer, PathBuf)> = self.all_layer_dirs();
        report.scanned_dirs = dirs.len();

        for (layer, dir) in dirs {
            let sub = self.scan_directory(&dir, layer).await;
            match sub {
                Ok(s) => {
                    report.found_skills += s.found;
                    report.loaded_skills += s.loaded;
                    report.skipped += s.skipped;
                    report.warnings.extend(s.warnings);
                }
                Err(e) => {
                    report.errors.push(e);
                    if self.config.fail_fast {
                        report.elapsed = start.elapsed();
                        return Err("scan aborted: fail_fast".to_string());
                    }
                }
            }
        }

        if let Ok(mut last) = self.last_scan.write() {
            *last = Some(Instant::now());
        }
        report.elapsed = start.elapsed();
        info!(
            found = report.found_skills,
            loaded = report.loaded_skills,
            skipped = report.skipped,
            errors = report.errors.len(),
            "Skill scan complete"
        );
        Ok(report)
    }

    /// Scan a single layer directory for `SKILL.md` files.
    async fn scan_directory(
        &self,
        dir: &Path,
        layer: SkillLayer,
    ) -> Result<DirScanResult, SkillLoadError> {
        if !dir.exists() {
            return Ok(DirScanResult::default());
        }
        if !dir.is_dir() {
            return Err(SkillLoadError::Io {
                path: dir.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotADirectory,
                    "registered skill path is not a directory",
                ),
            });
        }

        // Walk the tree in a blocking task so the async runtime is not held.
        let max_depth = self.config.max_depth;
        let follow_links = self.config.follow_links;
        let ignore_hidden = self.config.ignore_hidden;
        let base = dir.to_path_buf();
        let candidates: Vec<PathBuf> = tokio::task::spawn_blocking(move || {
            walk_skill_files(&base, max_depth, follow_links, ignore_hidden)
        })
        .await
        .map_err(|e| SkillLoadError::Internal(format!("scan task panicked: {e}")))?;

        let mut result = DirScanResult::default();
        result.found = candidates.len();

        for path in candidates {
            match self.load_skill_cached(&path, layer).await {
                Ok(Some(_)) => result.loaded += 1,
                Ok(None) => result.skipped += 1,
                Err(e) => {
                    if self.config.fail_fast {
                        return Err(e);
                    }
                    warn!("Failed to load skill {:?}: {}", path, e);
                    result.errors += 1;
                }
            }
        }
        Ok(result)
    }

    /// Load a single skill file, using the mtime cache to skip unchanged files.
    async fn load_skill_cached(
        &self,
        path: &Path,
        layer: SkillLayer,
    ) -> Result<Option<SkillSpec>, SkillLoadError> {
        let mtime = file_mtime_ns(path)?;

        if let Some(entry) = self.cache.get(path) {
            if entry.mtime_ns == mtime && !self.config.revalidate {
                // Unchanged; re-insert with the layer now assigned.
                if let Some(spec) = &entry.spec {
                    return Ok(Some(self.install_spec(spec.clone(), layer)));
                }
                return Ok(None);
            }
        }

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| SkillLoadError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;

        match self.parse_skill_md(&content, layer, Some(path.to_path_buf())) {
            Ok((spec, warnings)) => {
                for w in warnings {
                    warn!("{}", w);
                }
                let result = self.install_spec(spec.clone(), layer);
                self.cache.insert(
                    path.to_path_buf(),
                    DirCacheEntry {
                        mtime_ns: mtime,
                        spec: Some(spec),
                    },
                );
                debug!("Loaded skill '{}' from {:?}", result.id, path);
                Ok(Some(result))
            }
            Err(e) => {
                self.cache.insert(
                    path.to_path_buf(),
                    DirCacheEntry {
                        mtime_ns: mtime,
                        spec: None,
                    },
                );
                Err(e)
            }
        }
    }

    /// Insert a spec honoring layer priority. Returns the effective spec
    /// (the inserted one, or the existing higher-priority one).
    fn install_spec(&self, spec: SkillSpec, layer: SkillLayer) -> SkillSpec {
        let id = spec.id.clone();
        let mut spec = spec;
        spec.layer = layer;

        let existing_priority = self
            .skills
            .get(&id)
            .map(|existing| existing.layer.priority());

        let should_insert = match existing_priority {
            None => true,
            Some(p) => p <= layer.priority(),
        };

        if should_insert {
            self.skills.insert(id.clone(), spec.clone());

            let mut layer_skills = self.layers.entry(layer).or_default();
            if !layer_skills.contains(&id) {
                layer_skills.push(id.clone());
            }
            // Remove the id from any lower layers so the index stays consistent.
            let mut stale_layers: Vec<SkillLayer> = Vec::new();
            for entry in self.layers.iter() {
                let key = *entry.key();
                if key != layer && entry.value().contains(&id) {
                    stale_layers.push(key);
                }
            }
            for key in stale_layers {
                if let Some(mut list) = self.layers.get_mut(&key) {
                    list.retain(|s| s != &id);
                }
            }
        }
        spec
    }

    /// Remove a skill by id, returning the removed spec.
    pub fn remove_skill(&self, id: &str) -> Option<SkillSpec> {
        let removed = self.skills.remove(id).map(|(_, v)| v)?;
        for entry in self.layers.iter_mut() {
            entry.value_mut().retain(|s| s != id);
        }
        Some(removed)
    }

    /// Remove all skills belonging to a layer.
    pub fn remove_layer(&self, layer: SkillLayer) -> usize {
        let ids: Vec<String> = self
            .layers
            .get(&layer)
            .map(|v| v.clone())
            .unwrap_or_default();
        let mut removed = 0;
        for id in &ids {
            if let Some(spec) = self.skills.get(id) {
                if spec.layer == layer {
                    self.skills.remove(id);
                    removed += 1;
                }
            }
        }
        self.layers.remove(&layer);
        removed
    }

    /// Clear every loaded skill and the cache.
    pub fn clear(&self) {
        self.skills.clear();
        self.layers.clear();
        self.cache.clear();
        info!("Skill loader cleared");
    }

    // -----------------------------------------------------------------------
    // Registration of pre-parsed specs
    // -----------------------------------------------------------------------

    /// Register pre-built skill specs (e.g. bundled skills) into this loader,
    /// honoring layer priority: a higher-priority layer overrides an existing
    /// lower-priority skill with the same id.
    ///
    /// Returns the number of specs that were actually inserted.
    pub async fn register_skills(&self, specs: Vec<SkillSpec>) -> usize {
        let mut inserted = 0usize;
        for spec in specs {
            let layer = spec.layer;
            let id = spec.id.clone();
            let existing_priority = self.skills.get(&id).map(|s| s.layer.priority());
            let should_insert = match existing_priority {
                None => true,
                Some(p) => p <= layer.priority(),
            };
            if should_insert {
                self.install_spec(spec, layer);
                inserted += 1;
            }
        }
        info!("Registered {} skills programmatically", inserted);
        inserted
    }

    // -----------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------

    /// Parse SKILL.md content into a [`SkillSpec`].
    ///
    /// Returns the spec plus any non-fatal warnings.
    pub fn parse_skill_md(
        &self,
        content: &str,
        layer: SkillLayer,
        source_path: Option<PathBuf>,
    ) -> Result<(SkillSpec, Vec<String>), SkillLoadError> {
        let (frontmatter, body) = extract_frontmatter(content)?;
        let manifest: SkillManifest = serde_yaml::from_str(&frontmatter).map_err(|e| {
            let path = source_path.clone().unwrap_or_default();
            SkillLoadError::Yaml {
                path,
                message: e.to_string(),
            }
        })?;

        let path = source_path.unwrap_or_default();
        let spec = manifest_to_spec(
            manifest,
            layer,
            path.clone(),
            body,
            frontmatter,
        )?;

        let mut warnings = Vec::new();
        if spec.description.is_empty() {
            warnings.push(format!(
                "skill '{}' at {} has no description",
                spec.id,
                path.display()
            ));
        }
        if spec.is_meta() && spec.steps.is_empty() {
            warnings.push(format!(
                "meta-skill '{}' at {} has no steps",
                spec.id,
                path.display()
            ));
        }
        Ok((spec, warnings))
    }

    /// Extract and parse just the frontmatter of a SKILL.md file.
    pub fn parse_frontmatter(content: &str) -> Result<SkillManifest, SkillLoadError> {
        let (frontmatter, _) = extract_frontmatter(content)?;
        serde_yaml::from_str(&frontmatter).map_err(|e| SkillLoadError::Yaml {
            path: PathBuf::new(),
            message: e.to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Get a skill by its ID.
    pub async fn get_skill(&self, id: &str) -> Option<SkillSpec> {
        self.skills.get(id).map(|s| s.value().clone())
    }

    /// Get all skills, optionally filtered by layer.
    pub async fn get_skills(&self, layer_filter: Option<SkillLayer>) -> Vec<SkillSpec> {
        self.get_skills_filtered(&SkillFilter {
            layer: layer_filter,
            ..SkillFilter::default()
        })
    }

    /// Get skills matching arbitrary filter criteria.
    pub fn get_skills_filtered(&self, filter: &SkillFilter) -> Vec<SkillSpec> {
        let all: Vec<SkillSpec> = self.skills.iter().map(|s| s.value().clone()).collect();
        filter.apply(&all)
    }

    /// Get skills matching filter criteria, additionally enforcing the
    /// `eligible_only` flag with a shared eligibility checker.
    pub fn get_skills_filtered_with(
        &self,
        filter: &SkillFilter,
        checker: &EligibilityChecker,
    ) -> Vec<SkillSpec> {
        let all: Vec<SkillSpec> = self.skills.iter().map(|s| s.value().clone()).collect();
        all.into_iter()
            .filter(|s| {
                filter.matches(s)
                    && (!filter.eligible_only
                        || checker.is_eligible(&s.requires).unwrap_or(false))
            })
            .collect()
    }

    /// Get all meta-skills (kind = Meta or MetaSop).
    pub async fn get_meta_skills(&self) -> Vec<SkillSpec> {
        self.skills
            .iter()
            .filter(|s| s.value().is_meta())
            .map(|s| s.value().clone())
            .collect()
    }

    /// Get skills belonging to a specific layer.
    pub async fn get_layer_skills(&self, layer: SkillLayer) -> Vec<SkillSpec> {
        self.get_skills(Some(layer)).await
    }

    /// Get the ID list for a layer.
    pub async fn layer_ids(&self, layer: SkillLayer) -> Vec<String> {
        self.layers
            .get(&layer)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Check whether a skill id exists in the registry.
    pub async fn contains(&self, id: &str) -> bool {
        self.skills.contains_key(id)
    }

    /// Total number of loaded skills.
    pub async fn count(&self) -> usize {
        self.skills.len()
    }

    /// Count of skills per layer.
    pub async fn counts_by_layer(&self) -> HashMap<SkillLayer, usize> {
        let mut counts = HashMap::new();
        for entry in self.layers.iter() {
            counts.insert(*entry.key(), entry.value().len());
        }
        counts
    }

    /// When the last full scan completed, if any.
    pub fn last_scan_time(&self) -> Option<Instant> {
        self.last_scan.read().ok().copied().flatten()
    }

    /// Number of cache entries (files tracked).
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Search the catalog by relevance to a free-text query.
    pub fn search(&self, query: &str, limit: usize) -> Vec<SkillMatch> {
        let all: Vec<SkillSpec> = self.skills.iter().map(|s| s.value().clone()).collect();
        let mut ranked = rank_skills(query, &all);
        ranked.truncate(limit);
        ranked
    }

    /// Compute the set of skills that reference a given tool name.
    pub fn skills_using_tool(&self, tool_name: &str) -> Vec<SkillSpec> {
        self.skills
            .iter()
            .filter(|s| s.value().tool_names().iter().any(|t| t == tool_name))
            .map(|s| s.value().clone())
            .collect()
    }

    /// Compute the set of skills that reference a given sub-skill id
    /// (used by `skill_exec` resolution).
    pub fn dependents_of(&self, skill_id: &str) -> Vec<SkillSpec> {
        self.skills
            .iter()
            .filter(|s| {
                s.value()
                    .steps
                    .iter()
                    .any(|step| step.skill.as_deref() == Some(skill_id))
            })
            .map(|s| s.value().clone())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Hot reload
    // -----------------------------------------------------------------------

    /// Start hot-reload file watching for all registered directories.
    ///
    /// The watcher debounces events and reloads the affected `SKILL.md` files
    /// in the background. Returns an error if the OS watcher cannot be created.
    pub fn start_hot_reload(&mut self) -> Result<(), String> {
        if self.watcher.is_some() {
            return Ok(()); // already running
        }

        let skills = self.skills.clone();
        let layers = self.layers.clone();
        let layer_dirs = self.layer_dirs.clone();
        let cache = self.cache.clone();
        let reloading = self.reloading.clone();
        let config = self.config.clone();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<notify::Event>();

        let mut watcher = notify::recommended_watcher(
            move |res: Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let _ = tx.send(event);
                }
            },
        )
        .map_err(|e| format!("Failed to create watcher: {e}"))?;

        let dirs = self.all_layer_dirs();
        for (layer, dir) in &dirs {
            if dir.exists() {
                watcher
                    .watch(dir, RecursiveMode::Recursive)
                    .map_err(|e| format!("Failed to watch {:?}: {}", dir, e))?;
                info!("Watching directory for skill changes: {:?}", dir);
            }
        }

        // Background task: debounce events and reload changed skills.
        tokio::spawn(async move {
            let mut pending: HashMap<PathBuf, ()> = HashMap::new();
            let mut last_event = Instant::now();
            let mut first = true;

            while let Some(event) = rx.recv().await {
                let is_skill_event = event
                    .paths
                    .iter()
                    .any(|p| p.file_name().is_some_and(|n| n == "SKILL.md"));
                if !is_skill_event {
                    continue;
                }

                for path in &event.paths {
                    pending.insert(path.clone(), ());
                }
                last_event = Instant::now();

                if first {
                    first = false;
                    continue;
                }

                // Debounce: only flush once events quiet down.
                if last_event.elapsed() < config.hot_reload_debounce {
                    // Drain more events without flushing yet.
                    while let Ok(e) = rx.try_recv() {
                        for path in &e.paths {
                            if path.file_name().is_some_and(|n| n == "SKILL.md") {
                                pending.insert(path.clone(), ());
                                last_event = Instant::now();
                            }
                        }
                    }
                    continue;
                }

                if reloading.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    continue;
                }

                let paths: Vec<PathBuf> = pending.drain().map(|(p, _)| p).collect();
                let skills = skills.clone();
                let layers = layers.clone();
                let cache = cache.clone();
                let layer_dirs = layer_dirs.clone();
                let reloading = reloading.clone();

                tokio::spawn(async move {
                    for path in paths {
                        // A path that no longer exists on disk was removed; a
                        // path that exists was created or modified. This is
                        // robust regardless of the exact notify event kind.
                        let is_remove = !path.exists();
                        let layer =
                            determine_layer_from_path(&path, &layer_dirs).unwrap_or(SkillLayer::Extra);
                        if is_remove {
                            // Find and remove any skill whose source is this file.
                            let ids: Vec<String> = skills
                                .iter()
                                .filter(|s| s.value().source_path.as_deref() == Some(path.to_str().unwrap_or("")))
                                .map(|s| s.key().clone())
                                .collect();
                            for id in ids {
                                skills.remove(&id);
                                for mut l in layers.iter_mut() {
                                    l.value_mut().retain(|s| s != &id);
                                }
                            }
                            cache.remove(&path);
                            info!("Hot-reload removed skill file: {:?}", path);
                        } else {
                            match tokio::fs::read_to_string(&path).await {
                                Ok(content) => {
                                    let layer = determine_layer_from_path(&path, &layer_dirs)
                                        .unwrap_or(SkillLayer::Extra);
                                    let mtime = file_mtime_ns(&path).unwrap_or(0);
                                    let loader = Reloader {
                                        skills: skills.clone(),
                                        layers: layers.clone(),
                                        cache: cache.clone(),
                                    };
                                    loader.reload(&path, &content, layer, mtime);
                                }
                                Err(e) => {
                                    warn!("Hot-reload read failed for {:?}: {}", path, e);
                                }
                            }
                        }
                    }
                    reloading.store(false, std::sync::atomic::Ordering::SeqCst);
                });
            }
        });

        self.watcher = Some(watcher);
        info!("Skill hot-reload started");
        Ok(())
    }

    /// Whether hot-reload is currently active.
    pub fn is_watching(&self) -> bool {
        self.watcher.is_some()
    }

    /// Stop hot-reload (drops the watcher; already-spawned tasks finish).
    pub fn stop_hot_reload(&mut self) {
        self.watcher = None;
        info!("Skill hot-reload stopped");
    }
}

/// Minimal loader-like handle used by the hot-reload task to install specs.
struct Reloader {
    skills: Arc<DashMap<String, SkillSpec>>,
    layers: Arc<DashMap<SkillLayer, Vec<String>>>,
    cache: Arc<DashMap<PathBuf, DirCacheEntry>>,
}

impl Reloader {
    fn reload(&self, path: &Path, content: &str, layer: SkillLayer, mtime: u64) {
        let Some(manifest) = frontmatter_parse_lenient(content) else {
            warn!("Hot-reload: could not parse {:?}", path);
            return;
        };
        let (frontmatter, body) = match extract_frontmatter(content) {
            Ok(pair) => pair,
            Err(_) => {
                warn!("Hot-reload: malformed frontmatter in {:?}", path);
                return;
            }
        };
        let path_buf = path.to_path_buf();
        match manifest_to_spec(manifest, layer, path_buf.clone(), body, frontmatter) {
            Ok(spec) => {
                let id = spec.id.clone();
                let existing_priority = self.skills.get(&id).map(|s| s.layer.priority());
                if existing_priority.map_or(true, |p| p <= layer.priority()) {
                    self.skills.insert(id.clone(), spec.clone());
                    let mut layer_skills = self.layers.entry(layer).or_default();
                    if !layer_skills.contains(&id) {
                        layer_skills.push(id.clone());
                    }
                }
                self.cache.insert(
                    path_buf,
                    DirCacheEntry {
                        mtime_ns: mtime,
                        spec: Some(spec),
                    },
                );
                info!("Hot-reload updated skill '{}' from {:?}", id, path);
            }
            Err(e) => {
                warn!("Hot-reload rejected {:?}: {}", path, e);
                self.cache.insert(
                    path_buf,
                    DirCacheEntry {
                        mtime_ns: mtime,
                        spec: None,
                    },
                );
            }
        }
    }
}

/// Parse frontmatter leniently, returning a manifest or `None`.
fn frontmatter_parse_lenient(content: &str) -> Option<crate::types::SkillManifest> {
    let (frontmatter, _) = extract_frontmatter(content).ok()?;
    serde_yaml::from_str(&frontmatter).ok()
}

/// Result of scanning a single directory.
#[derive(Debug, Default)]
struct DirScanResult {
    found: usize,
    loaded: usize,
    skipped: usize,
    errors: usize,
    warnings: Vec<LoadWarning>,
}

/// Walk a directory tree collecting `SKILL.md` paths.
fn walk_skill_files(
    root: &Path,
    max_depth: usize,
    follow_links: bool,
    ignore_hidden: bool,
) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let walker = WalkDir::new(root)
        .max_depth(max_depth)
        .follow_links(follow_links)
        .into_iter()
        .filter_entry(|e| {
            if !ignore_hidden {
                return true;
            }
            let name = e.file_name().to_string_lossy();
            // Keep hidden dirs only at the root itself (the root may be hidden).
            !(name.starts_with('.') && name != ".agents" && e.depth() > 0)
        });
    for entry in walker.filter_map(|e| e.ok()) {
        if entry.file_name() == "SKILL.md" {
            out.push(entry.into_path());
        }
    }
    out
}

/// Extract the YAML frontmatter block and markdown body from SKILL.md content.
///
/// The content may begin with an optional UTF-8 BOM and may use CRLF or LF line
/// endings. Returns `(frontmatter, body)`.
pub fn extract_frontmatter(content: &str) -> Result<(String, String), SkillLoadError> {
    let content = content.trim_start_matches('\u{feff}');
    let normalized = content.replace("\r\n", "\n");
    let trimmed = normalized.trim_start();

    if !trimmed.starts_with("---") {
        return Err(SkillLoadError::MissingFrontmatter {
            path: PathBuf::new(),
        });
    }

    // Find the closing delimiter: a line that is exactly `---`.
    let after_first = &trimmed[3..];
    let closing = find_frontmatter_close(after_first);
    let (frontmatter, body) = match closing {
        Some(end) => {
            let fm = &after_first[..end];
            let rest = &after_first[end..];
            let body = rest
                .trim_start_matches("---")
                .trim_start_matches('\n')
                .trim_start_matches('\r');
            (fm.to_string(), body.to_string())
        }
        None => {
            // Some manifests omit the closing delimiter at EOF.
            if after_first.trim().is_empty() {
                return Err(SkillLoadError::UnclosedFrontmatter {
                    path: PathBuf::new(),
                });
            }
            (after_first.to_string(), String::new())
        }
    };

    Ok((frontmatter, body))
}

/// Find the byte offset of the closing `---` delimiter line.
///
/// `after_first` is the text immediately following the opening `---`.
/// Returns the byte offset where the closing `---` line begins, so that
/// `after_first[..offset]` is exactly the frontmatter (including its trailing
/// newline).
fn find_frontmatter_close(after_first: &str) -> Option<usize> {
    let mut offset = 0usize;
    for line in after_first.split('\n') {
        let trimmed = line.trim();
        if trimmed == "---" || trimmed == "..." {
            return Some(offset);
        }
        offset += line.len() + 1;
    }
    None
}

/// Convert a parsed manifest into a validated [`SkillSpec`].
///
/// Missing `id` is derived from the file stem of the source path when possible.
pub fn manifest_to_spec(
    manifest: SkillManifest,
    layer: SkillLayer,
    source_path: PathBuf,
    body: String,
    raw_frontmatter: String,
) -> Result<SkillSpec, SkillLoadError> {
    let mut spec = SkillSpec::new(
        manifest.id.unwrap_or_default(),
        manifest.name.unwrap_or_default(),
        manifest.description.unwrap_or_default(),
        layer,
    );

    spec.kind = manifest.kind.unwrap_or(SkillKind::Skill);
    spec.version = manifest.version;
    spec.author = string_from_value(manifest.author);
    spec.license = manifest.license;
    spec.homepage = manifest.homepage;
    spec.tags = manifest.tags;
    spec.steps = manifest.steps;
    spec.outputs = manifest.outputs;
    spec.raw_frontmatter = raw_frontmatter;
    spec.source_path = Some(source_path.to_string_lossy().to_string());
    spec.body = body;
    spec.allowed_tools = manifest.allowed_tools;
    spec.disable_model_invocation = manifest.disable_model_invocation;
    spec.contexts = manifest.contexts;
    spec.args = manifest.args;
    spec.dependencies = manifest.dependencies;

    if let Some(requires) = manifest.requires {
        spec.requires = requires;
    }
    if let Some(metadata) = manifest.metadata {
        // Merge nested metadata.tags into the top-level tags.
        for tag in &metadata.tags {
            if !spec.tags.contains(tag) {
                spec.tags.push(tag.clone());
            }
        }
        if spec.allowed_tools.is_empty() {
            spec.allowed_tools = metadata.allowed_tools.clone();
        }
        spec.metadata = Some(metadata);
    }

    spec.visibility = manifest
        .visibility
        .as_deref()
        .and_then(SkillVisibility::from_str_loose)
        .unwrap_or(SkillVisibility::Personal);
    spec.scope = manifest
        .scope
        .as_deref()
        .and_then(SkillScope::from_str_loose)
        .unwrap_or(SkillScope::Global);

    // Derive an id from the directory name when the frontmatter omitted one.
    if spec.id.is_empty() {
        spec.id = source_path
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_lowercase().replace(' ', "_"))
            .filter(|s| !s.is_empty())
            .ok_or_else(|| SkillLoadError::Validation {
                path: source_path.clone(),
                issues: vec!["skill id is required".to_string()],
            })?;
    }
    if spec.name.is_empty() {
        spec.name = spec.id.clone();
    }

    let issues = spec.validation_issues();
    if !issues.is_empty() {
        // A meta-skill with no steps is tolerated (it may be a stub); all other
        // validation failures are fatal.
        let fatal: Vec<String> = issues
            .iter()
            .filter(|i| !i.contains("has no steps"))
            .cloned()
            .collect();
        if !fatal.is_empty() {
            return Err(SkillLoadError::Validation {
                path: source_path,
                issues: fatal,
            });
        }
    }

    Ok(spec)
}

/// Coerce a JSON author value into a plain string.
fn string_from_value(value: Option<serde_json::Value>) -> Option<String> {
    match value {
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Object(map)) => {
            map.get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        }
        _ => None,
    }
}

/// Modification time of a file in nanoseconds since the Unix epoch.
fn file_mtime_ns(path: &Path) -> Result<u64, SkillLoadError> {
    let meta = std::fs::metadata(path).map_err(|e| SkillLoadError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let modified = meta.modified().map_err(|e| SkillLoadError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(mtime_to_ns(modified))
}

fn mtime_to_ns(t: SystemTime) -> u64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as u64,
        Err(e) => e.duration().as_nanos() as u64,
    }
}

/// Determine which layer a path belongs to, by checking the registered layer
/// directories.
fn determine_layer_from_path(
    path: &Path,
    layer_dirs: &DashMap<SkillLayer, Vec<PathBuf>>,
) -> Option<SkillLayer> {
    for entry in layer_dirs.iter() {
        for dir in entry.value() {
            if path.starts_with(dir) {
                return Some(*entry.key());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SkillRequires, StepType};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "opensquilla-skills-test-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_skill(dir: &Path, skill_id: &str, extra_yaml: &str) -> PathBuf {
        let skill_dir = dir.join(skill_id);
        std::fs::create_dir_all(&skill_dir).unwrap();
        let path = skill_dir.join("SKILL.md");
        let content = format!(
            "---\nid: {skill_id}\nname: {skill_id}\ndescription: test skill\nversion: 1.0.0\n{extra_yaml}\n---\n\nBody text.\n"
        );
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn extract_frontmatter_basic() {
        let content = "---\nid: foo\nname: Foo\n---\n\nHello body";
        let (fm, body) = extract_frontmatter(content).unwrap();
        assert!(fm.contains("id: foo"));
        assert!(body.contains("Hello body"));
    }

    #[test]
    fn extract_frontmatter_crlf_and_bom() {
        let content = "\u{feff}---\r\nid: foo\r\nname: Foo\r\n---\r\n\r\nBody";
        let (fm, body) = extract_frontmatter(content).unwrap();
        assert!(fm.contains("id: foo"));
        assert!(body.contains("Body"));
    }

    #[test]
    fn extract_frontmatter_missing() {
        let content = "no frontmatter here";
        assert!(extract_frontmatter(content).is_err());
    }

    #[test]
    fn parse_skill_roundtrip() {
        let content = r#"---
id: my-skill
name: My Skill
description: Does things
version: 1.2.0
tags: [a, b]
requires:
  os: [linux, macos]
  binaries: [git]
metadata:
  always: true
  classification: coding
---
# Body
"#;
        let loader = SkillLoader::new();
        let (spec, warnings) = loader
            .parse_skill_md(content, SkillLayer::Personal, None)
            .unwrap();
        assert_eq!(spec.id, "my-skill");
        assert_eq!(spec.layer, SkillLayer::Personal);
        assert_eq!(spec.tags, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(spec.requires.os.as_deref(), Some(&vec!["linux".to_string(), "macos".to_string()]));
        assert_eq!(spec.requires.binaries.as_deref(), Some(&vec!["git".to_string()]));
        assert!(spec.is_always());
        assert!(spec.body.contains("# Body"));
        assert!(warnings.is_empty());
    }

    #[test]
    fn layer_priority_override() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let loader = SkillLoader::new();
        let bundled = SkillSpec::new(
            "same".into(),
            "Same".into(),
            "bundled desc".into(),
            SkillLayer::Bundled,
        );
        let personal = SkillSpec::new(
            "same".into(),
            "Same".into(),
            "personal desc".into(),
            SkillLayer::Personal,
        );
        rt.block_on(async {
            loader.register_skills(vec![bundled.clone()]).await;
            loader.register_skills(vec![personal.clone()]).await;
            let got = loader.get_skill("same").await.unwrap();
            assert_eq!(got.layer, SkillLayer::Personal);
            assert_eq!(got.description, "personal desc");

            // Lower priority does not override.
            loader.register_skills(vec![bundled.clone()]).await;
            let got = loader.get_skill("same").await.unwrap();
            assert_eq!(got.layer, SkillLayer::Personal);
        });
    }

    #[test]
    fn scan_directory_loads_skills() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let dir = temp_dir("scan");
        write_skill(&dir, "alpha", "");
        write_skill(&dir, "beta", "kind: meta\n");
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        write_skill(&dir.join("nested"), "gamma", "");

        let loader = SkillLoader::new();
        loader.register_layer_dir(SkillLayer::Personal, dir.clone());
        rt.block_on(async {
            let report = loader.scan_all_report().await.unwrap();
            assert_eq!(report.loaded_skills, 3);
            assert_eq!(loader.count().await, 3);
            let metas = loader.get_meta_skills().await;
            assert_eq!(metas.len(), 1);
        });
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scan_respects_max_depth() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let dir = temp_dir("depth");
        write_skill(&dir, "top", "");
        let deep = dir.join("a/b/c/d");
        std::fs::create_dir_all(&deep).unwrap();
        write_skill(&deep, "deep-skill", "");

        let config = LoaderConfig {
            max_depth: 2,
            ..LoaderConfig::default()
        };
        let loader = SkillLoader::with_config(config);
        loader.register_layer_dir(SkillLayer::Personal, dir.clone());
        rt.block_on(async {
            let report = loader.scan_all_report().await.unwrap();
            // Only the top-level skill is within depth 2.
            assert_eq!(report.loaded_skills, 1);
        });
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalid_frontmatter_is_error() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let dir = temp_dir("bad");
        let bad = dir.join("badsyntax");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(
            bad.join("SKILL.md"),
            "---\nid: [unclosed\nname: x\n---\n",
        )
        .unwrap();
        let loader = SkillLoader::new();
        loader.register_layer_dir(SkillLayer::Personal, dir.clone());
        rt.block_on(async {
            let report = loader.scan_all_report().await.unwrap();
            assert_eq!(report.loaded_skills, 0);
            assert!(!report.errors.is_empty());
        });
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn remove_skill_and_layer() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let loader = SkillLoader::new();
        rt.block_on(async {
            loader
                .register_skills(vec![SkillSpec::new(
                    "x".into(),
                    "X".into(),
                    "d".into(),
                    SkillLayer::Bundled,
                )])
                .await;
            assert_eq!(loader.count().await, 1);
            assert!(loader.remove_skill("x").is_some());
            assert_eq!(loader.count().await, 0);
        });
    }

    #[test]
    fn search_finds_relevant() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let loader = SkillLoader::new();
        rt.block_on(async {
            loader
                .register_skills(vec![
                    SkillSpec::new("git".into(), "Git".into(), "version control".into(), SkillLayer::Bundled),
                    SkillSpec::new("web".into(), "Web".into(), "search online".into(), SkillLayer::Bundled),
                ])
                .await;
            let hits = loader.search("git", 5);
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].skill.id, "git");
        });
    }

    #[test]
    fn skill_filter_eligible_only_uses_requires() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let loader = SkillLoader::new();
        rt.block_on(async {
            let mut needs_binary = SkillSpec::new(
                "needs-binary".into(),
                "NeedsBinary".into(),
                "d".into(),
                SkillLayer::Bundled,
            );
            needs_binary.requires = SkillRequires {
                binaries: Some(vec!["definitely-not-a-real-binary-xyz".to_string()]),
                ..SkillRequires::default()
            };
            loader
                .register_skills(vec![
                    SkillSpec::new("plain".into(), "Plain".into(), "d".into(), SkillLayer::Bundled),
                    needs_binary,
                ])
                .await;
            let filter = SkillFilter {
                eligible_only: true,
                ..SkillFilter::default()
            };
            // Without a checker, eligible_only is advisory and both are kept.
            let hits = loader.get_skills_filtered(&filter);
            assert_eq!(hits.len(), 2);

            // With a checker, the skill requiring a missing binary is dropped.
            let checker = crate::eligibility::EligibilityChecker::new();
            let hits = loader.get_skills_filtered_with(&filter, &checker);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].id, "plain");
        });
    }

    #[test]
    fn meta_skill_with_steps_parses() {
        let content = r#"---
id: workflow
name: Workflow
kind: meta
description: A DAG
steps:
  - id: s1
    name: Classify
    type: llm_classify
    output_choices: [yes, no]
  - id: s2
    name: Chat
    type: llm_chat
    prompt: "hello"
    depends_on: [s1]
---
"#;
        let loader = SkillLoader::new();
        let (spec, warnings) = loader
            .parse_skill_md(content, SkillLayer::Managed, None)
            .unwrap();
        assert!(spec.is_meta());
        assert_eq!(spec.steps.len(), 2);
        assert_eq!(spec.steps[0].step_type, StepType::LlmClassify);
        assert!(warnings.is_empty());
    }
}
