//! Final-diff contract diagnostics for coding-agent runs.
//!
//! Mirrors the Python backend's `engine/final_diff_contract.py`. Classifies the
//! paths in a final workspace diff (source / scratch / test-like / docs /
//! generated / diagnostic-source-like) and detects suspicious final states such
//! as a diff with no source changes, lost source mutations, or scratch
//! artifacts polluting the diff.

use regex::Regex;
use std::sync::OnceLock;

/// One classification bucket for a changed path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinalDiffPathKind {
    /// No classification (empty or `/dev/null`).
    Unknown,
    /// A test file or test directory.
    TestLike,
    /// A scratch / temp / debug artifact.
    Scratch,
    /// A documentation file.
    Docs,
    /// A generated / derived artifact.
    Generated,
    /// A real source file.
    Source,
}

impl FinalDiffPathKind {
    /// The wire-string spelling used by the Python runtime.
    pub fn as_str(self) -> &'static str {
        match self {
            FinalDiffPathKind::Unknown => "unknown",
            FinalDiffPathKind::TestLike => "test-like",
            FinalDiffPathKind::Scratch => "scratch",
            FinalDiffPathKind::Docs => "docs",
            FinalDiffPathKind::Generated => "generated",
            FinalDiffPathKind::Source => "source",
        }
    }
}

fn regexes() -> &'static Vec<(FinalDiffPathKind, Vec<Regex>)> {
    static REGEXES: OnceLock<Vec<(FinalDiffPathKind, Vec<Regex>)>> = OnceLock::new();
    REGEXES
        .get_or_init(|| {
            let patterns: &[(FinalDiffPathKind, &[&str])] = &[
                (
                    FinalDiffPathKind::TestLike,
                    &[
                        r"(?i)(^|/)(test|tests|__tests__)(/|$)",
                        r"(?i)(^|/)[^/]+\.(spec|test)\.[^/]+$",
                        r"(?i)(^|/)test_[^/]+\.[^/]+$",
                        r"(?i)(^|/)[^/]+_test\.[^/]+$",
                        r"(?i)(^|/)[^/]*test[^/]*\.(py|js|ts|tsx|rb|php|sh|txt|java|go|zsh)$",
                    ],
                ),
                (
                    FinalDiffPathKind::Scratch,
                    &[
                        r"(?i)(^|/)(tmp|temp|scratch)(/|$)",
                        r"(?i)(^|/)\.?(pytest_cache|mypy_cache|ruff_cache|phpunit\.cache)(/|$)",
                        r"(?i)^[^/]*(debug|repro|reproduce|scratch|verify|inspect|investigate)[^/]*\.(py|js|mjs|cjs|ts|rb|php|sh|txt|md|json|yaml|yml|patch|diff|zsh)$",
                        r"(?i)^[^/]*(debug|repro|reproduce|scratch)[^/]*/",
                        r"(?i)(^|/)[^/]*\.(patch|diff)$",
                    ],
                ),
                (
                    FinalDiffPathKind::Docs,
                    &[
                        r"(?i)(^|/)(docs?|documentation|manual)(/|$)",
                        r"(?i)(^|/)readme(\.[^/]*)?$",
                    ],
                ),
                (
                    FinalDiffPathKind::Generated,
                    &[
                        r"(?i)(^|/)(dist|build|target|generated|gen)(/|$)",
                        r"(?i)(^|/)[^/]*(generated|prebuilt|bundle|min)\.[^/]+$",
                    ],
                ),
            ];
            patterns
                .iter()
                .map(|(kind, regexes)| {
                    let compiled = regexes
                        .iter()
                        .map(|pattern| Regex::new(pattern).expect("valid final-diff pattern"))
                        .collect();
                    (*kind, compiled)
                })
                .collect()
        })
}

fn diagnostic_source_like_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS
        .get_or_init(|| {
            [
                r"(?i)^\.?(php_cs|php-cs|php-cs-fixer)[^/]*\.(php|dist|json|ya?ml)$",
                r"(?i)^[^/]*(check|verify|inspect|investigate|trace|analy[sz]e|analysis)[^/]*\.(py|js|ts|rb|php|sh|txt|md|json|yaml|yml|zsh)$",
                r"(?i)(^|/)[^/]*(?:[_-](?:test|repro|reproduce|debug|scratch)|(?:test|repro|reproduce|debug|scratch)[_-])[^/]*/",
                r"^(?:[^/]+/){7,}[^/]+\.txt$",
            ]
            .into_iter()
            .map(|pattern| Regex::new(pattern).expect("valid diagnostic pattern"))
            .collect()
        })
        .as_slice()
}

/// Normalize a path the way the Python runtime does before classification.
pub fn normalize_final_diff_path(path: &str) -> String {
    let text = path.trim().replace('\\', "/");
    if text == "/dev/null" {
        return text;
    }
    let text = if let Some(rest) = text.strip_prefix("a/") {
        rest.to_string()
    } else if let Some(rest) = text.strip_prefix("b/") {
        rest.to_string()
    } else {
        text
    };
    // Split into components, resolving "." and "..", collapsing duplicates.
    let mut components: Vec<&str> = Vec::new();
    for component in text.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                components.pop();
            }
            _ => components.push(component),
        }
    }
    let joined = components.join("/");
    // lstrip("/") — classification never cares about the leading root.
    joined.trim_start_matches('/').to_string()
}

/// Classify a changed path for final-patch diagnostics.
///
/// The classifier is intentionally conservative for nested source trees:
/// root-level debug/repro/check artifacts are scratch, while paths under
/// source directories remain source unless they match standard test/generated/
/// doc locations.
pub fn classify_final_diff_path(relative_path: &str) -> FinalDiffPathKind {
    let normalized = normalize_final_diff_path(relative_path);
    if normalized.is_empty() || normalized == "/dev/null" {
        return FinalDiffPathKind::Unknown;
    }
    for (kind, compiled) in regexes() {
        if compiled.iter().any(|re| re.is_match(&normalized)) {
            return *kind;
        }
    }
    FinalDiffPathKind::Source
}

/// Structured summary of whether the current final diff looks actionable.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalDiffContractObservation {
    /// All paths present in the final diff.
    pub diff_paths: Vec<String>,
    /// Diff paths classified as source.
    pub source_paths: Vec<String>,
    /// Diff paths classified as scratch.
    pub scratch_paths: Vec<String>,
    /// Diff paths classified as test-like.
    pub test_like_paths: Vec<String>,
    /// Diff paths classified as docs.
    pub docs_paths: Vec<String>,
    /// Diff paths classified as generated.
    pub generated_paths: Vec<String>,
    /// Source-classified paths that look like diagnostic artifacts.
    pub diagnostic_source_like_paths: Vec<String>,
    /// Source paths that are not diagnostic-source-like.
    pub actionable_source_paths: Vec<String>,
    /// Candidate source paths (touched, changed receipts, or recent reads).
    pub candidate_source_paths: Vec<String>,
    /// Candidate source paths absent from the current diff.
    pub candidate_source_missing_paths: Vec<String>,
    /// Candidate actionable source paths absent from the current diff.
    pub candidate_actionable_source_missing_paths: Vec<String>,
    /// Source paths read earlier in the run.
    pub read_source_paths: Vec<String>,
    /// Read source paths absent from the current diff.
    pub read_source_missing_paths: Vec<String>,
    /// Read actionable source paths absent from the current diff.
    pub read_actionable_source_missing_paths: Vec<String>,
    /// Diff paths that were also mutation targets.
    pub mutation_overlap_paths: Vec<String>,
    /// Source paths with successful mutation receipts.
    pub changed_source_receipt_paths: Vec<String>,
    /// Changed-source receipt paths absent from the current diff.
    pub lost_source_mutation_paths: Vec<String>,
    /// The raw source-diff candidates passed in.
    pub source_diff_candidates: Vec<serde_json::Value>,
    /// Candidate ids that are recoverable given lost source mutations.
    pub recoverable_candidate_ids: Vec<String>,
    /// Detected contract violations.
    pub triggers: Vec<String>,
}

impl FinalDiffContractObservation {
    /// Whether the observation is suspicious.
    pub fn suspicious(&self) -> bool {
        !self.triggers.is_empty()
    }

    /// The first trigger, or `final_diff_contract_ok`.
    pub fn primary_reason(&self) -> &str {
        self.triggers.first().map(|s| s.as_str()).unwrap_or("final_diff_contract_ok")
    }

    /// Render the observation as a JSON event-details payload.
    pub fn to_event_details(&self) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        let list = |v: &[String]| {
            serde_json::Value::Array(v.iter().map(|s| serde_json::Value::String(s.clone())).collect())
        };
        obj.insert("diff_paths".into(), list(&self.diff_paths));
        obj.insert("source_paths".into(), list(&self.source_paths));
        obj.insert("scratch_paths".into(), list(&self.scratch_paths));
        obj.insert("test_like_paths".into(), list(&self.test_like_paths));
        obj.insert("docs_paths".into(), list(&self.docs_paths));
        obj.insert("generated_paths".into(), list(&self.generated_paths));
        obj.insert("diagnostic_source_like_paths".into(), list(&self.diagnostic_source_like_paths));
        obj.insert("actionable_source_paths".into(), list(&self.actionable_source_paths));
        obj.insert("candidate_source_paths".into(), list(&self.candidate_source_paths));
        obj.insert("candidate_source_missing_paths".into(), list(&self.candidate_source_missing_paths));
        obj.insert(
            "candidate_actionable_source_missing_paths".into(),
            list(&self.candidate_actionable_source_missing_paths),
        );
        obj.insert("read_source_paths".into(), list(&self.read_source_paths));
        obj.insert("read_source_missing_paths".into(), list(&self.read_source_missing_paths));
        obj.insert(
            "read_actionable_source_missing_paths".into(),
            list(&self.read_actionable_source_missing_paths),
        );
        obj.insert("mutation_overlap_paths".into(), list(&self.mutation_overlap_paths));
        obj.insert("changed_source_receipt_paths".into(), list(&self.changed_source_receipt_paths));
        obj.insert("lost_source_mutation_paths".into(), list(&self.lost_source_mutation_paths));
        obj.insert(
            "source_diff_candidate_count".into(),
            serde_json::Value::from(self.source_diff_candidates.len()),
        );
        obj.insert("recoverable_candidate_ids".into(), list(&self.recoverable_candidate_ids));
        obj.insert("recoverable_candidate_count".into(), serde_json::Value::from(self.recoverable_candidate_ids.len()));
        obj.insert("triggers".into(), list(&self.triggers));
        obj.insert("source_file_count".into(), serde_json::Value::from(self.source_paths.len()));
        obj.insert("scratch_file_count".into(), serde_json::Value::from(self.scratch_paths.len()));
        obj.insert("test_like_file_count".into(), serde_json::Value::from(self.test_like_paths.len()));
        obj.insert(
            "diagnostic_source_like_count".into(),
            serde_json::Value::from(self.diagnostic_source_like_paths.len()),
        );
        obj.insert("actionable_source_count".into(), serde_json::Value::from(self.actionable_source_paths.len()));
        obj.insert(
            "diagnostic_source_like_only".into(),
            serde_json::Value::Bool(!self.source_paths.is_empty() && self.actionable_source_paths.is_empty()),
        );
        obj.insert("candidate_source_count".into(), serde_json::Value::from(self.candidate_source_paths.len()));
        obj.insert(
            "candidate_actionable_source_missing_count".into(),
            serde_json::Value::from(self.candidate_actionable_source_missing_paths.len()),
        );
        obj.insert("read_source_count".into(), serde_json::Value::from(self.read_source_paths.len()));
        obj.insert("read_source_missing_count".into(), serde_json::Value::from(self.read_source_missing_paths.len()));
        obj.insert(
            "read_actionable_source_missing_count".into(),
            serde_json::Value::from(self.read_actionable_source_missing_paths.len()),
        );
        obj.insert("mutation_overlap_count".into(), serde_json::Value::from(self.mutation_overlap_paths.len()));
        obj.insert(
            "changed_source_receipt_count".into(),
            serde_json::Value::from(self.changed_source_receipt_paths.len()),
        );
        obj.insert("lost_source_mutation_count".into(), serde_json::Value::from(self.lost_source_mutation_paths.len()));
        obj.insert("suspicious".into(), serde_json::Value::Bool(self.suspicious()));
        obj.insert("primary_reason".into(), serde_json::Value::String(self.primary_reason().to_string()));
        serde_json::Value::Object(obj)
    }
}

/// Build a [`FinalDiffContractObservation`] from run records.
///
/// `read_records`, `write_records`, `mutation_records`, `mutation_receipts` and
/// `source_diff_candidates` are JSON objects matching the Python runtime's
/// record shapes (`relative_path`, `paths`, `classification`, `changed`,
/// `lost`, `restored`, `candidate_id`, ...).
#[allow(clippy::too_many_arguments)]
pub fn build_final_diff_contract_observation(
    diff_paths: &[String],
    read_records: &[serde_json::Value],
    write_records: &[serde_json::Value],
    mutation_records: &[serde_json::Value],
    mutation_receipts: &[serde_json::Value],
    source_diff_candidates: &[serde_json::Value],
    known_scratch_paths: &[String],
) -> FinalDiffContractObservation {
    let normalized_diff_paths = unique_paths(diff_paths);
    let known_scratch_set: std::collections::HashSet<String> = unique_paths(known_scratch_paths).into_iter().collect();

    let mut by_kind: std::collections::HashMap<FinalDiffPathKind, Vec<String>> = Default::default();
    for path in &normalized_diff_paths {
        let kind = if known_scratch_set.contains(path) {
            FinalDiffPathKind::Scratch
        } else {
            classify_final_diff_path(path)
        };
        by_kind.entry(kind).or_default().push(path.clone());
    }
    let kind_list = |kind| by_kind.get(&kind).cloned().unwrap_or_default();

    let all_write_mutation: Vec<serde_json::Value> =
        write_records.iter().chain(mutation_records.iter()).cloned().collect();
    let touched_paths = paths_from_records(&all_write_mutation, Some(&[FinalDiffPathKind::Scratch]))
        .into_iter()
        .filter(|path| !known_scratch_set.contains(path))
        .collect::<Vec<_>>();
    let changed_source_receipts = changed_source_paths_from_receipts(mutation_receipts)
        .into_iter()
        .filter(|path| !known_scratch_set.contains(path))
        .collect::<Vec<_>>();
    let touched_source_paths = source_paths_from_records(&all_write_mutation, &known_scratch_set);
    let read_source_paths = source_paths_from_records(read_records, &known_scratch_set);
    let candidate_source_paths = if !touched_source_paths.is_empty() {
        touched_source_paths.clone()
    } else if !changed_source_receipts.is_empty() {
        changed_source_receipts.clone()
    } else {
        read_source_paths.iter().rev().take(10).rev().cloned().collect()
    };

    let source_paths = kind_list(FinalDiffPathKind::Source);
    let source_set: std::collections::HashSet<String> = source_paths.iter().cloned().collect();
    let diagnostic_source_like_paths = source_paths
        .iter()
        .filter(|path| looks_diagnostic_source_like_path(path))
        .cloned()
        .collect::<Vec<_>>();
    let diagnostic_source_like_set: std::collections::HashSet<String> =
        diagnostic_source_like_paths.iter().cloned().collect();
    let actionable_source_paths = source_paths
        .iter()
        .filter(|path| !diagnostic_source_like_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let candidate_missing = candidate_source_paths
        .iter()
        .filter(|path| !source_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let read_source_missing = read_source_paths
        .iter()
        .filter(|path| !source_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let actionable_source_set: std::collections::HashSet<String> =
        actionable_source_paths.iter().cloned().collect();
    let candidate_actionable_missing = candidate_source_paths
        .iter()
        .filter(|path| !actionable_source_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let read_actionable_missing = read_source_paths
        .iter()
        .filter(|path| !actionable_source_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();

    let mutation_paths: std::collections::HashSet<String> =
        paths_from_records(mutation_records, None).into_iter().collect();
    let mutation_overlap = normalized_diff_paths
        .iter()
        .filter(|path| mutation_paths.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let lost_source_mutations = changed_source_receipts
        .iter()
        .filter(|path| !source_set.contains(*path))
        .cloned()
        .collect::<Vec<_>>();
    let normalized_candidates = source_diff_candidates.to_vec();
    let recoverable_candidate_ids = recoverable_candidate_ids(&normalized_candidates, &lost_source_mutations);

    let scratch_paths = kind_list(FinalDiffPathKind::Scratch);
    let test_like_paths = kind_list(FinalDiffPathKind::TestLike);
    let docs_paths = kind_list(FinalDiffPathKind::Docs);
    let generated_paths = kind_list(FinalDiffPathKind::Generated);

    let mut triggers: Vec<String> = Vec::new();
    if !lost_source_mutations.is_empty() {
        triggers.push("source_mutation_lost_before_final".to_string());
    }
    if normalized_diff_paths.is_empty() && !touched_paths.is_empty() {
        triggers.push("workspace_writes_without_final_diff".to_string());
    }
    if !normalized_diff_paths.is_empty() && source_paths.is_empty() {
        triggers.push("final_diff_without_source".to_string());
    }
    if !normalized_diff_paths.is_empty()
        && !touched_source_paths.is_empty()
        && !candidate_missing.is_empty()
        && {
            let touched_set: std::collections::HashSet<String> = touched_source_paths.iter().cloned().collect();
            touched_set.is_disjoint(&source_set)
        }
    {
        triggers.push("candidate_source_drift".to_string());
    }
    if !scratch_paths.is_empty() {
        triggers.push("scratch_artifact_in_final_diff".to_string());
    }
    if !diagnostic_source_like_paths.is_empty() {
        triggers.push("diagnostic_source_like_in_final_diff".to_string());
    }
    if test_like_pollution_is_suspicious(test_like_paths.len(), source_paths.len()) {
        triggers.push("test_like_heavy_final_diff".to_string());
    }
    if !generated_paths.is_empty() {
        triggers.push("generated_artifact_in_final_diff".to_string());
    }

    FinalDiffContractObservation {
        diff_paths: normalized_diff_paths,
        source_paths,
        scratch_paths,
        test_like_paths,
        docs_paths,
        generated_paths,
        diagnostic_source_like_paths,
        actionable_source_paths,
        candidate_source_paths,
        candidate_source_missing_paths: candidate_missing,
        candidate_actionable_source_missing_paths: candidate_actionable_missing,
        read_source_paths,
        read_source_missing_paths: read_source_missing,
        read_actionable_source_missing_paths: read_actionable_missing,
        mutation_overlap_paths: mutation_overlap,
        changed_source_receipt_paths: changed_source_receipts,
        lost_source_mutation_paths: lost_source_mutations,
        source_diff_candidates: normalized_candidates,
        recoverable_candidate_ids,
        triggers: unique_strings(&triggers),
    }
}

/// Render the recovery message shown to the model.
pub fn final_diff_contract_recovery_message(observation: &FinalDiffContractObservation) -> String {
    let reason = observation.primary_reason().replace('_', " ");
    let diff_text = render_path_list(&observation.diff_paths);
    let source_text = render_path_list(&observation.source_paths);
    let candidate_text = render_path_list(&observation.candidate_source_missing_paths);
    let lost_source_text = render_path_list(&observation.lost_source_mutation_paths);
    let pollution_paths: Vec<String> = observation
        .scratch_paths
        .iter()
        .chain(observation.test_like_paths.iter())
        .chain(observation.diagnostic_source_like_paths.iter())
        .cloned()
        .collect();
    let pollution_text = render_path_list(&pollution_paths);

    let mut message = String::new();
    message.push_str("[Runtime final-diff check]\n");
    message.push_str("The model is about to finish, but the current repository diff looks suspicious: ");
    message.push_str(&reason);
    message.push_str(". Current diff paths: ");
    message.push_str(&diff_text);
    message.push_str(".");
    if !source_text.is_empty() {
        message.push_str(&format!(" Current source diff paths: {source_text}."));
    }
    if !candidate_text.is_empty() {
        message.push_str(&format!(
            " Source candidate(s) seen earlier but absent from the current diff: {candidate_text}."
        ));
    }
    if !lost_source_text.is_empty() {
        message.push_str(&format!(
            " Successful source edit receipt(s) exist, but these source path(s) are absent from the current diff: {lost_source_text}."
        ));
    }
    if !observation.recoverable_candidate_ids.is_empty() {
        let recoverable = render_path_list(&observation.recoverable_candidate_ids);
        message.push_str(&format!(
            " Recoverable source diff candidate(s): {recoverable}. Inspect `git diff`; keep or recreate the source patch if it is still correct, or explicitly explain why the previous source edit should be discarded."
        ));
    }
    if !pollution_paths.is_empty() {
        message.push_str(&format!(
            " Suspicious scratch/debug/repro/test-like/diagnostic/source-like paths in the current diff: {pollution_text}."
        ));
    }
    message.push_str(
        " Before finalizing, inspect `git diff`, keep the smallest necessary source patch, and remove temporary scratch/debug/repro files. Keep test or documentation changes only if they are intentional and travel with the source fix.",
    );
    message
}

fn paths_from_records(records: &[serde_json::Value], excluded: Option<&[FinalDiffPathKind]>) -> Vec<String> {
    let excluded_set: std::collections::HashSet<FinalDiffPathKind> =
        excluded.map(|kinds| kinds.iter().copied().collect()).unwrap_or_default();
    let mut paths: Vec<String> = Vec::new();
    for record in records {
        let Some(obj) = record.as_object() else {
            continue;
        };
        if excluded_set.contains(&FinalDiffPathKind::Scratch)
            && obj.get("classification").and_then(|v| v.as_str()) == Some("scratch")
        {
            continue;
        }
        if let Some(relative_path) = obj.get("relative_path").and_then(|v| v.as_str()) {
            if !relative_path.is_empty() {
                paths.push(relative_path.to_string());
                continue;
            }
        }
        if let Some(items) = obj.get("paths").and_then(|v| v.as_array()) {
            for item in items {
                let Some(nested) = item.as_object() else {
                    continue;
                };
                if excluded_set.contains(&FinalDiffPathKind::Scratch)
                    && nested.get("classification").and_then(|v| v.as_str()) == Some("scratch")
                {
                    continue;
                }
                if let Some(nested_path) = nested.get("relative_path").and_then(|v| v.as_str()) {
                    if !nested_path.is_empty() {
                        paths.push(nested_path.to_string());
                    }
                }
            }
        }
    }
    unique_paths(&paths)
}

fn source_paths_from_records(
    records: &[serde_json::Value],
    known_scratch_paths: &std::collections::HashSet<String>,
) -> Vec<String> {
    paths_from_records(records, Some(&[FinalDiffPathKind::Scratch]))
        .into_iter()
        .filter(|path| {
            !known_scratch_paths.contains(path)
                && classify_final_diff_path(path) == FinalDiffPathKind::Source
        })
        .collect()
}

fn changed_source_paths_from_receipts(receipts: &[serde_json::Value]) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for receipt in receipts {
        let Some(obj) = receipt.as_object() else {
            continue;
        };
        if obj.get("changed").and_then(|v| v.as_bool()) != Some(true) {
            continue;
        }
        if obj.get("classification").and_then(|v| v.as_str()) != Some("source") {
            continue;
        }
        if let Some(relative_path) = obj.get("relative_path").and_then(|v| v.as_str()) {
            if !relative_path.is_empty() {
                paths.push(relative_path.to_string());
            }
        }
    }
    unique_paths(&paths)
}

fn recoverable_candidate_ids(
    candidates: &[serde_json::Value],
    lost_source_paths: &[String],
) -> Vec<String> {
    let lost_set: std::collections::HashSet<String> = unique_paths(lost_source_paths).into_iter().collect();
    if lost_set.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<String> = Vec::new();
    for candidate in candidates {
        let Some(obj) = candidate.as_object() else {
            continue;
        };
        if obj.get("lost").and_then(|v| v.as_bool()) != Some(true)
            || obj.get("restored").and_then(|v| v.as_bool()) == Some(true)
        {
            continue;
        }
        let candidate_paths: Vec<String> = obj
            .get("paths")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        let unique_candidate_paths = unique_paths(&candidate_paths);
        if !unique_candidate_paths.iter().any(|path| lost_set.contains(path)) {
            continue;
        }
        if let Some(candidate_id) = obj.get("candidate_id").and_then(|v| v.as_str()) {
            result.push(candidate_id.to_string());
        }
    }
    unique_strings(&result)
}

fn test_like_pollution_is_suspicious(test_like_count: usize, source_count: usize) -> bool {
    if test_like_count == 0 {
        return false;
    }
    if source_count == 0 {
        return true;
    }
    test_like_count >= 3 && test_like_count > source_count * 2
}

fn looks_diagnostic_source_like_path(path: &str) -> bool {
    diagnostic_source_like_patterns().iter().any(|re| re.is_match(path))
}

fn render_path_list(paths: &[String]) -> String {
    render_path_list_with_limit(paths, 8)
}

fn render_path_list_with_limit(paths: &[String], limit: usize) -> String {
    if paths.is_empty() {
        return "<none>".to_string();
    }
    let mut rendered = paths[..paths.len().min(limit)].join(", ");
    if paths.len() > limit {
        rendered.push_str(", ...");
    }
    rendered
}

fn unique_paths(paths: &[String]) -> Vec<String> {
    let normalized: Vec<String> = paths.iter().map(|path| normalize_final_diff_path(path)).collect();
    unique_strings(&normalized)
}

fn unique_strings(values: &[String]) -> Vec<String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for value in values {
        if value.is_empty() || seen.contains(value) {
            continue;
        }
        seen.insert(value.clone());
        result.push(value.clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_test_like() {
        assert_eq!(classify_final_diff_path("tests/test_a.py"), FinalDiffPathKind::TestLike);
        assert_eq!(classify_final_diff_path("src/foo.spec.ts"), FinalDiffPathKind::TestLike);
        assert_eq!(classify_final_diff_path("src/foo_test.py"), FinalDiffPathKind::TestLike);
    }

    #[test]
    fn test_classify_scratch() {
        assert_eq!(classify_final_diff_path("tmp/debug.log"), FinalDiffPathKind::Scratch);
        assert_eq!(classify_final_diff_path("repro.py"), FinalDiffPathKind::Scratch);
        assert_eq!(classify_final_diff_path("debug_issue.sh"), FinalDiffPathKind::Scratch);
        // A nested path under a source tree stays source unless it matches the
        // standard test/generated/doc locations.
        assert_eq!(classify_final_diff_path("src/debug_issue.py"), FinalDiffPathKind::Source);
    }

    #[test]
    fn test_classify_docs_generated_source() {
        assert_eq!(classify_final_diff_path("docs/guide.md"), FinalDiffPathKind::Docs);
        assert_eq!(classify_final_diff_path("dist/bundle.js"), FinalDiffPathKind::Generated);
        assert_eq!(classify_final_diff_path("src/lib.rs"), FinalDiffPathKind::Source);
    }

    #[test]
    fn test_classify_unknown_and_dev_null() {
        assert_eq!(classify_final_diff_path(""), FinalDiffPathKind::Unknown);
        assert_eq!(classify_final_diff_path("/dev/null"), FinalDiffPathKind::Unknown);
    }

    #[test]
    fn test_normalize_path_variants() {
        assert_eq!(normalize_final_diff_path("a/foo/bar"), "foo/bar");
        assert_eq!(normalize_final_diff_path("./foo//bar"), "foo/bar");
        assert_eq!(normalize_final_diff_path("b/x.rs"), "x.rs");
        assert_eq!(normalize_final_diff_path("/etc/hosts"), "etc/hosts");
        assert_eq!(normalize_final_diff_path("foo\\bar"), "foo/bar");
    }

    #[test]
    fn test_final_diff_without_source_trigger() {
        let observation = build_final_diff_contract_observation(
            &["scratch.txt".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert!(observation.suspicious());
        assert!(observation.triggers.contains(&"final_diff_without_source".to_string()));
    }

    #[test]
    fn test_lost_source_mutation_trigger() {
        let receipts = vec![serde_json::json!({
            "changed": true,
            "classification": "source",
            "relative_path": "src/lib.rs",
        })];
        let observation = build_final_diff_contract_observation(
            &["tests/other_test.py".to_string()],
            &[],
            &[],
            &[],
            &receipts,
            &[],
            &[],
        );
        assert!(observation.triggers.contains(&"source_mutation_lost_before_final".to_string()));
        assert_eq!(observation.changed_source_receipt_paths, vec!["src/lib.rs"]);
    }

    #[test]
    fn test_clean_source_diff_no_triggers() {
        let observation = build_final_diff_contract_observation(
            &["src/lib.rs".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert!(!observation.suspicious());
        assert_eq!(observation.primary_reason(), "final_diff_contract_ok");
        assert_eq!(observation.source_paths, vec!["src/lib.rs"]);
    }

    #[test]
    fn test_diagnostic_source_like_detected() {
        // "analysis.py" is classified as source (not scratch) but matches the
        // diagnostic-source-like pattern.
        let observation = build_final_diff_contract_observation(
            &["analysis.py".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        assert!(observation.triggers.contains(&"diagnostic_source_like_in_final_diff".to_string()));
        assert_eq!(observation.diagnostic_source_like_paths, vec!["analysis.py"]);
        assert!(observation.actionable_source_paths.is_empty());
    }

    #[test]
    fn test_known_scratch_paths_override_classification() {
        let observation = build_final_diff_contract_observation(
            &["src/custom_generated.rs".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &["src/custom_generated.rs".to_string()],
        );
        assert!(observation.triggers.contains(&"scratch_artifact_in_final_diff".to_string()));
        assert_eq!(observation.scratch_paths, vec!["src/custom_generated.rs"]);
    }

    #[test]
    fn test_recovery_message_renders() {
        let observation = build_final_diff_contract_observation(
            &["scratch/debug.log".to_string(), "src/lib.rs".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let message = final_diff_contract_recovery_message(&observation);
        assert!(message.starts_with("[Runtime final-diff check]\n"));
        assert!(message.contains("scratch_artifact_in_final_diff".replace('_', " ").as_str()));
    }

    #[test]
    fn test_to_event_details_counts() {
        let observation = build_final_diff_contract_observation(
            &["src/lib.rs".to_string()],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[],
        );
        let details = observation.to_event_details();
        assert_eq!(details["source_file_count"], 1);
        assert_eq!(details["primary_reason"], "final_diff_contract_ok");
    }
}
