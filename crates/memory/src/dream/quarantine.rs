//! Generated-artifact quarantine rules for Dream.
//!
//! Port of `src/opensquilla/memory/dream/quarantine.py`. The evidence-gated
//! pipeline must never treat dream-generated artifacts (its own state dirs,
//! logs, `dream-*.jsonl` receipts) as candidate sources.

/// Normalize a path to POSIX separators with any leading `.`/`/` characters
/// stripped, matching the Python `path.replace("\\", "/").lstrip("./")`.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
        .trim_start_matches(&['.', '/'][..])
        .to_string()
}

/// A path that the dream pipeline must never treat as a candidate source.
///
/// Port of the Python rules (`memory/.dream_cursor`, `memory/.dream*`,
/// `logs/*`, `dream-*.jsonl`) from quarantine.py.
pub fn is_quarantined_path(path: &str) -> bool {
    let normalized = normalize_path(path);
    if normalized == "memory/.dream_cursor" {
        return true;
    }
    if normalized.starts_with("memory/.dream") {
        return true;
    }
    if normalized == "logs" || normalized.starts_with("logs/") {
        return true;
    }
    let name = normalized.rsplit('/').next().unwrap_or("");
    name.starts_with("dream-") && name.ends_with(".jsonl")
}

/// Text that marks a document as dream-generated (must not be re-ingested).
///
/// Port of marker detection (`opensquilla-dream-promotion:`, `dream receipt`)
/// from quarantine.py.
pub fn is_quarantined_text(text: &str) -> bool {
    let lowered = text.to_lowercase();
    lowered.contains("opensquilla-dream-promotion:") || lowered.contains("dream receipt")
}
