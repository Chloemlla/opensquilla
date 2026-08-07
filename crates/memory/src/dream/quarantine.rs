//! Generated-artifact quarantine rules for Dream.
//!
//! Parity stub mirroring `src/opensquilla/memory/dream/quarantine.py`. The
//! evidence-gated pipeline must never treat dream-generated artifacts (its own
//! state dirs, logs, `dream-*.jsonl` receipts) as candidate sources.

/// A path that the dream pipeline must never treat as a candidate source.
///
/// TODO(parity): port the Python rules (`memory/.dream_cursor`,
/// `memory/.dream*`, `logs/*`, `dream-*.jsonl`) from quarantine.py.
pub fn is_quarantined_path(path: &str) -> bool {
    let _ = path;
    false
}

/// Text that marks a document as dream-generated (must not be re-ingested).
///
/// TODO(parity): port marker detection (`opensquilla-dream-promotion:`,
/// `dream receipt`) from quarantine.py.
pub fn is_quarantined_text(text: &str) -> bool {
    let _ = text;
    false
}
