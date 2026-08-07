//! TokenJuice rule and reduction types.
//!
//! Mirrors the Python `tokenjuice.types` dataclasses (`Rule`, `Reduction`) plus
//! the loosely-typed `match` / `transforms` / `filters` / `summarize` / `failure`
//! / `counters` / `outputMatches` rule sub-objects. The Python `Rule` keeps those
//! fields as free-form dicts (the JSON schema is intentionally permissive); we
//! preserve the same flexibility by serde-deriving the typed leaf structs while
//! still tolerating missing or extra keys.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A named regex capture whose match count is appended to the reduction
/// summary (e.g. `{"name": "error", "pattern": "error", "flags": "i"}`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Counter {
    /// Label emitted in the summary line (e.g. `"error: 3"`).
    pub name: String,
    /// Regex applied to each line; a line counts if it matches anywhere.
    pub pattern: String,
    /// Optional regex flags: `i` (case-insensitive), `m` (multiline).
    #[serde(default)]
    pub flags: Option<String>,
}

/// An output substitution: if `pattern` matches anywhere in the (post-ANSI)
/// text, the entire reduction is replaced by `message`.
///
/// Python loads this from the rule's `outputMatches` list; the bundled JSON
/// files name it `matchOutput`, so we deserialize from either key (see
/// [`Rule::output_matches`]).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OutputMatch {
    pub pattern: String,
    pub message: String,
    #[serde(default)]
    pub flags: Option<String>,
}

/// Match criteria selecting when a rule applies.
///
/// All fields are optional; an empty [`RuleMatch`] matches everything (the
/// generic fallback). Field names mirror the Python JSON keys exactly.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RuleMatch {
    /// Tool names this rule applies to (e.g. `["exec"]`).
    #[serde(default)]
    pub tool_names: Vec<String>,
    /// Required first argv element (e.g. `["git"]`, `["npm"]`).
    #[serde(default)]
    pub argv0: Vec<String>,
    /// Git subcommands allowed (e.g. `["status"]`). Only enforced when the
    /// strict matcher env lever is enabled.
    #[serde(default)]
    pub git_subcommands: Vec<String>,
    /// Groups of tokens that must ALL be present in argv; ANY group may match.
    #[serde(default)]
    pub argv_includes: Vec<Vec<String>>,
    /// Like `argv_includes` but only enforced under the strict matcher.
    #[serde(default)]
    pub argv_includes_any: Vec<Vec<String>>,
    /// Substrings that must all appear (case-insensitive) in the command text.
    #[serde(default)]
    pub command_includes: Vec<String>,
    /// Substrings of which at least one must appear (case-insensitive) in the
    /// command text.
    #[serde(default)]
    pub command_includes_any: Vec<String>,
    /// Regex that must match the command text.
    #[serde(default)]
    pub command_regex: Option<String>,
    /// Exit codes this rule applies to.
    #[serde(default)]
    pub exit_codes: Vec<i64>,
    /// Regex that must match the tool output (multiline).
    #[serde(default)]
    pub output_regex: Option<String>,
}

/// Line transforms applied before windowing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Transforms {
    /// Strip ANSI escape sequences.
    #[serde(default)]
    pub strip_ansi: bool,
    /// Remove leading/trailing blank lines.
    #[serde(default)]
    pub trim_empty_edges: bool,
    /// Collapse runs of identical adjacent lines to a single line.
    #[serde(default)]
    pub dedupe_adjacent: bool,
}

/// Line filters applied after transforms, before windowing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Filters {
    /// Regex patterns; a line is dropped if ANY matches.
    #[serde(default)]
    pub skip_patterns: Vec<String>,
    /// Regex patterns; if present, only lines matching ANY are kept.
    #[serde(default)]
    pub keep_patterns: Vec<String>,
}

/// Head/tail window sizes for the success path.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Summarize {
    #[serde(default)]
    pub head: Option<i64>,
    #[serde(default)]
    pub tail: Option<i64>,
}

/// Head/tail window sizes for the failure path (non-zero exit).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Failure {
    /// Whether to use the failure window at all on non-zero exit.
    #[serde(default)]
    pub preserve_on_failure: bool,
    #[serde(default)]
    pub head: Option<i64>,
    #[serde(default)]
    pub tail: Option<i64>,
}

/// A TokenJuice reduction rule.
///
/// Serialized form matches the bundled JSON rule files. Unknown top-level keys
/// (e.g. `description`) are ignored. `matchOutput` is accepted as an alias for
/// `outputMatches` so the bundled JSON works as-is.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Rule {
    /// Unique rule id, e.g. `"git/status"` or `"generic/fallback"`.
    pub id: String,
    /// Rule family for grouping (e.g. `"git-status"`).
    #[serde(default = "default_family")]
    pub family: String,
    /// Match criteria. An absent/empty object matches everything.
    #[serde(rename = "match", default)]
    pub r#match: RuleMatch,
    /// Pre-window line transforms.
    #[serde(default)]
    pub transforms: Transforms,
    /// Skip/keep line filters.
    #[serde(default)]
    pub filters: Filters,
    /// Success head/tail window sizes.
    #[serde(default)]
    pub summarize: Summarize,
    /// Failure head/tail window sizes.
    #[serde(default)]
    pub failure: Failure,
    /// Named regex counters appended to the summary.
    #[serde(default)]
    pub counters: Vec<Counter>,
    /// Output substitutions. Read from `outputMatches` (Python loader key) and
    /// `matchOutput` (bundled JSON key); both are merged.
    #[serde(default)]
    pub output_matches: Vec<OutputMatch>,
    /// Text emitted when every line is filtered away.
    #[serde(default)]
    pub on_empty: Option<String>,
    /// Which line set counters run over: `"preKeep"` (before keep-patterns) or
    /// `"postKeep"` (after). Defaults to `"postKeep"`.
    #[serde(default = "default_counter_source")]
    pub counter_source: String,
    /// Higher priority rules are tried first within a family.
    #[serde(default)]
    pub priority: i64,
}

impl Rule {
    /// Read output substitutions from a raw JSON object, accepting both the
    /// `outputMatches` (loader) and `matchOutput` (bundled JSON) keys. Used by
    /// the loose loader when an unknown-key-tolerant parse is wanted.
    pub fn output_matches_from(value: &Value) -> Vec<OutputMatch> {
        let mut out = Vec::new();
        for key in ["outputMatches", "matchOutput"] {
            if let Some(arr) = value.get(key).and_then(Value::as_array) {
                for entry in arr {
                    if let Ok(item) = serde_json::from_value::<OutputMatch>(entry.clone()) {
                        out.push(item);
                    }
                }
            }
        }
        out
    }
}

fn default_family() -> String {
    "generic".to_string()
}

fn default_counter_source() -> String {
    "postKeep".to_string()
}

/// The compact summary produced by reducing a tool result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reduction {
    /// The reduced text to inline back into the LLM context.
    pub inline_text: String,
    /// Length of the original tool result, in chars.
    pub raw_chars: usize,
    /// Length of the reduced text, in chars.
    pub reduced_chars: usize,
    /// `reduced_chars / max(1, raw_chars)`.
    pub ratio: f64,
    /// Id of the rule that produced this reduction, if any.
    #[serde(default)]
    pub reducer: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_deserializes_fallback() {
        let json = r#"{
            "id": "generic/fallback",
            "family": "generic",
            "match": {},
            "transforms": {"stripAnsi": true, "dedupeAdjacent": true, "trimEmptyEdges": true},
            "summarize": {"head": 200, "tail": 200},
            "failure": {"preserveOnFailure": true, "head": 50, "tail": 50},
            "counters": [{"name": "error", "pattern": "error", "flags": "i"}]
        }"#;
        let rule: Rule = serde_json::from_str(json).unwrap();
        assert_eq!(rule.id, "generic/fallback");
        assert!(rule.transforms.strip_ansi);
        assert!(rule.transforms.dedupe_adjacent);
        assert_eq!(rule.summarize.head, Some(200));
        assert!(rule.failure.preserve_on_failure);
        assert_eq!(rule.counters.len(), 1);
        assert_eq!(rule.counters[0].name, "error");
        assert_eq!(rule.counter_source, "postKeep");
    }

    #[test]
    fn rule_reads_match_output_alias() {
        // Bundled JSON uses `matchOutput`; the loader key is `outputMatches`.
        let raw: Value = serde_json::from_str(
            r#"{"id":"x","matchOutput":[{"pattern":"up to date","message":"ok","flags":"i"}]}"#,
        )
        .unwrap();
        let matches = Rule::output_matches_from(&raw);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].message, "ok");
    }

    #[test]
    fn reduction_ratio_round_trip() {
        let r = Reduction {
            inline_text: "ab".to_string(),
            raw_chars: 10,
            reduced_chars: 2,
            ratio: 0.2,
            reducer: Some("generic/fallback".to_string()),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: Reduction = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
