//! Helpers for refusing provider-only projected tool arguments.
//!
//! Mirrors the Python `opensquilla.tools.projected_arguments` module: detects
//! provider-compacted placeholder text inside tool-call arguments so the
//! dispatcher can refuse to execute a call that would operate on a
//! placeholder rather than real content.

use regex::Regex;
use serde_json::Value;

/// Prefix for instantiated tool-use argument projection strings.
pub const TOOL_ARGUMENT_PROJECTION_PREFIX: &str = "[tool_use_argument_projection]\n";
/// Prefix for historical tool-argument omission markers.
pub const HISTORICAL_TOOL_ARGUMENT_PROJECTION_PREFIX: &str = "[historical_tool_argument_omitted]\n";
/// Prefix for invalid provider-context projection strings.
pub const INVALID_PROVIDER_CONTEXT_PROJECTION_PREFIX: &str =
    "[invalid_provider_context_projection:";
/// Prefix for provider-request tool-input compaction strings.
pub const PROVIDER_REQUEST_TOOL_INPUT_COMPACTED_PREFIX: &str =
    "[provider_request_tool_input_compacted:";
/// Object key that marks invalid provider-context arguments.
pub const INVALID_PROVIDER_CONTEXT_ARGUMENTS_KEY: &str = "_invalid_provider_context_arguments";
/// Object keys that mark compacted tool arguments.
pub const COMPACTED_TOOL_ARGUMENT_MARKERS: &[&str] = &[
    "_opensquilla_compacted_tool_arguments",
    "_opensquilla_compacted_tool_input",
];

/// Matches instantiated provider-request compaction markers anywhere inside a
/// string argument, not only at char 0. Requires the numeric fields the
/// marker producers always fill in ("<n> chars", "original_chars=<n>",
/// ":<n>:<hash>") so template literals and prose that merely name a marker
/// prefix do not match.
fn compacted_marker_substring_re() -> &'static Regex {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(
            r"\[provider_request_[a-z0-9_]*(?:compacted|omitted):[^\]\n]*(?:\d+ chars|original_chars=\d+)|\[opensquilla_compacted:[A-Za-z0-9_.-]+:\d+:[0-9a-f]{8,64}\]",
        )
        .expect("valid compacted-marker regex")
    });
    &RE
}

/// A single projected tool-argument match found during a recursive scan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectedToolArgumentMatch {
    /// Classification of the projection kind.
    pub kind: String,
    /// JSON path to the offending value ("" for a top-level string).
    pub path: String,
}

/// Whether a value is a provider-context marker (bool-ish `true`).
pub fn is_provider_context_marker_value(value: &Value) -> bool {
    match value {
        Value::Bool(true) => true,
        Value::String(s) => {
            let normalized = s.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "true" | "1" | "yes" | "on")
        }
        _ => false,
    }
}

fn projection_string_kind(value: &str) -> Option<&'static str> {
    let stripped = value.trim_start();
    if stripped.starts_with(PROVIDER_REQUEST_TOOL_INPUT_COMPACTED_PREFIX) {
        return Some("provider_request_projection_string");
    }
    if stripped.starts_with(TOOL_ARGUMENT_PROJECTION_PREFIX)
        || stripped.starts_with(HISTORICAL_TOOL_ARGUMENT_PROJECTION_PREFIX)
        || stripped.starts_with(INVALID_PROVIDER_CONTEXT_PROJECTION_PREFIX)
    {
        return Some("projection_string");
    }
    if compacted_marker_substring_re().is_match(value) {
        return Some("compacted_marker_substring");
    }
    None
}

/// Recursively scan a JSON value for projected/compacted placeholder markers.
///
/// Returns the first match found, with a JSON path describing its location.
pub fn find_projected_tool_argument(
    value: &Value,
    path: &str,
) -> Option<ProjectedToolArgumentMatch> {
    match value {
        Value::String(s) => {
            let kind = projection_string_kind(s)?;
            Some(ProjectedToolArgumentMatch {
                kind: kind.to_string(),
                path: path.to_string(),
            })
        }
        Value::Object(map) => {
            let mut marker_keys: Vec<&str> = COMPACTED_TOOL_ARGUMENT_MARKERS.to_vec();
            marker_keys.push(INVALID_PROVIDER_CONTEXT_ARGUMENTS_KEY);
            for key in marker_keys {
                if let Some(nested) = map.get(key) {
                    if is_provider_context_marker_value(nested) {
                        let nested_path = if path.is_empty() {
                            key.to_string()
                        } else {
                            format!("{path}.{key}")
                        };
                        return Some(ProjectedToolArgumentMatch {
                            kind: "provider_context_argument_marker".to_string(),
                            path: nested_path,
                        });
                    }
                }
            }
            for (key, nested) in map {
                let nested_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                if let Some(matched) = find_projected_tool_argument(nested, &nested_path) {
                    return Some(matched);
                }
            }
            None
        }
        Value::Array(items) => {
            for (index, nested) in items.iter().enumerate() {
                let nested_path = if path.is_empty() {
                    format!("[{index}]")
                } else {
                    format!("{path}[{index}]")
                };
                if let Some(matched) = find_projected_tool_argument(nested, &nested_path) {
                    return Some(matched);
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_projection_prefix() {
        let value = json!(format!(
            "{TOOL_ARGUMENT_PROJECTION_PREFIX}the file contents were here"
        ));
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.kind, "projection_string");
        assert_eq!(matched.path, "");
    }

    #[test]
    fn detects_provider_request_compacted_prefix() {
        let value = json!(format!(
            "{PROVIDER_REQUEST_TOOL_INPUT_COMPACTED_PREFIX}123 chars]"
        ));
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.kind, "provider_request_projection_string");
    }

    #[test]
    fn detects_compacted_marker_substring() {
        let value =
            json!("prefix [provider_request_tool_input_compacted:foo original_chars=12] suffix");
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.kind, "compacted_marker_substring");

        let value2 = json!("[opensquilla_compacted:file_edit:7:a1b2c3d4e5f6a7b8]");
        assert_eq!(
            find_projected_tool_argument(&value2, "").map(|m| m.kind),
            Some("compacted_marker_substring".to_string())
        );
    }

    #[test]
    fn marker_missing_digits_does_not_match() {
        // Template literal / prose that merely names the prefix must not match.
        let value = json!("see [provider_request_tool_input_compacted: ...] docs");
        assert!(find_projected_tool_argument(&value, "").is_none());
    }

    #[test]
    fn detects_object_marker_keys() {
        let value = json!({
            "path": "src/main.rs",
            "old_text": "hello",
            "_invalid_provider_context_arguments": true,
        });
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.kind, "provider_context_argument_marker");
        assert_eq!(matched.path, "_invalid_provider_context_arguments");
    }

    #[test]
    fn detects_compacted_object_marker() {
        let value = json!({
            "path": "a.txt",
            "_opensquilla_compacted_tool_input": "yes",
        });
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.kind, "provider_context_argument_marker");
        assert_eq!(matched.path, "_opensquilla_compacted_tool_input");
    }

    #[test]
    fn recurses_into_nested_paths() {
        let value = json!({
            "outer": {
                "inner": [
                    {"ok": 1},
                    format!("{TOOL_ARGUMENT_PROJECTION_PREFIX}placeholder")
                ]
            }
        });
        let matched = find_projected_tool_argument(&value, "").expect("match");
        assert_eq!(matched.path, "outer.inner[1]");
        assert_eq!(matched.kind, "projection_string");
    }

    #[test]
    fn clean_arguments_pass() {
        let value = json!({"path": "src/main.rs", "old_text": "hello", "new_text": "hi"});
        assert!(find_projected_tool_argument(&value, "").is_none());
    }

    #[test]
    fn marker_value_truthiness() {
        assert!(is_provider_context_marker_value(&json!(true)));
        assert!(is_provider_context_marker_value(&json!("TRUE")));
        assert!(is_provider_context_marker_value(&json!("1")));
        assert!(!is_provider_context_marker_value(&json!(false)));
        assert!(!is_provider_context_marker_value(&json!("no")));
        assert!(!is_provider_context_marker_value(&json!(0)));
    }
}
