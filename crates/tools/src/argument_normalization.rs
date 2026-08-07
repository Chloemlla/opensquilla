//! Shared canonicalization for provider-emitted tool arguments.
//!
//! Mirrors the Python `opensquilla.tools.argument_normalization` module:
//! maps common model-emitted argument aliases (`file_path`/`filePath` ->
//! `path`, `old_string` -> `old_text`, ...) to canonical names without
//! guessing values. Conflicting alias/canonical pairs are reported so the
//! caller can reject rather than silently drop content.

use serde_json::{Map, Value};

/// A conflicting alias/canonical value pair for a tool argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgumentAliasConflict {
    /// The alias key that carried a conflicting value.
    pub alias: String,
    /// The canonical key the alias maps to.
    pub canonical: String,
    /// The other key the alias conflicts with.
    pub conflicting_with: String,
}

/// A single alias that was successfully remapped to its canonical key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedAlias {
    /// The alias key that was removed.
    pub alias: String,
    /// The canonical key that received the value.
    pub canonical: String,
}

/// Result of canonicalizing common model-emitted argument aliases.
#[derive(Debug, Clone)]
pub struct ToolArgumentNormalizationResult {
    /// The normalized arguments (aliases remapped or dropped).
    pub arguments: Map<String, Value>,
    /// Aliases that were applied (removed, value moved to canonical).
    pub aliases_applied: Vec<AppliedAlias>,
    /// Conflicting alias/canonical pairs.
    pub conflicts: Vec<ToolArgumentAliasConflict>,
}

impl ToolArgumentNormalizationResult {
    /// Whether any alias was applied.
    pub fn changed(&self) -> bool {
        !self.aliases_applied.is_empty()
    }

    /// Whether any conflicting alias/canonical pair was found.
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

/// Canonical name per tool for provider-emitted argument aliases.
///
/// Key: alias name. Value: canonical name. Missing tools have no aliases.
fn tool_argument_aliases(tool_name: &str) -> &'static [(&'static str, &'static str)] {
    match tool_name {
        "edit_file" => &[
            ("file_path", "path"),
            ("filePath", "path"),
            ("old_string", "old_text"),
            ("oldString", "old_text"),
            ("oldText", "old_text"),
            ("new_string", "new_text"),
            ("newString", "new_text"),
            ("newText", "new_text"),
        ],
        "read_file" | "write_file" => &[("file_path", "path"), ("filePath", "path")],
        _ => &[],
    }
}

/// Map known tool argument aliases to canonical names without guessing values.
pub fn canonicalize_tool_arguments(
    tool_name: &str,
    arguments: &Map<String, Value>,
) -> ToolArgumentNormalizationResult {
    let aliases = tool_argument_aliases(tool_name);
    let mut normalized = arguments.clone();
    if aliases.is_empty() {
        return ToolArgumentNormalizationResult {
            arguments: normalized,
            aliases_applied: Vec::new(),
            conflicts: Vec::new(),
        };
    }

    // Group present aliases by their canonical name.
    let mut aliases_by_canonical: Vec<(&'static str, Vec<&'static str>)> = Vec::new();
    for (alias, canonical) in aliases.iter().copied() {
        if normalized.contains_key(alias) {
            match aliases_by_canonical
                .iter_mut()
                .find(|(c, _)| *c == canonical)
            {
                Some((_, present)) => present.push(alias),
                None => aliases_by_canonical.push((canonical, vec![alias])),
            }
        }
    }

    let mut aliases_applied: Vec<AppliedAlias> = Vec::new();
    let mut conflicts: Vec<ToolArgumentAliasConflict> = Vec::new();

    for (canonical, present_aliases) in aliases_by_canonical {
        let canonical_key = canonical.to_string();
        if normalized.contains_key(&canonical_key) {
            let canonical_value = normalized
                .get(&canonical_key)
                .expect("contains_key checked above")
                .clone();
            for alias in present_aliases {
                if normalized.get(alias) != Some(&canonical_value) {
                    conflicts.push(ToolArgumentAliasConflict {
                        alias: alias.to_string(),
                        canonical: canonical_key.clone(),
                        conflicting_with: canonical_key.clone(),
                    });
                    continue;
                }
                normalized.remove(alias);
                aliases_applied.push(AppliedAlias {
                    alias: alias.to_string(),
                    canonical: canonical_key.clone(),
                });
            }
            continue;
        }

        let selected_alias = present_aliases[0];
        let selected_value = normalized
            .get(selected_alias)
            .expect("present alias")
            .clone();
        let mut local_conflicts: Vec<ToolArgumentAliasConflict> = Vec::new();
        for alias in &present_aliases[1..] {
            if normalized.get(*alias) != Some(&selected_value) {
                local_conflicts.push(ToolArgumentAliasConflict {
                    alias: alias.to_string(),
                    canonical: canonical_key.clone(),
                    conflicting_with: selected_alias.to_string(),
                });
            }
        }

        if !local_conflicts.is_empty() {
            conflicts.extend(local_conflicts);
            continue;
        }
        normalized.insert(canonical_key.clone(), selected_value);
        for alias in present_aliases {
            normalized.remove(alias);
            aliases_applied.push(AppliedAlias {
                alias: alias.to_string(),
                canonical: canonical_key.clone(),
            });
        }
    }

    ToolArgumentNormalizationResult {
        arguments: normalized,
        aliases_applied,
        conflicts,
    }
}

/// Render alias conflicts without leaking argument values.
pub fn format_alias_conflicts(conflicts: &[ToolArgumentAliasConflict]) -> Vec<String> {
    conflicts
        .iter()
        .map(|conflict| {
            format!(
                "{} conflicts with {} for canonical argument {}",
                conflict.alias, conflict.conflicting_with, conflict.canonical
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn map(value: Value) -> Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    #[test]
    fn no_aliases_for_unknown_tool() {
        let result = canonicalize_tool_arguments("some_tool", &map(json!({"file_path": "x"})));
        assert!(!result.changed());
        assert!(!result.has_conflicts());
        assert!(result.arguments.contains_key("file_path"));
    }

    #[test]
    fn remaps_alias_to_canonical() {
        let result = canonicalize_tool_arguments(
            "edit_file",
            &map(json!({"file_path": "src/main.rs", "old_text": "a"})),
        );
        assert!(result.changed());
        assert!(!result.has_conflicts());
        assert_eq!(result.arguments["path"], "src/main.rs");
        assert!(!result.arguments.contains_key("file_path"));
        assert_eq!(result.aliases_applied.len(), 1);
    }

    #[test]
    fn multiple_aliases_collapse_to_one_canonical() {
        let result = canonicalize_tool_arguments(
            "edit_file",
            &map(json!({"old_string": "a", "oldString": "a", "new_text": "b"})),
        );
        assert!(result.changed());
        assert!(!result.has_conflicts());
        assert_eq!(result.arguments["old_text"], "a");
        assert!(!result.arguments.contains_key("old_string"));
        assert!(!result.arguments.contains_key("oldString"));
    }

    #[test]
    fn alias_conflicts_with_canonical_are_reported() {
        let result = canonicalize_tool_arguments(
            "edit_file",
            &map(json!({"path": "src/a.rs", "file_path": "src/b.rs"})),
        );
        assert!(!result.changed());
        assert!(result.has_conflicts());
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(result.conflicts[0].alias, "file_path");
        assert_eq!(result.conflicts[0].canonical, "path");
        // Canonical value is preserved; the conflicting alias is left in place
        // for the caller to reject.
        assert_eq!(result.arguments["path"], "src/a.rs");
        assert!(result.arguments.contains_key("file_path"));
    }

    #[test]
    fn conflicting_aliases_leave_canonical_absent() {
        let result = canonicalize_tool_arguments(
            "edit_file",
            &map(json!({"old_string": "a", "oldString": "b"})),
        );
        assert!(!result.changed());
        assert!(result.has_conflicts());
        assert!(!result.arguments.contains_key("old_text"));
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(result.conflicts[0].conflicting_with, "old_string");
    }

    #[test]
    fn format_conflicts_without_leaking_values() {
        let conflicts = vec![ToolArgumentAliasConflict {
            alias: "file_path".to_string(),
            canonical: "path".to_string(),
            conflicting_with: "path".to_string(),
        }];
        let rendered = format_alias_conflicts(&conflicts);
        assert_eq!(rendered.len(), 1);
        assert!(rendered[0].contains("file_path conflicts with path"));
    }

    #[test]
    fn canonical_present_with_equal_alias_is_dropped() {
        let result = canonicalize_tool_arguments(
            "read_file",
            &map(json!({"path": "a.txt", "file_path": "a.txt"})),
        );
        assert!(result.changed());
        assert!(!result.has_conflicts());
        assert_eq!(result.arguments["path"], "a.txt");
        assert!(!result.arguments.contains_key("file_path"));
    }
}
