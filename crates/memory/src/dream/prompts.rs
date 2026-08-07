//! Dream provider prompts and constrained patch parsing.
//!
//! Parity mirroring `src/opensquilla/memory/dream/prompts.py`. The provider
//! prompt asks the LLM to return a constrained JSON `operations` array;
//! parsing validates the operations against the ranked candidates.

use crate::dream::models::{PromotionCandidate, PromotionPatch, PromotionPatchOperation};
use serde_json::{Map, Value};
use std::collections::HashSet;

/// Build the LLM prompt that asks for a MEMORY.md promotion patch.
///
/// Mirrors `promotion_patch_prompt` from prompts.py: the current MEMORY.md
/// block, the three allowed operations (`upsert` / `merge` / `skip`), and the
/// ranked candidates (id, score, reasons, snippet).
pub fn promotion_patch_prompt(
    current_memory_md: &str,
    candidates: &[PromotionCandidate],
) -> String {
    let candidate_lines: Vec<String> = candidates
        .iter()
        .map(|candidate| {
            format!(
                "- candidate_id: {}\n  score: {:.3}\n  reasons: {}\n  snippet: {}",
                candidate.candidate_id,
                candidate.score,
                candidate.reasons.join(", "),
                candidate.snippet
            )
        })
        .collect();

    let mut prompt = String::new();
    prompt.push_str(
        "You are updating OpenSquilla MEMORY.md as curated long-term memory.\n\
         Return JSON only with an operations array. Do not write dated logs, scores, \
         or source metadata into MEMORY.md.\n\n\
         Allowed operations:\n\
         - {\"op\":\"upsert\",\"candidate_ids\":[\"...\"],\"section\":\"User Preferences\",\"memory_id\":\"mem_short_stable_id\",\"text\":\"- durable memory\"}\n\
         - {\"op\":\"merge\",\"candidate_ids\":[\"...\"],\"section\":\"Project Practices\",\"memory_id\":\"mem_short_stable_id\",\"text\":\"- consolidated memory\"}\n\
         - {\"op\":\"skip\",\"candidate_ids\":[\"...\"],\"reason\":\"not durable\"}\n\n\
         Current MEMORY.md:\n<<<\n",
    );
    prompt.push_str(current_memory_md);
    prompt.push_str("\n>>>\n\nRanked candidates:\n");
    prompt.push_str(&candidate_lines.join("\n\n"));
    prompt.push_str("\n\nJSON:");
    prompt
}

/// First 300 characters of `text`, for error messages (char-boundary safe).
fn preview(text: &str) -> String {
    text.chars().take(300).collect()
}

/// True when the JSON value is Python-falsy (`None`, `False`, zero, or an
/// empty string / array / object).
fn is_falsy(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(value) => !value,
        Value::Number(number) => number.as_f64() == Some(0.0),
        Value::String(text) => text.is_empty(),
        Value::Array(values) => values.is_empty(),
        Value::Object(entries) => entries.is_empty(),
    }
}

/// Stringify a JSON field the way Python's `str(raw.get(key) or default)`
/// would: absent or falsy values fall back to `default`.
fn field_or_default(raw: &Map<String, Value>, key: &str, default: &str) -> String {
    match raw.get(key) {
        None => default.to_string(),
        Some(value) if is_falsy(value) => default.to_string(),
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
    }
}

/// Extract the first JSON object from `text`, tolerating markdown fences and
/// prose around it. Mirrors `_json_payload` from prompts.py (first `{` to
/// last `}`).
fn json_payload(text: &str) -> Result<Value, String> {
    let Some(start) = text.find('{') else {
        return Err(format!(
            "Dream response did not contain JSON: {}",
            preview(text)
        ));
    };
    let Some(end) = text.rfind('}') else {
        return Err(format!(
            "Dream response did not contain JSON: {}",
            preview(text)
        ));
    };
    let payload: Value = serde_json::from_str(&text[start..=end])
        .map_err(|error| format!("Dream response did not contain valid JSON: {error}"))?;
    if !payload.is_object() {
        return Err("Dream response JSON must be an object".to_string());
    }
    Ok(payload)
}

/// Parse an LLM JSON response into a constrained [`PromotionPatch`].
///
/// Mirrors `parse_promotion_patch` from prompts.py: only `upsert` / `merge` /
/// `skip` operations are kept, candidate ids are validated against the ranked
/// candidates, and the `["auto"]` sentinel expands to all candidate ids.
/// Returns `Err` when the response contains no usable operations.
pub fn parse_promotion_patch(
    text: &str,
    candidates: &[PromotionCandidate],
) -> std::result::Result<PromotionPatch, String> {
    let payload = json_payload(text)?;
    let candidate_ids: HashSet<String> = candidates
        .iter()
        .map(|candidate| candidate.candidate_id.clone())
        .collect();

    let operations_raw = match payload.get("operations").cloned().unwrap_or_default() {
        Value::Null => Vec::new(),
        Value::Array(items) => items,
        _ => return Err("Dream operations must be a list".to_string()),
    };

    let mut operations: Vec<PromotionPatchOperation> = Vec::new();
    for raw in operations_raw {
        let Some(raw) = raw.as_object() else {
            continue;
        };
        let op = field_or_default(raw, "op", "");
        if !matches!(op.as_str(), "upsert" | "merge" | "skip") {
            continue;
        }

        let mut ids: Vec<String> = raw
            .get("candidate_ids")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if ids.len() == 1 && ids[0] == "auto" {
            let mut all_ids: Vec<String> = candidate_ids.iter().cloned().collect();
            all_ids.sort();
            ids = all_ids;
        }
        ids.retain(|id| candidate_ids.contains(id));
        if ids.is_empty() {
            continue;
        }

        let replaces_memory_id = raw
            .get("replaces_memory_id")
            .and_then(Value::as_str)
            .map(String::from);
        let replaces_memory_ids: Vec<String> = raw
            .get("replaces_memory_ids")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let expected_old_text_sha256 = raw
            .get("expected_old_text_sha256")
            .and_then(Value::as_str)
            .map(String::from);
        let reason = match raw.get("reason") {
            None | Some(Value::Null) => None,
            Some(Value::String(text)) => Some(text.clone()),
            Some(other) => Some(other.to_string()),
        };

        operations.push(PromotionPatchOperation {
            op,
            candidate_ids: ids,
            section: field_or_default(raw, "section", "Long-Term Memory"),
            memory_id: field_or_default(raw, "memory_id", ""),
            text: field_or_default(raw, "text", ""),
            replaces_memory_id,
            replaces_memory_ids,
            expected_old_text_sha256,
            reason,
        });
    }

    if operations.is_empty() {
        return Err("Dream response contained no valid operations".to_string());
    }
    Ok(PromotionPatch { operations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn candidate(id: &str) -> PromotionCandidate {
        PromotionCandidate {
            candidate_id: id.to_string(),
            source_path: format!("memory/{id}.md"),
            snippet: format!("snippet for {id}"),
            snippet_sha256: format!("sha256-{id}"),
            claim_sha256: format!("claim-{id}"),
            score: 0.8,
            reasons: vec!["recurring".to_string(), "strong".to_string()],
            signal_counts: HashMap::new(),
        }
    }

    #[test]
    fn prompt_lists_current_memory_candidates_and_allowed_ops() {
        let prompt = promotion_patch_prompt("# Existing memory", &[candidate("cand_1")]);
        assert!(
            prompt
                .starts_with("You are updating OpenSquilla MEMORY.md as curated long-term memory.")
        );
        assert!(prompt.contains("Allowed operations:"));
        assert!(
            prompt
                .contains(r#"{"op":"upsert","candidate_ids":["..."],"section":"User Preferences""#)
        );
        assert!(prompt.contains("Current MEMORY.md:\n<<<\n# Existing memory\n>>>"));
        assert!(prompt.contains("- candidate_id: cand_1"));
        assert!(prompt.contains("  score: 0.800"));
        assert!(prompt.contains("  reasons: recurring, strong"));
        assert!(prompt.contains("  snippet: snippet for cand_1"));
        assert!(prompt.ends_with("\n\nJSON:"));
    }

    #[test]
    fn parse_keeps_valid_ops_and_filters_unknown() {
        let candidates = vec![candidate("cand_1"), candidate("cand_2")];
        let text = r#"{
            "operations": [
                {"op":"upsert","candidate_ids":["cand_1"],"section":"User Preferences","memory_id":"mem_1","text":"- durable memory"},
                {"op":"skip","candidate_ids":["cand_2"],"reason":"not durable"},
                {"op":"delete","candidate_ids":["cand_1"]},
                {"op":"upsert","candidate_ids":["missing"]}
            ]
        }"#;
        let patch = parse_promotion_patch(text, &candidates).unwrap();
        assert_eq!(patch.operations.len(), 2);
        assert_eq!(patch.operations[0].op, "upsert");
        assert_eq!(patch.operations[0].candidate_ids, ["cand_1"]);
        assert_eq!(patch.operations[0].section, "User Preferences");
        assert_eq!(patch.operations[0].memory_id, "mem_1");
        assert_eq!(patch.operations[0].text, "- durable memory");
        assert_eq!(patch.operations[1].op, "skip");
        assert_eq!(patch.operations[1].reason.as_deref(), Some("not durable"));
    }

    #[test]
    fn parse_tolerates_markdown_fences() {
        let text = "Sure, here you go:\n```json\n{\"operations\":[{\"op\":\"upsert\",\"candidate_ids\":[\"cand_1\"],\"section\":\"User Preferences\",\"memory_id\":\"mem_1\",\"text\":\"- durable memory\"}]}\n```";
        let patch = parse_promotion_patch(text, &[candidate("cand_1")]).unwrap();
        assert_eq!(patch.operations.len(), 1);
    }

    #[test]
    fn parse_auto_expands_to_all_sorted_candidate_ids() {
        let candidates = vec![candidate("cand_b"), candidate("cand_a")];
        let text = r#"{"operations":[{"op":"merge","candidate_ids":["auto"],"section":"Project Practices","memory_id":"mem_m","text":"- merged"}]}"#;
        let patch = parse_promotion_patch(text, &candidates).unwrap();
        assert_eq!(patch.operations[0].candidate_ids, ["cand_a", "cand_b"]);
    }

    #[test]
    fn parse_applies_defaults_for_absent_fields() {
        let text = r#"{"operations":[{"op":"upsert","candidate_ids":["cand_1"]}]}"#;
        let patch = parse_promotion_patch(text, &[candidate("cand_1")]).unwrap();
        let op = &patch.operations[0];
        assert_eq!(op.section, "Long-Term Memory");
        assert_eq!(op.memory_id, "");
        assert_eq!(op.text, "");
        assert_eq!(op.reason, None);
        assert!(op.replaces_memory_id.is_none());
        assert!(op.expected_old_text_sha256.is_none());
    }

    #[test]
    fn parse_preserves_empty_reason_string() {
        let text = r#"{"operations":[{"op":"skip","candidate_ids":["cand_1"],"reason":""}]}"#;
        let patch = parse_promotion_patch(text, &[candidate("cand_1")]).unwrap();
        assert_eq!(patch.operations[0].reason.as_deref(), Some(""));
    }

    #[test]
    fn parse_errors_when_no_json_object() {
        let err = parse_promotion_patch("No durable memories found.", &[candidate("cand_1")])
            .unwrap_err();
        assert!(err.contains("did not contain JSON"));
    }

    #[test]
    fn parse_errors_when_operations_not_a_list() {
        let text = r#"{"operations":"upsert"}"#;
        let err = parse_promotion_patch(text, &[candidate("cand_1")]).unwrap_err();
        assert!(err.contains("must be a list"));
    }

    #[test]
    fn parse_errors_when_no_valid_operations() {
        let text = r#"{"operations":[{"op":"delete","candidate_ids":["cand_1"]}]}"#;
        let err = parse_promotion_patch(text, &[candidate("cand_1")]).unwrap_err();
        assert!(err.contains("no valid operations"));
    }

    #[test]
    fn parse_errors_on_empty_operations_array() {
        let err =
            parse_promotion_patch(r#"{"operations":[]}"#, &[candidate("cand_1")]).unwrap_err();
        assert!(err.contains("no valid operations"));
    }
}
