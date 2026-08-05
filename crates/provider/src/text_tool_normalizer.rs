//! Text-to-tool dialect normalization.
//!
//! Some models (particularly DeepSeek, Qwen, MiniMax, and certain open-weight
//! models) emit tool calls as structured text rather than native JSON. This
//! module detects which text dialect a response uses, extracts the embedded
//! tool calls, and normalizes them into the standard
//! [`opensquilla_core::types::ToolCall`] type.
//!
//! Supported dialects:
//!
//! - `OpenAi` — `{"function_call": {"name": ..., "arguments": ...}}`.
//! - `Anthropic` — `{"type": "tool_use", "name": ..., "input": ...}`.
//! - `DeepSeekDsml` —
//!   `<tool_call><tool_name>...</tool_name><parameters>...</parameters></tool_call>`.
//! - `Xml` — `<tool_call name="..." arguments='...'/>` or
//!   `<invoke name="..."><parameter name="...">...</parameter></invoke>`.
//! - `JsonBlock` — a fenced ` ```json ` block carrying
//!   `{"function": "...", "parameters": {...}}`.

use opensquilla_core::types::ToolCall;
use regex::Regex;
use serde::Deserialize;
use serde_json::Deserializer;
use std::sync::LazyLock;

/// Text tool-call dialects recognized by the normalizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolDialect {
    /// OpenAI-style: `function_call` with JSON arguments.
    OpenAi,
    /// Anthropic-style: `tool_use` content blocks.
    Anthropic,
    /// DeepSeek DSML-style: `<tool_call>` XML tags.
    DeepSeekDsml,
    /// XML-style: `<tool_call name="..." arguments='...'/>` or
    /// `<invoke name="...">`.
    Xml,
    /// JSON block-style: ` ```json {"function": "...", "parameters": {...}} ``` `.
    JsonBlock,
}

/// A normalizer for one text tool-call dialect.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallNormalizer {
    dialect: ToolDialect,
}

impl Default for ToolCallNormalizer {
    fn default() -> Self {
        Self::new(ToolDialect::OpenAi)
    }
}

impl ToolCallNormalizer {
    /// Create a normalizer for the given dialect.
    pub fn new(dialect: ToolDialect) -> Self {
        Self { dialect }
    }

    /// Extract tool calls embedded in raw text using the configured dialect.
    pub fn extract_tool_calls(&self, text: &str) -> Vec<ToolCall> {
        match self.dialect {
            ToolDialect::OpenAi => extract_openai(text),
            ToolDialect::Anthropic => extract_anthropic(text),
            ToolDialect::DeepSeekDsml => extract_dsml(text),
            ToolDialect::Xml => extract_xml(text),
            ToolDialect::JsonBlock => extract_json_block(text),
        }
    }

    /// Normalize extracted tool calls to canonical form.
    ///
    /// Repairs stringified JSON arguments, fills in missing ids, coerces
    /// non-object inputs to `{}`, trims names, and deduplicates calls.
    pub fn normalize(&self, calls: Vec<ToolCall>) -> Vec<ToolCall> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for mut call in calls {
            // Repair arguments: if input is a stringified JSON, parse it.
            if let Some(s) = call.input.as_str() {
                call.input = parse_json_object(s).unwrap_or_else(|| serde_json::json!({"raw": s}));
            }
            if !call.input.is_object() {
                call.input = serde_json::json!({});
            }
            call.name = call.name.trim().to_string();
            if call.id.is_empty() {
                call.id = generate_call_id();
            }
            let key = (call.name.clone(), call.input.clone());
            if seen.insert(key) {
                out.push(call);
            }
        }
        out
    }

    /// Extract tool calls from text and normalize them in one step.
    pub fn extract_and_normalize(&self, text: &str) -> Vec<ToolCall> {
        let calls = self.extract_tool_calls(text);
        self.normalize(calls)
    }

    /// Detect which dialect a piece of text uses.
    ///
    /// Detection is heuristic and order-sensitive: DSML and XML tags take
    /// priority over JSON fences, which take priority over bare JSON objects.
    pub fn detect_dialect(text: &str) -> ToolDialect {
        let t = text.trim();
        if t.is_empty() {
            return ToolDialect::JsonBlock;
        }
        // DeepSeek DSML: <tool_call><tool_name>...</tool_name>...
        if t.contains("<tool_call>") && t.contains("<tool_name>") {
            return ToolDialect::DeepSeekDsml;
        }
        // XML: self-closing <tool_call ...> or <invoke ...>
        if t.contains("<tool_call ") || t.contains("<tool_call\n") || t.contains("<invoke") {
            return ToolDialect::Xml;
        }
        // JSON code block fence
        if t.contains("```") {
            return ToolDialect::JsonBlock;
        }
        // Anthropic tool_use content block
        if t.contains("\"tool_use\"") {
            return ToolDialect::Anthropic;
        }
        // OpenAI function_call / function + arguments JSON
        if t.contains("function_call")
            || (t.contains("\"arguments\"")
                && (t.contains("\"name\"") || t.contains("\"function\"")))
            || (t.contains("\"function\"") && t.contains("\"parameters\""))
        {
            return ToolDialect::OpenAi;
        }
        ToolDialect::JsonBlock
    }
}

// ---------------------------------------------------------------------------
// Per-dialect extraction
// ---------------------------------------------------------------------------

static DSML_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?s)<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>",
    )
    .expect("invalid DSML regex")
});

static XML_SELF_CLOSING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"<tool_call\s+name\s*=\s*"([^"]*)"\s+arguments\s*=\s*'([^']*)'\s*/?>|<tool_call\s+name\s*=\s*'([^']*)'\s+arguments\s*=\s*"([^"]*)"\s*/?>"#,
    )
    .expect("invalid XML self-closing regex")
});

static XML_SELF_CLOSING_REV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"<tool_call\s+arguments\s*=\s*'([^']*)'\s+name\s*=\s*"([^"]*)"\s*/?>|<tool_call\s+arguments\s*=\s*"([^"]*)"\s+name\s*=\s*'([^']*)'\s*/?>"#,
    )
    .expect("invalid XML self-closing (rev) regex")
});

static XML_NESTED_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<tool_call\s+name\s*=\s*"([^"]*)"\s*>(.*?)</tool_call>"#)
        .expect("invalid XML nested regex")
});

static XML_NESTED_SQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<tool_call\s+name\s*=\s*'([^']*)'\s*>(.*?)</tool_call>"#)
        .expect("invalid XML nested (single-quote) regex")
});

static XML_INVOKE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<invoke\s+name\s*=\s*"([^"]*)"\s*>(.*?)</invoke>"#)
        .expect("invalid XML invoke regex")
});

static JSON_BLOCK_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```(?:json)?\s*\r?\n?(.*?)```").expect("invalid JSON block regex")
});

fn extract_dsml(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for cap in DSML_RE.captures_iter(text) {
        let name = cap[1].trim().to_string();
        let raw_params = cap[2].trim().to_string();
        let args = if raw_params.trim_start().starts_with('{') {
            parse_json_object(&raw_params).unwrap_or_else(|| serde_json::json!({"raw": raw_params}))
        } else {
            parse_xml_parameters(&raw_params)
        };
        calls.push(ToolCall::new("", name, args));
    }
    calls
}

fn extract_xml(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Self-closing: <tool_call name="..." arguments='...'/> (either order).
    for cap in XML_SELF_CLOSING_RE.captures_iter(text) {
        let (name, args_raw) = if let (Some(n), Some(a)) = (cap.get(1), cap.get(2)) {
            (n.as_str().to_string(), a.as_str().to_string())
        } else if let (Some(n), Some(a)) = (cap.get(3), cap.get(4)) {
            (n.as_str().to_string(), a.as_str().to_string())
        } else {
            continue;
        };
        let args = serde_json::from_str(&args_raw).unwrap_or_else(|_| serde_json::json!({"raw": args_raw}));
        calls.push(ToolCall::new("", name, args));
    }
    for cap in XML_SELF_CLOSING_REV_RE.captures_iter(text) {
        let (args_raw, name) = if let (Some(a), Some(n)) = (cap.get(1), cap.get(2)) {
            (a.as_str().to_string(), n.as_str().to_string())
        } else if let (Some(a), Some(n)) = (cap.get(3), cap.get(4)) {
            (a.as_str().to_string(), n.as_str().to_string())
        } else {
            continue;
        };
        let args = serde_json::from_str(&args_raw).unwrap_or_else(|_| serde_json::json!({"raw": args_raw}));
        calls.push(ToolCall::new("", name, args));
    }

    // Nested: <tool_call name="...">...</tool_call> with parameter children.
    extract_nested_xml_calls(&XML_NESTED_RE, text, &mut calls);
    extract_nested_xml_calls(&XML_NESTED_SQ_RE, text, &mut calls);

    // <invoke name="...">...</invoke> (MiniMax-style).
    extract_nested_xml_calls(&XML_INVOKE_RE, text, &mut calls);

    calls
}

fn extract_nested_xml_calls(re: &Regex, text: &str, calls: &mut Vec<ToolCall>) {
    for cap in re.captures_iter(text) {
        let name = cap[1].trim().to_string();
        let body = cap[2].trim().to_string();
        let args = if body.trim_start().starts_with('{') {
            parse_json_object(&body).unwrap_or_else(|| serde_json::json!({"raw": body}))
        } else {
            parse_xml_parameters(&body)
        };
        calls.push(ToolCall::new("", name, args));
    }
}

fn extract_json_block(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for cap in JSON_BLOCK_RE.captures_iter(text) {
        let content = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
        if content.is_empty() {
            continue;
        }
        if let Some(value) = parse_json_object(content) {
            if let Some(call) = tool_call_from_json_value(&value) {
                calls.push(call);
            }
        }
    }
    calls
}

fn extract_openai(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for value in scan_json_objects(text) {
        if let Some(call) = tool_call_from_json_value(&value) {
            calls.push(call);
        }
    }
    if calls.is_empty() {
        // Fallback: the whole text is (nearly) a single tool-call JSON object.
        if let Some(value) = parse_json_object(text) {
            if let Some(call) = tool_call_from_json_value(&value) {
                calls.push(call);
            }
        }
    }
    calls
}

fn extract_anthropic(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    for value in scan_json_objects(text) {
        // Direct tool_use object.
        let is_tool_use = value
            .get("type")
            .and_then(|v| v.as_str())
            .map(|t| t.eq_ignore_ascii_case("tool_use"))
            .unwrap_or(false);
        if is_tool_use {
            let name = value.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let input = value.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
            if !name.is_empty() {
                calls.push(ToolCall::new("", name, input));
            }
            continue;
        }
        // Wrapped {"tool_use": {...}}.
        if let Some(inner) = value.get("tool_use") {
            let name = inner.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let input = inner.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
            if !name.is_empty() {
                calls.push(ToolCall::new("", name, input));
            }
        }
    }
    if calls.is_empty() {
        // Fallback: the whole text is (nearly) a single tool_use object.
        if let Some(value) = parse_json_object(text) {
            let is_tool_use = value
                .get("type")
                .and_then(|v| v.as_str())
                .map(|t| t.eq_ignore_ascii_case("tool_use"))
                .unwrap_or(false);
            if is_tool_use {
                let name = value.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let input = value.get("input").cloned().unwrap_or_else(|| serde_json::json!({}));
                if !name.is_empty() {
                    calls.push(ToolCall::new("", name, input));
                }
            }
        }
    }
    calls
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

/// Build a `ToolCall` from a recognized JSON object shape.
///
/// Accepts `{"function_call": {"name", "arguments"}}`,
/// `{"function" | "name" | "tool", "parameters" | "arguments" | "input"}`, and
/// the Anthropic `{"type": "tool_use", "name", "input"}` shape.
fn tool_call_from_json_value(value: &serde_json::Value) -> Option<ToolCall> {
    if let Some(fc) = value.get("function_call") {
        let name = fc.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if !name.is_empty() {
            let args = fc
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            return Some(ToolCall::new("", name, normalize_args(args)));
        }
    }
    let name = value
        .get("function")
        .or_else(|| value.get("name"))
        .or_else(|| value.get("tool"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if !name.is_empty() {
        let args = value
            .get("parameters")
            .or_else(|| value.get("arguments"))
            .or_else(|| value.get("input"))
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        return Some(ToolCall::new("", name, normalize_args(args)));
    }
    None
}

/// Parse stringified JSON arguments into an object, preserving the raw string
/// when it cannot be parsed.
fn normalize_args(args: serde_json::Value) -> serde_json::Value {
    if let Some(s) = args.as_str() {
        return parse_json_object(s).unwrap_or_else(|| serde_json::json!({"raw": s}));
    }
    args
}

/// Scan a string for top-level JSON objects, parsing each in turn.
///
/// Used by the JSON-shaped dialects (`OpenAi`, `Anthropic`) where the tool
/// call may be embedded in prose or alongside other objects.
fn scan_json_objects(text: &str) -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    while start < bytes.len() {
        if bytes[start] == b'{' {
            let rest = &text[start..];
            let mut de = Deserializer::from_str(rest);
            if let Ok(value) = serde_json::Value::deserialize(&mut de) {
                let consumed = de.byte_offset();
                results.push(value);
                if consumed == 0 {
                    break;
                }
                start += consumed;
                continue;
            }
        }
        start += 1;
    }
    results
}

// ---------------------------------------------------------------------------
// XML / JSON argument parsing
// ---------------------------------------------------------------------------

static NAMED_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)<parameter\s+name\s*=\s*"([^"]+)"\s*>(.*?)</parameter>"#)
        .expect("invalid named parameter regex")
});

static PLAIN_PARAM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"<([^/>]+)>([^<]*)</([^>]+)>").expect("invalid plain parameter regex")
});

/// Convert XML-style parameters into a JSON object.
///
/// Supports both `<parameter name="key">value</parameter>` and plain
/// `<key>value</key>` child elements. Values are typed (number, boolean,
/// null) where they parse unambiguously.
fn parse_xml_parameters(xml: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();

    // Named <parameter name="...">value</parameter> first.
    let mut found = false;
    for cap in NAMED_PARAM_RE.captures_iter(xml) {
        found = true;
        let key = cap[1].trim().to_string();
        let value = cap[2].trim().to_string();
        map.insert(key, typed_value(&value));
    }
    if found {
        return serde_json::Value::Object(map);
    }

    // Plain <tag>value</tag>.
    for cap in PLAIN_PARAM_RE.captures_iter(xml) {
        let key = cap[1].trim().to_string();
        // Only accept a pair when the closing tag matches the opening tag.
        if cap[3].trim() != key {
            continue;
        }
        let value = cap[2].trim().to_string();
        map.insert(key, typed_value(&value));
    }
    serde_json::Value::Object(map)
}

/// Interpret a raw XML parameter value as a JSON value.
fn typed_value(raw: &str) -> serde_json::Value {
    if let Ok(n) = raw.parse::<f64>() {
        serde_json::json!(n)
    } else if raw == "true" {
        serde_json::json!(true)
    } else if raw == "false" {
        serde_json::json!(false)
    } else if raw == "null" {
        serde_json::Value::Null
    } else {
        serde_json::json!(raw)
    }
}

/// Parse a string as JSON, falling back to a lightweight repair pass.
fn parse_json_object(s: &str) -> Option<serde_json::Value> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(v);
    }
    repair_json(trimmed)
}

static KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)").expect("invalid key repair regex")
});

static TRAILING_COMMA_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r",\s*([\}\]])").expect("invalid trailing comma regex")
});

/// Lightweight JSON repair for malformed tool-call fragments.
///
/// Wraps unquoted object keys, removes trailing commas, and (as a last resort)
/// replaces single quotes with double quotes outside of existing string
/// literals.
fn repair_json(input: &str) -> Option<serde_json::Value> {
    let mut candidate = input.trim().to_string();
    if candidate.is_empty() {
        return None;
    }
    candidate = KEY_RE
        .replace_all(&candidate, "${1}\"${2}\"${3}")
        .to_string();
    candidate = TRAILING_COMMA_RE
        .replace_all(&candidate, "${1}")
        .to_string();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&candidate) {
        return Some(v);
    }
    let single_quoted = replace_single_quotes(&candidate);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&single_quoted) {
        return Some(v);
    }
    None
}

/// Replace single quotes with double quotes only outside existing
/// double-quoted string literals.
fn replace_single_quotes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_double = false;
    let mut escaped = false;
    for ch in s.chars() {
        if in_double {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_double = false;
            }
            continue;
        }
        if ch == '"' {
            in_double = true;
            out.push(ch);
        } else if ch == '\'' {
            out.push('"');
        } else {
            out.push(ch);
        }
    }
    out
}

/// Generate a pseudo-random `call_<hex>` id without external crates.
fn generate_call_id() -> String {
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let hex = b"0123456789abcdef";
    let mut out = String::with_capacity(17);
    out.push_str("call_");
    for _ in 0..12 {
        // xorshift64*
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        seed = seed.wrapping_mul(0x2545_F491_4F6C_DD1D);
        out.push(hex[((seed >> 33) & 15) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_dsml() {
        let n = ToolCallNormalizer::new(ToolDialect::DeepSeekDsml);
        let text = "<tool_call>\n<tool_name>get_weather</tool_name>\n<parameters>\n<location>NYC</location>\n<units>celsius</units>\n</parameters>\n</tool_call>";
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
        assert_eq!(calls[0].input["units"], "celsius");
    }

    #[test]
    fn test_extract_dsml_inline_json_parameters() {
        let n = ToolCallNormalizer::new(ToolDialect::DeepSeekDsml);
        let text = "<tool_call><tool_name>f</tool_name><parameters>{\"a\": 1}</parameters></tool_call>";
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "f");
        assert_eq!(calls[0].input["a"], 1);
    }

    #[test]
    fn test_extract_xml_self_closing() {
        let n = ToolCallNormalizer::new(ToolDialect::Xml);
        let text = r#"<tool_call name="get_weather" arguments='{"location":"NYC"}' />"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_xml_self_closing_reversed_attrs() {
        let n = ToolCallNormalizer::new(ToolDialect::Xml);
        let text = r#"<tool_call arguments='{"location":"NYC"}' name="get_weather"/>"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_xml_nested_parameters() {
        let n = ToolCallNormalizer::new(ToolDialect::Xml);
        let text = r#"<tool_call name="get_weather"><parameter name="location">NYC</parameter><parameter name="units">celsius</parameter></tool_call>"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
        assert_eq!(calls[0].input["units"], "celsius");
    }

    #[test]
    fn test_extract_xml_invoke() {
        let n = ToolCallNormalizer::new(ToolDialect::Xml);
        let text = r#"<invoke name="get_weather"><parameter name="location">NYC</parameter></invoke>"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_json_block() {
        let n = ToolCallNormalizer::new(ToolDialect::JsonBlock);
        let text = "Some text\n```json\n{\"function\": \"get_weather\", \"parameters\": {\"location\": \"NYC\"}}\n```\nMore text";
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_openai_function_call() {
        let n = ToolCallNormalizer::new(ToolDialect::OpenAi);
        let text = r#"Before {"function_call": {"name": "get_weather", "arguments": {"location": "NYC"}}} after"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_openai_stringified_arguments() {
        let n = ToolCallNormalizer::new(ToolDialect::OpenAi);
        let text = r#"{"function_call": {"name": "get_weather", "arguments": "{\"location\": \"NYC\"}"}}"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_extract_anthropic_tool_use() {
        let n = ToolCallNormalizer::new(ToolDialect::Anthropic);
        let text = r#"{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "NYC"}}"#;
        let calls = n.extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input["location"], "NYC");
    }

    #[test]
    fn test_detect_dialect() {
        assert_eq!(
            ToolCallNormalizer::detect_dialect(
                "<tool_call><tool_name>x</tool_name><parameters></parameters></tool_call>"
            ),
            ToolDialect::DeepSeekDsml
        );
        assert_eq!(
            ToolCallNormalizer::detect_dialect(r#"<tool_call name="x" arguments='{}'/>"#),
            ToolDialect::Xml
        );
        assert_eq!(
            ToolCallNormalizer::detect_dialect("```json\n{\"function\":\"x\",\"parameters\":{}}\n```"),
            ToolDialect::JsonBlock
        );
        assert_eq!(
            ToolCallNormalizer::detect_dialect(r#"{"type":"tool_use","name":"x","input":{}}"#),
            ToolDialect::Anthropic
        );
        assert_eq!(
            ToolCallNormalizer::detect_dialect(r#"{"function_call":{"name":"x","arguments":{}}}"#),
            ToolDialect::OpenAi
        );
    }

    #[test]
    fn test_normalize_repairs_and_dedupes() {
        let n = ToolCallNormalizer::new(ToolDialect::OpenAi);
        let calls = vec![
            ToolCall::new("", "get_weather", serde_json::json!({"location": "NYC"})),
            ToolCall::new("", "get_weather", serde_json::json!({"location": "NYC"})), // duplicate
            ToolCall::new("", "get_weather", serde_json::json!(42)), // non-object input
        ];
        let normalized = n.normalize(calls);
        assert_eq!(normalized.len(), 2);
        assert!(!normalized[0].id.is_empty());
        assert_eq!(normalized[0].name, "get_weather");
        assert!(normalized[0].input.is_object());
        // Non-object input coerced to {}.
        assert_eq!(normalized[1].input, serde_json::json!({}));
    }

    #[test]
    fn test_normalize_stringified_args() {
        let n = ToolCallNormalizer::new(ToolDialect::OpenAi);
        let calls = vec![ToolCall::new(
            "c1",
            "get_weather",
            serde_json::json!(r#"{"location":"NYC"}"#),
        )];
        let normalized = n.normalize(calls);
        assert_eq!(normalized[0].input["location"], "NYC");
    }

    #[test]
    fn test_repair_json_unquoted_keys() {
        let repaired = parse_json_object(r#"{location: "NYC"}"#);
        assert_eq!(repaired, Some(serde_json::json!({"location": "NYC"})));
    }

    #[test]
    fn test_repair_json_trailing_comma() {
        let repaired = parse_json_object(r#"{"a": 1, "b": 2,}"#);
        assert_eq!(repaired, Some(serde_json::json!({"a": 1, "b": 2})));
    }

    #[test]
    fn test_typed_xml_values() {
        let xml = "<location>NYC</location>\n<temperature>25.5</temperature>\n<enabled>true</enabled>\n<nothing>null</nothing>";
        let params = parse_xml_parameters(xml);
        assert_eq!(params["location"], "NYC");
        assert_eq!(params["temperature"], 25.5);
        assert_eq!(params["enabled"], true);
        assert!(params["nothing"].is_null());
    }
}
