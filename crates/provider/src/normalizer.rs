//! Text tool call normalizer.
//!
//! Some models (particularly DeepSeek and certain open-weight models) emit
//! tool calls as structured text rather than native JSON. This module
//! normalizes those text-based tool call formats — DSML (DeepSeek Markup
//! Language), XML, and JSON code blocks — into the standard `ToolCall` type.

use opensquilla_core::types::ToolCall;
use regex::Regex;

/// A discovered tool call embedded in text.
#[derive(Debug, Clone)]
pub struct TextToolCall {
    /// The tool name.
    pub name: String,
    /// The raw arguments string (before JSON parsing).
    pub arguments_raw: String,
    /// The parsed arguments (if valid JSON).
    pub arguments: serde_json::Value,
    /// The span in the original text where the tool call was found.
    pub span: (usize, usize),
}

/// Detect and extract tool calls from model-generated text.
///
/// Supports the following formats:
///
/// **DSML (DeepSeek Markup Language):**
/// ```dsml
/// <tool_call>
///   <tool_name>get_weather</tool_name>
///   <parameters>
///     <location>NYC</location>
///   </parameters>
/// </tool_call>
/// ```
///
/// **XML-style:**
/// ```xml
/// <tool_call name="get_weather" arguments='{"location":"NYC"}' />
/// ```
///
/// **JSON code block:**
/// ```json
/// {"name": "get_weather", "arguments": {"location": "NYC"}}
/// ```
pub struct ToolCallNormalizer {
    dsml_pattern: Regex,
    xml_pattern: Regex,
    json_block_pattern: Regex,
}

impl Default for ToolCallNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallNormalizer {
    /// Create a new normalizer with the default pattern set.
    pub fn new() -> Self {
        Self {
            dsml_pattern: Regex::new(
                r"(?s)<tool_call>\s*<tool_name>(.*?)</tool_name>\s*<parameters>(.*?)</parameters>\s*</tool_call>"
            ).expect("Invalid DSML regex"),
            xml_pattern: Regex::new(
                r#"<tool_call\s+name="([^"]*)"\s+arguments='([^']*)'\s*/?>|<tool_call\s+name='([^']*)'\s+arguments="([^"]*)"\s*/?>"#
            ).expect("Invalid XML regex"),
            json_block_pattern: Regex::new(
                r"(?s)```(?:json)?\s*\n(.*?)\n?```"
            ).expect("Invalid JSON block regex"),
        }
    }

    /// Extract all text-based tool calls from the given text.
    ///
    /// Returns a list of `TextToolCall` structs, one per detected tool call.
    pub fn extract(&self, text: &str) -> Vec<TextToolCall> {
        let mut results = Vec::new();

        // Try DSML format
        results.extend(self.extract_dsml(text));

        // Try XML format
        results.extend(self.extract_xml(text));

        // Try JSON code block format
        results.extend(self.extract_json_block(text));

        results
    }

    /// Extract all tool calls and convert them to the standard `ToolCall` type.
    pub fn normalize(&self, text: &str) -> Vec<ToolCall> {
        self.extract(text)
            .into_iter()
            .map(|tc| ToolCall::new("", tc.name, tc.arguments))
            .collect()
    }

    /// Strip text tool call markup from the text, returning clean text.
    pub fn strip_tool_calls(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Strip DSML blocks
        result = self
            .dsml_pattern
            .replace_all(&result, "")
            .to_string();

        // Strip XML-style tags
        result = self
            .xml_pattern
            .replace_all(&result, "")
            .to_string();

        // Strip JSON code blocks that look like tool calls
        result = self
            .json_block_pattern
            .replace_all(&result, |caps: &regex::Captures| {
                let content = &caps[1];
                if content.contains("\"name\"") && content.contains("\"arguments\"") {
                    String::new()
                } else {
                    caps[0].to_string()
                }
            })
            .to_string();

        // Clean up extra whitespace
        result = result.trim().to_string();

        result
    }

    fn extract_dsml(&self, text: &str) -> Vec<TextToolCall> {
        let mut results = Vec::new();

        for cap in self.dsml_pattern.captures_iter(text) {
            let name = cap[1].trim().to_string();
            let raw_params = cap[2].trim().to_string();

            // Try to parse parameters as inline JSON
            let args = if raw_params.starts_with('{') {
                serde_json::from_str(&raw_params).unwrap_or(serde_json::json!({"raw": raw_params}))
            } else {
                // Convert XML-style parameters to JSON
                let params = parse_xml_parameters(&raw_params);
                serde_json::json!(params)
            };

            let start = cap.get(0).map(|m| m.start()).unwrap_or(0);
            let end = cap.get(0).map(|m| m.end()).unwrap_or(0);

            results.push(TextToolCall {
                name,
                arguments_raw: raw_params,
                arguments: args,
                span: (start, end),
            });
        }

        results
    }

    fn extract_xml(&self, text: &str) -> Vec<TextToolCall> {
        let mut results = Vec::new();

        for cap in self.xml_pattern.captures_iter(text) {
            let (name, args_raw) = if let (Some(n), Some(a)) = (cap.get(1), cap.get(2)) {
                (n.as_str().to_string(), a.as_str().to_string())
            } else if let (Some(n), Some(a)) = (cap.get(3), cap.get(4)) {
                (n.as_str().to_string(), a.as_str().to_string())
            } else {
                continue;
            };

            let args = serde_json::from_str(&args_raw)
                .unwrap_or(serde_json::json!({"raw": args_raw}));

            let start = cap.get(0).map(|m| m.start()).unwrap_or(0);
            let end = cap.get(0).map(|m| m.end()).unwrap_or(0);

            results.push(TextToolCall {
                name,
                arguments_raw: args_raw,
                arguments: args,
                span: (start, end),
            });
        }

        results
    }

    fn extract_json_block(&self, text: &str) -> Vec<TextToolCall> {
        let mut results = Vec::new();

        for cap in self.json_block_pattern.captures_iter(text) {
            let content = &cap[1];
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
                let name = value["name"].as_str().unwrap_or("").to_string();
                let args = value["arguments"].clone();
                if args.is_null() {
                    continue;
                }

                let start = cap.get(0).map(|m| m.start()).unwrap_or(0);
                let end = cap.get(0).map(|m| m.end()).unwrap_or(0);

                results.push(TextToolCall {
                    name,
                    arguments_raw: content.to_string(),
                    arguments: args,
                    span: (start, end),
                });
            }
        }

        results
    }
}

/// Parse simple XML-style parameters into a JSON object.
///
/// Converts:
/// ```xml
/// <location>NYC</location>
/// <units>celsius</units>
/// ```
/// into `{"location": "NYC", "units": "celsius"}`
fn parse_xml_parameters(xml: &str) -> serde_json::Value {
    // Rust's `regex` crate does not support backreferences, so we match
    // `<tag>content</closing>` and verify the closing tag matches the opening
    // tag in code.
    let re = Regex::new(r"<([^/>]+)>([^<]*)</([^>]+)>").expect("Invalid XML param regex");
    let mut map = serde_json::Map::new();

    for cap in re.captures_iter(xml) {
        let key = cap[1].trim().to_string();
        // Only accept the pair if the closing tag matches the opening tag.
        if cap[3].trim() != key {
            continue;
        }
        let value = cap[2].trim().to_string();

        // Try to parse as number
        if let Ok(n) = value.parse::<f64>() {
            map.insert(key, serde_json::json!(n));
        } else if value == "true" {
            map.insert(key, serde_json::json!(true));
        } else if value == "false" {
            map.insert(key, serde_json::json!(false));
        } else {
            map.insert(key, serde_json::json!(value));
        }
    }

    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_dsml() {
        let normalizer = ToolCallNormalizer::new();
        let text = r#"
Some text before
<tool_call>
<tool_name>get_weather</tool_name>
<parameters>
<location>NYC</location>
<units>celsius</units>
</parameters>
</tool_call>
Some text after"#;

        let calls = normalizer.extract(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "NYC");
        assert_eq!(calls[0].arguments["units"], "celsius");
    }

    #[test]
    fn test_normalize_xml() {
        let normalizer = ToolCallNormalizer::new();
        let text = r#"<tool_call name="get_weather" arguments='{"location":"NYC"}' />"#;
        let calls = normalizer.extract(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "NYC");
    }

    #[test]
    fn test_normalize_json_block() {
        let normalizer = ToolCallNormalizer::new();
        let text = "Some text\n```json\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}\n```\nMore text";
        let calls = normalizer.extract(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "NYC");
    }

    #[test]
    fn test_strip_tool_calls() {
        let normalizer = ToolCallNormalizer::new();
        let text = "Hello <tool_call><tool_name>ping</tool_name><parameters></parameters></tool_call> World";
        let cleaned = normalizer.strip_tool_calls(text);
        assert_eq!(cleaned, "Hello  World");
    }

    #[test]
    fn test_parse_xml_parameters() {
        let xml = "<location>NYC</location>\n<temperature>25.5</temperature>\n<enabled>true</enabled>";
        let params = parse_xml_parameters(xml);
        assert_eq!(params["location"], "NYC");
        assert_eq!(params["temperature"], 25.5);
        assert_eq!(params["enabled"], true);
    }
}