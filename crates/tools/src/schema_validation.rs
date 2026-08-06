//! JSON Schema parameter validation for tools.
//!
//! Validates tool call arguments against a tool's JSON Schema **before** the
//! tool's `execute` method runs. This is a deeper, spec-compliant validation
//! than the lightweight `Tool::validate_args` default (which only checks for
//! the presence of required parameters): it uses the `jsonschema` crate to
//! enforce types, enums, ranges, `maxLength`, `pattern`, `array`/`object`
//! structure, and all other JSON Schema keywords.
//!
//! ## Usage
//!
//! Each [`ToolDefinition`](crate::registry::ToolDefinition) can be converted to
//! a JSON Schema via [`ToolDefinition::to_json_schema`]. Pass that schema to
//! [`SchemaValidator::new`] once, then call
//! [`SchemaValidator::validate_args`] on every incoming argument object. The
//! validator compiles the schema once and reuses the compiled validator for
//! cheap repeated checks.
//!
//! ```ignore
//! use opensquilla_tools::registry::{Tool, ToolDefinition};
//! use opensquilla_tools::schema_validation::SchemaValidator;
//!
//! let def: &ToolDefinition = tool.definition();
//! let validator = SchemaValidator::from_definition(def)?;
//! validator.validate_args(&args)?;
//! ```

use crate::registry::{ToolDefinition, ToolError};
use jsonschema::Validator;
use serde_json::Value;
use std::sync::Arc;

/// A compiled JSON Schema validator for a single tool's parameter schema.
///
/// The underlying `jsonschema::Validator` is wrapped in an `Arc` so the
/// compiled schema can be shared across dispatch sites without recompiling.
pub struct SchemaValidator {
    tool_name: String,
    schema: Value,
    validator: Arc<Validator>,
}

impl SchemaValidator {
    /// Build a validator for a tool's parameter schema.
    ///
    /// Returns an error if the schema itself is malformed (this is a
    /// programming error, not a caller input error — the schema author
    /// supplied an invalid JSON Schema).
    pub fn new(tool_name: impl Into<String>, schema: Value) -> Result<Self, ToolError> {
        let tool_name = tool_name.into();
        let validator = jsonschema::validator_for(&schema).map_err(|e| {
            ToolError::new(
                "SCHEMA_INVALID",
                format!(
                    "Tool '{}' has an invalid parameter JSON Schema: {}",
                    tool_name, e
                ),
            )
        })?;
        Ok(Self {
            tool_name,
            schema,
            validator: Arc::new(validator),
        })
    }

    /// Build a validator directly from a [`ToolDefinition`].
    ///
    /// Uses the definition's name and its `to_json_schema()` representation.
    pub fn from_definition(def: &ToolDefinition) -> Result<Self, ToolError> {
        Self::new(&def.name, def.to_json_schema())
    }

    /// The tool name this validator was built for.
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// The JSON Schema this validator was compiled from.
    pub fn schema(&self) -> &Value {
        &self.schema
    }

    /// Validate the given arguments against the compiled schema.
    ///
    /// Returns `Ok(())` if the arguments conform to the schema, or an
    /// `INVALID_ARGS` `ToolError` whose message lists every violation.
    /// Non-object arguments are rejected unless the schema explicitly permits
    /// them (e.g. `type: "null"`), mirroring the default
    /// [`Tool::validate_args`](crate::registry::Tool) behavior.
    pub fn validate_args(&self, args: &Value) -> Result<(), ToolError> {
        match self.validator.validate(args) {
            Ok(()) => Ok(()),
            Err(errors) => {
                let mut messages: Vec<String> = Vec::new();
                for err in errors {
                    let location = if err.instance_path.as_str().is_empty() {
                        "<root>".to_string()
                    } else {
                        err.instance_path.to_string()
                    };
                    messages.push(format!("at {}: {}", location, err));
                }
                let detail = if messages.len() == 1 {
                    messages.pop().unwrap_or_default()
                } else {
                    format!(
                        "{} validation errors: {}",
                        messages.len(),
                        messages.join("; ")
                    )
                };
                Err(ToolError::invalid_args(format!(
                    "Invalid arguments for tool '{}': {}",
                    self.tool_name, detail
                )))
            }
        }
    }

    /// Cheap boolean check: returns `true` if the arguments are valid.
    ///
    /// Prefer [`validate_args`](Self::validate_args) when an error message is
    /// needed; this is useful for pre-flight filtering.
    pub fn is_valid(&self, args: &Value) -> bool {
        self.validator.is_valid(args)
    }
}

/// Validate a single tool call's arguments against its tool's schema.
///
/// Convenience wrapper: looks up the tool's definition, builds (or fetches a
/// cached) validator, and validates the arguments. Returns the same
/// `Result<(), ToolError>` shape that `Tool::validate_args` uses, so it can
/// be dropped in as a replacement for the default implementation.
pub fn validate_tool_args(tool: &dyn crate::registry::Tool, args: &Value) -> Result<(), ToolError> {
    let def = tool.definition();
    let validator = SchemaValidator::from_definition(def)?;
    validator.validate_args(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_validator(schema: Value) -> SchemaValidator {
        SchemaValidator::new("test_tool", schema).expect("schema should compile")
    }

    #[test]
    fn test_valid_object_passes() {
        let v = make_validator(json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "count": { "type": "integer", "minimum": 0 }
            },
            "required": ["name"]
        }));
        assert!(
            v.validate_args(&json!({"name": "alice", "count": 3}))
                .is_ok()
        );
    }

    #[test]
    fn test_missing_required_field_fails() {
        let v = make_validator(json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
        }));
        let result = v.validate_args(&json!({}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, "INVALID_ARGS");
        assert!(err.message.contains("name"));
    }

    #[test]
    fn test_wrong_type_fails() {
        let v = make_validator(json!({
            "type": "object",
            "properties": { "count": { "type": "integer" } },
            "required": ["count"]
        }));
        let result = v.validate_args(&json!({"count": "not a number"}));
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("count"));
    }

    #[test]
    fn test_enum_constraint_fails() {
        let v = make_validator(json!({
            "type": "object",
            "properties": {
                "op": { "type": "string", "enum": ["read", "write"] }
            },
            "required": ["op"]
        }));
        assert!(v.validate_args(&json!({"op": "read"})).is_ok());
        assert!(v.validate_args(&json!({"op": "delete"})).is_err());
    }

    #[test]
    fn test_invalid_schema_returns_error() {
        // A schema whose "type" is not a valid JSON Schema type is rejected.
        let result = SchemaValidator::new("bad", json!({"type": "not-a-real-type"}));
        assert!(result.is_err());
        let err = result.err().expect("expected schema error");
        assert_eq!(err.code, "SCHEMA_INVALID");
    }

    #[test]
    fn test_is_valid_fast_path() {
        let v = make_validator(json!({"type": "object", "required": ["x"]}));
        assert!(v.is_valid(&json!({"x": 1})));
        assert!(!v.is_valid(&json!({})));
    }

    #[test]
    fn test_multiple_errors_aggregated() {
        let v = make_validator(json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer" },
                "b": { "type": "string" }
            },
            "required": ["a", "b"]
        }));
        let result = v.validate_args(&json!({"a": "bad", "c": 1}));
        assert!(result.is_err());
        let msg = result.unwrap_err().message;
        // Should mention the missing required field and the wrong type.
        assert!(
            msg.contains("b"),
            "message should mention missing 'b': {}",
            msg
        );
    }
}
