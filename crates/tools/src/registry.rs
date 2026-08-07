//! Tool trait and registry for the OpenSquilla tool system.
//!
//! Defines the core `Tool` trait that all built-in tools implement,
//! the `ToolRegistry` that manages tool lifecycle and lookup,
//! and associated types for parameter definitions and execution results.

use opensquilla_core::ToolCall;
use opensquilla_core::error::{AppError, AppResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

/// A structured error that can occur during tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    /// A machine-readable error code.
    pub code: String,
    /// A human-readable error message.
    pub message: String,
    /// Optional details for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ToolError {
    /// Create a new tool error with a code and message.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            details: None,
        }
    }

    /// Attach additional error details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    /// Create a tool-not-found error.
    pub fn not_found(name: impl Into<String>) -> Self {
        Self::new(
            "TOOL_NOT_FOUND",
            format!("Tool '{}' not found", name.into()),
        )
    }

    /// Create an invalid-arguments error.
    pub fn invalid_args(message: impl Into<String>) -> Self {
        Self::new("INVALID_ARGS", message)
    }

    /// Create an execution-failed error.
    pub fn execution_failed(message: impl Into<String>) -> Self {
        Self::new("EXECUTION_FAILED", message)
    }

    /// Create a timeout error.
    pub fn timeout(duration_secs: u64) -> Self {
        Self::new(
            "TIMEOUT",
            format!("Tool execution timed out after {}s", duration_secs),
        )
    }

    /// Create a permission-denied error.
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Self::new("PERMISSION_DENIED", message)
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for ToolError {}

impl From<ToolError> for AppError {
    fn from(err: ToolError) -> Self {
        AppError::new(&err.code, &err.message)
            .with_details(serde_json::json!({ "tool_error": err.details }))
    }
}

/// The result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// The text content of the tool result.
    pub content: String,
    /// Whether the tool execution resulted in an error.
    #[serde(default)]
    pub is_error: bool,
    /// Optional structured data returned by the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    /// Optional MIME type of the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl ToolOutput {
    /// Create a successful tool output with text content.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            data: None,
            mime_type: None,
        }
    }

    /// Create a successful tool output with structured data.
    pub fn success_with_data(content: impl Into<String>, data: serde_json::Value) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            data: Some(data),
            mime_type: None,
        }
    }

    /// Create an error tool output.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            is_error: true,
            data: None,
            mime_type: None,
        }
    }

    /// Set the MIME type for this output.
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Attach structured data.
    pub fn with_data(mut self, data: serde_json::Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// A type alias for the result of a tool execution.
pub type ToolResult<T = ToolOutput> = std::result::Result<T, ToolError>;

/// A JSON Schema definition for a tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    /// The JSON Schema type (e.g., "string", "integer", "boolean", "array", "object").
    pub param_type: String,
    /// A description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the parameter is required.
    #[serde(default)]
    pub required: bool,
    /// Default value if not provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub default: Option<serde_json::Value>,
    /// Enum values if the parameter is constrained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Items schema for array types.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ParameterDefinition>>,
    /// Properties schema for object types.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, ParameterDefinition>>,
}

impl ParameterDefinition {
    /// Create a new string parameter definition.
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            param_type: "string".to_string(),
            description: Some(description.into()),
            required: false,
            default: None,
            enum_values: None,
            items: None,
            properties: None,
        }
    }

    /// Create a required string parameter.
    pub fn required_string(description: impl Into<String>) -> Self {
        let mut p = Self::string(description);
        p.required = true;
        p
    }

    /// Create an integer parameter.
    pub fn integer(description: impl Into<String>) -> Self {
        Self {
            param_type: "integer".to_string(),
            description: Some(description.into()),
            required: false,
            default: None,
            enum_values: None,
            items: None,
            properties: None,
        }
    }

    /// Create a boolean parameter.
    pub fn boolean(description: impl Into<String>) -> Self {
        Self {
            param_type: "boolean".to_string(),
            description: Some(description.into()),
            required: false,
            default: None,
            enum_values: None,
            items: None,
            properties: None,
        }
    }

    /// Create an array parameter.
    pub fn array(description: impl Into<String>, items: ParameterDefinition) -> Self {
        Self {
            param_type: "array".to_string(),
            description: Some(description.into()),
            required: false,
            default: None,
            enum_values: None,
            items: Some(Box::new(items)),
            properties: None,
        }
    }

    /// Mark the parameter as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set a default value.
    pub fn default(mut self, value: serde_json::Value) -> Self {
        self.default = Some(value);
        self
    }

    /// Set enum values.
    pub fn enum_values(mut self, values: Vec<String>) -> Self {
        self.enum_values = Some(values);
        self
    }
}

/// Full definition of a tool, including its metadata and parameter schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The name of the tool (used by the model to invoke it).
    pub name: String,
    /// A description of what the tool does.
    pub description: String,
    /// The parameter schema for the tool, as a JSON Schema object.
    pub parameters: HashMap<String, ParameterDefinition>,
    /// The category of the tool (e.g., "shell", "filesystem", "web").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Whether this tool requires user confirmation.
    #[serde(default)]
    pub requires_confirmation: bool,
    /// The risk level of this tool (0 = safe, 1 = low, 2 = medium, 3 = high).
    #[serde(default)]
    pub risk_level: u8,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: HashMap<String, ParameterDefinition>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            category: None,
            requires_confirmation: false,
            risk_level: 0,
        }
    }

    /// Set the tool category.
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Require user confirmation before execution.
    pub fn with_confirmation(mut self) -> Self {
        self.requires_confirmation = true;
        self
    }

    /// Set the risk level.
    pub fn risk_level(mut self, level: u8) -> Self {
        self.risk_level = level;
        self
    }

    /// Convert this definition to a JSON Schema representation for LLM consumption.
    pub fn to_json_schema(&self) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for (name, param) in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert(
                "type".to_string(),
                serde_json::Value::String(param.param_type.clone()),
            );
            if let Some(ref desc) = param.description {
                prop.insert(
                    "description".to_string(),
                    serde_json::Value::String(desc.clone()),
                );
            }
            if let Some(ref default) = param.default {
                prop.insert("default".to_string(), default.clone());
            }
            if let Some(ref enum_vals) = param.enum_values {
                let vals: Vec<serde_json::Value> = enum_vals
                    .iter()
                    .map(|v| serde_json::Value::String(v.clone()))
                    .collect();
                prop.insert("enum".to_string(), serde_json::Value::Array(vals));
            }
            if param.required {
                required.push(serde_json::Value::String(name.clone()));
            }
            properties.insert(name.clone(), serde_json::Value::Object(prop));
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        })
    }
}

/// The core trait that all tools must implement.
///
/// Provides the tool's identity (name, description), its parameter schema,
/// and the execution method that performs the tool's work.
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// Return the tool's definition (name, description, parameters).
    fn definition(&self) -> &ToolDefinition;

    /// Return the name of this tool.
    fn name(&self) -> &str {
        &self.definition().name
    }

    /// Return a description of this tool.
    fn description(&self) -> &str {
        &self.definition().description
    }

    /// Return the parameter schema for this tool.
    fn parameters(&self) -> &HashMap<String, ParameterDefinition> {
        &self.definition().parameters
    }

    /// Execute the tool with the given input arguments.
    ///
    /// The `args` parameter contains the parsed JSON arguments from the model.
    /// Returns a `ToolOutput` on success, or a `ToolError` on failure.
    async fn execute(&self, args: serde_json::Value) -> ToolResult;

    /// Validate the input arguments against the tool's parameter schema.
    ///
    /// Returns an error if validation fails.
    fn validate_args(&self, args: &serde_json::Value) -> std::result::Result<(), ToolError> {
        let def = self.definition();
        if let Some(obj) = args.as_object() {
            for (name, param) in &def.parameters {
                if param.required && !obj.contains_key(name) {
                    return Err(ToolError::invalid_args(format!(
                        "Missing required parameter '{}' for tool '{}'",
                        name,
                        self.name()
                    )));
                }
            }
        } else if !def.parameters.is_empty() && !args.is_null() {
            return Err(ToolError::invalid_args(format!(
                "Expected object arguments for tool '{}', got {:?}",
                self.name(),
                args
            )));
        }
        Ok(())
    }
}

/// A registry that manages all available tools.
///
/// Tools are registered by name and can be looked up for execution.
/// The registry also provides tool definitions for LLM consumption.
#[derive(Default)]
pub struct ToolRegistry {
    /// The map of tool names to their implementations.
    tools: HashMap<String, Arc<dyn Tool>>,
    /// Category index for tools.
    categories: HashMap<String, Vec<String>>,
}

impl ToolRegistry {
    /// Create a new, empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            categories: HashMap::new(),
        }
    }

    /// Build a registry containing every built-in tool.
    ///
    /// File-system scoped tools (filesystem, patch, git, image, pdf, artifacts)
    /// are rooted at the current working directory. The memory, session,
    /// messaging, and cron tools use fresh in-memory backends so the registry
    /// is self-contained and safe to build anywhere.
    ///
    /// This mirrors the Python `tools.registry.get_default_registry()` entry
    /// point. Callers that need persistent or externally-shared backends
    /// should construct the individual tools and register them explicitly.
    pub fn with_builtins() -> std::result::Result<Self, ToolError> {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::with_builtins_in(cwd)
    }

    /// Build a registry containing every built-in tool, rooted at
    /// `working_dir` for all file-system scoped tools.
    pub fn with_builtins_in(working_dir: PathBuf) -> std::result::Result<Self, ToolError> {
        let mut registry = ToolRegistry::new();

        // Shell and code execution.
        registry.register(crate::shell::ExecCommandTool::new(30))?;
        registry.register(crate::shell::BackgroundProcessTool::new())?;
        registry.register(crate::shell::EnhancedExecTool::default())?;
        registry.register(crate::code_exec::CodeExecTool::default())?;

        // Filesystem, patch, git, diff, and archive (all rooted at the
        // working directory).
        registry.register(crate::filesystem::FilesystemTool::new(working_dir.clone()))?;
        registry.register(crate::patch::ApplyPatchTool::new(working_dir.clone()))?;
        registry.register(crate::patch::ReversePatchTool::new(working_dir.clone()))?;
        registry.register(crate::patch::ThreeWayMergeTool::new(working_dir.clone()))?;
        registry.register(crate::patch::ResolveConflictsTool::new(working_dir.clone()))?;
        registry.register(crate::git::GitTool::new(working_dir.clone()))?;
        registry.register(crate::diff::DiffTool::new(working_dir.clone()))?;
        registry.register(crate::diff::DirDiffTool::new(working_dir.clone()))?;
        registry.register(crate::archive::ArchiveTool::new(working_dir.clone()))?;

        // System process monitoring.
        registry.register(crate::process_monitor::ProcessMonitorTool::new())?;

        // Web.
        registry.register(crate::web::WebSearchTool::default())?;
        registry.register(crate::web::WebFetchTool::default())?;
        registry.register(crate::web::HttpRequestTool::default())?;
        registry.register(crate::web::WebExtractTool::default())?;

        // Media: image, pdf, tts, transcription, plus the combined media dispatcher.
        registry.register(crate::media::ImageTool::new(working_dir.clone()))?;
        registry.register(crate::media::PdfTool::new(working_dir.clone()))?;
        registry.register(crate::media::TtsTool::new(None))?;
        registry.register(crate::media::TranscriptionTool::new(
            working_dir.clone(),
            None,
        ))?;
        registry.register(crate::media::MediaTool::new(working_dir.clone(), None))?;

        // Artifact generation.
        registry.register(crate::artifacts::ArtifactTool::new(working_dir.clone()))?;
        registry.register(crate::artifacts::GenerateMarkdownTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::artifacts::GenerateJsonTool::new(working_dir.clone()))?;

        // File authoring: pdf/xlsx/csv/html generation + readers.
        // Note: generate_json and generate_markdown are already registered by
        // the artifacts module above, so they are intentionally not duplicated.
        registry.register(crate::file_authoring::GeneratePdfTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::file_authoring::GenerateXlsxTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::file_authoring::GenerateCsvTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::file_authoring::GenerateHtmlTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::file_authoring::ReadXlsxTool::new(
            working_dir.clone(),
        ))?;
        registry.register(crate::file_authoring::ReadCsvTool::new(working_dir))?;

        // Memory tools share a single in-memory store.
        let memory_store = opensquilla_memory::MemoryStore::in_memory().map_err(|e| {
            ToolError::new(
                "MEMORY_ERROR",
                format!("Failed to open in-memory memory store: {}", e),
            )
        })?;
        registry.register(crate::memory_tools::MemorySaveTool::new(
            memory_store.clone(),
        ))?;
        registry.register(crate::memory_tools::MemorySearchTool::new(
            memory_store.clone(),
        ))?;
        registry.register(crate::memory_tools::MemoryDeleteTool::new(
            memory_store.clone(),
        ))?;
        registry.register(crate::memory_tools::MemoryListTool::new(memory_store))?;

        // Session tools share a single in-memory storage handle.
        let session_storage = Arc::new(opensquilla_session::SessionStorage::in_memory().map_err(
            |e| {
                ToolError::new(
                    "SESSION_ERROR",
                    format!("Failed to open in-memory session storage: {}", e),
                )
            },
        )?);
        registry.register(crate::session_tools::SessionCreateTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_tools::SessionListTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_tools::SessionGetTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_tools::SessionSwitchTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_tools::SessionExportTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_tools::SessionDeleteTool::from_arc(
            session_storage.clone(),
        ))?;

        // Multi-session RPC tools share the same session storage handle.
        registry.register(crate::session_rpc_tools::SessionsSendTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_rpc_tools::SessionsSpawnTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_rpc_tools::SessionsYieldTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::session_rpc_tools::SessionsHistoryTool::from_arc(
            session_storage.clone(),
        ))?;

        // Messaging.
        registry.register(crate::messaging::SendMessageTool::default())?;

        // Cron tools share a single scheduler engine (in-memory job store).
        let engine = Arc::new(crate::cron_tool::build_scheduler_engine(None)?);
        registry.register(crate::cron_tool::ScheduleTaskTool::from_arc(engine.clone()))?;
        registry.register(crate::cron_tool::ListTasksTool::from_arc(engine.clone()))?;
        registry.register(crate::cron_tool::CancelTaskTool::from_arc(engine))?;

        // Skill management tools share a single in-memory skill loader plus a
        // community hub rooted at a temp managed dir. Mutation tools operate
        // on a temp workspace dir so the registry is self-contained.
        let skill_loader = Arc::new(opensquilla_skills::SkillLoader::new());
        let skill_workspace_dir = std::env::temp_dir().join("opensquilla-skills-workspace");
        let skill_hub = Arc::new(
            opensquilla_skills::SkillHub::new(
                std::env::temp_dir().join("opensquilla-skills-managed"),
            )
            .map_err(|e| {
                ToolError::new("SKILL_ERROR", format!("Failed to create skill hub: {}", e))
            })?,
        );
        registry.register(crate::skill_tools::SkillListTool::from_arc(
            skill_loader.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillViewTool::from_arc(
            skill_loader.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillSearchCommunityTool::from_arc(
            skill_loader.clone(),
            skill_hub.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillInstallCommunityTool::from_arc(
            skill_hub,
        ))?;
        registry.register(crate::skill_tools::InstallSkillDepsTool::from_arc(
            skill_loader.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillCreateTool::from_arc(
            skill_loader.clone(),
            skill_workspace_dir.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillEditTool::from_arc(
            skill_loader.clone(),
        ))?;
        registry.register(crate::skill_tools::SkillDeleteTool::from_arc(skill_loader))?;

        // Plan-control tools share the session storage already built above.
        registry.register(crate::plan_control::SubmitPlanTool::from_arc(
            session_storage.clone(),
        ))?;
        registry.register(crate::plan_control::RequestUserInputTool::new())?;
        registry.register(crate::plan_control::PlanRunCheckpointTool::from_arc(
            session_storage.clone(),
        ))?;

        // Router control tool (enabled with a fresh in-memory hold store).
        registry.register(crate::router_control::RouterControlTool::enabled())?;

        Ok(registry)
    }

    /// Register a tool in the registry.
    ///
    /// Returns an error if a tool with the same name is already registered.
    pub fn register(&mut self, tool: impl Tool + 'static) -> std::result::Result<(), ToolError> {
        let tool = Arc::new(tool);
        let name = tool.name().to_string();
        let category = tool.definition().category.clone();

        if self.tools.contains_key(&name) {
            return Err(ToolError::new(
                "DUPLICATE_TOOL",
                format!("Tool '{}' is already registered", name),
            ));
        }

        if let Some(ref cat) = category {
            self.categories
                .entry(cat.clone())
                .or_default()
                .push(name.clone());
        }

        self.tools.insert(name, tool);
        Ok(())
    }

    /// Register a tool from an Arc-wrapped implementation.
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) -> std::result::Result<(), ToolError> {
        let name = tool.name().to_string();
        let category = tool.definition().category.clone();

        if self.tools.contains_key(&name) {
            return Err(ToolError::new(
                "DUPLICATE_TOOL",
                format!("Tool '{}' is already registered", name),
            ));
        }

        if let Some(ref cat) = category {
            self.categories
                .entry(cat.clone())
                .or_default()
                .push(name.clone());
        }

        self.tools.insert(name, tool);
        Ok(())
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Check if a tool with the given name exists.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get all registered tool names.
    pub fn tool_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get all tool definitions for LLM consumption.
    pub fn definitions(&self) -> Vec<serde_json::Value> {
        let mut defs: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|tool| {
                let def = tool.definition();
                serde_json::json!({
                    "name": def.name,
                    "description": def.description,
                    "parameters": def.to_json_schema(),
                    "category": def.category,
                })
            })
            .collect();
        defs.sort_by(|a, b| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or(""))
        });
        defs
    }

    /// Get tools grouped by category.
    pub fn by_category(&self) -> HashMap<&str, Vec<&dyn Tool>> {
        let mut result: HashMap<&str, Vec<&dyn Tool>> = HashMap::new();
        for tool in self.tools.values() {
            let cat = tool
                .definition()
                .category
                .as_deref()
                .unwrap_or("uncategorized");
            result.entry(cat).or_default().push(tool.as_ref());
        }
        result
    }

    /// Get all tools in a specific category.
    pub fn get_category(&self, category: &str) -> Vec<Arc<dyn Tool>> {
        self.categories
            .get(category)
            .map(|names| {
                names
                    .iter()
                    .filter_map(|n| self.tools.get(n).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a tool from the registry by name.
    pub fn remove(&mut self, name: &str) -> bool {
        if let Some(tool) = self.tools.remove(name) {
            if let Some(ref cat) = tool.definition().category {
                if let Some(names) = self.categories.get_mut(cat) {
                    names.retain(|n| n != name);
                }
            }
            true
        } else {
            false
        }
    }

    /// Iterate over all registered tools.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &dyn Tool)> {
        self.tools
            .iter()
            .map(|(name, tool)| (name.as_str(), tool.as_ref()))
    }
}

/// The input for a tool execution, extracted from a model's ToolCall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// The tool call that triggered this execution.
    pub call: ToolCall,
    /// The parsed arguments as a JSON value.
    pub args: serde_json::Value,
}

impl ToolInput {
    /// Create a new tool input from a tool call.
    pub fn from_call(call: ToolCall) -> Self {
        let args = call.input.clone();
        Self { call, args }
    }

    /// Get a string argument by name.
    pub fn get_string(&self, name: &str) -> Option<String> {
        self.args
            .get(name)
            .and_then(|v| v.as_str().map(String::from))
    }

    /// Get an integer argument by name.
    pub fn get_i64(&self, name: &str) -> Option<i64> {
        self.args.get(name).and_then(|v| v.as_i64())
    }

    /// Get a boolean argument by name.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.args.get(name).and_then(|v| v.as_bool())
    }

    /// Get a float argument by name.
    pub fn get_f64(&self, name: &str) -> Option<f64> {
        self.args.get(name).and_then(|v| v.as_f64())
    }

    /// Get an array argument by name.
    pub fn get_array(&self, name: &str) -> Option<Vec<serde_json::Value>> {
        self.args.get(name).and_then(|v| v.as_array().cloned())
    }

    /// Get an object argument by name.
    pub fn get_object(&self, name: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
        self.args.get(name).and_then(|v| v.as_object().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> &ToolDefinition {
            static DEFINITION: std::sync::LazyLock<ToolDefinition> =
                std::sync::LazyLock::new(|| {
                    ToolDefinition::new(
                        "echo",
                        "Echo back the input text",
                        HashMap::from([(
                            "text".to_string(),
                            ParameterDefinition::required_string("The text to echo"),
                        )]),
                    )
                });
            &DEFINITION
        }

        async fn execute(&self, args: serde_json::Value) -> ToolResult {
            let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Ok(ToolOutput::success(text))
        }
    }

    #[tokio::test]
    async fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        assert!(registry.register(EchoTool).is_ok());
        assert!(registry.has_tool("echo"));
        assert_eq!(registry.len(), 1);
        let tool = registry.get("echo").unwrap();
        assert_eq!(tool.name(), "echo");
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let mut registry = ToolRegistry::new();
        registry.register(EchoTool).unwrap();
        let result = registry.register(EchoTool);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "DUPLICATE_TOOL");
    }

    #[tokio::test]
    async fn test_validate_args() {
        let tool = EchoTool;
        assert!(tool.validate_args(&json!({"text": "hello"})).is_ok());
        assert!(tool.validate_args(&json!({})).is_err());
        assert!(tool.validate_args(&json!({"other": "value"})).is_err());
    }

    #[tokio::test]
    async fn test_execute_tool() {
        let tool = EchoTool;
        let result = tool.execute(json!({"text": "hello world"})).await.unwrap();
        assert_eq!(result.content, "hello world");
        assert!(!result.is_error);
    }

    #[test]
    fn test_with_builtins_registers_all_tools() {
        let registry = ToolRegistry::with_builtins().expect("built-in registry");
        let names = registry.tool_names();

        // Core execution + filesystem + web + git + media + artifacts.
        for expected in [
            "exec_command",
            "background_process",
            "execute_code",
            "filesystem",
            "apply_patch",
            "web_search",
            "web_fetch",
            "http_request",
            "git",
            "image",
            "pdf",
            "tts",
            "media",
            "publish_artifact",
            "generate_markdown",
            "generate_json",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing built-in tool '{}' (registered: {:?})",
                expected,
                names
            );
        }

        // The expanded tool families must all be present.
        for expected in [
            // Shell enhancements.
            "exec_command_enhanced",
            // Filesystem enhancements.
            "reverse_patch",
            "merge_three_way",
            "resolve_conflicts",
            // Diffing.
            "diff_files",
            "diff_directories",
            // Archives.
            "archive",
            // Process monitoring.
            "process_monitor",
            // Web extraction.
            "web_extract",
            // Media transcription.
            "transcribe_audio",
            // File authoring.
            "generate_pdf",
            "generate_xlsx",
            "generate_csv",
            "generate_html",
            "read_xlsx",
            "read_csv",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing expanded tool '{}' (registered: {:?})",
                expected,
                names
            );
        }

        // The four new tool families must all be present.
        for expected in [
            "memory_save",
            "memory_search",
            "memory_delete",
            "memory_list",
            "session_create",
            "session_list",
            "session_get",
            "session_switch",
            "session_export",
            "session_delete",
            "sessions_send",
            "sessions_spawn",
            "sessions_yield",
            "sessions_history",
            "send_message",
            "schedule_task",
            "list_tasks",
            "cancel_task",
            // Skill management tools.
            "skill_list",
            "skill_view",
            "skill_search_community",
            "skill_install_community",
            "install_skill_deps",
            "skill_create",
            "skill_edit",
            "skill_delete",
            // Plan-control tools.
            "submit_plan",
            "request_user_input",
            "plan_run_checkpoint",
            // Router control tool.
            "router_control",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing built-in tool '{}' (registered: {:?})",
                expected,
                names
            );
        }

        // No duplicate registrations are allowed, so the registry length must
        // equal the number of unique tool names.
        assert_eq!(names.len(), registry.len());
    }
}
