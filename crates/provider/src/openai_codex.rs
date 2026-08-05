//! OpenAI Codex backend.
//!
//! The Codex backend streams from the ChatGPT backend `codex/responses`
//! endpoint — an OpenAI Responses-flavored SSE protocol authenticated with the
//! operator's ChatGPT subscription (Bearer access token + optional
//! `chatgpt-account-id` header) instead of a platform API key. It is
//! Responses-item compatible, so it reuses the input-item builder and SSE
//! parser from the [`crate::openai_responses`] module.
//!
//! Beyond the streaming chat surface ([`ChatProvider`] / [`Provider`]), this
//! backend provides code-oriented conveniences:
//!
//! * [`OpenAICodexProvider::execute_code`] — runs generated code in a
//!   sandboxed subprocess (via the `opensquilla-sandbox` crate), collecting
//!   stdout, stderr, the exit code, and any files the run produced.
//! * [`OpenAICodexProvider::generate_code`] — asks the model to write code for
//!   a prompt and extracts a fenced code block plus its explanation.
//! * [`OpenAICodexProvider::debug_code`] — asks the model to review code and
//!   parses a structured issue/fix report.

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use opensquilla_core::types::{
    ChatMessage, ContentBlock, MessageRole, ToolCall, ToolDefinition, Usage,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::openai_responses::{build_responses_input_items, parse_responses_sse_event};
use crate::stream::{SseStream, ToolCallBuffer};
use crate::types::{
    ChatConfig, ChatProvider, Provider, ProviderError, ProviderResponse, ProviderResult,
    StreamEvent,
};

/// Default ChatGPT backend base URL for the Codex endpoint.
pub const CODEX_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
/// The default model for Codex requests.
pub const CODEX_DEFAULT_MODEL: &str = "codex-latest";
/// Originator header/UA the Codex CLI uses so the backend recognizes the
/// credential owner.
pub const CODEX_ORIGINATOR: &str = "codex_cli_rs";

/// Known Codex models and their display names.
pub const KNOWN_CODEX_MODELS: &[(&str, &str)] = &[
    ("codex-latest", "Codex (latest)"),
    ("codex-mini-latest", "Codex Mini (latest)"),
    ("gpt-4o-codex", "GPT-4o Codex"),
    ("gpt-5.5", "GPT-5.5"),
    ("gpt-5", "GPT-5"),
    ("o4-mini", "GPT-4o mini"),
    ("o3", "GPT-3 (o3)"),
];

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a Codex backend.
///
/// `base_url` defaults to the ChatGPT backend
/// (`https://chatgpt.com/backend-api`). `api_key` is the OAuth access token;
/// set `account_id` via [`OpenAICodexProvider::with_account_id`] for
/// multi-account subscriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexConfig {
    /// The OAuth access token (or platform API key for proxy deployments).
    pub api_key: String,
    /// The base URL (e.g. `https://chatgpt.com/backend-api`).
    pub base_url: String,
    /// The default model (e.g. `codex-latest`, `gpt-4o-codex`).
    pub default_model: String,
}

impl CodexConfig {
    /// Create a Codex config with default base URL and model.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: CODEX_DEFAULT_BASE_URL.into(),
            default_model: CODEX_DEFAULT_MODEL.into(),
        }
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(&base_url.into());
        self
    }

    /// Override the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

/// Normalize a Codex base URL.
///
/// `chatgpt.com` / `chat.openai.com` hosts are given the `/backend-api` path
/// prefix when absent; all other hosts (including OpenAI-platform and proxy
/// endpoints) are returned unchanged.
pub fn normalize_base_url(base_url: &str) -> String {
    let base = (base_url.trim().to_owned())
        .trim_end_matches('/')
        .to_string();
    let host_only = base.to_lowercase();
    if (host_only.contains("chatgpt.com") || host_only.contains("chat.openai.com"))
        && !host_only.contains("/backend-api")
    {
        format!("{base}/backend-api")
    } else {
        base
    }
}

// ---------------------------------------------------------------------------
// Code execution / generation types
// ---------------------------------------------------------------------------

/// A file supplied to or produced by a code run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFile {
    /// The path relative to the execution workspace.
    pub path: String,
    /// The file contents.
    pub content: String,
}

/// A request to execute code in a sandboxed subprocess.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionRequest {
    /// The language of the code (`python`, `node`, `bash`, `rust`, `go`, ...).
    pub language: String,
    /// The source code to run.
    pub code: String,
    /// Additional files to place in the workspace before execution. Paths are
    /// resolved relative to the workspace root and must not escape it.
    #[serde(default)]
    pub files: Vec<CodeFile>,
    /// Optional CPU-time limit in seconds (overrides the sandbox default).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Whether the sandbox allows network access (strict policy denies it).
    #[serde(default)]
    pub network_allowed: bool,
}

impl CodeExecutionRequest {
    /// Create a request for the given language and source code.
    pub fn new(language: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            code: code.into(),
            files: Vec::new(),
            timeout_secs: None,
            network_allowed: false,
        }
    }

    /// Attach workspace files to the request.
    pub fn with_files(mut self, files: Vec<CodeFile>) -> Self {
        self.files = files;
        self
    }

    /// Set the execution timeout in seconds.
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = Some(timeout_secs);
        self
    }
}

/// The result of a sandboxed code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Captured standard output.
    pub stdout: String,
    /// Captured standard error.
    pub stderr: String,
    /// The process exit code (`-1` on timeout).
    pub exit_code: i32,
    /// Wall-clock duration of the run in milliseconds.
    pub duration_ms: u64,
    /// Files present in the workspace after the run (excluding the input main
    /// file).
    pub files: Vec<CodeFile>,
    /// The sandbox level the code ran under (e.g. `"strict"`).
    pub sandbox_level: String,
    /// Whether the run was terminated by the sandbox timeout.
    pub timed_out: bool,
}

/// Code produced by the model for a generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCode {
    /// The extracted source code (from the first fenced code block).
    pub code: String,
    /// Any surrounding explanation text.
    pub explanation: String,
    /// The requested language.
    pub language: String,
}

/// A single issue found by a code review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDebugIssue {
    /// Human-readable description of the issue.
    pub description: String,
    /// Optional source location (file:line or a snippet).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// Optional suggested fix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    /// Severity (`error`, `warning`, `info`, ...).
    #[serde(default = "default_severity")]
    pub severity: String,
}

fn default_severity() -> String {
    "info".to_string()
}

/// A structured code-review report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeDebugReport {
    /// The issues found by the review.
    pub issues: Vec<CodeDebugIssue>,
    /// Corrected code, when the model provides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_code: Option<String>,
    /// A short summary of the review.
    #[serde(default)]
    pub summary: String,
}

// ---------------------------------------------------------------------------
// The concrete provider
// ---------------------------------------------------------------------------

/// Provider for the OpenAI Codex endpoint.
///
/// The provider implements the canonical [`Provider`] trait and the streaming
/// [`ChatProvider`] entrypoint. Because the Codex backend streams only, the
/// non-streaming [`Provider::send_message`] collects the stream.
pub struct OpenAICodexProvider {
    name: String,
    api_base: String,
    api_key: String,
    default_model: String,
    account_id: Option<String>,
    originator: String,
    client: Client,
}

impl std::fmt::Debug for OpenAICodexProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICodexProvider")
            .field("name", &self.name)
            .field("api_base", &self.api_base)
            .field("api_key", &"[redacted]")
            .field("default_model", &self.default_model)
            .field("account_id", &self.account_id)
            .field("originator", &self.originator)
            .finish_non_exhaustive()
    }
}

impl OpenAICodexProvider {
    /// Create a new Codex provider.
    ///
    /// * `name` – A label for this provider instance (e.g. `"openai_codex"`).
    /// * `api_base` – The base URL. `chatgpt.com` / `chat.openai.com` hosts
    ///   are normalized to include `/backend-api`.
    /// * `api_key` – The OAuth access token.
    pub fn new(
        name: impl Into<String>,
        api_base: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::from_config(
            name.into(),
            CodexConfig::new(api_key).with_base_url(api_base.into()),
        )
    }

    /// Create a provider from a [`CodexConfig`].
    pub fn from_config(name: impl Into<String>, config: CodexConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to create reqwest Client");
        Self {
            name: name.into(),
            api_base: config.base_url,
            api_key: config.api_key,
            default_model: config.default_model,
            account_id: None,
            originator: CODEX_ORIGINATOR.into(),
            client,
        }
    }

    /// Set the `chatgpt-account-id` header for multi-account subscriptions.
    pub fn with_account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Override the default model.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// The configured API base URL.
    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    /// The default model for this provider.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// The full Codex responses endpoint URL.
    pub fn responses_url(&self) -> String {
        format!("{}/codex/responses", self.api_base.trim_end_matches('/'))
    }

    /// The effective request model, preferring the config's model.
    fn effective_model(&self, config: &ChatConfig) -> &str {
        if config.model.is_empty() {
            &self.default_model
        } else {
            &config.model
        }
    }

    /// Build the Codex request body.
    ///
    /// The Codex endpoint is Responses-item compatible: it carries
    /// `instructions`, an `input` array, `tools`, `tool_choice`,
    /// `parallel_tool_calls`, `store: false`, and `include:
    /// ["reasoning.encrypted_content"]`. The ChatGPT backend rejects
    /// `max_output_tokens`, so it is not emitted (a debug log notes when a
    /// caller requested one).
    fn build_request_body(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> serde_json::Value {
        if config.max_tokens > 0 && !config.extra.contains_key("max_output_tokens") {
            debug!(
                target = "provider",
                provider = %self.name,
                requested_max_tokens = config.max_tokens,
                "Codex backend does not accept max_output_tokens; dropping it"
            );
        }

        let input_items = build_responses_input_items(messages);
        let input: Vec<serde_json::Value> = input_items.iter().map(|item| item.to_json()).collect();
        let system_text: String = messages
            .iter()
            .filter(|m| m.role == MessageRole::System)
            .map(|m| m.text_content())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut body = serde_json::json!({
            "model": self.effective_model(config),
            "instructions": system_text,
            "input": input,
            "tool_choice": config.extra.get("tool_choice").cloned().unwrap_or_else(|| serde_json::json!("auto")),
            "parallel_tool_calls": true,
            "store": false,
            "stream": true,
            "include": ["reasoning.encrypted_content"],
        });

        if !tools.is_empty() {
            let tool_defs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "strict": false,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tool_defs);
        }

        if let Some(obj) = body.as_object_mut() {
            for (k, v) in &config.extra {
                obj.insert(k.clone(), v.clone());
            }
        }
        body
    }

    /// Send the request to the Codex endpoint and normalize errors.
    async fn post_stream(&self, body: &serde_json::Value) -> ProviderResult<reqwest::Response> {
        let mut request = self
            .client
            .post(self.responses_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("originator", &self.originator)
            .header("User-Agent", &self.originator)
            .json(body);
        if let Some(account) = &self.account_id {
            request = request.header("chatgpt-account-id", account);
        }
        let resp = request.send().await.map_err(ProviderError::Network)?;
        let status = resp.status();
        if !status.is_success() {
            let error_text = resp.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(ProviderError::Auth(error_text));
            }
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Err(ProviderError::RateLimited(error_text));
            }
            return Err(ProviderError::Provider(format!(
                "HTTP {status}: {error_text}"
            )));
        }
        Ok(resp)
    }

    /// List the well-known Codex models.
    pub fn list_models(&self) -> Vec<String> {
        KNOWN_CODEX_MODELS
            .iter()
            .map(|(id, _)| id.to_string())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Code generation
    // -----------------------------------------------------------------------

    /// Generate code for a prompt in the given language.
    ///
    /// Builds a Codex chat request with a code-generation system prompt,
    /// collects the stream, and extracts the first fenced code block as the
    /// generated source.
    pub async fn generate_code(
        &self,
        config: &ChatConfig,
        prompt: &str,
        language: &str,
    ) -> Result<GeneratedCode, ProviderError> {
        let messages = build_generation_messages(prompt, language);
        let response = self.send_message(config, &messages, &[]).await?;
        let full_text = response
            .content
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");
        let (code, explanation) = extract_code_blocks(&full_text);
        Ok(GeneratedCode {
            code,
            explanation,
            language: language.to_string(),
        })
    }

    /// Debug a snippet of code.
    ///
    /// Builds a review prompt, collects the stream, and parses either a JSON
    /// report (`{issues, summary, suggested_code}`) or a markdown-style
    /// listing into a [`CodeDebugReport`].
    pub async fn debug_code(
        &self,
        config: &ChatConfig,
        code: &str,
        language: &str,
    ) -> Result<CodeDebugReport, ProviderError> {
        let messages = build_debug_messages(code, language);
        let response = self.send_message(config, &messages, &[]).await?;
        let full_text = response
            .content
            .iter()
            .map(|m| m.text_content())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(parse_debug_report(&full_text))
    }

    // -----------------------------------------------------------------------
    // Sandboxed code execution
    // -----------------------------------------------------------------------

    /// Execute code in a sandboxed subprocess.
    ///
    /// The source is written to a temporary workspace, additional files are
    /// laid down beside it, and the appropriate interpreter/compiler is run
    /// under a strict [`opensquilla_sandbox::SandboxPolicy`]. For compiled
    /// languages the workspace binary is compiled and then executed. Any files
    /// present after the run are collected into [`CodeExecutionResult::files`]
    /// (the input main file is excluded).
    ///
    /// The workspace directory is always cleaned up, including on failure.
    pub async fn execute_code(
        &self,
        request: &CodeExecutionRequest,
    ) -> Result<CodeExecutionResult, ProviderError> {
        let extension = language_extension(&request.language)?;
        let workdir = create_workdir()?;
        let main_name = format!("main.{extension}");
        let main_file = workdir.join(&main_name);

        // Write inputs.
        std::fs::write(&main_file, &request.code)
            .map_err(|e| ProviderError::Internal(format!("failed to write main file: {e}")))?;
        for file in &request.files {
            let target = safe_join(&workdir, &file.path)?;
            std::fs::write(&target, &file.content).map_err(|e| {
                ProviderError::Internal(format!(
                    "failed to write workspace file {}: {e}",
                    file.path
                ))
            })?;
        }

        let policy = {
            let mut base = SandboxPolicy::build_policy(SandboxLevel::Strict, None);
            base.resource_limits.cpu_time_secs =
                request.timeout_secs.or(base.resource_limits.cpu_time_secs);
            if request.network_allowed {
                base.network = NetworkPolicy::Host;
            }
            base
        };

        let plan = execution_plan(&request.language, &workdir, &main_file)?;

        // Compiled languages: compile first; a failed compile returns the
        // compiler output directly.
        if let ExecutionPlan::Compiled {
            compile_command,
            compile_args,
            ..
        } = &plan
        {
            let mut sandbox = PlatformSandbox::new();
            let compile_args_ref: Vec<&str> = compile_args.iter().map(|s| s.as_str()).collect();
            let compile_result = sandbox
                .execute(compile_command, &compile_args_ref, &policy)
                .await
                .map_err(ProviderError::Internal)?;
            if compile_result.exit_code != 0 {
                let result = CodeExecutionResult {
                    stdout: compile_result.stdout,
                    stderr: compile_result.stderr,
                    exit_code: compile_result.exit_code,
                    duration_ms: compile_result.duration_ms,
                    files: collect_files(&workdir, &main_name),
                    sandbox_level: "strict".into(),
                    timed_out: false,
                };
                let _ = std::fs::remove_dir_all(&workdir);
                return Ok(result);
            }
        }

        let (command, args) = plan.to_command()?;
        let args_ref: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let mut sandbox = PlatformSandbox::new();
        let start = std::time::Instant::now();
        let run_result = sandbox
            .execute(&command, &args_ref, &policy)
            .await
            .map_err(ProviderError::Internal)?;
        let duration_ms = run_result
            .duration_ms
            .max(start.elapsed().as_millis() as u64);

        let files = collect_files(&workdir, &main_name);
        let timed_out = run_result.exit_code == -1 && !run_result.stderr.is_empty();
        let result = CodeExecutionResult {
            stdout: run_result.stdout,
            stderr: run_result.stderr,
            exit_code: run_result.exit_code,
            duration_ms,
            files,
            sandbox_level: "strict".into(),
            timed_out,
        };

        let _ = std::fs::remove_dir_all(&workdir);
        Ok(result)
    }
}

/// Build the messages for a code-generation request.
fn build_generation_messages(prompt: &str, language: &str) -> Vec<ChatMessage> {
    let system = format!(
        "You are a senior {language} engineer. Write production-quality code \
         that satisfies the user's request. Respond with a short explanation \
         and a single fenced code block (```{language}) containing the complete \
         {language} source."
    );
    vec![
        ChatMessage::system(system),
        ChatMessage::user(prompt.to_string()),
    ]
}

/// Build the messages for a code-review request.
fn build_debug_messages(code: &str, language: &str) -> Vec<ChatMessage> {
    let system = "You are an expert code reviewer. Analyze the provided code and \
         list concrete issues, each with a severity, a location, and a suggested fix. \
         Provide corrected code when practical. Respond as JSON with exactly these \
         keys: issues (array of {\"description\", \"location\", \"fix\", \"severity\"}), \
         summary, suggested_code.";
    let user = format!("```{language}\n{code}\n```\n\nPlease review this code.");
    vec![ChatMessage::system(system), ChatMessage::user(user)]
}

/// Extract the first fenced code block (```lang ... ```) and the surrounding
/// explanation text.
fn extract_code_blocks(text: &str) -> (String, String) {
    let mut code = String::new();
    let mut explanation = String::new();
    let mut in_block = false;
    let mut block_seen = false;

    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            if in_block {
                in_block = false;
                block_seen = true;
            } else {
                in_block = true;
            }
            continue;
        }
        if in_block {
            if !code.is_empty() {
                code.push('\n');
            }
            code.push_str(line);
        } else if !block_seen {
            if !explanation.is_empty() {
                explanation.push('\n');
            }
            explanation.push_str(line);
        }
    }
    (code.trim().to_string(), explanation.trim().to_string())
}

/// Parse a code-review response into a [`CodeDebugReport`].
///
/// Prefers a JSON object with `issues` / `summary` / `suggested_code`. Falls
/// back to a loose markdown interpretation (bullet lines become issues; fenced
/// blocks become suggested code).
fn parse_debug_report(text: &str) -> CodeDebugReport {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(obj) = value.as_object() {
            let issues = obj
                .get("issues")
                .and_then(serde_json::Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|item| {
                            let description = item
                                .get("description")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            if description.is_empty() {
                                return None;
                            }
                            Some(CodeDebugIssue {
                                description,
                                location: item
                                    .get("location")
                                    .and_then(serde_json::Value::as_str)
                                    .map(String::from),
                                fix: item
                                    .get("fix")
                                    .and_then(serde_json::Value::as_str)
                                    .map(String::from),
                                severity: item
                                    .get("severity")
                                    .and_then(serde_json::Value::as_str)
                                    .unwrap_or("info")
                                    .to_string(),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let summary = obj
                .get("summary")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("")
                .to_string();
            let suggested_code = obj
                .get("suggested_code")
                .and_then(serde_json::Value::as_str)
                .map(String::from);
            return CodeDebugReport {
                issues,
                suggested_code,
                summary,
            };
        }
    }

    // Markdown fallback.
    let mut issues = Vec::new();
    let mut suggested = String::new();
    let mut in_block = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_block = !in_block;
            continue;
        }
        if in_block {
            if !suggested.is_empty() {
                suggested.push('\n');
            }
            suggested.push_str(line);
        } else if (trimmed.starts_with("- ") || trimmed.starts_with("* ")) && trimmed.len() > 2 {
            issues.push(CodeDebugIssue {
                description: trimmed[2..].trim().to_string(),
                location: None,
                fix: None,
                severity: "info".to_string(),
            });
        }
    }
    CodeDebugReport {
        issues,
        suggested_code: if suggested.trim().is_empty() {
            None
        } else {
            Some(suggested.trim().to_string())
        },
        summary: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Language / execution planning
// ---------------------------------------------------------------------------

/// Resolve a language name to a source file extension.
fn language_extension(language: &str) -> Result<&'static str, ProviderError> {
    match language.to_lowercase().as_str() {
        "python" | "python3" | "py" => Ok("py"),
        "javascript" | "js" | "node" => Ok("js"),
        "typescript" | "ts" => Ok("ts"),
        "bash" | "sh" | "shell" => Ok("sh"),
        "ruby" | "rb" => Ok("rb"),
        "php" => Ok("php"),
        "perl" => Ok("pl"),
        "go" | "golang" => Ok("go"),
        "rust" | "rs" => Ok("rs"),
        "c" => Ok("c"),
        "cpp" | "c++" => Ok("cpp"),
        _ => {
            return Err(ProviderError::Config(format!(
                "Unsupported code language: {language}"
            )));
        }
    }
}

/// How to execute a piece of code.
enum ExecutionPlan {
    /// Run a single command with arguments.
    Direct { command: String, args: Vec<String> },
    /// Compile to a workspace binary, then run it.
    Compiled {
        compile_command: String,
        compile_args: Vec<String>,
        run_command: String,
        run_args: Vec<String>,
    },
}

impl ExecutionPlan {
    /// The final run step as a `(command, args)` pair.
    fn to_command(&self) -> Result<(String, Vec<String>), ProviderError> {
        match self {
            Self::Direct { command, args } => Ok((command.clone(), args.clone())),
            Self::Compiled {
                run_command,
                run_args,
                ..
            } => Ok((run_command.clone(), run_args.clone())),
        }
    }
}

/// Build the execution plan for a language.
fn execution_plan(
    language: &str,
    workdir: &Path,
    main_file: &Path,
) -> Result<ExecutionPlan, ProviderError> {
    let lang = language.to_lowercase();
    let main = main_file.to_string_lossy().to_string();
    let bin = |name: &str| workdir.join(name).to_string_lossy().to_string();
    match lang.as_str() {
        "python" | "python3" | "py" => Ok(ExecutionPlan::Direct {
            command: python_command().into(),
            args: vec![main],
        }),
        "javascript" | "js" | "node" => Ok(ExecutionPlan::Direct {
            command: "node".into(),
            args: vec![main],
        }),
        "typescript" | "ts" => Ok(ExecutionPlan::Direct {
            command: "npx".into(),
            args: vec!["tsx".into(), main],
        }),
        "bash" | "sh" | "shell" => Ok(ExecutionPlan::Direct {
            command: "bash".into(),
            args: vec![main],
        }),
        "ruby" | "rb" => Ok(ExecutionPlan::Direct {
            command: "ruby".into(),
            args: vec![main],
        }),
        "php" => Ok(ExecutionPlan::Direct {
            command: "php".into(),
            args: vec![main],
        }),
        "perl" => Ok(ExecutionPlan::Direct {
            command: "perl".into(),
            args: vec![main],
        }),
        "go" | "golang" => Ok(ExecutionPlan::Direct {
            command: "go".into(),
            args: vec!["run".into(), main],
        }),
        "rust" | "rs" => Ok(ExecutionPlan::Compiled {
            compile_command: "rustc".into(),
            compile_args: vec![main.clone(), "-o".into(), bin("program")],
            run_command: bin("program"),
            run_args: vec![],
        }),
        "c" => Ok(ExecutionPlan::Compiled {
            compile_command: "cc".into(),
            compile_args: vec![main.clone(), "-o".into(), bin("program")],
            run_command: bin("program"),
            run_args: vec![],
        }),
        "cpp" | "c++" => Ok(ExecutionPlan::Compiled {
            compile_command: "c++".into(),
            compile_args: vec![main.clone(), "-o".into(), bin("program")],
            run_command: bin("program"),
            run_args: vec![],
        }),
        _ => Err(ProviderError::Config(format!(
            "Unsupported code language: {language}"
        ))),
    }
}

/// The Python interpreter name for this platform.
fn python_command() -> &'static str {
    if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    }
}

/// Create a unique temporary workspace directory.
fn create_workdir() -> Result<PathBuf, ProviderError> {
    let nonce = format!(
        "osq_codex_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let workdir = std::env::temp_dir().join(nonce);
    std::fs::create_dir_all(&workdir)
        .map_err(|e| ProviderError::Internal(format!("failed to create codex workspace: {e}")))?;
    Ok(workdir)
}

/// Join a relative path under a workspace root, rejecting traversal.
fn safe_join(workdir: &Path, rel: &str) -> Result<PathBuf, ProviderError> {
    let candidate = Path::new(rel);
    if candidate.is_absolute()
        || candidate.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(ProviderError::Config(format!(
            "Unsafe codex workspace path: {rel}"
        )));
    }
    Ok(workdir.join(candidate))
}

/// Collect the files present in a workspace directory, excluding the input
/// main file.
fn collect_files(workdir: &Path, exclude_main: &str) -> Vec<CodeFile> {
    let mut out = Vec::new();
    let mut stack = vec![workdir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|n| n.to_str()) != Some(exclude_main) {
                let content = std::fs::read_to_string(&path).unwrap_or_default();
                let rel = path
                    .strip_prefix(workdir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();
                out.push(CodeFile { path: rel, content });
            }
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

// ---------------------------------------------------------------------------
// Sandbox backend selection
// ---------------------------------------------------------------------------

use opensquilla_sandbox::{NetworkPolicy, SandboxLevel, SandboxPolicy};

#[cfg(target_os = "linux")]
type PlatformSandbox = opensquilla_sandbox::LinuxSandbox;
#[cfg(target_os = "macos")]
type PlatformSandbox = opensquilla_sandbox::MacOsSandbox;
#[cfg(target_os = "windows")]
type PlatformSandbox = opensquilla_sandbox::WindowsSandbox;
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
type PlatformSandbox = opensquilla_sandbox::NoopSandbox;

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl Provider for OpenAICodexProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn supported_models(&self) -> Vec<String> {
        self.list_models()
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let mut stream = self.chat(config, messages, tools).await?;

        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut buffer = ToolCallBuffer::default();
        let mut usage = Usage::default();
        let mut stop_reason: Option<String> = None;

        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Text { text: t } => text.push_str(&t),
                StreamEvent::Reasoning { .. } => {}
                StreamEvent::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    if let Some(tc) = buffer.accumulate(&id, &name, &arguments) {
                        tool_calls.push(tc);
                    }
                }
                StreamEvent::Done {
                    usage: u,
                    stop_reason: s,
                } => {
                    if let Some(u) = u {
                        usage = u;
                    }
                    stop_reason = s;
                }
                StreamEvent::Error { message } => {
                    return Err(ProviderError::Provider(message));
                }
            }
        }
        tool_calls.extend(buffer.flush());

        let mut msg = ChatMessage {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text(text)],
            name: None,
            tool_call_id: None,
            tool_calls: None,
            tool_result: None,
        };
        if !tool_calls.is_empty() {
            msg.tool_calls = Some(tool_calls);
        }

        info!(
            target = "provider",
            provider = %self.name,
            model = %config.model,
            "Codex response collected"
        );

        Ok(ProviderResponse {
            content: vec![msg],
            usage,
            model: config.model.clone(),
            stop_reason,
        })
    }

    async fn stream_chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.chat(config, messages, tools).await
    }
}

#[async_trait]
impl ChatProvider for OpenAICodexProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let body = self.build_request_body(config, messages, tools);

        debug!(
            target = "provider",
            provider = %self.name,
            model = %config.model,
            "Sending Codex streaming request"
        );

        let resp = self.post_stream(&body).await?;
        let stream = SseStream::new(resp, parse_responses_sse_event);
        Ok(Box::new(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn provider() -> OpenAICodexProvider {
        OpenAICodexProvider::new(
            "openai_codex",
            "https://chatgpt.com/backend-api",
            "oauth-token",
        )
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_base_url_chatgpt() {
        assert_eq!(
            normalize_base_url("https://chatgpt.com"),
            "https://chatgpt.com/backend-api"
        );
        assert_eq!(
            normalize_base_url("https://chatgpt.com/backend-api/"),
            "https://chatgpt.com/backend-api"
        );
        assert_eq!(
            normalize_base_url("https://chat.openai.com"),
            "https://chat.openai.com/backend-api"
        );
        // Non-chatgpt hosts are unchanged.
        assert_eq!(
            normalize_base_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn test_codex_config_defaults() {
        let cfg = CodexConfig::new("key");
        assert_eq!(cfg.base_url, CODEX_DEFAULT_BASE_URL);
        assert_eq!(cfg.default_model, CODEX_DEFAULT_MODEL);
        let cfg = cfg.with_default_model("gpt-4o-codex");
        assert_eq!(cfg.default_model, "gpt-4o-codex");
    }

    #[test]
    fn test_responses_url() {
        assert_eq!(
            provider().responses_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        let p = OpenAICodexProvider::new("openai_codex", "https://api.openai.com/v1", "key");
        assert_eq!(
            p.responses_url(),
            "https://api.openai.com/v1/codex/responses"
        );
    }

    #[test]
    fn test_known_models() {
        let models = provider().list_models();
        assert!(models.contains(&"codex-latest".to_string()));
        assert!(models.contains(&"gpt-4o-codex".to_string()));
    }

    // -----------------------------------------------------------------------
    // Request building
    // -----------------------------------------------------------------------

    #[test]
    fn test_build_request_body_shape() {
        let p = provider();
        let config = ChatConfig {
            model: "codex-latest".into(),
            ..Default::default()
        };
        let messages = vec![
            ChatMessage::system("You are a coding assistant."),
            ChatMessage::user("write a function"),
        ];
        let body = p.build_request_body(&config, &messages, &[]);
        assert_eq!(body["model"], "codex-latest");
        assert_eq!(body["instructions"], "You are a coding assistant.");
        assert_eq!(body["stream"].as_bool(), Some(true));
        assert_eq!(body["store"].as_bool(), Some(false));
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"].as_bool(), Some(true));
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
        assert_eq!(body["input"].as_array().unwrap().len(), 2);
        // max_output_tokens is rejected by the backend and never emitted.
        assert!(body.get("max_output_tokens").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn test_build_request_body_tools() {
        let p = provider();
        let config = ChatConfig {
            model: "codex-latest".into(),
            ..Default::default()
        };
        let tools = vec![ToolDefinition {
            name: "search".into(),
            description: "Search".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = p.build_request_body(&config, &[ChatMessage::user("go")], &tools);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "search");
        assert_eq!(tools[0]["strict"].as_bool(), Some(false));
    }

    #[test]
    fn test_build_request_body_reasoning_effort() {
        let p = provider();
        let config = ChatConfig {
            model: "codex-latest".into(),
            extra: HashMap::from([(
                "reasoning".into(),
                serde_json::json!({"effort": "high", "summary": "auto"}),
            )]),
            ..Default::default()
        };
        let body = p.build_request_body(&config, &[ChatMessage::user("hi")], &[]);
        assert_eq!(body["reasoning"]["effort"], "high");
    }

    #[test]
    fn test_build_request_body_includes_tool_input_roundtrip() {
        let p = provider();
        let config = ChatConfig {
            model: "codex-latest".into(),
            ..Default::default()
        };
        let user = ChatMessage::user("weather in NYC");
        let mut user = user;
        user.content.push(ContentBlock::ToolUse(ToolCall::new(
            "call_1",
            "get_weather",
            serde_json::json!({"city": "NYC"}),
        )));
        let body = p.build_request_body(&config, &[user], &[]);
        let input = body["input"].as_array().unwrap();
        let fc = input.iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["call_id"], "call_1");
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["arguments"], r#"{"city":"NYC"}"#);
    }

    // -----------------------------------------------------------------------
    // SSE parsing (Codex uses the Responses protocol)
    // -----------------------------------------------------------------------

    #[test]
    fn test_codex_stream_uses_responses_parser() {
        let data = r#"{"type":"response.output_text.delta","delta":"fn main()"}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Text { text }) => assert_eq!(text, "fn main()"),
            _ => panic!("Expected Text event"),
        }
    }

    #[test]
    fn test_codex_completed_event() {
        let data = r#"{"type":"response.completed","response":{"status":"completed","model":"codex-latest","usage":{"input_tokens":5,"output_tokens":9}}}"#;
        let event = parse_responses_sse_event(data);
        match event.unwrap() {
            Ok(StreamEvent::Done { usage, stop_reason }) => {
                let usage = usage.unwrap();
                assert_eq!(usage.input_tokens, 5);
                assert_eq!(usage.output_tokens, 9);
                assert_eq!(stop_reason, Some("completed".into()));
            }
            _ => panic!("Expected Done event"),
        }
    }

    // -----------------------------------------------------------------------
    // Code generation / debug parsing (pure functions)
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_code_blocks() {
        let text = "Here is the code:\n```rust\nfn main() {}\n```\nThat's it.";
        let (code, explanation) = extract_code_blocks(text);
        assert_eq!(code, "fn main() {}");
        assert_eq!(explanation, "Here is the code:");
    }

    #[test]
    fn test_extract_code_blocks_no_fence() {
        let (code, explanation) = extract_code_blocks("plain code text");
        assert_eq!(code, "");
        assert_eq!(explanation, "plain code text");
    }

    #[test]
    fn test_parse_debug_report_json() {
        let text = r#"{"issues":[{"description":"null deref","location":"main.rs:3","fix":"check before use","severity":"error"}],"summary":"one issue","suggested_code":"fn main(){}"}"#;
        let report = parse_debug_report(text);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].description, "null deref");
        assert_eq!(report.issues[0].location.as_deref(), Some("main.rs:3"));
        assert_eq!(report.summary, "one issue");
        assert_eq!(report.suggested_code.as_deref(), Some("fn main(){}"));
    }

    #[test]
    fn test_parse_debug_report_markdown() {
        let text = "- Memory leak in loop\n- Unused import\n```\nfn main() {}\n```";
        let report = parse_debug_report(text);
        assert_eq!(report.issues.len(), 2);
        assert_eq!(report.issues[0].description, "Memory leak in loop");
        assert_eq!(report.suggested_code.as_deref(), Some("fn main() {}"));
    }

    #[test]
    fn test_parse_debug_report_invalid_json_falls_back() {
        let report = parse_debug_report("this is not json but a note");
        assert!(report.issues.is_empty());
        assert!(report.suggested_code.is_none());
    }

    #[test]
    fn test_generation_messages_shape() {
        let messages = build_generation_messages("add", "rust");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::System);
        assert!(messages[0].text_content().contains("rust"));
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[1].text_content(), "add");
    }

    // -----------------------------------------------------------------------
    // Language resolution
    // -----------------------------------------------------------------------

    #[test]
    fn test_language_extension() {
        assert_eq!(language_extension("python").unwrap(), "py");
        assert_eq!(language_extension("node").unwrap(), "js");
        assert_eq!(language_extension("bash").unwrap(), "sh");
        assert_eq!(language_extension("rust").unwrap(), "rs");
        assert_eq!(language_extension("C++").unwrap(), "cpp");
        assert!(language_extension("brainfuck").is_err());
    }

    #[test]
    fn test_execution_plan_direct() {
        let workdir = Path::new("/tmp/w");
        let main = workdir.join("main.py");
        let plan = execution_plan("python", workdir, &main).unwrap();
        match plan {
            ExecutionPlan::Direct { command, args } => {
                assert_eq!(args, vec!["/tmp/w/main.py".to_string()]);
                assert!(!command.is_empty());
            }
            _ => panic!("expected direct plan"),
        }
    }

    #[test]
    fn test_execution_plan_compiled() {
        let workdir = Path::new("/tmp/w");
        let main = workdir.join("main.rs");
        let plan = execution_plan("rust", workdir, &main).unwrap();
        match plan {
            ExecutionPlan::Compiled {
                compile_command, ..
            } => {
                assert_eq!(compile_command, "rustc");
            }
            _ => panic!("expected compiled plan"),
        }
    }

    #[test]
    fn test_execution_plan_to_command() {
        let workdir = Path::new("/tmp/w");
        let main = workdir.join("main.js");
        let plan = execution_plan("node", workdir, &main).unwrap();
        let (command, args) = plan.to_command().unwrap();
        assert_eq!(command, "node");
        assert_eq!(args, vec!["/tmp/w/main.js".to_string()]);
    }

    #[test]
    fn test_safe_join_rejects_traversal() {
        let workdir = Path::new("/tmp/w");
        assert!(safe_join(workdir, "out.txt").is_ok());
        assert!(safe_join(workdir, "../escape.txt").is_err());
        assert!(safe_join(workdir, "/abs/path.txt").is_err());
        assert!(safe_join(workdir, "a/b/c.txt").is_ok());
    }

    #[test]
    fn test_python_command_platform() {
        // Just verifies the helper returns a non-empty value.
        assert!(!python_command().is_empty());
    }

    #[test]
    fn test_collect_files_excludes_main() {
        let workdir = std::env::temp_dir().join(format!("osq_test_files_{}", std::process::id()));
        std::fs::create_dir_all(&workdir).unwrap();
        std::fs::write(workdir.join("main.py"), "print(1)").unwrap();
        std::fs::write(workdir.join("out.txt"), "hello").unwrap();
        std::fs::create_dir_all(workdir.join("sub")).unwrap();
        std::fs::write(workdir.join("sub/data.txt"), "data").unwrap();

        let files = collect_files(&workdir, "main.py");
        let paths: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
        assert!(!paths.iter().any(|p| p == "main.py"));
        assert!(paths.contains(&"out.txt".to_string()));
        assert!(paths.contains(&"sub/data.txt".to_string()));

        let _ = std::fs::remove_dir_all(&workdir);
    }
}
