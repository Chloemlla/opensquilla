//! Code execution tool: execute_code.
//!
//! Executes code in a subprocess with language detection, timeout,
//! and output capture. Supports multiple programming languages by
//! dispatching to the appropriate interpreter or compiler.

use crate::registry::{
    ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult,
};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;
use tokio::process::Command;
use tokio::time::Duration;

/// Language configuration for code execution.
#[derive(Debug, Clone)]
struct LanguageConfig {
    /// The name of the language.
    name: &'static str,
    /// The command to execute the code.
    command: &'static str,
    /// Arguments to pass before the code file.
    args: &'static [&'static str],
    /// File extension for the source file.
    extension: &'static str,
    /// Whether to write code to a temp file first.
    needs_file: bool,
}

/// Supported languages and their execution configurations.
const LANGUAGES: &[LanguageConfig] = &[
    LanguageConfig {
        name: "python",
        command: "python3",
        args: &[],
        extension: "py",
        needs_file: true,
    },
    LanguageConfig {
        name: "python3",
        command: "python3",
        args: &[],
        extension: "py",
        needs_file: true,
    },
    LanguageConfig {
        name: "python2",
        command: "python2",
        args: &[],
        extension: "py",
        needs_file: true,
    },
    LanguageConfig {
        name: "javascript",
        command: "node",
        args: &[],
        extension: "js",
        needs_file: true,
    },
    LanguageConfig {
        name: "js",
        command: "node",
        args: &[],
        extension: "js",
        needs_file: true,
    },
    LanguageConfig {
        name: "typescript",
        command: "npx",
        args: &["ts-node"],
        extension: "ts",
        needs_file: true,
    },
    LanguageConfig {
        name: "ts",
        command: "npx",
        args: &["ts-node"],
        extension: "ts",
        needs_file: true,
    },
    LanguageConfig {
        name: "ruby",
        command: "ruby",
        args: &[],
        extension: "rb",
        needs_file: true,
    },
    LanguageConfig {
        name: "perl",
        command: "perl",
        args: &[],
        extension: "pl",
        needs_file: true,
    },
    LanguageConfig {
        name: "php",
        command: "php",
        args: &[],
        extension: "php",
        needs_file: true,
    },
    LanguageConfig {
        name: "bash",
        command: "bash",
        args: &[],
        extension: "sh",
        needs_file: true,
    },
    LanguageConfig {
        name: "sh",
        command: "sh",
        args: &[],
        extension: "sh",
        needs_file: true,
    },
    LanguageConfig {
        name: "rust",
        command: "rustc",
        args: &[],
        extension: "rs",
        needs_file: true,
    },
    LanguageConfig {
        name: "go",
        command: "go",
        args: &["run"],
        extension: "go",
        needs_file: true,
    },
    LanguageConfig {
        name: "r",
        command: "Rscript",
        args: &[],
        extension: "R",
        needs_file: true,
    },
    LanguageConfig {
        name: "sqlite",
        command: "sqlite3",
        args: &[],
        extension: "sql",
        needs_file: true,
    },
];

/// Detect the language configuration from a language name or code snippet.
fn detect_language(language: &str) -> Option<&'static LanguageConfig> {
    let lang = language.to_lowercase();
    LANGUAGES.iter().find(|c| c.name == lang)
}

/// Tool for executing code in various languages.
pub struct CodeExecTool {
    /// Default timeout in seconds.
    timeout_secs: u64,
    /// Maximum output size in bytes.
    max_output_size: u64,
    /// Temporary directory for code files.
    temp_dir: Option<std::path::PathBuf>,
}

impl CodeExecTool {
    /// Create a new code execution tool.
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout_secs,
            max_output_size: 1024 * 1024, // 1 MB
            temp_dir: None,
        }
    }

    /// Set a custom temp directory.
    pub fn with_temp_dir(mut self, dir: std::path::PathBuf) -> Self {
        self.temp_dir = Some(dir);
        self
    }

    /// Execute code in the given language.
    async fn execute_code(
        &self,
        code: &str,
        language: &str,
        timeout: Option<u64>,
    ) -> ToolResult<ToolOutput> {
        let config = detect_language(language).ok_or_else(|| {
            ToolError::invalid_args(format!(
                "Unsupported language: '{}'. Supported: {}",
                language,
                LANGUAGES
                    .iter()
                    .map(|c| c.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        let timeout_secs = timeout.unwrap_or(self.timeout_secs);
        let start = Instant::now();

        if config.needs_file {
            // Write code to a temporary file.
            let temp_dir = self
                .temp_dir
                .clone()
                .unwrap_or_else(|| std::env::temp_dir());
            let file_name = format!("exec_{}.{}", uuid::Uuid::new_v4(), config.extension);
            let file_path = temp_dir.join(&file_name);

            tokio::fs::write(&file_path, code).await.map_err(|e| {
                ToolError::new("IO_ERROR", format!("Failed to write temp file: {}", e))
            })?;

            // Build the command.
            let mut cmd = Command::new(config.command);
            cmd.args(config.args);
            cmd.arg(&file_path);
            cmd.kill_on_drop(true);

            let result =
                tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

            // Clean up the temp file.
            tokio::fs::remove_file(&file_path).await.ok();

            let duration_ms = start.elapsed().as_millis() as u64;

            match result {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);

                    // Truncate output if too large.
                    let content = if output.status.success() {
                        truncate_output(&stdout, self.max_output_size)
                    } else {
                        let error_msg = format!(
                            "Exit code {}:\n{}",
                            exit_code,
                            truncate_output(&stderr, self.max_output_size)
                        );
                        error_msg
                    };

                    let data = serde_json::json!({
                        "language": language,
                        "exit_code": exit_code,
                        "duration_ms": duration_ms,
                        "stdout_length": stdout.len(),
                        "stderr_length": stderr.len(),
                    });

                    Ok(ToolOutput::success(content)
                        .with_data(data)
                        .with_mime_type("text/plain"))
                }
                Ok(Err(e)) => Err(ToolError::new(
                    "IO_ERROR",
                    format!("Failed to execute process: {}", e),
                )),
                Err(_) => Err(ToolError::timeout(timeout_secs)),
            }
        } else {
            // Inline execution (e.g., for eval-style languages).
            let mut cmd = Command::new(config.command);
            cmd.args(config.args);
            cmd.arg("-e");
            cmd.arg(code);
            cmd.kill_on_drop(true);

            let result =
                tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

            let duration_ms = start.elapsed().as_millis() as u64;

            match result {
                Ok(Ok(output)) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);

                    let content = if output.status.success() {
                        truncate_output(&stdout, self.max_output_size)
                    } else {
                        format!(
                            "Exit code {}:\n{}",
                            exit_code,
                            truncate_output(&stderr, self.max_output_size)
                        )
                    };

                    let data = serde_json::json!({
                        "language": language,
                        "exit_code": exit_code,
                        "duration_ms": duration_ms,
                    });

                    Ok(ToolOutput::success(content).with_data(data))
                }
                Ok(Err(e)) => Err(ToolError::new(
                    "IO_ERROR",
                    format!("Failed to execute code: {}", e),
                )),
                Err(_) => Err(ToolError::timeout(timeout_secs)),
            }
        }
    }
}

impl Default for CodeExecTool {
    fn default() -> Self {
        Self::new(30)
    }
}

/// Truncate output to a maximum size.
fn truncate_output(output: &str, max_size: u64) -> String {
    let max = max_size as usize;
    if output.len() > max {
        let mut truncated = output[..max].to_string();
        truncated.push_str(&format!(
            "\n... (output truncated, {} bytes shown of {})",
            max,
            output.len()
        ));
        truncated
    } else {
        output.to_string()
    }
}

#[async_trait]
impl Tool for CodeExecTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "execute_code",
                concat!(
                    "Execute code in a specified programming language. ",
                    "Supports Python, JavaScript/TypeScript, Ruby, Bash, Rust, Go, R, PHP, Perl, and SQLite. ",
                    "Code is executed in a subprocess with a timeout.",
),
                HashMap::from([
                    (
                        "code".to_string(),
                        ParameterDefinition::required_string("The code to execute"),
                    ),
                    (
                        "language".to_string(),
                        ParameterDefinition::required_string("The programming language")
                            .enum_values(vec![
                                "python".to_string(),
                                "javascript".to_string(),
                                "typescript".to_string(),
                                "ruby".to_string(),
                                "bash".to_string(),
                                "rust".to_string(),
                                "go".to_string(),
                                "r".to_string(),
                                "php".to_string(),
                                "perl".to_string(),
                                "sqlite".to_string(),
                            ]),
                    ),
                    (
                        "timeout".to_string(),
                        ParameterDefinition::integer("Timeout in seconds")
                            .default(serde_json::json!(30)),
                    ),
                ]),
            )
            .category("code")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let code = params["code"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'code' parameter"))?;

        let language = params["language"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'language' parameter"))?;

        let timeout = params["timeout"].as_i64().map(|t| t as u64);

        if code.len() > 100_000 {
            return Err(ToolError::new(
                "CODE_TOO_LONG",
                format!("Code too long: {} characters (max 100000)", code.len()),
            ));
        }

        self.execute_code(code, language, timeout).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detect_language() {
        assert!(detect_language("python").is_some());
        assert!(detect_language("javascript").is_some());
        assert!(detect_language("bash").is_some());
        assert!(detect_language("rust").is_some());
        assert!(detect_language("cobol").is_none());
    }

    #[tokio::test]
    async fn test_execute_python() {
        let tool = CodeExecTool::new(10);
        let result = tool
            .execute(serde_json::json!({
                "code": "print('hello from python')",
                "language": "python",
            }))
            .await;
        // Python may not be installed in the test environment.
        if let Ok(output) = result {
            assert!(output.content.contains("hello from python"));
        }
    }

    #[tokio::test]
    async fn test_unsupported_language() {
        let tool = CodeExecTool::new(10);
        let result = tool
            .execute(serde_json::json!({
                "code": "print('hello')",
                "language": "cobol",
            }))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }

    #[tokio::test]
    async fn test_missing_params() {
        let tool = CodeExecTool::new(10);
        let result = tool.execute(serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_truncation() {
        let short = "hello";
        assert_eq!(truncate_output(short, 100), "hello");

        let long = "a".repeat(1000);
        let truncated = truncate_output(&long, 10);
        assert!(truncated.len() < 100);
        assert!(truncated.contains("truncated"));
    }
}
