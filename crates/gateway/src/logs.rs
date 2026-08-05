//! Logs RPC handlers.
//!
//! Provides `rpc_logs` for log file inspection: tailing recent lines,
//! filtering by level, and listing available log files.

use opensquilla_core::error::AppError;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::rpc::{RpcRegistry, rpc_handler};

/// A log entry parsed from a line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub line: String,
    pub level: Option<String>,
    pub line_number: u64,
}

/// Result of a log read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogReadResult {
    pub path: String,
    pub exists: bool,
    pub total_lines: u64,
    pub entries: Vec<LogEntry>,
    pub truncated: bool,
}

/// Resolve a log file path from params, falling back to a default location.
fn resolve_log_path(params: &serde_json::Value) -> Result<PathBuf, AppError> {
    if let Some(path) = params.get("path").and_then(|v| v.as_str()) {
        return Ok(PathBuf::from(path));
    }
    // Default: platform log directory
    if let Some(data_dir) = dirs::data_dir() {
        let candidate = data_dir
            .join("opensquilla")
            .join("logs")
            .join("opensquilla.log");
        return Ok(candidate);
    }
    Err(AppError::bad_request(
        "Missing 'path' parameter and no default log directory available",
    ))
}

/// Parse a log level from a line (looks for common level tokens).
fn parse_level(line: &str) -> Option<String> {
    for level in &["ERROR", "WARN", "WARNING", "INFO", "DEBUG", "TRACE"] {
        if line.contains(level) {
            return Some(level.to_string());
        }
    }
    None
}

/// Read the last N lines of a file efficiently.
fn tail_lines(path: &PathBuf, max_lines: usize) -> Result<(u64, Vec<String>), AppError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| AppError::internal(format!("Failed to read log file: {e}")))?;
    let lines: Vec<&str> = content.lines().collect();
    let total = lines.len() as u64;
    let start = lines.len().saturating_sub(max_lines);
    let tail: Vec<String> = lines[start..].iter().map(|s| s.to_string()).collect();
    Ok((total, tail))
}

/// Register logs RPC handlers on the given registry.
pub fn register_logs_handlers(registry: &mut RpcRegistry) {
    // logs.tail — tail the last N lines of a log file
    registry.register(rpc_handler("logs.tail", {
        move |params| {
            let path = resolve_log_path(params)?;
            let max_lines = params.get("lines").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
            let level_filter = params
                .get("level")
                .and_then(|v| v.as_str())
                .map(String::from);

            if !path.exists() {
                return Ok(serde_json::to_value(LogReadResult {
                    path: path.display().to_string(),
                    exists: false,
                    total_lines: 0,
                    entries: vec![],
                    truncated: false,
                })
                .map_err(|e| AppError::internal(e.to_string()))?);
            }

            let (total, lines) = tail_lines(&path, max_lines)?;
            let mut entries = Vec::new();
            let start_line = total.saturating_sub(lines.len() as u64);
            for (i, line) in lines.iter().enumerate() {
                let level = parse_level(line);
                if let Some(ref filter) = level_filter {
                    if level.as_deref() != Some(filter.as_str()) {
                        continue;
                    }
                }
                entries.push(LogEntry {
                    line: line.clone(),
                    level,
                    line_number: start_line + i as u64 + 1,
                });
            }

            let result = LogReadResult {
                path: path.display().to_string(),
                exists: true,
                total_lines: total,
                truncated: total > max_lines as u64,
                entries,
            };
            Ok(serde_json::to_value(result).map_err(|e| AppError::internal(e.to_string()))?)
        }
    }));

    // logs.head — read the first N lines of a log file
    registry.register(rpc_handler("logs.head", {
        move |params| {
            let path = resolve_log_path(params)?;
            let max_lines = params.get("lines").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

            if !path.exists() {
                return Ok(serde_json::to_value(LogReadResult {
                    path: path.display().to_string(),
                    exists: false,
                    total_lines: 0,
                    entries: vec![],
                    truncated: false,
                })
                .map_err(|e| AppError::internal(e.to_string()))?);
            }

            let content = std::fs::read_to_string(&path)
                .map_err(|e| AppError::internal(format!("Failed to read log file: {e}")))?;
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len() as u64;
            let head: Vec<String> = lines
                .iter()
                .take(max_lines)
                .map(|s| s.to_string())
                .collect();

            let entries: Vec<LogEntry> = head
                .iter()
                .enumerate()
                .map(|(i, line)| LogEntry {
                    line: line.clone(),
                    level: parse_level(line),
                    line_number: i as u64 + 1,
                })
                .collect();

            let result = LogReadResult {
                path: path.display().to_string(),
                exists: true,
                total_lines: total,
                truncated: total > max_lines as u64,
                entries,
            };
            Ok(serde_json::to_value(result).map_err(|e| AppError::internal(e.to_string()))?)
        }
    }));

    // logs.search — search for lines containing a substring
    registry.register(rpc_handler("logs.search", {
        move |params| {
            let path = resolve_log_path(params)?;
            let pattern = params
                .get("pattern")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::bad_request("Missing 'pattern' parameter"))?;
            let max_results = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

            if !path.exists() {
                return Err(AppError::not_found(format!(
                    "Log file '{}' does not exist",
                    path.display()
                )));
            }

            let content = std::fs::read_to_string(&path)
                .map_err(|e| AppError::internal(format!("Failed to read log file: {e}")))?;
            let mut entries = Vec::new();
            for (i, line) in content.lines().enumerate() {
                if line.contains(pattern) {
                    entries.push(LogEntry {
                        line: line.to_string(),
                        level: parse_level(line),
                        line_number: i as u64 + 1,
                    });
                    if entries.len() >= max_results {
                        break;
                    }
                }
            }

            let result = LogReadResult {
                path: path.display().to_string(),
                exists: true,
                total_lines: content.lines().count() as u64,
                truncated: entries.len() >= max_results,
                entries,
            };
            Ok(serde_json::to_value(result).map_err(|e| AppError::internal(e.to_string()))?)
        }
    }));

    // logs.list — list available log files in a directory
    registry.register(rpc_handler("logs.list", {
        move |params| {
            let dir_str = params
                .get("dir")
                .and_then(|v| v.as_str())
                .map(String::from)
                .or_else(|| {
                    dirs::data_dir().map(|d| {
                        d.join("opensquilla")
                            .join("logs")
                            .to_string_lossy()
                            .to_string()
                    })
                })
                .ok_or_else(|| {
                    AppError::bad_request("Missing 'dir' parameter and no default log directory")
                })?;
            let dir = PathBuf::from(&dir_str);

            if !dir.exists() {
                return Ok(serde_json::json!({
                    "dir": dir_str,
                    "exists": false,
                    "files": [],
                }));
            }

            let mut files: Vec<serde_json::Value> = Vec::new();
            let read_dir = std::fs::read_dir(&dir)
                .map_err(|e| AppError::internal(format!("Failed to read directory: {e}")))?;
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let modified = entry
                        .metadata()
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs());
                    files.push(serde_json::json!({
                        "name": name,
                        "size_bytes": size,
                        "modified_unix": modified,
                    }));
                }
            }
            files.sort_by(|a, b| {
                a["name"]
                    .as_str()
                    .unwrap_or("")
                    .cmp(b["name"].as_str().unwrap_or(""))
            });

            Ok(serde_json::json!({
                "dir": dir_str,
                "exists": true,
                "files": files,
                "count": files.len(),
            }))
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logs_tail_missing_file() {
        let mut registry = RpcRegistry::new();
        register_logs_handlers(&mut registry);

        let r = registry
            .dispatch(
                "logs.tail",
                serde_json::json!({"path": "/nonexistent/xyz.log", "lines": 10}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["exists"], false);
    }

    #[tokio::test]
    async fn test_logs_tail_real_file() {
        let mut registry = RpcRegistry::new();
        register_logs_handlers(&mut registry);

        let temp = std::env::temp_dir().join(format!("osq-test-{}.log", uuid::Uuid::new_v4()));
        std::fs::write(&temp, "INFO line1\nERROR line2\nDEBUG line3\n").unwrap();

        let r = registry
            .dispatch(
                "logs.tail",
                serde_json::json!({"path": temp.to_string_lossy(), "lines": 10}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        assert_eq!(resp["exists"], true);
        assert_eq!(resp["total_lines"], 3);
        let entries = resp["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 3);

        std::fs::remove_file(&temp).ok();
    }

    #[tokio::test]
    async fn test_logs_search() {
        let mut registry = RpcRegistry::new();
        register_logs_handlers(&mut registry);

        let temp = std::env::temp_dir().join(format!("osq-search-{}.log", uuid::Uuid::new_v4()));
        std::fs::write(&temp, "INFO hello\nERROR world\nINFO again\n").unwrap();

        let r = registry
            .dispatch(
                "logs.search",
                serde_json::json!({"path": temp.to_string_lossy(), "pattern": "world"}),
            )
            .await;
        let resp = r.unwrap().unwrap();
        let entries = resp["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 1);

        std::fs::remove_file(&temp).ok();
    }
}
