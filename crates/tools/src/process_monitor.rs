//! System process monitoring tool: a cross-platform `ps` equivalent.
//!
//! Provides a tool for listing system processes, inspecting per-PID resource
//! usage, and finding processes by name. Uses platform-specific APIs via
//! `tokio::process::Command` to parse `ps` (Unix) or `tasklist`/`wmic`
//! (Windows) output, avoiding the need for heavy system-binding crates.
//!
//! This is the Rust counterpart of the Python `process_monitor.py` module.

use crate::registry::{ParameterDefinition, Tool, ToolDefinition, ToolError, ToolOutput, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

/// Information about a single system process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// The process ID.
    pub pid: u32,
    /// The parent process ID (0 if unknown).
    pub ppid: u32,
    /// The process name / command.
    pub name: String,
    /// CPU usage percentage (0.0 if unknown).
    pub cpu_percent: f64,
    /// Memory usage in bytes (RSS).
    pub memory_bytes: u64,
    /// Memory usage percentage (0.0 if unknown).
    pub memory_percent: f64,
    /// The process status (e.g., "running", "sleeping").
    pub status: String,
    /// The user that owns the process (if available).
    pub user: Option<String>,
    /// The command line that started the process (if available).
    pub command_line: Option<String>,
    /// The process start time (epoch seconds, if available).
    pub start_time: Option<u64>,
}

/// Aggregated system resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    /// Total CPU cores (logical).
    pub cpu_cores: usize,
    /// Total system memory in bytes.
    pub total_memory: u64,
    /// Used system memory in bytes.
    pub used_memory: u64,
    /// Memory usage percentage.
    pub memory_percent: f64,
    /// Number of processes running.
    pub process_count: usize,
}

/// Tool for monitoring system processes.
pub struct ProcessMonitorTool {
    /// Whether this tool runs on Windows.
    is_windows: bool,
}

impl ProcessMonitorTool {
    /// Create a new process monitor tool.
    pub fn new() -> Self {
        Self {
            is_windows: cfg!(target_os = "windows"),
        }
    }

    /// List all processes on the system.
    async fn list_processes(&self) -> ToolResult<Vec<ProcessInfo>> {
        if self.is_windows {
            self.list_processes_windows().await
        } else {
            self.list_processes_unix().await
        }
    }

    /// List processes on Unix systems using `ps`.
    async fn list_processes_unix(&self) -> ToolResult<Vec<ProcessInfo>> {
        let output = tokio::process::Command::new("ps")
            .args([
                "-eo",
                "pid=,ppid=,pcpu=,rss=,pmem=,stat=,user=,etime=,comm=",
            ])
            .output()
            .await
            .map_err(|e| {
                ToolError::new(
                    "PROCESS_ERROR",
                    format!("Failed to run ps: {}", e),
                )
            })?;

        if !output.status.success() {
            return Err(ToolError::new(
                "PROCESS_ERROR",
                format!(
                    "ps failed with exit code {}: {}",
                    output.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let mut processes = Vec::new();
        for line in text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 7 {
                continue;
            }
            let pid = parts[0].parse::<u32>().unwrap_or(0);
            let ppid = parts[1].parse::<u32>().unwrap_or(0);
            let cpu_percent = parts[2].parse::<f64>().unwrap_or(0.0);
            let rss_kb: u64 = parts[3].parse::<u64>().unwrap_or(0);
            let memory_bytes = rss_kb * 1024;
            let memory_percent = parts[4].parse::<f64>().unwrap_or(0.0);
            let status = parts[5].to_string();
            let user = parts.get(6).map(|s| s.to_string());
            let name = parts.get(8).map(|s| s.to_string()).unwrap_or_default();

            processes.push(ProcessInfo {
                pid,
                ppid,
                name,
                cpu_percent,
                memory_bytes,
                memory_percent,
                status,
                user,
                command_line: None,
                start_time: None,
            });
        }
        Ok(processes)
    }

    /// List processes on Windows using `tasklist`.
    async fn list_processes_windows(&self) -> ToolResult<Vec<ProcessInfo>> {
        let output = tokio::process::Command::new("tasklist")
            .args(["/FO", "CSV", "/NH"])
            .output()
            .await
            .map_err(|e| {
                ToolError::new(
                    "PROCESS_ERROR",
                    format!("Failed to run tasklist: {}", e),
                )
            })?;

        if !output.status.success() {
            return Err(ToolError::new(
                "PROCESS_ERROR",
                format!(
                    "tasklist failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }

        let text = String::from_utf8_lossy(&output.stdout);
        let mut processes = Vec::new();
        for line in text.lines() {
            // tasklist CSV format: "Name","PID","SessionName","Session#","MemUsage"
            let fields: Vec<String> = parse_csv_line(line);
            if fields.len() < 5 {
                continue;
            }
            let name = fields[0].clone();
            let pid = fields[1].parse::<u32>().unwrap_or(0);
            let mem_str = fields[4].replace(',', "").replace('K', "").trim().to_string();
            let memory_bytes: u64 = mem_str.parse::<u64>().unwrap_or(0) * 1024;

            processes.push(ProcessInfo {
                pid,
                ppid: 0, // tasklist doesn't show PPID
                name,
                cpu_percent: 0.0, // tasklist doesn't show CPU%
                memory_bytes,
                memory_percent: 0.0,
                status: "running".to_string(),
                user: None,
                command_line: None,
                start_time: None,
            });
        }
        Ok(processes)
    }

    /// Get detailed info about a specific process.
    async fn get_process(&self, pid: u32) -> ToolResult<ProcessInfo> {
        let processes = self.list_processes().await?;
        processes
            .into_iter()
            .find(|p| p.pid == pid)
            .ok_or_else(|| ToolError::not_found(format!("Process {} not found", pid)))
    }

    /// Find processes by name (substring match, case-insensitive).
    async fn find_by_name(&self, name: &str) -> ToolResult<Vec<ProcessInfo>> {
        let processes = self.list_processes().await?;
        let lower = name.to_lowercase();
        Ok(processes
            .into_iter()
            .filter(|p| p.name.to_lowercase().contains(&lower))
            .collect())
    }

    /// Kill a process by PID.
    async fn kill_process(&self, pid: u32, force: bool) -> ToolResult<()> {
        let (program, args) = if self.is_windows {
            if force {
                ("taskkill", vec!["/F", "/PID", &pid.to_string()])
            } else {
                ("taskkill", vec!["/PID", &pid.to_string()])
            }
        } else if force {
            ("kill", vec!["-9", &pid.to_string()])
        } else {
            ("kill", vec![&pid.to_string()])
        };

        let output = tokio::process::Command::new(program)
            .args(&args)
            .output()
            .await
            .map_err(|e| {
                ToolError::new("PROCESS_ERROR", format!("Failed to run kill: {}", e))
            })?;

        if !output.status.success() {
            return Err(ToolError::new(
                "PROCESS_ERROR",
                format!(
                    "Failed to kill process {}: {}",
                    pid,
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }
        Ok(())
    }

    /// Get system resource summary.
    async fn system_resources(&self) -> ToolResult<SystemResources> {
        let processes = self.list_processes().await?;
        let process_count = processes.len();
        let used_memory: u64 = processes.iter().map(|p| p.memory_bytes).sum();

        let (cpu_cores, total_memory) = if self.is_windows {
            self.system_info_windows().await.unwrap_or((1, 0))
        } else {
            self.system_info_unix().await.unwrap_or((1, 0))
        };

        let memory_percent = if total_memory > 0 {
            (used_memory as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        Ok(SystemResources {
            cpu_cores,
            total_memory,
            used_memory,
            memory_percent,
            process_count,
        })
    }

    /// Get CPU cores and total memory on Unix.
    async fn system_info_unix(&self) -> ToolResult<(usize, u64)> {
        // CPU cores: nproc
        let nproc = tokio::process::Command::new("nproc")
            .output()
            .await
            .ok()
            .and_then(|o| {
                if o.status.success() {
                    String::from_utf8_lossy(&o.stdout)
                        .trim()
                        .parse::<usize>()
                        .ok()
                } else {
                    None
                }
            })
            .unwrap_or(1);

        // Total memory: read /proc/meminfo (Linux only).
        let total_memory = tokio::fs::read_to_string("/proc/meminfo")
            .await
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|n| n.parse::<u64>().ok())
                            .map(|kb| kb * 1024)
                    })
            })
            .unwrap_or(0);

        Ok((nproc, total_memory))
    }

    /// Get CPU cores and total memory on Windows via wmic.
    async fn system_info_windows(&self) -> ToolResult<(usize, u64)> {
        let cpu_cores = tokio::process::Command::new("wmic")
            .args(["cpu", "get", "NumberOfLogicalProcessors"])
            .output()
            .await
            .ok()
            .and_then(|o| {
                let text = String::from_utf8_lossy(&o.stdout);
                text.lines()
                    .nth(1)
                    .and_then(|l| l.trim().parse::<usize>().ok())
            })
            .unwrap_or(1);

        let total_memory = tokio::process::Command::new("wmic")
            .args(["ComputerSystem", "get", "TotalPhysicalMemory"])
            .output()
            .await
            .ok()
            .and_then(|o| {
                let text = String::from_utf8_lossy(&o.stdout);
                text.lines()
                    .nth(1)
                    .and_then(|l| l.trim().parse::<u64>().ok())
            })
            .unwrap_or(0);

        Ok((cpu_cores, total_memory))
    }

    /// Get the process tree (children of a given PID).
    async fn process_tree(&self, root_pid: u32) -> ToolResult<Vec<ProcessInfo>> {
        let processes = self.list_processes().await?;
        let mut tree = Vec::new();
        let mut queue = vec![root_pid];
        let mut visited = std::collections::HashSet::new();

        while let Some(pid) = queue.pop() {
            if !visited.insert(pid) {
                continue;
            }
            for p in &processes {
                if p.pid == pid {
                    tree.push(p.clone());
                }
                if p.ppid == pid && p.pid != root_pid {
                    queue.push(p.pid);
                }
            }
        }

        if tree.is_empty() {
            return Err(ToolError::not_found(format!(
                "Process {} not found",
                root_pid
            )));
        }
        Ok(tree)
    }

    /// Monitor a process for a duration, sampling CPU/memory.
    async fn monitor_process(
        &self,
        pid: u32,
        duration_secs: u64,
        interval_secs: u64,
    ) -> ToolResult<Vec<ProcessInfo>> {
        let mut samples = Vec::new();
        let start = Instant::now();
        let interval = interval_secs.max(1);
        let duration = duration_secs.max(interval);

        while start.elapsed().as_secs() < duration {
            if let Ok(info) = self.get_process(pid).await {
                samples.push(info);
            } else {
                // Process may have exited.
                break;
            }
            let remaining = duration.saturating_sub(start.elapsed().as_secs());
            let sleep = interval.min(remaining);
            if sleep == 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(sleep)).await;
        }

        if samples.is_empty() {
            return Err(ToolError::not_found(format!(
                "Process {} not found",
                pid
            )));
        }
        Ok(samples)
    }
}

impl Default for ProcessMonitorTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a simple CSV line (handling quoted fields with commas).
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        match ch {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => {
                fields.push(current.trim().trim_matches('"').to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    fields.push(current.trim().trim_matches('"').to_string());
    fields
}

#[async_trait]
impl Tool for ProcessMonitorTool {
    fn definition(&self) -> &ToolDefinition {
        static DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(|| {
            ToolDefinition::new(
                "process_monitor",
                "Monitor system processes: list all processes, get process details by PID, "
                    + "find processes by name, kill processes, view system resources, "
                    + "and get process trees. Cross-platform (ps on Unix, tasklist on Windows).",
                HashMap::from([
                    (
                        "operation".to_string(),
                        ParameterDefinition::required_string("The operation to perform")
                            .enum_values(vec![
                                "list".to_string(),
                                "get".to_string(),
                                "find".to_string(),
                                "kill".to_string(),
                                "resources".to_string(),
                                "tree".to_string(),
                                "monitor".to_string(),
                            ]),
                    ),
                    (
                        "pid".to_string(),
                        ParameterDefinition::integer("Process ID (for get, kill, tree, monitor)"),
                    ),
                    (
                        "name".to_string(),
                        ParameterDefinition::string("Process name to search for (for find)"),
                    ),
                    (
                        "force".to_string(),
                        ParameterDefinition::boolean("Force kill (for kill operation)")
                            .default(serde_json::json!(false)),
                    ),
                    (
                        "duration".to_string(),
                        ParameterDefinition::integer("Duration in seconds to monitor (for monitor)")
                            .default(serde_json::json!(10)),
                    ),
                    (
                        "interval".to_string(),
                        ParameterDefinition::integer("Sampling interval in seconds (for monitor)")
                            .default(serde_json::json!(1)),
                    ),
                ]),
            )
            .category("system")
            .risk_level(3)
            .with_confirmation()
        });
        &DEF
    }

    async fn execute(&self, params: Value) -> ToolResult {
        let operation = params["operation"]
            .as_str()
            .ok_or_else(|| ToolError::invalid_args("Missing 'operation' parameter"))?;

        match operation {
            "list" => {
                let processes = self.list_processes().await?;
                let count = processes.len();
                let content = processes
                    .iter()
                    .map(|p| {
                        format!(
                            "{:>7} {:>7} {:>6.1}% {:>12} {}",
                            p.pid, p.ppid, p.cpu_percent, p.memory_bytes, p.name
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let data = serde_json::json!({
                    "processes": processes,
                    "count": count,
                });

                Ok(ToolOutput::success(content).with_data(data))
            }
            "get" => {
                let pid = params["pid"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'pid' for get"))?
                    as u32;
                let process = self.get_process(pid).await?;
                let content = serde_json::to_string_pretty(&process).unwrap_or_default();
                Ok(ToolOutput::success(content).with_data(serde_json::to_value(&process).unwrap_or_default()))
            }
            "find" => {
                let name = params["name"]
                    .as_str()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'name' for find"))?;
                let processes = self.find_by_name(name).await?;
                let content = processes
                    .iter()
                    .map(|p| format!("{:>7} {:>12} {}", p.pid, p.memory_bytes, p.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                let data = serde_json::json!({
                    "processes": processes,
                    "count": processes.len(),
                    "query": name,
                });
                Ok(ToolOutput::success(content).with_data(data))
            }
            "kill" => {
                let pid = params["pid"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'pid' for kill"))?
                    as u32;
                let force = params["force"].as_bool().unwrap_or(false);
                self.kill_process(pid, force).await?;
                let data = serde_json::json!({
                    "pid": pid,
                    "force": force,
                    "killed": true,
                });
                Ok(ToolOutput::success(format!(
                    "Killed process {} ({})",
                    pid,
                    if force { "forced" } else { "graceful" }
                ))
                .with_data(data))
            }
            "resources" => {
                let resources = self.system_resources().await?;
                let content = format!(
                    "CPU cores: {}\nTotal memory: {} bytes\nUsed memory: {} bytes ({:.1}%)\nProcesses: {}",
                    resources.cpu_cores,
                    resources.total_memory,
                    resources.used_memory,
                    resources.memory_percent,
                    resources.process_count
                );
                let data = serde_json::to_value(&resources).unwrap_or_default();
                Ok(ToolOutput::success(content).with_data(data))
            }
            "tree" => {
                let pid = params["pid"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'pid' for tree"))?
                    as u32;
                let tree = self.process_tree(pid).await?;
                let content = tree
                    .iter()
                    .map(|p| format!("{:>7} (ppid: {:>7}) {}", p.pid, p.ppid, p.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                let data = serde_json::json!({
                    "root_pid": pid,
                    "processes": tree,
                    "count": tree.len(),
                });
                Ok(ToolOutput::success(content).with_data(data))
            }
            "monitor" => {
                let pid = params["pid"]
                    .as_i64()
                    .ok_or_else(|| ToolError::invalid_args("Missing 'pid' for monitor"))?
                    as u32;
                let duration = params["duration"].as_i64().unwrap_or(10) as u64;
                let interval = params["interval"].as_i64().unwrap_or(1) as u64;
                let samples = self.monitor_process(pid, duration, interval).await?;

                // Compute averages.
                let avg_cpu: f64 =
                    samples.iter().map(|p| p.cpu_percent).sum::<f64>() / samples.len() as f64;
                let avg_mem: u64 =
                    samples.iter().map(|p| p.memory_bytes).sum::<u64>() / samples.len() as u64;
                let max_mem: u64 = samples.iter().map(|p| p.memory_bytes).max().unwrap_or(0);

                let content = samples
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        format!(
                            "sample {}: pid={} cpu={:.1}% mem={}",
                            i, p.pid, p.cpu_percent, p.memory_bytes
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let data = serde_json::json!({
                    "pid": pid,
                    "duration_secs": duration,
                    "interval_secs": interval,
                    "samples": samples.len(),
                    "avg_cpu_percent": avg_cpu,
                    "avg_memory_bytes": avg_mem,
                    "max_memory_bytes": max_mem,
                    "data": samples,
                });

                Ok(ToolOutput::success(content).with_data(data))
            }
            other => Err(ToolError::invalid_args(format!("Unknown operation: {}", other))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_csv_line() {
        let line = r#""chrome.exe","1234","Console","1","25,600 K""#;
        let fields = parse_csv_line(line);
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0], "chrome.exe");
        assert_eq!(fields[1], "1234");
        assert_eq!(fields[4], "25,600 K");
    }

    #[test]
    fn test_parse_csv_simple() {
        let fields = parse_csv_line("a,b,c");
        assert_eq!(fields, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn test_process_monitor_list() {
        let tool = ProcessMonitorTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "list"}))
            .await;
        // This may fail in restricted environments; just ensure no panic.
        if let Ok(output) = result {
            assert!(output.data.is_some());
        }
    }

    #[tokio::test]
    async fn test_process_monitor_resources() {
        let tool = ProcessMonitorTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "resources"}))
            .await;
        if let Ok(output) = result {
            assert!(output.data.is_some());
        }
    }

    #[tokio::test]
    async fn test_process_monitor_get_invalid_pid() {
        let tool = ProcessMonitorTool::new();
        let result = tool
            .execute(serde_json::json!({
                "operation": "get",
                "pid": 99999999,
            }))
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_monitor_missing_pid() {
        let tool = ProcessMonitorTool::new();
        let result = tool
            .execute(serde_json::json!({"operation": "get"}))
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, "INVALID_ARGS");
    }
}
