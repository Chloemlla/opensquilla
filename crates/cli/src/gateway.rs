//! Gateway lifecycle commands.
//!
//! Implements the `gateway` subcommand (Mode C). The gateway is the axum-based
//! in-process server built by [`opensquilla_gateway::Gateway`]. `start` runs it
//! in the foreground on the configured host:port, `status` probes the TCP port
//! (plus any PID file), and `stop`/`restart` manage the process via a small
//! runtime PID file stored in the data directory.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_gateway::Gateway;
use tokio::net::TcpStream;
use tracing::info;

use crate::util;

/// Path to the runtime PID file used by `gateway stop`.
fn pid_path() -> PathBuf {
    util::data_dir().join("gateway.pid")
}

/// Start the gateway in the foreground.
///
/// Acquires a PID file, then runs the axum server until the process receives a
/// shutdown signal. A stale PID file from a crashed run is cleared first.
pub async fn start_gateway() -> Result<()> {
    start_gateway_opts(false).await
}

/// Start the gateway, optionally in the background (detached).
pub async fn start_gateway_opts(detach: bool) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;
    let addr = format!("{host}:{port}");

    if port_open(&host, port).await {
        anyhow::bail!("Gateway is already running on {addr}");
    }

    if detach {
        // Relaunch the current executable with a detach marker.
        let exe = std::env::current_exe().context("Failed to locate current executable")?;
        let status = std::process::Command::new(&exe)
            .arg("gateway")
            .arg("start")
            .arg("--detach=false")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn detached gateway")?;
        println!("Gateway launching in background (pid {})...", status.id());
        println!("  Logs are written to the data directory.");
        return Ok(());
    }

    // Clear any stale PID file left by a crashed process.
    let path = pid_path();
    if path.exists() {
        std::fs::remove_file(&path).ok();
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| anyhow::anyhow!("Failed to create PID file at {}: {e}", path.display()))?;
    writeln!(file, "{}", std::process::id()).context("Failed to write PID file")?;
    drop(file);

    println!("Starting gateway on {addr} (Ctrl+C to stop)...");
    info!(addr = %addr, "Gateway starting");

    let gateway = Gateway::new(config.gateway.clone());
    let result = gateway
        .serve()
        .await
        .map_err(|e| anyhow::anyhow!("Gateway failed: {e}"));

    let _ = std::fs::remove_file(&path);
    result
}

/// Show recent gateway logs.
pub async fn show_logs(lines: usize, follow: bool) -> Result<()> {
    let log_path = util::data_dir().join("gateway.log");
    if !log_path.exists() {
        println!("No gateway log file found at {}", log_path.display());
        return Ok(());
    }

    let contents = std::fs::read_to_string(&log_path)
        .with_context(|| format!("Failed to read {}", log_path.display()))?;
    let all_lines: Vec<&str> = contents.lines().collect();
    let start = all_lines.len().saturating_sub(lines);
    for line in &all_lines[start..] {
        println!("{line}");
    }

    if follow {
        println!();
        println!("Following log (Ctrl+C to stop)...");
        // Simple follow: re-read every second for new content.
        let mut last_line_count = all_lines.len();
        loop {
            tokio::time::sleep(Duration::from_millis(1000)).await;
            if let Ok(contents) = std::fs::read_to_string(&log_path) {
                let new_lines: Vec<&str> = contents.lines().collect();
                if new_lines.len() > last_line_count {
                    for line in &new_lines[last_line_count..] {
                        println!("{line}");
                    }
                }
                last_line_count = new_lines.len();
            }
        }
    }
    Ok(())
}

/// Show gateway metrics.
pub async fn show_metrics() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;
    let addr = format!("{host}:{port}");

    if !port_open(&host, port).await {
        anyhow::bail!("Gateway is not running on {addr}");
    }

    // Query the metrics endpoint.
    let client = crate::rpc::RpcClient::from_config(&config);
    match client.gateway_metrics().await {
        Ok(metrics) => {
            println!("Gateway Metrics");
            println!("{:-<60}", "");
            if let Some(obj) = metrics.as_object() {
                for (k, v) in obj {
                    println!("  {:<24} {}", k, v);
                }
            } else {
                println!("  {metrics}");
            }
        }
        Err(_) => {
            // Fall back to local info.
            println!("Gateway is running on {addr}");
            if let Some(pid) = read_pid(&pid_path()) {
                println!("PID: {pid}");
            }
        }
    }
    Ok(())
}

/// Show gateway configuration info.
pub async fn show_info() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let g = &config.gateway;

    println!("Gateway Configuration");
    println!("{:-<60}", "");
    crate::table::KeyValue::new()
        .entry("Host", g.host.clone())
        .entry("Port", g.port.to_string())
        .entry("Max connections", g.max_connections.to_string())
        .entry("Request timeout (secs)", g.request_timeout_secs.to_string())
        .entry("CORS origins", g.cors_origins.join(", "))
        .print();
    println!("{:-<60}", "");
    Ok(())
}

/// Stop a running gateway by terminating the recorded process.
pub async fn stop_gateway() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;

    if !port_open(&host, port).await {
        println!("Gateway is not running.");
        let _ = std::fs::remove_file(pid_path());
        return Ok(());
    }

    let path = pid_path();
    let pid = read_pid(&path).ok_or_else(|| {
        anyhow::anyhow!(
            "Gateway is running but no PID file found at {}",
            path.display()
        )
    })?;

    println!("Stopping gateway (pid {pid})...");
    kill_process(pid)?;

    // Wait for the port to close, up to ~5 seconds.
    for _ in 0..50 {
        if !port_open(&host, port).await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let _ = std::fs::remove_file(&path);
    println!("Gateway stopped.");
    Ok(())
}

/// Report whether the gateway is currently listening on its port.
pub async fn gateway_status() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;
    let addr = format!("{host}:{port}");

    println!("Gateway:   {addr}");
    if port_open(&host, port).await {
        println!("Status:    running");
        if let Some(pid) = read_pid(&pid_path()) {
            println!("PID:       {pid}");
        }
    } else {
        println!("Status:    not running");
        if pid_path().exists() {
            println!(
                "Note:      stale PID file present at {}",
                pid_path().display()
            );
        }
    }
    Ok(())
}

/// Stop the gateway if running, then start it again.
pub async fn restart_gateway() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;

    if port_open(&host, port).await {
        stop_gateway().await?;
    } else {
        println!("Gateway was not running.");
        let _ = std::fs::remove_file(pid_path());
    }

    println!();
    start_gateway().await
}

/// Probe whether a TCP connection can be established to the given address.
async fn port_open(host: &str, port: u16) -> bool {
    TcpStream::connect((host, port)).await.is_ok()
}

/// Read the recorded PID from a runtime PID file.
fn read_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Terminate a process by PID using platform-native tooling.
fn kill_process(pid: u32) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("taskkill")
            .args(["/F", "/PID", &pid.to_string()])
            .status()
            .context("Failed to run taskkill")?;
        if !status.success() {
            anyhow::bail!("taskkill failed to terminate pid {pid}");
        }
    }
    #[cfg(unix)]
    {
        let status = std::process::Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .context("Failed to run kill")?;
        if !status.success() {
            anyhow::bail!("kill failed to terminate pid {pid}");
        }
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        let _ = pid;
        anyhow::bail!("Process termination is not supported on this platform");
    }
    Ok(())
}
