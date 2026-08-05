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
    let config = Config::load().context("Failed to load configuration")?;
    let host = config.gateway.host.clone();
    let port = config.gateway.port;
    let addr = format!("{host}:{port}");

    if port_open(&host, port).await {
        anyhow::bail!("Gateway is already running on {addr}");
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
