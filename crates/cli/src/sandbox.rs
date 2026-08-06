//! Sandbox management commands.
//!
//! Implements the `sandbox` subcommand. Execution goes through the platform
//! sandbox backend selected at compile time; the policy command prints the
//! effective `SandboxPolicy`, and the test command verifies that a trivial
//! command can run inside the sandbox.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_sandbox::policy::{SandboxLevel, SandboxPolicy, SandboxResult};
use std::time::Instant;

/// Show the effective sandbox policy.
pub async fn show_policy() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let policy = effective_policy(&config);

    println!("Sandbox policy");
    println!("{:-<50}", "");
    println!("  Level:            {:?}", policy.level);
    println!("  Audit enabled:    {}", policy.audit_enabled);
    println!("  Env allowlist:    {}", policy.env_allowlist.join(", "));
    println!();
    println!("  Filesystem:");
    println!(
        "    read allowed:  {}",
        join_paths(&policy.filesystem.read_allowed)
    );
    println!(
        "    write allowed: {}",
        join_paths(&policy.filesystem.write_allowed)
    );
    println!(
        "    denied:        {}",
        join_paths(&policy.filesystem.denied)
    );
    println!("    tmp writable:  {}", policy.filesystem.tmp_writable);
    println!("    home readable: {}", policy.filesystem.home_readable);
    println!();
    println!("  Network:          {:?}", policy.network);
    println!();
    println!("  Resource limits:");
    println!(
        "    cpu (secs):     {}",
        opt(policy.resource_limits.cpu_time_secs)
    );
    println!(
        "    memory (bytes): {}",
        opt(policy.resource_limits.memory_bytes)
    );
    println!(
        "    max processes:  {}",
        opt(policy.resource_limits.max_processes)
    );
    println!(
        "    file (bytes):   {}",
        opt(policy.resource_limits.file_size_bytes)
    );
    println!("{:-<50}", "");
    Ok(())
}

/// Test that the sandbox can execute a trivial command.
pub async fn test_sandbox() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let policy = effective_policy(&config);
    let level = policy.level;

    println!("Testing sandbox (level {:?})...", level);
    let start = Instant::now();
    let result = run_in_sandbox("echo", &["sandbox-ok"], &policy)
        .await
        .map_err(|e| anyhow::anyhow!("Sandbox test failed: {e}"))?;
    let duration_ms = start.elapsed().as_millis();

    let stdout = result.stdout.trim();
    let exit = result.exit_code;

    if exit == 0 && stdout.contains("sandbox-ok") {
        println!("OK: sandbox executed 'echo sandbox-ok' in {duration_ms} ms");
        Ok(())
    } else {
        anyhow::bail!(
            "Sandbox test failed: exit={exit} stdout={stdout:?} stderr={:?}",
            result.stderr
        )
    }
}

/// Execute a command inside the sandbox and print its output.
pub async fn exec_sandbox(command: Vec<String>) -> Result<()> {
    exec_sandbox_opts(command, None, Vec::new()).await
}

/// Execute a command inside the sandbox with working directory and env vars.
pub async fn exec_sandbox_opts(
    command: Vec<String>,
    workdir: Option<String>,
    envs: Vec<String>,
) -> Result<()> {
    if command.is_empty() {
        anyhow::bail!("No command provided");
    }
    let config = Config::load().context("Failed to load configuration")?;
    let policy = effective_policy(&config);

    let program = command[0].clone();
    let args: Vec<&str> = command[1..].iter().map(String::as_str).collect();
    let env_pairs: Vec<(String, String)> = envs
        .iter()
        .filter_map(|e| e.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    println!("Executing in sandbox: {program} {}", args.join(" "));
    if let Some(ref wd) = workdir {
        println!("Working directory: {wd}");
    }
    if !env_pairs.is_empty() {
        println!("Environment: {} var(s)", env_pairs.len());
    }
    let result = run_in_sandbox_opts(&program, &args, &env_pairs, workdir.as_deref(), &policy)
        .await
        .map_err(|e| anyhow::anyhow!("Sandbox execution failed: {e}"))?;
    print_result(&result);
    Ok(())
}

/// Show the sandbox audit log.
pub async fn show_audit_log(limit: usize) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let policy = effective_policy(&config);

    if !policy.audit_enabled {
        println!("Audit logging is disabled in the current policy.");
        return Ok(());
    }

    // Run a no-op command to generate an audit entry, then display.
    println!("Audit log (last {limit} entries):");
    println!("{:-<60}", "");
    // In a full implementation, this would read from a persistent audit log.
    // For now, we show the policy's audit configuration.
    println!("  Audit enabled: {}", policy.audit_enabled);
    println!("  Level: {:?}", policy.level);
    println!();
    println!("  (Persistent audit log requires a running gateway.)");
    Ok(())
}

/// Validate the sandbox policy.
pub async fn validate_policy() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let policy = effective_policy(&config);

    match policy.validate() {
        Ok(()) => {
            println!("{} Sandbox policy is valid.", crate::table::ok());
            println!("  Level: {:?}", policy.level);
            println!("  Audit: {}", policy.audit_enabled);
            Ok(())
        }
        Err(e) => {
            println!("{} Sandbox policy is invalid: {e}", crate::table::fail());
            anyhow::bail!("Policy validation failed: {e}");
        }
    }
}

/// Run a command through the platform sandbox backend.
async fn run_in_sandbox(
    command: &str,
    args: &[&str],
    policy: &SandboxPolicy,
) -> std::result::Result<SandboxResult, String> {
    run_in_sandbox_opts(command, args, &[], None, policy).await
}

/// Run a command through the platform sandbox backend with env and workdir.
#[allow(clippy::too_many_arguments)]
async fn run_in_sandbox_opts(
    command: &str,
    args: &[&str],
    env: &[(String, String)],
    workdir: Option<&str>,
    policy: &SandboxPolicy,
) -> std::result::Result<SandboxResult, String> {
    let env_map: std::collections::HashMap<String, String> =
        env.iter().cloned().collect();
    #[cfg(target_os = "linux")]
    {
        let mut sb = opensquilla_sandbox::linux::LinuxSandbox::new();
        sb.execute_with_env(command, args, env_map, workdir, policy)
            .await
    }
    #[cfg(target_os = "macos")]
    {
        let mut sb = opensquilla_sandbox::macos::MacOsSandbox::new();
        sb.execute_with_env(command, args, env_map, workdir, policy)
            .await
    }
    #[cfg(target_os = "windows")]
    {
        let mut sb = opensquilla_sandbox::windows::WindowsSandbox::new();
        sb.execute_with_env(command, args, env_map, workdir, policy)
            .await
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        let mut sb = opensquilla_sandbox::noop::NoopSandbox::new();
        sb.execute_with_env(command, args, env_map, workdir, policy)
            .await
    }
}

/// Build the effective sandbox policy from configuration or defaults.
fn effective_policy(config: &Config) -> SandboxPolicy {
    let mut policy = SandboxPolicy::default();
    if let Some(sb) = config.sandbox.as_ref() {
        if !sb.enabled {
            // A disabled sandbox still uses the default policy at the least
            // restrictive level for development workflows.
            policy.level = SandboxLevel::Standard;
        }
    }
    policy
}

fn print_result(result: &SandboxResult) {
    println!("Exit code: {}", result.exit_code);
    println!("Duration:  {} ms", result.duration_ms);
    if !result.stdout.is_empty() {
        println!("--- stdout ---");
        print!("{}", result.stdout);
        if !result.stdout.ends_with('\n') {
            println!();
        }
    }
    if !result.stderr.is_empty() {
        println!("--- stderr ---");
        print!("{}", result.stderr);
        if !result.stderr.ends_with('\n') {
            println!();
        }
    }
    if !result.audit_log.is_empty() {
        println!("--- audit log ---");
        for entry in &result.audit_log {
            println!("  [{}] {}", entry.timestamp.to_rfc3339(), entry.action);
        }
    }
}

fn join_paths(paths: &[String]) -> String {
    if paths.is_empty() {
        "(none)".to_string()
    } else {
        paths.join(", ")
    }
}

fn opt(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "(unset)".to_string(),
    }
}
