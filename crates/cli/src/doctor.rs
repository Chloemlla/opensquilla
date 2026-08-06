//! Diagnostics commands.
//!
//! Implements the `doctor` subcommand against the recovery crate's
//! [`HealthCheck`]. `doctor` runs the full subsystem suite; `doctor check
//! <subsystem>` filters the full results to a single named subsystem.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_recovery::health::{HealthCheck, HealthIssue, HealthStatus, IssueSeverity};

/// Run the full set of health checks and print a report.
pub async fn run_doctor() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let health = HealthCheck::new(&config);

    println!("OpenSquilla Diagnostics");
    println!("{:=<60}", "");
    println!("Running full check...\n");

    let result = health.run_full_check().await;

    print_overall(&result.status, result.uptime_seconds, &result.timestamp);
    println!();

    if result.components.is_empty() {
        println!("No subsystem checks were reported.");
    } else {
        for component in &result.components {
            print_component(
                component.name.as_str(),
                &component.status,
                &component.description,
                component.latency_ms,
            );
        }
    }

    print_issues(&result.issues);
    Ok(())
}

/// Run the full check and print only the named subsystem.
pub async fn run_doctor_check(subsystem: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let health = HealthCheck::new(&config);
    let result = health.run_full_check().await;

    let needle = subsystem.to_lowercase();
    let matches: Vec<_> = result
        .components
        .iter()
        .filter(|c| c.name.to_lowercase().contains(&needle))
        .collect();

    if matches.is_empty() {
        let available: Vec<&str> = result.components.iter().map(|c| c.name.as_str()).collect();
        anyhow::bail!(
            "No subsystem matching '{subsystem}'. Available: {}",
            if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            }
        );
    }

    println!("Subsystem check: {subsystem}");
    println!("{:-<60}", "");
    for component in &matches {
        print_component(
            &component.name,
            &component.status,
            &component.description,
            component.latency_ms,
        );
        for (k, v) in &component.details {
            println!("      {k}: {v}");
        }
    }
    println!();

    let related: Vec<_> = result
        .issues
        .iter()
        .filter(|i| i.component.to_lowercase().contains(&needle))
        .collect();
    if related.is_empty() {
        println!("No issues reported for this subsystem.");
    } else {
        println!("Issues:");
        for issue in &related {
            print_issue(issue);
        }
    }
    Ok(())
}

fn print_overall(
    status: &HealthStatus,
    uptime_seconds: u64,
    timestamp: &chrono::DateTime<chrono::Utc>,
) {
    let icon = match status {
        HealthStatus::Healthy => "[OK]",
        HealthStatus::Degraded => "[!!]",
        HealthStatus::Unhealthy => "[FAIL]",
    };
    println!("Overall Status: {icon} {:?}", status);
    println!("Uptime:         {} seconds", uptime_seconds);
    println!("Checked at:     {}", timestamp.to_rfc3339());
}

fn print_component(name: &str, status: &HealthStatus, description: &str, latency_ms: u64) {
    let icon = match status {
        HealthStatus::Healthy => "[OK]",
        HealthStatus::Degraded => "[!!]",
        HealthStatus::Unhealthy => "[FAIL]",
    };
    println!("  {icon} {name}: {description} ({latency_ms} ms)");
}

fn print_issues(issues: &[HealthIssue]) {
    if issues.is_empty() {
        println!("\nNo issues found.");
        return;
    }
    println!("\nIssues ({}):", issues.len());
    for issue in issues {
        print_issue(issue);
    }
}

fn print_issue(issue: &HealthIssue) {
    let severity = match issue.severity {
        IssueSeverity::Critical => "CRITICAL",
        IssueSeverity::Warning => "WARNING",
        IssueSeverity::Info => "INFO",
    };
    println!("  - [{severity}] {}: {}", issue.component, issue.message);
    if let Some(suggestion) = &issue.suggestion {
        println!("      suggestion: {suggestion}");
    }
}

/// Run the full set of health checks and print a JSON report.
pub async fn run_doctor_json() -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let health = HealthCheck::new(&config);
    let result = health.run_full_check().await;

    let report = serde_json::json!({
        "status": format!("{:?}", result.status).to_lowercase(),
        "timestamp": result.timestamp.to_rfc3339(),
        "uptime_seconds": result.uptime_seconds,
        "components": result.components.iter().map(|c| serde_json::json!({
            "name": c.name,
            "status": format!("{:?}", c.status).to_lowercase(),
            "description": c.description,
            "latency_ms": c.latency_ms,
            "details": c.details,
        })).collect::<Vec<_>>(),
        "issues": result.issues.iter().map(|i| serde_json::json!({
            "component": i.component,
            "severity": format!("{:?}", i.severity).to_lowercase(),
            "message": i.message,
            "suggestion": i.suggestion,
        })).collect::<Vec<_>>(),
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).context("Failed to serialize report")?
    );
    Ok(())
}

/// Run the full check and print a JSON report filtered to a subsystem.
pub async fn run_doctor_check_json(subsystem: String) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let health = HealthCheck::new(&config);
    let result = health.run_full_check().await;

    let needle = subsystem.to_lowercase();
    let matches: Vec<_> = result
        .components
        .iter()
        .filter(|c| c.name.to_lowercase().contains(&needle))
        .collect();

    if matches.is_empty() {
        let available: Vec<&str> = result.components.iter().map(|c| c.name.as_str()).collect();
        anyhow::bail!(
            "No subsystem matching '{subsystem}'. Available: {}",
            if available.is_empty() {
                "none".to_string()
            } else {
                available.join(", ")
            }
        );
    }

    let report = serde_json::json!({
        "subsystem": subsystem,
        "components": matches.iter().map(|c| serde_json::json!({
            "name": c.name,
            "status": format!("{:?}", c.status).to_lowercase(),
            "description": c.description,
            "latency_ms": c.latency_ms,
            "details": c.details,
        })).collect::<Vec<_>>(),
        "issues": result.issues.iter()
            .filter(|i| i.component.to_lowercase().contains(&needle))
            .map(|i| serde_json::json!({
                "component": i.component,
                "severity": format!("{:?}", i.severity).to_lowercase(),
                "message": i.message,
                "suggestion": i.suggestion,
            })).collect::<Vec<_>>(),
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&report).context("Failed to serialize report")?
    );
    Ok(())
}
