use chrono::{DateTime, Utc};
use cron::Schedule;
use std::str::FromStr;
use tracing::warn;

/// Error type for cron expression parsing.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Invalid cron expression '{0}': {1}")]
    InvalidCron(String, String),

    #[error("Unsupported cron expression format: {0}")]
    UnsupportedFormat(String),
}

/// Wrapper around the cron crate's Schedule for 5-field POSIX expressions.
///
/// Supports standard 5-field POSIX cron syntax:
///   minute (0-59) hour (0-23) day-of-month (1-31) month (1-12) day-of-week (0-6)
/// Also supports extensions like */N for step values and comma-separated lists.
#[derive(Debug, Clone)]
pub struct CronParser {
    expression: String,
    schedule: Schedule,
}

impl CronParser {
    /// Parse a 5-field POSIX cron expression.
    ///
    /// The cron crate expects 6 or 7 fields (seconds + minutes + ...).
    /// We prepend "0 " to make it a 6-field expression (seconds=0).
    pub fn new(expression: &str) -> Result<Self, ParseError> {
        let trimmed = expression.trim();
        let field_count = trimmed.split_whitespace().count();

        let normalized = if field_count == 5 {
            format!("0 {}", trimmed)
        } else if field_count == 6 || field_count == 7 {
            trimmed.to_string()
        } else {
            return Err(ParseError::InvalidCron(
                trimmed.to_string(),
                format!("expected 5 fields, got {}", field_count),
            ));
        };

        let schedule = Schedule::from_str(&normalized).map_err(|e| {
            ParseError::InvalidCron(trimmed.to_string(), e.to_string())
        })?;

        Ok(Self {
            expression: trimmed.to_string(),
            schedule,
        })
    }

    /// Compute the next scheduled time after the given timestamp.
    pub fn next_after(&self, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
        self.schedule.after(&after).next()
    }

    /// Compute the next scheduled time from now.
    pub fn next(&self) -> Option<DateTime<Utc>> {
        self.next_after(Utc::now())
    }

    /// Compute the next N scheduled times after the given timestamp.
    pub fn next_n_after(&self, after: DateTime<Utc>, n: usize) -> Vec<DateTime<Utc>> {
        self.schedule.after(&after).take(n).collect()
    }

    /// Get the raw expression string.
    pub fn expression(&self) -> &str {
        &self.expression
    }

    /// Validate a cron expression without creating a parser.
    pub fn validate(expression: &str) -> Result<(), ParseError> {
        let _ = CronParser::new(expression)?;
        Ok(())
    }
}

/// Validate a schedule kind's time expression.
pub fn validate_schedule_expression(kind: &super::types::ScheduleKind) -> Option<String> {
    match kind {
        super::types::ScheduleKind::Cron(expr) => {
            match CronParser::new(expr) {
                Ok(_) => None,
                Err(e) => Some(e.to_string()),
            }
        }
        super::types::ScheduleKind::At(dt) => {
            if *dt < Utc::now() {
                Some("Scheduled time is in the past".to_string())
            } else {
                None
            }
        }
        super::types::ScheduleKind::Every(secs) => {
            if *secs == 0 {
                Some("Interval must be greater than 0 seconds".to_string())
            } else {
                None
            }
        }
    }
}

/// Compute the next run time for a given schedule kind.
pub fn compute_next_run(kind: &super::types::ScheduleKind) -> Option<DateTime<Utc>> {
    match kind {
        super::types::ScheduleKind::Cron(expr) => {
            match CronParser::new(expr) {
                Ok(parser) => parser.next(),
                Err(e) => {
                    warn!("Failed to compute next run for cron '{}': {}", expr, e);
                    None
                }
            }
        }
        super::types::ScheduleKind::At(dt) => {
            if *dt > Utc::now() {
                Some(*dt)
            } else {
                None
            }
        }
        super::types::ScheduleKind::Every(secs) => {
            Some(Utc::now() + chrono::Duration::seconds(*secs as i64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_every_minute() {
        let parser = CronParser::new("* * * * *").unwrap();
        let next = parser.next();
        assert!(next.is_some());
    }

    #[test]
    fn test_parse_daily_at_midnight() {
        let parser = CronParser::new("0 0 * * *").unwrap();
        let next = parser.next();
        assert!(next.is_some());
    }

    #[test]
    fn test_parse_hourly() {
        let parser = CronParser::new("0 * * * *").unwrap();
        let next = parser.next();
        assert!(next.is_some());
    }

    #[test]
    fn test_parse_every_30_minutes() {
        let parser = CronParser::new("*/30 * * * *").unwrap();
        let next = parser.next();
        assert!(next.is_some());
    }

    #[test]
    fn test_parse_invalid_expression() {
        assert!(CronParser::new("invalid").is_err());
    }

    #[test]
    fn test_validate_good_cron() {
        assert!(validate_schedule_expression(&super::super::types::ScheduleKind::Cron(
            "0 9 * * 1-5".to_string()
        )).is_none());
    }

    #[test]
    fn test_validate_bad_cron() {
        assert!(validate_schedule_expression(&super::super::types::ScheduleKind::Cron(
            "bad cron".to_string()
        )).is_some());
    }
}
