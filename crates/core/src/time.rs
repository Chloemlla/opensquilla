use chrono::{DateTime, Utc};

pub fn now() -> DateTime<Utc> {
    Utc::now()
}

pub fn to_millis(dt: &DateTime<Utc>) -> i64 {
    dt.timestamp_millis()
}

pub fn from_millis(ms: i64) -> DateTime<Utc> {
    DateTime::from_timestamp_millis(ms).unwrap_or(Utc::now())
}