use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SessionKey {
    pub prefix: String,
    pub session_id: Uuid,
    pub suffix: Option<String>,
}

impl SessionKey {
    pub fn new(session_id: Uuid) -> Self {
        Self {
            prefix: String::from("session"),
            session_id,
            suffix: None,
        }
    }

    pub fn for_transcript(session_id: Uuid) -> Self {
        Self {
            prefix: String::from("transcript"),
            session_id,
            suffix: None,
        }
    }

    pub fn for_summary(session_id: Uuid) -> Self {
        Self {
            prefix: String::from("summary"),
            session_id,
            suffix: None,
        }
    }

    pub fn for_plan(session_id: Uuid) -> Self {
        Self {
            prefix: String::from("plan"),
            session_id,
            suffix: None,
        }
    }

    pub fn for_usage(session_id: Uuid) -> Self {
        Self {
            prefix: String::from("usage"),
            session_id,
            suffix: None,
        }
    }

    pub fn with_suffix(mut self, suffix: String) -> Self {
        self.suffix = Some(suffix);
        self
    }

    pub fn to_string(&self) -> String {
        match &self.suffix {
            Some(s) => format!("{}:{}:{}", self.prefix, self.session_id, s),
            None => format!("{}:{}", self.prefix, self.session_id),
        }
    }

    pub fn from_string(key: &str) -> Option<Self> {
        let parts: Vec<&str> = key.splitn(3, ':').collect();
        match parts.len() {
            2 => {
                let session_id = Uuid::parse_str(parts[1]).ok()?;
                Some(Self {
                    prefix: parts[0].to_string(),
                    session_id,
                    suffix: None,
                })
            }
            3 => {
                let session_id = Uuid::parse_str(parts[1]).ok()?;
                Some(Self {
                    prefix: parts[0].to_string(),
                    session_id,
                    suffix: Some(parts[2].to_string()),
                })
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for SessionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}