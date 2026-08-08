use regex::Regex;
use serde::{Deserialize, Serialize};

/// Redacts sensitive information such as API keys and tokens from strings.
///
/// This is used to prevent secrets from leaking into logs, error messages,
/// or other output channels.
#[derive(Debug, Clone)]
pub struct SecretRedactor {
    patterns: Vec<SecretPattern>,
    redaction_string: String,
}

/// A pattern for detecting secrets in text.
#[derive(Debug, Clone)]
pub struct SecretPattern {
    name: String,
    pattern: Regex,
    group_to_redact: usize,
}

impl SecretRedactor {
    /// Create a new redactor with default patterns.
    pub fn new() -> Self {
        Self {
            patterns: Self::default_patterns(),
            redaction_string: "****".to_string(),
        }
    }

    /// Create a redactor with a custom redaction string.
    pub fn with_redaction_string(redaction: &str) -> Self {
        Self {
            patterns: Self::default_patterns(),
            redaction_string: redaction.to_string(),
        }
    }

    /// Create a redactor with custom patterns.
    pub fn with_patterns(patterns: Vec<SecretPattern>) -> Self {
        Self {
            patterns,
            redaction_string: "****".to_string(),
        }
    }

    /// Redact all known secret patterns from the input text.
    pub fn redact(&self, text: &str) -> String {
        let mut result = text.to_string();

        for pattern in &self.patterns {
            result = pattern
                .pattern
                .replace_all(&result, |caps: &regex::Captures| {
                    let mut redacted = String::new();
                    if let Some(full) = caps.get(0) {
                        // Try to get the capture group to redact; fall back to full match
                        if let Some(target) = caps.get(pattern.group_to_redact) {
                            let prefix = &full.as_str()[..target.start() - full.start()];
                            let suffix = &full.as_str()[target.end() - full.start()..];
                            redacted = format!("{}{}{}", prefix, self.redaction_string, suffix);
                        } else {
                            redacted = self.redaction_string.clone();
                        }
                    }
                    redacted
                })
                .to_string();
        }

        result
    }

    /// Check if the text contains any secrets.
    pub fn contains_secrets(&self, text: &str) -> bool {
        self.patterns.iter().any(|p| p.pattern.is_match(text))
    }

    /// Add a custom secret pattern.
    pub fn add_pattern(&mut self, pattern: SecretPattern) {
        self.patterns.push(pattern);
    }

    /// Return the default set of secret patterns.
    fn default_patterns() -> Vec<SecretPattern> {
        vec![
            // OpenAI API keys: sk-... (48 chars)
            SecretPattern {
                name: "openai_api_key".to_string(),
                pattern: Regex::new(r"(sk-[A-Za-z0-9]{20,})(?:[^A-Za-z0-9]|$)").unwrap(),
                group_to_redact: 1,
            },
            // Anthropic API keys: sk-ant-...
            SecretPattern {
                name: "anthropic_api_key".to_string(),
                pattern: Regex::new(r"(sk-ant-[A-Za-z0-9]{20,})(?:[^A-Za-z0-9]|$)").unwrap(),
                group_to_redact: 1,
            },
            // Generic bearer tokens
            SecretPattern {
                name: "bearer_token".to_string(),
                pattern: Regex::new(r"(?i)(Bearer\s+)([A-Za-z0-9\-._~+/]{20,})(?:[^A-Za-z0-9\-._~+/]|$)")
                    .unwrap(),
                group_to_redact: 2,
            },
            // Generic API key headers
            SecretPattern {
                name: "api_key_header".to_string(),
                pattern: Regex::new(r#"(?i)((?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*['"]?)([A-Za-z0-9\-._~+/]{16,})['"]?"#)
                    .unwrap(),
                group_to_redact: 2,
            },
            // AWS access keys
            SecretPattern {
                name: "aws_access_key".to_string(),
                pattern: Regex::new(r"(AKIA[0-9A-Z]{16})(?:[^0-9A-Z]|$)").unwrap(),
                group_to_redact: 1,
            },
            // GitHub tokens
            SecretPattern {
                name: "github_token".to_string(),
                pattern: Regex::new(r"(gh[pousr]_[A-Za-z0-9_]{24,})(?:[^A-Za-z0-9_]|$)").unwrap(),
                group_to_redact: 1,
            },
            // JWT tokens
            SecretPattern {
                name: "jwt_token".to_string(),
                pattern: Regex::new(r"(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
                    .unwrap(),
                group_to_redact: 1,
            },
            // Slack tokens
            SecretPattern {
                name: "slack_token".to_string(),
                pattern: Regex::new(r"(xox[baprs]-[A-Za-z0-9-]{10,})(?:[^A-Za-z0-9-]|$)")
                    .unwrap(),
                group_to_redact: 1,
            },
            // Discord tokens
            SecretPattern {
                name: "discord_token".to_string(),
                pattern: Regex::new(r"([A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{27,})")
                    .unwrap(),
                group_to_redact: 1,
            },
        ]
    }
}

impl Default for SecretRedactor {
    fn default() -> Self {
        Self::new()
    }
}

/// The kind of a detected secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecretType {
    ApiKey,
    Token,
    Password,
    Certificate,
    PrivateKey,
    Custom,
}

impl SecretType {
    /// The canonical string token for this secret type.
    pub fn as_str(&self) -> &'static str {
        match self {
            SecretType::ApiKey => "api_key",
            SecretType::Token => "token",
            SecretType::Password => "password",
            SecretType::Certificate => "certificate",
            SecretType::PrivateKey => "private_key",
            SecretType::Custom => "custom",
        }
    }

    /// Parse a wire token back into a type (`custom` for unknown values).
    pub fn parse(token: &str) -> Self {
        match token.trim().to_lowercase().as_str() {
            "api_key" | "apikey" => SecretType::ApiKey,
            "token" => SecretType::Token,
            "password" | "passwd" => SecretType::Password,
            "certificate" | "cert" => SecretType::Certificate,
            "private_key" | "privatekey" => SecretType::PrivateKey,
            _ => SecretType::Custom,
        }
    }
}

/// A single detected secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretMatch {
    /// The type of the detected secret.
    pub secret_type: SecretType,
    /// Byte offset of the start of the match.
    pub location: usize,
    /// A short surrounding-context snippet (redacted at its center).
    pub context: String,
    /// A heuristic confidence in `[0, 1]`.
    pub confidence: f64,
}

/// Detects and redacts secrets in text and configuration.
///
/// Unlike the lower-level [`SecretRedactor`], the sanitizer exposes the full
/// lifecycle: [`SecretSanitizer::detect_secrets`] enumerates matches,
/// [`SecretSanitizer::redact_secrets`] replaces them, and
/// [`SecretSanitizer::is_secret_key`] classifies config key names.
#[derive(Debug, Clone)]
pub struct SecretSanitizer {
    patterns: Vec<SecretPattern>,
    redaction_string: String,
}

impl SecretSanitizer {
    /// Create a new sanitizer with the default pattern set.
    pub fn new() -> Self {
        Self {
            patterns: Self::default_patterns(),
            redaction_string: "[REDACTED]".to_string(),
        }
    }

    /// Create a sanitizer with a custom redaction string.
    pub fn with_redaction_string(redaction: impl Into<String>) -> Self {
        Self {
            patterns: Self::default_patterns(),
            redaction_string: redaction.into(),
        }
    }

    /// The redaction string used by `redact_secrets`.
    pub fn redaction_string(&self) -> &str {
        &self.redaction_string
    }

    /// Find all potential secrets in the text.
    pub fn detect_secrets(&self, text: &str) -> Vec<SecretMatch> {
        let mut matches = Vec::new();
        for pattern in &self.patterns {
            for m in pattern.pattern.find_iter(text) {
                matches.push(SecretMatch {
                    secret_type: secret_type_for(&pattern.name),
                    location: m.start(),
                    context: context_snippet(text, m.start(), m.end()),
                    confidence: confidence_for(&pattern.name),
                });
            }
        }
        matches
    }

    /// Replace all detected secrets with the configured redaction string.
    pub fn redact_secrets(&self, text: &str) -> String {
        let mut result = text.to_string();
        for pattern in &self.patterns {
            result = pattern
                .pattern
                .replace_all(&result, self.redaction_string.as_str())
                .to_string();
        }
        result
    }

    /// Check whether a config key name looks like it holds a secret.
    ///
    /// This is intentionally conservative (name-based, not value-based) and
    /// powers `is_secret_key`-style validation before values are logged.
    pub fn is_secret_key(&self, key_name: &str) -> bool {
        let lower = key_name.trim().to_lowercase();
        SECRET_KEY_TOKENS.iter().any(|t| lower.contains(t))
    }

    /// Whether the text contains any secret.
    pub fn contains_secrets(&self, text: &str) -> bool {
        self.patterns.iter().any(|p| p.pattern.is_match(text))
    }

    /// The default secret pattern set.
    fn default_patterns() -> Vec<SecretPattern> {
        vec![
            SecretPattern {
                name: "private_key".to_string(),
                pattern: Regex::new(r"-----BEGIN [A-Z ]*PRIVATE KEY-----").unwrap(),
                group_to_redact: 0,
            },
            SecretPattern {
                name: "certificate".to_string(),
                pattern: Regex::new(r"-----BEGIN CERTIFICATE-----").unwrap(),
                group_to_redact: 0,
            },
            SecretPattern {
                name: "openai_api_key".to_string(),
                pattern: Regex::new(r"(sk-[A-Za-z0-9]{20,})(?:[^A-Za-z0-9]|$)").unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "anthropic_api_key".to_string(),
                pattern: Regex::new(r"(sk-ant-[A-Za-z0-9]{20,})(?:[^A-Za-z0-9]|$)").unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "aws_access_key".to_string(),
                pattern: Regex::new(r"(AKIA[0-9A-Z]{16})(?:[^0-9A-Z]|$)").unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "github_token".to_string(),
                pattern: Regex::new(r"(gh[pousr]_[A-Za-z0-9_]{24,})(?:[^A-Za-z0-9_]|$)").unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "slack_token".to_string(),
                pattern: Regex::new(r"(xox[baprs]-[A-Za-z0-9-]{10,})(?:[^A-Za-z0-9-]|$)")
                    .unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "jwt_token".to_string(),
                pattern: Regex::new(r"(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
                    .unwrap(),
                group_to_redact: 1,
            },
            SecretPattern {
                name: "bearer_token".to_string(),
                pattern: Regex::new(r"(?i)(Bearer\s+)([A-Za-z0-9\-._~+/]{20,})(?:[^A-Za-z0-9\-._~+/]|$)")
                    .unwrap(),
                group_to_redact: 2,
            },
            SecretPattern {
                name: "api_key_header".to_string(),
                pattern: Regex::new(r#"(?i)((?:api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token)\s*[:=]\s*['"]?)([A-Za-z0-9\-._~+/]{16,})['"]?"#)
                    .unwrap(),
                group_to_redact: 2,
            },
            SecretPattern {
                name: "password".to_string(),
                pattern: Regex::new(r#"(?i)(password|passwd|pwd)\s*[:=]\s*['"]?([^\s'";&]+)"#)
                    .unwrap(),
                group_to_redact: 2,
            },
        ]
    }
}

impl Default for SecretSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Config key-name tokens that indicate a secret-bearing key.
const SECRET_KEY_TOKENS: &[&str] = &[
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "private_key",
    "credential",
];

/// Map a pattern name to its [`SecretType`].
fn secret_type_for(pattern_name: &str) -> SecretType {
    match pattern_name {
        "private_key" => SecretType::PrivateKey,
        "certificate" => SecretType::Certificate,
        "password" => SecretType::Password,
        "bearer_token" | "jwt_token" | "slack_token" | "github_token" => SecretType::Token,
        _ => SecretType::ApiKey,
    }
}

/// A heuristic confidence for a matched pattern.
fn confidence_for(pattern_name: &str) -> f64 {
    match pattern_name {
        "openai_api_key" | "anthropic_api_key" | "aws_access_key" | "private_key"
        | "certificate" => 0.95,
        "jwt_token" | "slack_token" | "github_token" | "bearer_token" => 0.9,
        "api_key_header" => 0.7,
        "password" => 0.5,
        _ => 0.5,
    }
}

/// Build a short context snippet around `[start, end)` with a redacted center.
fn context_snippet(text: &str, start: usize, end: usize) -> String {
    let text = text.as_bytes();
    let snippet_start = start.saturating_sub(24);
    let snippet_end = (end + 24).min(text.len());
    let prefix = String::from_utf8_lossy(&text[snippet_start..start]);
    let suffix = String::from_utf8_lossy(&text[end..snippet_end]);
    format!("{prefix}[REDACTED]{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_openai_key() {
        let redactor = SecretRedactor::new();
        let input = "My API key is sk-abcdefghijklmnopqrstuvwxyz123456 and I use it.";
        let result = redactor.redact(input);
        assert!(!result.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
        assert!(result.contains("****"));
    }

    #[test]
    fn test_redact_bearer_token() {
        let redactor = SecretRedactor::new();
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doeRta";
        let result = redactor.redact(input);
        assert!(!result.contains("eyJhbGciOiJIUzI1NiJ9"));
    }

    #[test]
    fn test_clean_text() {
        let redactor = SecretRedactor::new();
        let input = "Hello, this is a normal message without any secrets.";
        let result = redactor.redact(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_contains_secrets() {
        let redactor = SecretRedactor::new();
        assert!(redactor.contains_secrets("sk-abcdefghijklmnopqrstuvwxyz123456"));
        assert!(!redactor.contains_secrets("Hello world"));
    }
}

#[cfg(test)]
mod sanitizer_tests {
    use super::*;

    #[test]
    fn test_detect_secrets_api_key() {
        let sanitizer = SecretSanitizer::new();
        let matches = sanitizer.detect_secrets("key=sk-abcdefghijklmnopqrstuvwxyz123456");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].secret_type, SecretType::ApiKey);
        assert!(matches[0].confidence >= 0.9);
    }

    #[test]
    fn test_detect_secrets_token() {
        let sanitizer = SecretSanitizer::new();
        let matches = sanitizer.detect_secrets("token=ghp_abcdefghijklmnopqrstuvwxyz123456789");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].secret_type, SecretType::Token);
    }

    #[test]
    fn test_detect_private_key() {
        let sanitizer = SecretSanitizer::new();
        let matches = sanitizer.detect_secrets("-----BEGIN RSA PRIVATE KEY-----\n...");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].secret_type, SecretType::PrivateKey);
    }

    #[test]
    fn test_redact_secrets() {
        let sanitizer = SecretSanitizer::new();
        let input = "My key is sk-abcdefghijklmnopqrstuvwxyz123456 and it is secret.";
        let redacted = sanitizer.redact_secrets(input);
        assert!(!redacted.contains("sk-abcdefghijklmnopqrstuvwxyz123456"));
        assert!(redacted.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_leaves_clean_text() {
        let sanitizer = SecretSanitizer::new();
        let input = "The weather today is sunny with a high of 25 degrees.";
        assert_eq!(sanitizer.redact_secrets(input), input);
    }

    #[test]
    fn test_is_secret_key() {
        let sanitizer = SecretSanitizer::new();
        assert!(sanitizer.is_secret_key("provider.openai.api_key"));
        assert!(sanitizer.is_secret_key("auth_token"));
        assert!(sanitizer.is_secret_key("DATABASE_PASSWORD"));
        assert!(!sanitizer.is_secret_key("model.default"));
        assert!(!sanitizer.is_secret_key("compaction.message_limit"));
    }

    #[test]
    fn test_false_positive_reduction() {
        let sanitizer = SecretSanitizer::new();
        // A short value is not a token; a normal sentence is not a secret.
        assert!(!sanitizer.contains_secrets("Just a normal sentence."));
        let matches = sanitizer.detect_secrets("name=alice");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_custom_redaction_string() {
        let sanitizer = SecretSanitizer::with_redaction_string("***");
        let redacted = sanitizer.redact_secrets("token=ghp_abcdefghijklmnopqrstuvwxyz123456789");
        assert!(redacted.contains("***"));
    }

    #[test]
    fn test_secret_type_round_trip() {
        assert_eq!(SecretType::parse("api_key"), SecretType::ApiKey);
        assert_eq!(SecretType::parse("private_key"), SecretType::PrivateKey);
        assert_eq!(SecretType::parse("nonsense"), SecretType::Custom);
        assert_eq!(SecretType::ApiKey.as_str(), "api_key");
    }
}
