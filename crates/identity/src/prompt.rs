use serde::{Deserialize, Serialize};

use crate::Identity;

/// A prompt template used to render the terminal prompt for an identity.
///
/// Placeholders (replaced at render time):
/// - `{name}` — identity name
/// - `{version}` — identity software version
/// - `{workspace}` — workspace path
/// - `{hostname}` — machine hostname
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Short name of this template.
    pub name: String,
    /// Template string with `{name}`, `{version}`, `{workspace}`, `{hostname}`
    /// placeholders.
    pub template: String,
}

impl PromptTemplate {
    /// Create a new prompt template.
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
        }
    }

    /// A terminal-style prompt (similar to a PS1 string).
    pub fn default_terminal() -> Self {
        Self {
            name: "terminal".into(),
            template: "{name}@\\h {workspace}$ ".into(),
        }
    }

    /// A minimal prompt.
    pub fn minimal() -> Self {
        Self {
            name: "minimal".into(),
            template: "{name}> ".into(),
        }
    }

    /// Render this template for the given identity.
    pub fn render(&self, identity: &Identity) -> String {
        let mut out = self.template.clone();
        out = out.replace("{name}", &identity.name);
        out = out.replace("{version}", &identity.version);
        out = out.replace(
            "{workspace}",
            &identity.workspace_path.display().to_string(),
        );
        out = out.replace(
            "{hostname}",
            identity.hostname.as_deref().unwrap_or("unknown"),
        );
        out
    }
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self::default_terminal()
    }
}
