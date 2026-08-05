use serde::{Deserialize, Serialize};

/// Per-agent resource and capability limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLimits {
    /// Maximum tool calls per turn.
    pub max_tool_calls: u32,
    /// Maximum total output tokens per turn.
    pub max_output_tokens: u32,
    /// Maximum context window tokens.
    pub max_context_tokens: u32,
    /// Maximum subprocess runtime in seconds.
    pub max_subprocess_secs: u64,
    /// Whether file system writes are allowed.
    pub allow_fs_write: bool,
    /// Whether network access is allowed.
    pub allow_network: bool,
    /// Maximum memory usage in megabytes.
    pub max_memory_mb: u64,
}

impl Default for AgentLimits {
    fn default() -> Self {
        Self {
            max_tool_calls: 100,
            max_output_tokens: 8_192,
            max_context_tokens: 128_000,
            max_subprocess_secs: 300,
            allow_fs_write: true,
            allow_network: true,
            max_memory_mb: 2_048,
        }
    }
}

impl AgentLimits {
    /// A locked-down default for untrusted or sub-agents.
    pub fn restricted() -> Self {
        Self {
            max_tool_calls: 20,
            max_output_tokens: 2_048,
            max_context_tokens: 32_000,
            max_subprocess_secs: 60,
            allow_fs_write: false,
            allow_network: false,
            max_memory_mb: 512,
        }
    }
}
