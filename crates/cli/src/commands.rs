use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "opensquilla",
    version,
    about = "OpenSquilla - Open-source AI assistant platform"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Path to config file
    #[arg(short, long, global = true)]
    pub config: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start interactive chat session
    Chat {
        /// Session ID to resume
        #[arg(short, long)]
        session: Option<String>,

        /// Provider to use
        #[arg(short, long)]
        provider: Option<String>,

        /// Model to use
        #[arg(short, long)]
        model: Option<String>,

        /// Non-interactive mode: pass a single prompt
        prompt: Option<String>,

        /// Run with the engine directly instead of the gateway RPC path
        #[arg(long)]
        standalone: bool,
    },

    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// List and manage providers
    Providers {
        #[command(subcommand)]
        action: ProviderAction,
    },

    /// Manage sessions
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },

    /// Manage models
    Models {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// Manage memory
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Manage skills
    Skills {
        #[command(subcommand)]
        action: SkillAction,
    },

    /// Manage sandbox environments
    Sandbox {
        #[command(subcommand)]
        action: SandboxAction,
    },

    /// Manage channels (Discord, Slack, etc.)
    Channels {
        #[command(subcommand)]
        action: ChannelAction,
    },

    /// Manage scheduled tasks
    Scheduler {
        #[command(subcommand)]
        action: SchedulerAction,
    },

    /// Run diagnostics and health checks
    Doctor {
        /// Check a single subsystem (e.g. config, filesystem, network)
        #[arg(long)]
        subsystem: Option<String>,
    },

    /// Manage gateway lifecycle
    Gateway {
        #[command(subcommand)]
        action: GatewayAction,
    },

    /// Launch terminal UI
    Tui,
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Get a config value
    Get { key: String },
    /// Set a config value
    Set { key: String, value: String },
    /// Remove a config key
    Unset { key: String },
    /// Edit config in default editor
    Edit,
    /// Show all config
    List,
    /// Print the resolved config file path
    Path,
    /// Import config from a TOML, YAML, or JSON file
    Import { path: String },
    /// Export config to a TOML or JSON file
    Export { path: String },
}

#[derive(Subcommand, Debug)]
pub enum ProviderAction {
    /// List all configured providers
    List,
    /// Show provider status
    Status { name: Option<String> },
    /// Test a provider connection
    Test { name: Option<String> },
}

#[derive(Subcommand, Debug)]
pub enum SessionAction {
    /// List all sessions
    List,
    /// Show session details
    Show { id: String },
    /// Delete a session
    Delete { id: String },
    /// Archive a session
    Archive { id: String },
    /// Export session data
    Export { id: String, output: Option<String> },
    /// Create a new session
    Create { name: Option<String> },
    /// Show the full transcript of a session
    Messages { id: String },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
    /// List available models
    List { provider: Option<String> },
    /// Show model details
    Show { name: String },
}

#[derive(Subcommand, Debug)]
pub enum MemoryAction {
    /// List memory entries
    List {
        /// Optional session filter
        #[arg(short, long)]
        session: Option<String>,
    },
    /// Show memory entry
    Show { id: String },
    /// Delete memory entries
    Delete { id: String },
    /// Clear all memory
    Clear,
    /// Search memory entries
    Search { query: String },
    /// Run a memory consistency check
    Check,
    /// Trigger dream consolidation
    Dream,
}

#[derive(Subcommand, Debug)]
pub enum SkillAction {
    /// List installed skills
    List,
    /// Show skill details
    Show { name: String },
    /// Install a skill (path:, github:, or a hub identifier)
    Install { path: String },
    /// Uninstall a skill
    Uninstall { name: String },
    /// Search the skill hub
    Search { query: String },
}

#[derive(Subcommand, Debug)]
pub enum SandboxAction {
    /// Test the sandbox configuration
    Test,
    /// Show the effective sandbox policy
    Policy,
    /// Execute a command in the sandbox
    Exec { command: Vec<String> },
}

#[derive(Subcommand, Debug)]
pub enum ChannelAction {
    /// List configured channels
    List,
    /// Show channel status
    Status { name: Option<String> },
    /// Test a channel
    Test { name: String },
    /// Connect a channel
    Connect { kind: String },
    /// Disconnect a channel
    Disconnect { id: String },
}

#[derive(Subcommand, Debug)]
pub enum SchedulerAction {
    /// List scheduled tasks
    List,
    /// Show task details
    Show { id: String },
    /// Add a scheduled task (schedule: cron, every:N, at:RFC3339)
    Add {
        name: String,
        schedule: String,
        #[arg(long)]
        handler: Option<String>,
    },
    /// Cancel/remove a scheduled task
    Cancel { id: String },
    /// Remove a scheduled task (alias of cancel)
    Remove { id: String },
    /// Pause a scheduled task
    Pause { id: String },
    /// Resume a scheduled task
    Resume { id: String },
}

#[derive(Subcommand, Debug)]
pub enum GatewayAction {
    /// Start the gateway
    Start,
    /// Stop the gateway
    Stop,
    /// Show gateway status
    Status,
    /// Restart the gateway
    Restart,
}
