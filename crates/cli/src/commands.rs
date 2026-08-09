use clap::{Parser, Subcommand};

pub use crate::agent::AgentAction;
pub use crate::bundle::BundleAction;
pub use crate::code_task::CodeTaskAction;
pub use crate::cost::CostAction;
pub use crate::diagnostics::DiagnosticsAction;
pub use crate::dist::DistAction;
pub use crate::ensemble::EnsembleAction;
pub use crate::init::InitAction;
pub use crate::mcp_server::McpServerAction;
pub use crate::migrate::MigrateAction;
pub use crate::onboard::OnboardAction;
pub use crate::recovery::RecoveryAction;
pub use crate::router::RouterAction;
pub use crate::search::SearchAction;
pub use crate::status::StatusAction;
pub use crate::tools::ToolAction;
pub use crate::uninstall::UninstallAction;

#[derive(Parser, Debug)]
#[command(
    name = "opensquilla",
    version,
    about = "OpenSquilla - Open-source AI assistant platform",
    long_about = "OpenSquilla is an open-source AI assistant platform with a Rust/Tauri v2 backend.\n\
                  This CLI provides access to chat, agents, sessions, providers, models, memory,\n\
                  skills, sandbox, channels, scheduling, routing, and cost management."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Enable verbose (debug) logging
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Path to config file
    #[arg(short, long, global = true)]
    pub config: Option<String>,

    /// Output format: plain, json, or table
    #[arg(long, global = true, default_value = "plain")]
    pub output: String,

    /// Disable color output
    #[arg(long, global = true)]
    pub no_color: bool,
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

        /// Attach a file to the message (repeatable)
        #[arg(short = 'f', long = "file")]
        attach: Vec<String>,
    },

    /// Run an autonomous agent task
    Agent {
        #[command(subcommand)]
        action: AgentAction,
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

        /// Output JSON instead of formatted text
        #[arg(long)]
        json: bool,
    },

    /// Manage gateway lifecycle
    Gateway {
        #[command(subcommand)]
        action: GatewayAction,
    },

    /// Show cost and usage analytics
    Cost {
        #[command(subcommand)]
        action: CostAction,
    },

    /// Run the onboarding wizard
    Onboard {
        #[command(subcommand)]
        action: Option<OnboardAction>,
    },

    /// Manage model routing and calibration
    Router {
        #[command(subcommand)]
        action: RouterAction,
    },

    /// Initialize a new OpenSquilla project
    Init {
        #[command(subcommand)]
        action: Option<InitAction>,
    },

    /// Show a system status overview
    Status {
        #[command(subcommand)]
        action: Option<StatusAction>,
    },

    /// Run a web search
    Search {
        #[command(subcommand)]
        action: SearchAction,
    },

    /// Inspect and test the built-in tool registry
    Tools {
        #[command(subcommand)]
        action: ToolAction,
    },

    /// Collect diagnostics and runtime info
    Diagnostics {
        #[command(subcommand)]
        action: DiagnosticsAction,
    },

    /// Run OpenSquilla as an MCP server
    McpServer {
        #[command(subcommand)]
        action: McpServerAction,
    },

    /// Migrate config/sessions from OpenClaw, Hermes, or old OpenSquilla layouts
    Migrate {
        #[command(subcommand)]
        action: MigrateAction,
    },

    /// Crash recovery commands
    Recovery {
        #[command(subcommand)]
        action: RecoveryAction,
    },

    /// Provider ensemble management
    Ensemble {
        #[command(subcommand)]
        action: EnsembleAction,
    },

    /// Launch terminal UI
    Tui,

    /// Solve real-repository coding tasks with an agent
    CodeTask {
        #[command(subcommand)]
        action: CodeTaskAction,
    },

    /// Collect a diagnostics bundle
    Bundle {
        #[command(subcommand)]
        action: BundleAction,
    },

    /// Emit workspace-state.json
    Dist {
        #[command(subcommand)]
        action: DistAction,
    },

    /// Uninstall OpenSquilla (default: keep user data)
    Uninstall {
        #[command(subcommand)]
        action: UninstallAction,
    },
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
    /// Validate the configuration file
    Validate,
    /// Show the default configuration
    Defaults,
}

#[derive(Subcommand, Debug)]
pub enum ProviderAction {
    /// List all configured providers
    List,
    /// Show provider status
    Status { name: Option<String> },
    /// Test a provider connection
    Test { name: Option<String> },
    /// Add a new provider interactively
    Add {
        name: String,
        provider_type: String,
        #[arg(long)]
        api_key: Option<String>,
        #[arg(long)]
        base_url: Option<String>,
        #[arg(long)]
        model: Option<String>,
    },
    /// Remove a provider
    Remove { name: String },
    /// Set the default provider
    Default { name: String },
}

#[derive(Subcommand, Debug)]
pub enum SessionAction {
    /// List all sessions
    List {
        /// Filter by status
        #[arg(long)]
        status: Option<String>,
        /// Limit number of results
        #[arg(long, default_value = "50")]
        limit: u64,
    },
    /// Show session details
    Show { id: String },
    /// Delete a session
    Delete { id: String },
    /// Archive a session
    Archive { id: String },
    /// Export session data
    Export { id: String, output: Option<String> },
    /// Create a new session
    Create {
        name: Option<String>,
        /// Session mode: chat, plan, agent, batch
        #[arg(long)]
        mode: Option<String>,
    },
    /// Show the full transcript of a session
    Messages {
        id: String,
        /// Maximum number of messages to show
        #[arg(long)]
        limit: Option<u64>,
    },
    /// Fork a session into a new session
    Fork { id: String },
    /// Kill a session (force stop)
    Kill { id: String },
    /// Pause a session
    Pause { id: String },
    /// Resume a paused session
    Resume { id: String },
    /// Compact a session's context window
    Compact { id: String },
    /// Reset (clear) a session's transcript
    Reset { key: String },
    /// Search sessions by name or content
    Search {
        query: String,
        #[arg(long, default_value = "20")]
        limit: u64,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
    /// List available models
    List {
        provider: Option<String>,
        /// Show only models that support tools
        #[arg(long)]
        tools: bool,
        /// Show only models that support vision
        #[arg(long)]
        vision: bool,
    },
    /// Show model details
    Show { name: String },
    /// Compare two or more models side by side
    Compare { models: Vec<String> },
}

#[derive(Subcommand, Debug)]
pub enum MemoryAction {
    /// List memory entries
    List {
        /// Optional session filter
        #[arg(short, long)]
        session: Option<String>,
        /// Filter by memory type
        #[arg(long)]
        kind: Option<String>,
        /// Limit number of results
        #[arg(long, default_value = "100")]
        limit: u64,
    },
    /// Show memory entry
    Show { id: String },
    /// Delete memory entries
    Delete { id: String },
    /// Clear all memory
    Clear,
    /// Search memory entries
    Search {
        query: String,
        #[arg(long, default_value = "20")]
        limit: u64,
    },
    /// Run a memory consistency check
    Check,
    /// Trigger dream consolidation
    Dream,
    /// Add a memory entry manually
    Add {
        content: String,
        #[arg(long)]
        kind: Option<String>,
        #[arg(long)]
        importance: Option<f64>,
        #[arg(long)]
        tag: Vec<String>,
    },
    /// Export memory entries to JSON
    Export {
        output: String,
        #[arg(long)]
        kind: Option<String>,
    },
    /// Flush a session transcript into durable memory
    FlushSession {
        /// Session key to flush
        #[arg(long)]
        key: String,
        /// Write a flush receipt JSON to this path
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub enum SkillAction {
    /// List installed skills
    List,
    /// Show skill details
    Show { name: String },
    /// Install a skill (path:, github:, or a hub identifier)
    Install {
        source: String,
        /// Skip security scan
        #[arg(long)]
        no_scan: bool,
    },
    /// Uninstall a skill
    Uninstall { name: String },
    /// Search the skill hub
    Search {
        query: String,
        #[arg(long)]
        limit: Option<u64>,
    },
    /// Enable a skill
    Enable { name: String },
    /// Disable a skill
    Disable { name: String },
    /// Update an installed skill
    Update { name: String },
    /// Show skill metadata and trust level
    Info { name: String },
    /// Run a skill
    Run { name: String, input: Option<String> },
}

#[derive(Subcommand, Debug)]
pub enum SandboxAction {
    /// Test the sandbox configuration
    Test,
    /// Show the effective sandbox policy
    Policy,
    /// Execute a command in the sandbox
    Exec {
        command: Vec<String>,
        /// Working directory
        #[arg(long)]
        workdir: Option<String>,
        /// Set environment variable (KEY=VALUE)
        #[arg(long = "env")]
        envs: Vec<String>,
    },
    /// Show sandbox audit log
    Audit {
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Validate the sandbox policy
    Validate,
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
    /// Send a test message to a channel
    Send { name: String, message: String },
    /// Start all enabled channels
    Start,
    /// Stop all channels
    Stop,
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
        /// Agent to associate with the task
        #[arg(long)]
        agent: Option<String>,
        /// JSON payload for the task
        #[arg(long)]
        payload: Option<String>,
    },
    /// Cancel/remove a scheduled task
    Cancel { id: String },
    /// Remove a scheduled task (alias of cancel)
    Remove { id: String },
    /// Pause a scheduled task
    Pause { id: String },
    /// Resume a scheduled task
    Resume { id: String },
    /// Show execution history for a task
    History {
        id: String,
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show scheduler statistics
    Stats,
}

#[derive(Subcommand, Debug)]
pub enum GatewayAction {
    /// Start the gateway
    Start {
        /// Run in background (detach)
        #[arg(long)]
        detach: bool,
    },
    /// Stop the gateway
    Stop,
    /// Show gateway status
    Status,
    /// Restart the gateway
    Restart,
    /// Show gateway logs
    Logs {
        /// Number of lines to show
        #[arg(long, default_value = "50")]
        lines: usize,
        /// Follow log output
        #[arg(long)]
        follow: bool,
    },
    /// Show gateway metrics
    Metrics,
    /// Show gateway configuration
    Info,
}
