use clap::Parser;
use opensquilla_cli::commands::{
    ChannelAction, Command, ConfigAction, GatewayAction, MemoryAction, ModelAction, ProviderAction,
    SandboxAction, SchedulerAction, SessionAction, SkillAction,
};
use opensquilla_cli::{
    channels, chat, config, doctor, gateway, memory, models, providers, sandbox, scheduler,
    sessions, skills, tui,
};
use opensquilla_core::config::Config;
use tracing::info;

fn main() {
    let cli = opensquilla_cli::Cli::parse();

    // Initialize tracing
    let log_level = if cli.verbose { "debug" } else { "info" };
    opensquilla_observability::logging::init_logger(log_level)
        .expect("Failed to initialize logger");

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");

    let config = Config::load().unwrap_or_else(|_| {
        eprintln!("Warning: No config found, using defaults");
        Config::default()
    });

    let command = cli.command.unwrap_or(Command::Chat {
        session: None,
        provider: None,
        model: None,
        prompt: None,
        standalone: false,
    });

    rt.block_on(async move {
        if let Err(e) = dispatch(command, &config).await {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    });
}

async fn dispatch(command: Command, config: &Config) -> anyhow::Result<()> {
    match command {
        Command::Chat {
            session,
            provider,
            model,
            prompt,
            standalone,
        } => {
            info!("Dispatching chat command (standalone={standalone})");
            chat::run_chat(config.clone(), session, provider, model, prompt, standalone).await?;
        }
        Command::Config { action } => match action {
            ConfigAction::Get { key } => config::get_config(key).await?,
            ConfigAction::Set { key, value } => config::set_config(key, value).await?,
            ConfigAction::Unset { key } => config::unset_config(key).await?,
            ConfigAction::Edit => config::edit_config().await?,
            ConfigAction::List => config::list_config().await?,
            ConfigAction::Path => config::show_config_path().await?,
            ConfigAction::Import { path } => config::import_config(path).await?,
            ConfigAction::Export { path } => config::export_config(path).await?,
        },
        Command::Providers { action } => match action {
            ProviderAction::List => providers::list_providers().await?,
            ProviderAction::Status { name } => providers::show_provider_status(name).await?,
            ProviderAction::Test { name } => providers::test_provider(name).await?,
        },
        Command::Session { action } => match action {
            SessionAction::List => sessions::list_sessions().await?,
            SessionAction::Show { id } => sessions::show_session(id).await?,
            SessionAction::Delete { id } => sessions::delete_session(id).await?,
            SessionAction::Archive { id } => sessions::archive_session(id).await?,
            SessionAction::Export { id, output } => sessions::export_session(id, output).await?,
            SessionAction::Create { name } => sessions::create_session(name).await?,
            SessionAction::Messages { id } => sessions::show_messages(id).await?,
        },
        Command::Models { action } => match action {
            ModelAction::List { provider } => models::list_models(provider).await?,
            ModelAction::Show { name } => models::show_model(name).await?,
        },
        Command::Memory { action } => match action {
            MemoryAction::List { session } => memory::list_memory(session).await?,
            MemoryAction::Show { id } => memory::show_memory(id).await?,
            MemoryAction::Delete { id } => memory::delete_memory(id).await?,
            MemoryAction::Clear => memory::clear_memory().await?,
            MemoryAction::Search { query } => memory::search_memory(query).await?,
            MemoryAction::Check => memory::memory_check().await?,
            MemoryAction::Dream => memory::memory_dream().await?,
        },
        Command::Skills { action } => match action {
            SkillAction::List => skills::list_skills().await?,
            SkillAction::Show { name } => skills::show_skill(name).await?,
            SkillAction::Install { path } => skills::install_skill(path).await?,
            SkillAction::Uninstall { name } => skills::uninstall_skill(name).await?,
            SkillAction::Search { query } => skills::search_skills(query).await?,
        },
        Command::Sandbox { action } => match action {
            SandboxAction::Test => sandbox::test_sandbox().await?,
            SandboxAction::Policy => sandbox::show_policy().await?,
            SandboxAction::Exec { command } => sandbox::exec_sandbox(command).await?,
        },
        Command::Channels { action } => match action {
            ChannelAction::List => channels::list_channels().await?,
            ChannelAction::Status { name } => channels::show_channel_status(name).await?,
            ChannelAction::Test { name } => channels::test_channel(name).await?,
            ChannelAction::Connect { kind } => channels::connect_channel(kind).await?,
            ChannelAction::Disconnect { id } => channels::disconnect_channel(id).await?,
        },
        Command::Scheduler { action } => match action {
            SchedulerAction::List => scheduler::list_tasks().await?,
            SchedulerAction::Show { id } => scheduler::show_task(id).await?,
            SchedulerAction::Add {
                name,
                schedule,
                handler,
            } => scheduler::add_task(name, schedule, handler).await?,
            SchedulerAction::Cancel { id } => scheduler::cancel_task(id).await?,
            SchedulerAction::Remove { id } => scheduler::remove_task(id).await?,
            SchedulerAction::Pause { id } => scheduler::pause_task(id).await?,
            SchedulerAction::Resume { id } => scheduler::resume_task(id).await?,
        },
        Command::Doctor { subsystem } => match subsystem {
            Some(subsystem) => {
                info!("Running diagnostics for subsystem {subsystem}");
                doctor::run_doctor_check(subsystem).await?;
            }
            None => {
                info!("Running diagnostics");
                doctor::run_doctor().await?;
            }
        },
        Command::Gateway { action } => match action {
            GatewayAction::Start => gateway::start_gateway().await?,
            GatewayAction::Stop => gateway::stop_gateway().await?,
            GatewayAction::Status => gateway::gateway_status().await?,
            GatewayAction::Restart => gateway::restart_gateway().await?,
        },
        Command::Tui => {
            info!("Dispatching TUI command");
            tui::run_tui().await?;
        }
    }
    Ok(())
}
