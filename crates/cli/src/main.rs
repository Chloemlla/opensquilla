use clap::Parser;
use opensquilla_cli::commands::{
    AgentAction, ChannelAction, Command, ConfigAction, CostAction, DiagnosticsAction,
    EnsembleAction, GatewayAction, InitAction, McpServerAction, MigrateAction, ModelAction,
    MemoryAction, OnboardAction, ProviderAction, RecoveryAction, RouterAction, SandboxAction,
    SchedulerAction, SearchAction, SessionAction, SkillAction, StatusAction, ToolAction,
};
use opensquilla_cli::{
    agent, channels, chat, config, cost, diagnostics, doctor, ensemble, gateway, init, mcp_server,
    memory, migrate, models, onboard, providers, recovery, router, sandbox, scheduler, search,
    sessions, skills, status, tools, tui,
};
use opensquilla_core::config::Config;
use tracing::info;

fn main() {
    let cli = opensquilla_cli::Cli::parse();

    // Initialize tracing
    let log_level = if cli.verbose { "debug" } else { "info" };
    opensquilla_observability::logging::init_logger(log_level)
        .expect("Failed to initialize logger");

    // Honor --no-color by setting NO_COLOR.
    if cli.no_color {
        unsafe {
            std::env::set_var("NO_COLOR", "1");
        }
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");

    let config = if let Some(path) = &cli.config {
        Config::from_file(path).unwrap_or_else(|_| {
            eprintln!("Warning: Could not load config from {path}, using defaults");
            Config::default()
        })
    } else {
        Config::load().unwrap_or_else(|_| {
            eprintln!("Warning: No config found, using defaults");
            Config::default()
        })
    };

    let command = cli.command.unwrap_or(Command::Chat {
        session: None,
        provider: None,
        model: None,
        prompt: None,
        standalone: false,
        attach: Vec::new(),
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
            attach,
        } => {
            info!("Dispatching chat command (standalone={standalone})");
            chat::run_chat_with_attachments(
                config.clone(),
                session,
                provider,
                model,
                prompt,
                standalone,
                attach,
            )
            .await?;
        }
        Command::Agent { action } => match action {
            AgentAction::Run {
                goal,
                provider,
                model,
                max_rounds,
                session,
                system_prompt,
                stream,
                tools,
            } => {
                agent::agent_run(
                    goal,
                    provider,
                    model,
                    max_rounds,
                    session,
                    system_prompt,
                    stream,
                    tools,
                )
                .await?;
            }
            AgentAction::List => agent::agent_list().await?,
            AgentAction::Show { name } => agent::agent_show(name).await?,
            AgentAction::Create {
                name,
                system_prompt,
                provider,
                model,
                max_rounds,
            } => {
                agent::agent_create(name, system_prompt, provider, model, max_rounds).await?;
            }
            AgentAction::Delete { name } => agent::agent_delete(name).await?,
            AgentAction::Skill { skill, input } => agent::agent_skill(skill, input).await?,
        },
        Command::Config { action } => match action {
            ConfigAction::Get { key } => config::get_config(key).await?,
            ConfigAction::Set { key, value } => config::set_config(key, value).await?,
            ConfigAction::Unset { key } => config::unset_config(key).await?,
            ConfigAction::Edit => config::edit_config().await?,
            ConfigAction::List => config::list_config().await?,
            ConfigAction::Path => config::show_config_path().await?,
            ConfigAction::Import { path } => config::import_config(path).await?,
            ConfigAction::Export { path } => config::export_config(path).await?,
            ConfigAction::Validate => config::validate_config().await?,
            ConfigAction::Defaults => config::show_defaults().await?,
        },
        Command::Providers { action } => match action {
            ProviderAction::List => providers::list_providers().await?,
            ProviderAction::Status { name } => providers::show_provider_status(name).await?,
            ProviderAction::Test { name } => providers::test_provider(name).await?,
            ProviderAction::Add {
                name,
                provider_type,
                api_key,
                base_url,
                model,
            } => {
                providers::add_provider(
                    name,
                    provider_type,
                    api_key,
                    base_url,
                    model,
                )
                .await?;
            }
            ProviderAction::Remove { name } => providers::remove_provider(name).await?,
            ProviderAction::Default { name } => providers::set_default_provider(name).await?,
        },
        Command::Session { action } => match action {
            SessionAction::List { status, limit } => {
                sessions::list_sessions_filtered(status, limit).await?
            }
            SessionAction::Show { id } => sessions::show_session(id).await?,
            SessionAction::Delete { id } => sessions::delete_session(id).await?,
            SessionAction::Archive { id } => sessions::archive_session(id).await?,
            SessionAction::Export { id, output } => sessions::export_session(id, output).await?,
            SessionAction::Create { name, mode } => sessions::create_session(name, mode).await?,
            SessionAction::Messages { id, limit } => {
                sessions::show_messages(id, limit).await?
            }
            SessionAction::Fork { id } => sessions::fork_session(id).await?,
            SessionAction::Kill { id } => sessions::kill_session(id).await?,
            SessionAction::Pause { id } => sessions::pause_session(id).await?,
            SessionAction::Resume { id } => sessions::resume_session(id).await?,
            SessionAction::Compact { id } => sessions::compact_session_cmd(id).await?,
            SessionAction::Search { query, limit } => {
                sessions::search_sessions(query, limit).await?
            }
        },
        Command::Models { action } => match action {
            ModelAction::List {
                provider,
                tools,
                vision,
            } => models::list_models_filtered(provider, tools, vision).await?,
            ModelAction::Show { name } => models::show_model(name).await?,
            ModelAction::Compare { models: items } => {
                models::compare_models(items).await?
            }
        },
        Command::Memory { action } => match action {
            MemoryAction::List {
                session,
                kind,
                limit,
            } => memory::list_memory_filtered(session, kind, limit).await?,
            MemoryAction::Show { id } => memory::show_memory(id).await?,
            MemoryAction::Delete { id } => memory::delete_memory(id).await?,
            MemoryAction::Clear => memory::clear_memory().await?,
            MemoryAction::Search { query, limit } => {
                memory::search_memory_limited(query, limit).await?
            }
            MemoryAction::Check => memory::memory_check().await?,
            MemoryAction::Dream => memory::memory_dream().await?,
            MemoryAction::Add {
                content,
                kind,
                importance,
                tag,
            } => {
                memory::add_memory(content, kind, importance, tag).await?;
            }
            MemoryAction::Export { output, kind } => {
                memory::export_memory(output, kind).await?
            }
        },
        Command::Skills { action } => match action {
            SkillAction::List => skills::list_skills().await?,
            SkillAction::Show { name } => skills::show_skill(name).await?,
            SkillAction::Install { source, no_scan } => {
                skills::install_skill(source, no_scan).await?
            }
            SkillAction::Uninstall { name } => skills::uninstall_skill(name).await?,
            SkillAction::Search { query, limit } => {
                skills::search_skills_limited(query, limit).await?
            }
            SkillAction::Enable { name } => skills::enable_skill(name).await?,
            SkillAction::Disable { name } => skills::disable_skill(name).await?,
            SkillAction::Update { name } => skills::update_skill(name).await?,
            SkillAction::Info { name } => skills::info_skill(name).await?,
            SkillAction::Run { name, input } => skills::run_skill(name, input).await?,
        },
        Command::Sandbox { action } => match action {
            SandboxAction::Test => sandbox::test_sandbox().await?,
            SandboxAction::Policy => sandbox::show_policy().await?,
            SandboxAction::Exec {
                command,
                workdir,
                envs,
            } => sandbox::exec_sandbox_opts(command, workdir, envs).await?,
            SandboxAction::Audit { limit } => sandbox::show_audit_log(limit).await?,
            SandboxAction::Validate => sandbox::validate_policy().await?,
        },
        Command::Channels { action } => match action {
            ChannelAction::List => channels::list_channels().await?,
            ChannelAction::Status { name } => channels::show_channel_status(name).await?,
            ChannelAction::Test { name } => channels::test_channel(name).await?,
            ChannelAction::Connect { kind } => channels::connect_channel(kind).await?,
            ChannelAction::Disconnect { id } => channels::disconnect_channel(id).await?,
            ChannelAction::Send { name, message } => {
                channels::send_message(name, message).await?
            }
            ChannelAction::Start => channels::start_all_channels().await?,
            ChannelAction::Stop => channels::stop_all_channels().await?,
        },
        Command::Scheduler { action } => match action {
            SchedulerAction::List => scheduler::list_tasks().await?,
            SchedulerAction::Show { id } => scheduler::show_task(id).await?,
            SchedulerAction::Add {
                name,
                schedule,
                handler,
                agent,
                payload,
            } => {
                scheduler::add_task_opts(name, schedule, handler, agent, payload).await?
            }
            SchedulerAction::Cancel { id } => scheduler::cancel_task(id).await?,
            SchedulerAction::Remove { id } => scheduler::remove_task(id).await?,
            SchedulerAction::Pause { id } => scheduler::pause_task(id).await?,
            SchedulerAction::Resume { id } => scheduler::resume_task(id).await?,
            SchedulerAction::History { id, limit } => {
                scheduler::show_history(id, limit).await?
            }
            SchedulerAction::Stats => scheduler::show_stats().await?,
        },
        Command::Doctor { subsystem, json } => match subsystem {
            Some(subsystem) => {
                info!("Running diagnostics for subsystem {subsystem}");
                if json {
                    doctor::run_doctor_check_json(subsystem).await?;
                } else {
                    doctor::run_doctor_check(subsystem).await?;
                }
            }
            None => {
                info!("Running diagnostics");
                if json {
                    doctor::run_doctor_json().await?;
                } else {
                    doctor::run_doctor().await?;
                }
            }
        },
        Command::Gateway { action } => match action {
            GatewayAction::Start { detach } => gateway::start_gateway_opts(detach).await?,
            GatewayAction::Stop => gateway::stop_gateway().await?,
            GatewayAction::Status => gateway::gateway_status().await?,
            GatewayAction::Restart => gateway::restart_gateway().await?,
            GatewayAction::Logs { lines, follow } => {
                gateway::show_logs(lines, follow).await?
            }
            GatewayAction::Metrics => gateway::show_metrics().await?,
            GatewayAction::Info => gateway::show_info().await?,
        },
        Command::Cost { action } => cost::run_cost(action).await?,
        Command::Onboard { action } => {
            let action = action.unwrap_or(OnboardAction::Run);
            onboard::run_onboard(action).await?;
        }
        Command::Router { action } => router::run_router(action).await?,
        Command::Init { action } => {
            let action = action.unwrap_or(InitAction::Create {
                directory: None,
                name: None,
                force: false,
            });
            init::run_init(action).await?;
        }
        Command::Status { action } => {
            let action = action.unwrap_or(StatusAction::Full);
            status::run_status(action).await?;
        }
        Command::Search { action } => search::run_search(action).await?,
        Command::Tools { action } => tools::run_tool(action).await?,
        Command::Diagnostics { action } => diagnostics::run_diagnostics(action).await?,
        Command::McpServer { action } => mcp_server::run_mcp_server(action).await?,
        Command::Migrate { action } => migrate::run_migrate(action).await?,
        Command::Recovery { action } => recovery::run_recovery(action).await?,
        Command::Ensemble { action } => ensemble::run_ensemble(action).await?,
        Command::Tui => {
            info!("Dispatching TUI command");
            tui::run_tui().await?;
        }
    }
    Ok(())
}
