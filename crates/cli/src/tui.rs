//! # OpenSquilla TUI
//!
//! Terminal UI entry point for the `osq-tui` binary. Renders a full ratatui
//! interface with multiple tabs:
//!
//! - **Chat** — interactive chat with streamed responses
//! - **Sessions** — session list and management
//! - **Providers** — provider status and models
//! - **Channels** — channel runtime status
//! - **Cost** — token usage and spending dashboard
//! - **Logs** — gateway and runtime log viewer
//!
//! This file is both a library module (`opensquilla_cli::tui`) and the crate
//! root for the `osq-tui` binary target.
//!
//! Key bindings:
//! - `Enter`       send the current input (chat)
//! - `Esc`         quit
//! - `Ctrl+C`      cancel an in-flight stream (or quit when idle)
//! - `Tab`         cycle views
//! - `1`..`6`      jump to a specific view
//! - `Ctrl+Q`      quit

use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use futures::StreamExt;
use opensquilla_core::config::Config;
use opensquilla_provider::ChatConfig;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Tabs, Wrap};
use ratatui::{Frame, Terminal};
use tokio::sync::mpsc;
use tracing::info;

use opensquilla_cli::util;

/// Main entry point for the `osq-tui` binary.
fn main() {
    if let Err(e) = opensquilla_observability::logging::init_logger("info") {
        eprintln!("Warning: failed to initialize logger: {e}");
    }

    let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime");
    rt.block_on(async {
        if let Err(e) = run_tui().await {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    });
}

/// Run the interactive terminal UI loop.
pub async fn run_tui() -> Result<()> {
    info!("Starting OpenSquilla TUI");

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let backend = CrosstermBackend::new(&mut stdout);
    let mut terminal = Terminal::new(backend)?;
    execute!(io::stdout(), EnterAlternateScreen)?;

    let res = TuiApp::new().run(&mut terminal).await;

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    res
}

/// A single message in the on-screen transcript.
#[derive(Debug, Clone)]
struct ChatLine {
    role: String,
    content: String,
}

/// Events produced by the background streaming task.
#[derive(Debug, Clone)]
enum TuiEvent {
    /// A text delta from the provider stream.
    Delta(String),
    /// The stream completed successfully.
    Done,
    /// The stream failed.
    Error(String),
}

/// The selectable views in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum View {
    Chat,
    Sessions,
    Providers,
    Channels,
    Cost,
    Logs,
}

impl View {
    const ALL: [View; 6] = [
        View::Chat,
        View::Sessions,
        View::Providers,
        View::Channels,
        View::Cost,
        View::Logs,
    ];

    fn title(&self) -> &'static str {
        match self {
            View::Chat => "Chat",
            View::Sessions => "Sessions",
            View::Providers => "Providers",
            View::Channels => "Channels",
            View::Cost => "Cost",
            View::Logs => "Logs",
        }
    }

    fn index(&self) -> usize {
        View::ALL.iter().position(|v| v == self).unwrap_or(0)
    }

    fn from_index(idx: usize) -> Self {
        View::ALL[idx % View::ALL.len()]
    }
}

/// A row of session data for the sessions view.
#[derive(Debug, Clone)]
struct SessionRow {
    id: String,
    name: String,
    mode: String,
    status: String,
    messages: u64,
    tokens: u64,
    cost: f64,
}

/// A row of provider data for the providers view.
#[derive(Debug, Clone)]
struct ProviderRow {
    name: String,
    backend: String,
    models: usize,
    default_model: String,
    status: &'static str,
}

/// A row of channel data for the channels view.
#[derive(Debug, Clone)]
struct ChannelRow {
    name: String,
    channel_type: String,
    enabled: bool,
    status: &'static str,
}

/// Application state for the TUI loop.
pub struct TuiApp {
    config: Config,
    view: View,
    provider_name: String,
    model_name: String,
    session_id: String,
    messages: Vec<ChatLine>,
    input: String,
    status_message: String,
    help_visible: bool,
    streaming: bool,
    should_quit: bool,
    scroll: u16,
    tx: mpsc::Sender<TuiEvent>,
    rx: mpsc::Receiver<TuiEvent>,
    // Session list data.
    sessions: Vec<SessionRow>,
    session_scroll: u16,
    selected_session: usize,
    // Provider data.
    providers: Vec<ProviderRow>,
    // Channel data.
    channels: Vec<ChannelRow>,
    // Log viewer state.
    log_lines: Vec<String>,
    log_scroll: u16,
    log_auto_follow: bool,
    // Cost dashboard data.
    cost_total: f64,
    cost_sessions: u64,
    cost_tokens: u64,
    cost_avg_per_session: f64,
}

impl Default for TuiApp {
    fn default() -> Self {
        Self::new()
    }
}

impl TuiApp {
    /// Create a new TUI app. Session resolution happens when `run` starts.
    pub fn new() -> Self {
        let config = Config::load().unwrap_or_else(|_| {
            eprintln!("Warning: No config found, using defaults");
            Config::default()
        });
        let (tx, rx) = mpsc::channel(256);

        Self {
            config,
            view: View::Chat,
            provider_name: String::new(),
            model_name: String::new(),
            session_id: uuid::Uuid::new_v4().to_string(),
            messages: Vec::new(),
            input: String::new(),
            status_message: String::new(),
            help_visible: false,
            streaming: false,
            should_quit: false,
            scroll: 0,
            tx,
            rx,
            sessions: Vec::new(),
            session_scroll: 0,
            selected_session: 0,
            providers: Vec::new(),
            channels: Vec::new(),
            log_lines: Vec::new(),
            log_scroll: 0,
            log_auto_follow: true,
            cost_total: 0.0,
            cost_sessions: 0,
            cost_tokens: 0,
            cost_avg_per_session: 0.0,
        }
    }

    /// Run the main event loop until the user quits.
    pub async fn run(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<&mut io::Stdout>>,
    ) -> Result<()> {
        self.provider_name = util::default_provider(&self.config);
        self.model_name = util::default_model(&self.config);

        // Resolve a persistent session for the transcript.
        match util::build_session_manager(&self.config) {
            Ok(manager) => {
                if let Ok(session) = util::resolve_or_create_session(&manager, None).await {
                    self.session_id = session.id.to_string();
                }
            }
            Err(e) => {
                self.status_message = format!("session store unavailable: {e}");
            }
        }

        // Prime the background views with data.
        self.load_session_rows().await;
        self.load_provider_rows();
        self.load_channel_rows();
        self.load_cost_data().await;
        self.load_logs();

        self.status_message = format!(
            "session {} | {} / {} | Tab: views, Esc: quit",
            self.session_id, self.provider_name, self.model_name
        );

        while !self.should_quit {
            terminal.draw(|f| self.render(f))?;

            if event::poll(Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    self.handle_key(key);
                }
            }

            // Drain any stream deltas produced by the background task.
            while let Ok(evt) = self.rx.try_recv() {
                self.handle_stream_event(evt);
            }
        }
        Ok(())
    }

    fn render(&self, frame: &mut Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(3),
            ])
            .split(frame.size());

        // Status bar
        let status_style = Style::default()
            .fg(Color::White)
            .bg(Color::Blue)
            .add_modifier(Modifier::BOLD);
        let status_text = if self.streaming {
            format!(
                " {} | streaming... | Ctrl+C to cancel ",
                self.status_message
            )
        } else {
            format!(" {} | Enter: send, Tab: next view ", self.status_message)
        };
        frame.render_widget(
            Paragraph::new(Text::from(Line::from(Span::styled(
                status_text,
                status_style,
            )))),
            chunks[0],
        );

        // Tab bar
        self.render_tabs(frame, chunks[1]);

        // Main area.
        match self.view {
            View::Chat => {
                if self.help_visible {
                    let main = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Min(1), Constraint::Length(14)])
                        .split(chunks[2]);
                    self.render_messages(frame, main[0]);
                    self.render_help(frame, main[1]);
                } else {
                    self.render_messages(frame, chunks[2]);
                }
                self.render_input(frame, chunks[3]);
            }
            View::Sessions => {
                self.render_sessions(frame, chunks[2]);
                self.render_view_footer(frame, chunks[3], "↑/↓ scroll  ·  Enter: show messages");
            }
            View::Providers => {
                self.render_providers(frame, chunks[2]);
                self.render_view_footer(frame, chunks[3], "Provider configuration summary");
            }
            View::Channels => {
                self.render_channels(frame, chunks[2]);
                self.render_view_footer(frame, chunks[3], "Channel runtime status");
            }
            View::Cost => {
                self.render_cost(frame, chunks[2]);
                self.render_view_footer(frame, chunks[3], "Cost dashboard");
            }
            View::Logs => {
                self.render_logs(frame, chunks[2]);
                self.render_view_footer(frame, chunks[3], "↑/↓ scroll · Space: toggle follow");
            }
        }
    }

    fn render_tabs(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let titles: Vec<Line> = View::ALL
            .iter()
            .map(|v| {
                let label = v.title();
                if *v == self.view {
                    Line::from(Span::styled(
                        format!(" {label} "),
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(Span::styled(
                        format!(" {label} "),
                        Style::default().fg(Color::White),
                    ))
                }
            })
            .collect();
        let tabs = Tabs::new(titles)
            .select(self.view.index())
            .block(Block::default().borders(Borders::NONE))
            .style(Style::default().fg(Color::White));
        frame.render_widget(tabs, area);
    }

    fn render_view_footer(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect, hint: &str) {
        let hint_style = Style::default().fg(Color::DarkGray);
        let text = Text::from(Line::from(Span::styled(format!(" {hint} "), hint_style)));
        let widget = Paragraph::new(text).block(Block::default().borders(Borders::ALL));
        frame.render_widget(widget, area);
    }

    fn render_input(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let input_text = if self.input.is_empty() {
            Text::from(Line::from(Span::styled(
                " Type a message... (or /help)",
                Style::default().fg(Color::DarkGray),
            )))
        } else {
            Text::from(Line::from(Span::styled(
                format!("> {}", self.input),
                Style::default().fg(Color::Green),
            )))
        };
        let input_widget = Paragraph::new(input_text)
            .block(Block::default().borders(Borders::ALL).title("Input"))
            .wrap(Wrap { trim: true });
        frame.render_widget(input_widget, area);

        // Position the cursor inside the input box.
        let cursor_x =
            area.x + 2 + (self.input.chars().count() as u16).min(area.width.saturating_sub(4));
        let cursor_y = area.y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    fn render_messages(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for msg in &self.messages {
            let role_style = match msg.role.as_str() {
                "user" => Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
                "assistant" => Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
                _ => Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            };
            lines.push(Line::from(Span::styled(
                format!("[{}] ", msg.role),
                role_style,
            )));
            for line in msg.content.lines() {
                lines.push(Line::from(Span::raw(line.to_string())));
            }
            lines.push(Line::from(""));
        }

        let text = Text::from(lines);
        let scroll_offset = if self.streaming {
            // Follow the tail while streaming.
            u16::MAX
        } else {
            self.scroll
        };
        let widget = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Messages"))
            .scroll((scroll_offset, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(widget, area);
    }

    fn render_help(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let help_text = Text::from(vec![
            Line::from(" Help"),
            Line::from("  Enter       send message"),
            Line::from("  Esc / Ctrl+Q  quit"),
            Line::from("  Ctrl+C      cancel streaming"),
            Line::from("  Tab         cycle views"),
            Line::from(
                "  1..6        jump to view (1=Chat,2=Sessions,3=Providers,4=Channels,5=Cost,6=Logs)",
            ),
            Line::from("  /clear      clear the transcript"),
            Line::from("  /provider <p>  switch provider"),
            Line::from("  /model <m>     switch model"),
            Line::from("  /new        start a new session"),
            Line::from("  /status     show session status"),
            Line::from("  /sessions   list sessions in this pane"),
            Line::from("  /help       show this panel"),
        ]);
        let widget = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(widget, area);
    }

    // ------------------------------------------------------------------
    // View renderers
    // ------------------------------------------------------------------

    fn render_sessions(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        if self.sessions.is_empty() {
            let text = Text::from(Line::from(Span::styled(
                " No sessions found.",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Sessions"));
            frame.render_widget(widget, area);
            return;
        }

        let header = Row::new(vec![
            Cell::from(Span::styled(
                "ID",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Name",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Mode",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Status",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Msgs",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Tokens",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Cost",
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ]);

        let rows: Vec<Row> = self
            .sessions
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let selected = i == self.selected_session;
                let style = if selected {
                    Style::default().bg(Color::Blue).fg(Color::White)
                } else {
                    Style::default()
                };
                Row::new(vec![
                    Cell::from(Span::raw(truncate_str(&s.id, 12))),
                    Cell::from(Span::raw(&s.name)),
                    Cell::from(Span::raw(&s.mode)),
                    Cell::from(Span::styled(&s.status, status_color(&s.status))),
                    Cell::from(Span::raw(s.messages.to_string())),
                    Cell::from(Span::raw(s.tokens.to_string())),
                    Cell::from(Span::raw(format!("${:.4}", s.cost))),
                ])
                .style(style)
            })
            .collect();

        let widths = [
            Constraint::Length(14),
            Constraint::Length(22),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Length(10),
            Constraint::Length(10),
        ];
        let table = ratatui::widgets::Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title("Sessions"))
            .highlight_style(Style::default().bg(Color::Blue));
        frame.render_widget(table, area);
    }

    fn render_providers(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        if self.providers.is_empty() {
            let text = Text::from(Line::from(Span::styled(
                " No providers configured.",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Providers"));
            frame.render_widget(widget, area);
            return;
        }

        let header = Row::new(vec![
            Cell::from(Span::styled(
                "Provider",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Backend",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Models",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Default Model",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Status",
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ]);

        let rows: Vec<Row> = self
            .providers
            .iter()
            .map(|p| {
                Row::new(vec![
                    Cell::from(Span::raw(&p.name)),
                    Cell::from(Span::raw(&p.backend)),
                    Cell::from(Span::raw(p.models.to_string())),
                    Cell::from(Span::raw(&p.default_model)),
                    Cell::from(Span::styled(p.status, status_color(p.status))),
                ])
            })
            .collect();

        let widths = [
            Constraint::Length(18),
            Constraint::Length(16),
            Constraint::Length(8),
            Constraint::Min(16),
            Constraint::Length(10),
        ];
        let table = ratatui::widgets::Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title("Providers"));
        frame.render_widget(table, area);
    }

    fn render_channels(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        if self.channels.is_empty() {
            let text = Text::from(Line::from(Span::styled(
                " No channels configured.",
                Style::default().fg(Color::DarkGray),
            )));
            let widget = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Channels"));
            frame.render_widget(widget, area);
            return;
        }

        let header = Row::new(vec![
            Cell::from(Span::styled(
                "Name",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Type",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Enabled",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Cell::from(Span::styled(
                "Status",
                Style::default().add_modifier(Modifier::BOLD),
            )),
        ]);

        let rows: Vec<Row> = self
            .channels
            .iter()
            .map(|c| {
                Row::new(vec![
                    Cell::from(Span::raw(&c.name)),
                    Cell::from(Span::raw(&c.channel_type)),
                    Cell::from(Span::styled(
                        if c.enabled { "yes" } else { "no" },
                        if c.enabled {
                            Style::default().fg(Color::Green)
                        } else {
                            Style::default().fg(Color::Gray)
                        },
                    )),
                    Cell::from(Span::styled(c.status, status_color(c.status))),
                ])
            })
            .collect();

        let widths = [
            Constraint::Length(20),
            Constraint::Length(18),
            Constraint::Length(10),
            Constraint::Length(12),
        ];
        let table = ratatui::widgets::Table::new(rows, widths)
            .header(header)
            .block(Block::default().borders(Borders::ALL).title("Channels"));
        frame.render_widget(table, area);
    }

    fn render_cost(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(8), Constraint::Min(1)])
            .split(area);

        // Summary panel.
        let summary_text = Text::from(vec![
            Line::from("  Cost Summary"),
            Line::from(""),
            Line::from(vec![
                Span::styled("  Total sessions:  ", Style::default().fg(Color::Cyan)),
                Span::raw(self.cost_sessions.to_string()),
                Span::styled("   |   Total tokens:  ", Style::default().fg(Color::Cyan)),
                Span::raw(self.cost_tokens.to_string()),
            ]),
            Line::from(vec![
                Span::styled("  Total cost (USD):  ", Style::default().fg(Color::Cyan)),
                Span::styled(
                    format!("${:.4}", self.cost_total),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   |   Avg/session:  ", Style::default().fg(Color::Cyan)),
                Span::raw(format!("${:.4}", self.cost_avg_per_session)),
            ]),
        ]);
        let summary_widget = Paragraph::new(summary_text).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Cost Dashboard"),
        );
        frame.render_widget(summary_widget, chunks[0]);

        // Top sessions list.
        let mut sorted: Vec<&SessionRow> = self.sessions.iter().collect();
        sorted.sort_by(|a, b| {
            b.cost
                .partial_cmp(&a.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top: Vec<&SessionRow> = sorted.into_iter().take(10).collect();

        let items: Vec<ListItem> = top
            .iter()
            .map(|s| {
                ListItem::new(Line::from(vec![
                    Span::raw(format!("  {:<12} ", truncate_str(&s.id, 12))),
                    Span::styled(
                        format!("${:<10.4} ", s.cost),
                        Style::default().fg(Color::Green),
                    ),
                    Span::styled(
                        format!("{:<10} tokens", s.tokens),
                        Style::default().fg(Color::Yellow),
                    ),
                    Span::raw(truncate_str(&s.name, 20)),
                ]))
            })
            .collect();
        let list = List::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Top Sessions by Cost"),
        );
        frame.render_widget(list, chunks[1]);
    }

    fn render_logs(&self, frame: &mut Frame<'_>, area: ratatui::layout::Rect) {
        let lines: Vec<Line> = if self.log_lines.is_empty() {
            vec![Line::from(Span::styled(
                " No log entries.",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            self.log_lines
                .iter()
                .map(|l| {
                    let (style, content) = if l.contains("ERROR") || l.contains("error") {
                        (Style::default().fg(Color::Red), l.as_str())
                    } else if l.contains("WARN") || l.contains("warning") {
                        (Style::default().fg(Color::Yellow), l.as_str())
                    } else if l.contains("INFO") {
                        (Style::default().fg(Color::White), l.as_str())
                    } else {
                        (Style::default().fg(Color::Gray), l.as_str())
                    };
                    Line::from(Span::styled(content.to_string(), style))
                })
                .collect()
        };

        let scroll = if self.log_auto_follow {
            u16::MAX
        } else {
            self.log_scroll
        };
        let widget = Paragraph::new(lines)
            .block(Block::default().borders(Borders::ALL).title("Logs"))
            .scroll((scroll, 0));
        frame.render_widget(widget, area);
    }

    // ------------------------------------------------------------------
    // Data loaders
    // ------------------------------------------------------------------

    async fn load_session_rows(&mut self) {
        match util::build_session_manager(&self.config) {
            Ok(manager) => match manager.list_sessions(&util::default_agent_id(), 200, 0) {
                Ok(sessions) => {
                    self.sessions = sessions
                        .into_iter()
                        .map(|s| SessionRow {
                            id: s.id.to_string(),
                            name: s.name,
                            mode: format!("{:?}", s.mode).to_lowercase(),
                            status: format!("{:?}", s.status).to_lowercase(),
                            messages: s.message_count,
                            tokens: s.total_tokens,
                            cost: s.total_cost_usd,
                        })
                        .collect();
                }
                Err(_) => {}
            },
            Err(_) => {}
        }
    }

    fn load_provider_rows(&mut self) {
        match util::build_provider_registry(&self.config) {
            Ok(registry) => {
                let default = util::default_model(&self.config);
                self.providers = registry
                    .list()
                    .iter()
                    .filter_map(|name| {
                        registry.get(name).map(|p| {
                            let models = p.supported_models();
                            ProviderRow {
                                name: name.clone(),
                                backend: p.name().to_string(),
                                models: models.len(),
                                default_model: models
                                    .first()
                                    .cloned()
                                    .unwrap_or_else(|| default.clone()),
                                status: "configured",
                            }
                        })
                    })
                    .collect();
            }
            Err(_) => {}
        }
    }

    fn load_channel_rows(&mut self) {
        self.channels = self
            .config
            .channels
            .iter()
            .map(|c| ChannelRow {
                name: c.name.clone(),
                channel_type: c.channel_type.clone(),
                enabled: c.enabled,
                status: if c.enabled { "ready" } else { "disabled" },
            })
            .collect();
    }

    async fn load_cost_data(&mut self) {
        match util::build_session_manager(&self.config) {
            Ok(manager) => match manager.list_sessions(&util::default_agent_id(), 1000, 0) {
                Ok(sessions) => {
                    self.cost_sessions = sessions.len() as u64;
                    self.cost_tokens = sessions.iter().map(|s| s.total_tokens).sum();
                    self.cost_total = sessions.iter().map(|s| s.total_cost_usd).sum();
                    self.cost_avg_per_session = if sessions.is_empty() {
                        0.0
                    } else {
                        self.cost_total / sessions.len() as f64
                    };
                }
                Err(_) => {}
            },
            Err(_) => {}
        }
    }

    fn load_logs(&mut self) {
        let log_path = util::data_dir().join("gateway.log");
        let lines: Vec<String> = std::fs::read_to_string(log_path)
            .map(|c| c.lines().map(|l| l.to_string()).collect())
            .unwrap_or_default();
        // Keep the last 1000 lines.
        let start = lines.len().saturating_sub(1000);
        self.log_lines = lines[start..].to_vec();
    }

    // ------------------------------------------------------------------
    // Key handling
    // ------------------------------------------------------------------

    fn handle_key(&mut self, key: KeyEvent) {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            if self.streaming {
                self.status_message = "Stream cancelled".to_string();
                self.streaming = false;
            } else {
                self.should_quit = true;
            }
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('q') {
            self.should_quit = true;
            return;
        }

        // View switching keys apply globally, except in the chat view where
        // digits are typed into the input.
        if !key.modifiers.contains(KeyModifiers::CONTROL) && self.view != View::Chat {
            match key.code {
                KeyCode::Tab => {
                    let next = (self.view.index() + 1) % View::ALL.len();
                    self.view = View::from_index(next);
                    self.status_message = format!("View: {}", self.view.title());
                    return;
                }
                KeyCode::Char('1') => {
                    self.view = View::Chat;
                    self.status_message = "View: Chat".to_string();
                    return;
                }
                KeyCode::Char('2') => {
                    self.view = View::Sessions;
                    self.status_message = "View: Sessions".to_string();
                    return;
                }
                KeyCode::Char('3') => {
                    self.view = View::Providers;
                    self.status_message = "View: Providers".to_string();
                    return;
                }
                KeyCode::Char('4') => {
                    self.view = View::Channels;
                    self.status_message = "View: Channels".to_string();
                    return;
                }
                KeyCode::Char('5') => {
                    self.view = View::Cost;
                    self.status_message = "View: Cost".to_string();
                    return;
                }
                KeyCode::Char('6') => {
                    self.view = View::Logs;
                    self.status_message = "View: Logs".to_string();
                    return;
                }
                _ => {}
            }
        }

        // View-specific key handling.
        match self.view {
            View::Chat => self.handle_chat_key(key),
            View::Sessions => self.handle_sessions_key(key),
            View::Logs => self.handle_logs_key(key),
            _ => {}
        }
    }

    fn handle_chat_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                self.should_quit = true;
                self.status_message = "Quitting...".to_string();
            }
            KeyCode::Tab => {
                self.help_visible = !self.help_visible;
            }
            KeyCode::Enter => {
                let msg = std::mem::take(&mut self.input);
                let trimmed = msg.trim();
                if trimmed.is_empty() {
                    return;
                }
                if let Some(cmd) = trimmed.strip_prefix('/') {
                    self.handle_command(cmd);
                } else {
                    self.send_user_message(trimmed.to_string());
                }
            }
            KeyCode::Backspace => {
                self.input.pop();
            }
            KeyCode::Char(c) => {
                self.input.push(c);
            }
            KeyCode::Up => {
                self.scroll = self.scroll.saturating_add(1);
            }
            KeyCode::Down => {
                self.scroll = self.scroll.saturating_sub(1);
            }
            _ => {}
        }
    }

    fn handle_sessions_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => {
                self.selected_session = self.selected_session.saturating_sub(1);
            }
            KeyCode::Down => {
                if !self.sessions.is_empty() {
                    self.selected_session =
                        (self.selected_session + 1).min(self.sessions.len() - 1);
                }
            }
            KeyCode::Enter => {
                if let Some(row) = self.sessions.get(self.selected_session) {
                    self.view = View::Chat;
                    self.session_id = row.id.clone();
                    self.status_message = format!("Session {}", row.id);
                }
            }
            KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_logs_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up => {
                self.log_auto_follow = false;
                self.log_scroll = self.log_scroll.saturating_add(1);
            }
            KeyCode::Down => {
                self.log_auto_follow = false;
                self.log_scroll = self.log_scroll.saturating_sub(1);
            }
            KeyCode::Char(' ') => {
                self.log_auto_follow = !self.log_auto_follow;
                self.status_message = if self.log_auto_follow {
                    "Logs: following tail".to_string()
                } else {
                    "Logs: manual scroll".to_string()
                };
            }
            KeyCode::Esc => {
                self.should_quit = true;
            }
            _ => {}
        }
    }

    fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        match parts[0] {
            "help" => self.help_visible = true,
            "clear" => {
                self.messages.clear();
                self.scroll = 0;
                self.status_message = "Transcript cleared".to_string();
            }
            "provider" => {
                if let Some(name) = parts.get(1) {
                    self.provider_name = name.to_string();
                    self.status_message = format!("Provider set to {name}");
                } else {
                    self.status_message = format!("Provider: {}", self.provider_name);
                }
            }
            "model" => {
                if let Some(name) = parts.get(1) {
                    self.model_name = name.to_string();
                    self.status_message = format!("Model set to {name}");
                } else {
                    self.status_message = format!("Model: {}", self.model_name);
                }
            }
            "new" => {
                self.session_id = uuid::Uuid::new_v4().to_string();
                self.messages.clear();
                self.status_message = format!("New session {}", self.session_id);
            }
            "status" => {
                self.status_message = format!(
                    "session {} | {} / {} | {} messages",
                    self.session_id,
                    self.provider_name,
                    self.model_name,
                    self.messages.len()
                );
            }
            "sessions" => {
                self.view = View::Sessions;
                self.status_message = "View: Sessions".to_string();
            }
            "cost" => {
                self.view = View::Cost;
                self.status_message = "View: Cost".to_string();
            }
            "logs" => {
                self.view = View::Logs;
                self.status_message = "View: Logs".to_string();
            }
            "providers" => {
                self.view = View::Providers;
                self.status_message = "View: Providers".to_string();
            }
            "channels" => {
                self.view = View::Channels;
                self.status_message = "View: Channels".to_string();
            }
            _ => {
                self.status_message = format!("Unknown command: /{}", parts[0]);
            }
        }
    }

    /// Push a user message and start a background stream for the response.
    fn send_user_message(&mut self, content: String) {
        self.messages.push(ChatLine {
            role: "user".to_string(),
            content: content.clone(),
        });
        self.status_message = "Waiting for response...".to_string();
        self.streaming = true;

        let tx = self.tx.clone();
        let config = self.config.clone();
        let provider_name = self.provider_name.clone();
        let model_name = self.model_name.clone();

        tokio::spawn(async move {
            let result = stream_response(&config, &provider_name, &model_name, &content).await;
            match result {
                Ok(deltas) => {
                    for d in deltas {
                        if tx.send(TuiEvent::Delta(d)).await.is_err() {
                            return;
                        }
                    }
                    let _ = tx.send(TuiEvent::Done).await;
                }
                Err(e) => {
                    let _ = tx.send(TuiEvent::Error(e.to_string())).await;
                }
            }
        });
    }

    fn handle_stream_event(&mut self, evt: TuiEvent) {
        match evt {
            TuiEvent::Delta(text) => {
                // Append to the trailing assistant message, or start one.
                match self.messages.last_mut() {
                    Some(last) if last.role == "assistant" => {
                        last.content.push_str(&text);
                    }
                    _ => {
                        self.messages.push(ChatLine {
                            role: "assistant".to_string(),
                            content: text,
                        });
                    }
                }
            }
            TuiEvent::Done => {
                self.streaming = false;
                self.status_message = "Response complete".to_string();
            }
            TuiEvent::Error(e) => {
                self.streaming = false;
                self.status_message = format!("Error: {e}");
            }
        }
    }
}

/// Stream a single response from the provider, returning the collected text.
async fn stream_response(
    config: &Config,
    provider_name: &str,
    model_name: &str,
    content: &str,
) -> Result<Vec<String>> {
    let registry = util::build_provider_registry(config)?;
    let provider = registry
        .get(provider_name)
        .ok_or_else(|| anyhow::anyhow!("Provider '{provider_name}' is not configured"))?;

    let chat_config = ChatConfig {
        model: model_name.to_string(),
        temperature: 0.7,
        max_tokens: 2048,
        top_p: 1.0,
        stop_sequences: Vec::new(),
        stream: true,
        extra: Default::default(),
    };

    let user_msg = opensquilla_core::types::Message::user(content);
    let mut stream = provider
        .stream_chat(&chat_config, &[user_msg], &[])
        .await
        .map_err(|e| anyhow::anyhow!("Failed to open stream: {e}"))?;

    let mut deltas = Vec::new();
    while let Some(event) = stream.next().await {
        match event.map_err(|e| anyhow::anyhow!("Stream error: {e}"))? {
            opensquilla_provider::StreamEvent::Text { text } => {
                deltas.push(text);
            }
            opensquilla_provider::StreamEvent::Done { .. } => break,
            opensquilla_provider::StreamEvent::Error { message } => {
                anyhow::bail!("Stream error: {message}");
            }
            _ => {}
        }
    }
    Ok(deltas)
}

/// Map a status string to a color.
fn status_color(status: &str) -> Style {
    match status.to_lowercase().as_str() {
        "active" | "running" | "connected" | "configured" | "ready" | "healthy" => {
            Style::default().fg(Color::Green)
        }
        "paused" | "degraded" | "starting" => Style::default().fg(Color::Yellow),
        "killed" | "failed" | "error" | "disabled" | "stopped" => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::Gray),
    }
}

/// Truncate a string to a maximum character count.
fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}
