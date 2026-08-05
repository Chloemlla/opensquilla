//! # OpenSquilla TUI
//!
//! Terminal UI entry point for the `osq-tui` binary. Renders a full ratatui
//! chat interface with a status bar, a scrollable message list, an input line,
//! and a toggleable help panel. Responses are streamed to the UI in real time
//! through an internal mpsc channel fed by a background provider stream.
//!
//! This file is both a library module (`opensquilla_cli::tui`) and the crate
//! root for the `osq-tui` binary target.
//!
//! Key bindings:
//! - `Enter`      send the current input
//! - `Esc`        quit
//! - `Ctrl+C`     cancel an in-flight stream (or quit when idle)
//! - `Tab`        toggle the help panel
//! - `Ctrl+Q`     quit
//! - `/` commands are handled when the input starts with a slash

use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures::StreamExt;
use opensquilla_core::config::Config;
use opensquilla_provider::ChatConfig;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
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

/// Application state for the TUI loop.
pub struct TuiApp {
    config: Config,
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

        self.status_message = format!(
            "session {} | {} / {}",
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

    fn render(&self, frame: &mut Frame<CrosstermBackend<&mut io::Stdout>>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
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
            format!(
                " {} | Enter: send, Tab: help, Esc: quit ",
                self.status_message
            )
        };
        frame.render_widget(
            Paragraph::new(Text::from(Line::from(Span::styled(
                status_text,
                status_style,
            )))),
            chunks[0],
        );

        // Main area: messages and optional help panel.
        if self.help_visible {
            let main = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Min(1), Constraint::Length(12)])
                .split(chunks[1]);
            self.render_messages(frame, main[0]);
            self.render_help(frame, main[1]);
        } else {
            self.render_messages(frame, chunks[1]);
        }

        // Input area
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
        frame.render_widget(input_widget, chunks[2]);

        // Position the cursor inside the input box.
        let cursor_x = chunks[2].x
            + 2
            + (self.input.chars().count() as u16).min(chunks[2].width.saturating_sub(4));
        let cursor_y = chunks[2].y + 1;
        frame.set_cursor_position((cursor_x, cursor_y));
    }

    fn render_messages(
        &self,
        frame: &mut Frame<CrosstermBackend<&mut io::Stdout>>,
        area: ratatui::layout::Rect,
    ) {
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

    fn render_help(
        &self,
        frame: &mut Frame<CrosstermBackend<&mut io::Stdout>>,
        area: ratatui::layout::Rect,
    ) {
        let help_text = Text::from(vec![
            Line::from(" Help"),
            Line::from("  Enter       send message"),
            Line::from("  Esc / Ctrl+Q  quit"),
            Line::from("  Ctrl+C      cancel streaming"),
            Line::from("  Tab         toggle this panel"),
            Line::from("  /clear      clear the transcript"),
            Line::from("  /provider <p>  switch provider"),
            Line::from("  /model <m>     switch model"),
            Line::from("  /new        start a new session"),
            Line::from("  /status     show session status"),
            Line::from("  /help       show this panel"),
        ]);
        let widget = Paragraph::new(help_text)
            .block(Block::default().borders(Borders::ALL).title("Help"))
            .style(Style::default().fg(Color::White));
        frame.render_widget(widget, area);
    }

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
                    let _ = tx.send(TuiEvent::Error(e)).await;
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
