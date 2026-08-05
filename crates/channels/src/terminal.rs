//! Terminal channel adapter — interactive stdin/stdout I/O.
//!
//! This adapter reads bytes from stdin through a line editor and writes styled
//! output to stdout:
//!
//! - A [`LineEditor`] that supports backspace, left/right arrow movement,
//!   up/down history navigation, Home/End and Delete.
//! - Raw-mode toggling on Unix (via `stty`); on Windows the editor still
//!   works over line-buffered input.
//! - ANSI colors and styles for formatting outgoing messages.
//!
//! Incoming lines are turned into [`IncomingMessage`]s and queued for polling
//! via [`TerminalChannel::incoming_receiver`].

use crate::types::{Channel, ChannelConfig, ChannelType, IncomingMessage, OutgoingMessage};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// ANSI styling
// ---------------------------------------------------------------------------

/// ANSI reset sequence.
pub const ANSI_RESET: &str = "\x1b[0m";
/// ANSI bold.
pub const ANSI_BOLD: &str = "\x1b[1m";
/// ANSI dim.
pub const ANSI_DIM: &str = "\x1b[2m";
/// ANSI underline.
pub const ANSI_UNDERLINE: &str = "\x1b[4m";
/// ANSI red foreground.
pub const ANSI_RED: &str = "\x1b[31m";
/// ANSI green foreground.
pub const ANSI_GREEN: &str = "\x1b[32m";
/// ANSI yellow foreground.
pub const ANSI_YELLOW: &str = "\x1b[33m";
/// ANSI blue foreground.
pub const ANSI_BLUE: &str = "\x1b[34m";
/// ANSI magenta foreground.
pub const ANSI_MAGENTA: &str = "\x1b[35m";
/// ANSI cyan foreground.
pub const ANSI_CYAN: &str = "\x1b[36m";
/// ANSI gray foreground.
pub const ANSI_GRAY: &str = "\x1b[90m";

/// Wrap `text` in an ANSI `style` sequence, if enabled.
pub fn styled(text: &str, style: &str) -> String {
    format!("{style}{text}{ANSI_RESET}")
}

/// Color `text` in the given foreground color.
pub fn colorize(text: &str, color: &str) -> String {
    styled(text, color)
}

/// Format an outgoing message for terminal display.
pub fn format_outgoing(message: &OutgoingMessage, color: bool) -> String {
    let header = if color {
        format!(
            "{}[{}]{} {}",
            ANSI_CYAN, message.channel_id, ANSI_RESET, ANSI_GRAY
        )
    } else {
        format!("[{}] ", message.channel_id)
    };
    let mut out = format!("{header}{}{ANSI_RESET}", message.text);
    if !message.attachments.is_empty() {
        let names: Vec<&str> = message
            .attachments
            .iter()
            .filter_map(|a| a.url.as_deref())
            .collect();
        if !names.is_empty() {
            out.push_str(&format!(
                "\n  {}attachments: {}{ANSI_RESET}",
                ANSI_DIM,
                names.join(", ")
            ));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Raw mode
// ---------------------------------------------------------------------------

/// Enable raw-mode input so keystrokes are delivered without buffering or echo.
///
/// On Unix this shells out to `stty raw -echo`. Windows terminals do not
/// support a portable raw mode from a plain library, so this is a no-op there
/// and the line editor still functions over line-buffered input.
pub fn enable_raw_mode() -> Result<(), String> {
    #[cfg(unix)]
    {
        let status = std::process::Command::new("stty")
            .args(["raw", "-echo"])
            .status()
            .map_err(|e| format!("stty raw: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("stty raw failed".to_string())
        }
    }
    #[cfg(windows)]
    {
        warn!("Raw terminal mode is not supported on Windows; using line-buffered input");
        Ok(())
    }
}

/// Restore sane terminal settings after raw mode.
pub fn disable_raw_mode() -> Result<(), String> {
    #[cfg(unix)]
    {
        let status = std::process::Command::new("stty")
            .args(["sane"])
            .status()
            .map_err(|e| format!("stty sane: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err("stty sane failed".to_string())
        }
    }
    #[cfg(windows)]
    {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Line editor
// ---------------------------------------------------------------------------

/// The result of feeding a byte to a [`LineEditor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorAction {
    /// Keep reading (no complete line yet).
    Continue,
    /// A full line was submitted.
    Submit(String),
    /// The user pressed Ctrl+C.
    Interrupt,
    /// End of input (Ctrl+D / EOF).
    Eof,
}

/// A small, terminal-agnostic line editor.
///
/// Handles printable characters, backspace, left/right cursor movement,
/// up/down history, Home/End, Delete and Ctrl+C/D. Escape sequences are
/// recognized by collecting bytes until a CSI final byte (`0x40..=0x7e`).
#[derive(Debug)]
pub struct LineEditor {
    buffer: String,
    cursor: usize,
    history: Vec<String>,
    history_index: usize,
    escape_buf: Vec<u8>,
    in_escape: bool,
}

impl Default for LineEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl LineEditor {
    /// Create an empty editor.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: 0,
            escape_buf: Vec::new(),
            in_escape: false,
        }
    }

    /// The current buffer content.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Feed a single input byte.
    pub fn handle_input(&mut self, byte: u8) -> EditorAction {
        if self.in_escape {
            return self.handle_escape_byte(byte);
        }
        match byte {
            b'\r' | b'\n' => self.submit(),
            0x03 => EditorAction::Interrupt,
            0x04 => {
                if self.buffer.is_empty() {
                    EditorAction::Eof
                } else {
                    self.submit()
                }
            }
            0x08 | 0x7f => {
                self.backspace();
                EditorAction::Continue
            }
            0x09 => {
                // Tab: insert a couple of spaces as a soft indent.
                for _ in 0..2 {
                    self.insert(' ');
                }
                EditorAction::Continue
            }
            0x1b => {
                self.in_escape = true;
                self.escape_buf.clear();
                self.escape_buf.push(byte);
                EditorAction::Continue
            }
            b if b >= 0x20 && b < 0x7f => {
                self.insert(b as char);
                EditorAction::Continue
            }
            _ => EditorAction::Continue,
        }
    }

    fn handle_escape_byte(&mut self, byte: u8) -> EditorAction {
        self.escape_buf.push(byte);
        // A CSI sequence is `ESC '[' params* final`, where the final byte is
        // in `0x40..=0x7e`. The `[` introducer (0x5b) is itself in that range
        // but must not be treated as final, hence the `len >= 3` guard.
        if self.escape_buf.len() >= 3 && (0x40..=0x7e).contains(&byte) {
            let seq = std::mem::take(&mut self.escape_buf);
            self.in_escape = false;
            self.apply_escape(&seq);
        }
        EditorAction::Continue
    }

    fn apply_escape(&mut self, seq: &[u8]) {
        if seq.len() >= 3 && seq[0] == 0x1b && seq[1] == b'[' {
            match seq[2] {
                b'A' => self.history_prev(),
                b'B' => self.history_next(),
                b'C' => self.cursor_right(),
                b'D' => self.cursor_left(),
                b'H' => self.cursor = 0,
                b'F' => self.cursor = self.buffer.len(),
                b'1' => self.cursor = 0,                 // Home
                b'4' => self.cursor = self.buffer.len(), // End
                b'3' => {
                    if self.cursor < self.buffer.len() {
                        self.buffer.remove(self.cursor);
                    }
                }
                _ => {}
            }
        }
    }

    fn submit(&mut self) -> EditorAction {
        let line = std::mem::take(&mut self.buffer);
        self.cursor = 0;
        let trimmed = line.trim_end().to_string();
        if !trimmed.is_empty() && self.history.last() != Some(&trimmed) {
            self.history.push(trimmed.clone());
        }
        self.history_index = self.history.len();
        EditorAction::Submit(line)
    }

    fn insert(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.buffer.remove(self.cursor);
        }
    }

    fn cursor_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    fn cursor_right(&mut self) {
        if self.cursor < self.buffer.len() {
            self.cursor += 1;
        }
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_index == 0 {
            return;
        }
        self.history_index -= 1;
        self.buffer = self.history[self.history_index].clone();
        self.cursor = self.buffer.len();
    }

    fn history_next(&mut self) {
        if self.history_index >= self.history.len() {
            return;
        }
        self.history_index += 1;
        if self.history_index >= self.history.len() {
            self.buffer.clear();
            self.cursor = 0;
        } else {
            self.buffer = self.history[self.history_index].clone();
            self.cursor = self.buffer.len();
        }
    }
}

// ---------------------------------------------------------------------------
// Channel adapter
// ---------------------------------------------------------------------------

/// Terminal channel adapter using stdin/stdout for interactive CLI use.
pub struct TerminalChannel {
    config: ChannelConfig,
    sender: mpsc::UnboundedSender<String>,
    receiver: Option<mpsc::UnboundedReceiver<IncomingMessage>>,
    color: bool,
    prompt: String,
}

impl TerminalChannel {
    pub fn new(config: ChannelConfig) -> Result<Self, String> {
        let color = config
            .config
            .get("color")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let prompt = config
            .config
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("> ")
            .to_string();
        let raw_mode = config
            .config
            .get("raw_mode")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let (sender, mut rx) = mpsc::unbounded_channel::<String>();
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel::<IncomingMessage>();

        if raw_mode {
            if let Err(e) = enable_raw_mode() {
                warn!("Failed to enable raw mode: {e}");
            }
        }

        // Spawn the stdin reader driving the line editor.
        let channel_id = config.channel_id.clone();
        let user_name = config
            .config
            .get("user_name")
            .and_then(|v| v.as_str())
            .unwrap_or("User")
            .to_string();
        tokio::spawn(async move {
            let mut stdin = tokio::io::stdin();
            let mut editor = LineEditor::new();
            let mut buf = [0u8; 1];
            info!("Terminal channel started, reading stdin...");
            loop {
                let n = match stdin.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => n,
                    Err(e) => {
                        warn!("Terminal stdin error: {e}");
                        break;
                    }
                };
                if n == 0 {
                    break;
                }
                match editor.handle_input(buf[0]) {
                    EditorAction::Continue => {}
                    EditorAction::Submit(line) => {
                        let trimmed = line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let msg = make_incoming(&channel_id, &user_name, line);
                        if incoming_tx.send(msg).is_err() {
                            break;
                        }
                    }
                    EditorAction::Interrupt | EditorAction::Eof => break,
                }
            }
            let _ = disable_raw_mode();
        });

        // Spawn the stdout writer.
        let color = color;
        tokio::spawn(async move {
            let mut stdout = tokio::io::stdout();
            while let Some(text) = rx.recv().await {
                let _ = stdout.write_all(text.as_bytes()).await;
                let _ = stdout.write_all(b"\n").await;
                let _ = stdout.flush().await;
            }
        });

        Ok(Self {
            config,
            sender,
            receiver: Some(incoming_rx),
            color,
            prompt,
        })
    }

    /// Get the incoming message receiver for polling.
    pub fn incoming_receiver(&mut self) -> Option<mpsc::UnboundedReceiver<IncomingMessage>> {
        self.receiver.take()
    }

    /// The configured display prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Whether ANSI color output is enabled.
    pub fn color_enabled(&self) -> bool {
        self.color
    }
}

fn make_incoming(channel_id: &str, user_name: &str, text: String) -> IncomingMessage {
    IncomingMessage {
        id: uuid::Uuid::new_v4(),
        channel_id: channel_id.to_string(),
        channel_type: ChannelType::Terminal,
        user_id: "terminal_user".to_string(),
        user_name: Some(user_name.to_string()),
        text,
        thread_id: None,
        attachments: Vec::new(),
        timestamp: chrono::Utc::now(),
        raw: serde_json::Value::Null,
    }
}

#[async_trait::async_trait]
impl Channel for TerminalChannel {
    fn channel_type(&self) -> ChannelType {
        ChannelType::Terminal
    }

    fn channel_id(&self) -> &str {
        &self.config.channel_id
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    async fn send_message(&self, message: &OutgoingMessage) -> Result<(), String> {
        let text = format_outgoing(message, self.color);
        self.sender
            .send(text)
            .map_err(|e| format!("Send error: {e}"))
    }

    async fn send_typing(&self, _channel_id: &str) -> Result<(), String> {
        Ok(())
    }

    async fn set_webhook(&self, _url: &str) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn channel() -> TerminalChannel {
        TerminalChannel::new(ChannelConfig {
            channel_type: ChannelType::Terminal,
            channel_id: "term".to_string(),
            name: "test".to_string(),
            enabled: true,
            config: json!({ "color": false }),
        })
        .unwrap()
    }

    #[test]
    fn test_line_editor_basic() {
        let mut ed = LineEditor::new();
        assert_eq!(ed.handle_input(b'h'), EditorAction::Continue);
        assert_eq!(ed.handle_input(b'i'), EditorAction::Continue);
        assert_eq!(ed.buffer(), "hi");
        assert_eq!(
            ed.handle_input(b'\n'),
            EditorAction::Submit("hi".to_string())
        );
        assert_eq!(ed.buffer(), "");
    }

    #[test]
    fn test_line_editor_backspace() {
        let mut ed = LineEditor::new();
        for b in b"abc".iter() {
            ed.handle_input(*b);
        }
        ed.handle_input(0x7f);
        assert_eq!(ed.buffer(), "ab");
        ed.handle_input(0x08);
        assert_eq!(ed.buffer(), "a");
    }

    #[test]
    fn test_line_editor_arrow_keys() {
        let mut ed = LineEditor::new();
        for b in b"abc".iter() {
            ed.handle_input(*b);
        }
        // Move left twice: ESC [ D x2
        ed.handle_input(0x1b);
        ed.handle_input(b'[');
        ed.handle_input(b'D');
        ed.handle_input(0x1b);
        ed.handle_input(b'[');
        ed.handle_input(b'D');
        // Insert 'x' at cursor -> "axbc"
        ed.handle_input(b'x');
        assert_eq!(ed.buffer(), "axbc");
    }

    #[test]
    fn test_line_editor_history() {
        let mut ed = LineEditor::new();
        ed.handle_input(b'a');
        assert_eq!(
            ed.handle_input(b'\n'),
            EditorAction::Submit("a".to_string())
        );
        ed.handle_input(b'b');
        assert_eq!(
            ed.handle_input(b'\n'),
            EditorAction::Submit("b".to_string())
        );
        // Up twice -> "a"
        ed.handle_input(0x1b);
        ed.handle_input(b'[');
        ed.handle_input(b'A');
        ed.handle_input(0x1b);
        ed.handle_input(b'[');
        ed.handle_input(b'A');
        assert_eq!(ed.buffer(), "a");
    }

    #[test]
    fn test_line_editor_ctrl_c() {
        let mut ed = LineEditor::new();
        assert_eq!(ed.handle_input(0x03), EditorAction::Interrupt);
        assert_eq!(ed.handle_input(0x04), EditorAction::Eof);
    }

    #[test]
    fn test_styled_output() {
        assert_eq!(colorize("x", ANSI_RED), "\x1b[31mx\x1b[0m");
        assert_eq!(styled("y", ANSI_BOLD), "\x1b[1my\x1b[0m");
    }

    #[test]
    fn test_format_outgoing_plain() {
        let msg = OutgoingMessage::new(
            "term".to_string(),
            ChannelType::Terminal,
            "hello".to_string(),
        );
        let text = format_outgoing(&msg, false);
        assert!(text.contains("hello"));
        assert!(text.contains("[term]"));
    }

    #[tokio::test]
    async fn test_terminal_channel_constructs() {
        let c = channel();
        assert_eq!(c.channel_type(), ChannelType::Terminal);
        assert_eq!(c.channel_id(), "term");
    }
}
