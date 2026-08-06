//! # Rich terminal output formatting
//!
//! A lightweight table and formatting module that replaces Python's Rich
//! library. Provides table rendering with column alignment, colored output,
//! progress indicators, panels, and tree-style listings.
//!
//! All formatting degrades gracefully when stdout is not a TTY: colors and
//! box-drawing characters are omitted, leaving plain aligned text.

use std::fmt::Display;
use std::io::{self, IsTerminal};

// ---------------------------------------------------------------------------
// Color support
// ---------------------------------------------------------------------------

/// Terminal color codes for ANSI-aware output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Default,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    Gray,
    BrightRed,
    BrightGreen,
    BrightYellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    White,
}

impl Color {
    /// Return the ANSI foreground escape code for this color.
    fn fg_code(self) -> Option<&'static str> {
        match self {
            Color::Default => None,
            Color::Black => Some("30"),
            Color::Red => Some("31"),
            Color::Green => Some("32"),
            Color::Yellow => Some("33"),
            Color::Blue => Some("34"),
            Color::Magenta => Some("35"),
            Color::Cyan => Some("36"),
            Color::Gray => Some("90"),
            Color::BrightRed => Some("91"),
            Color::BrightGreen => Some("92"),
            Color::BrightYellow => Some("93"),
            Color::BrightBlue => Some("94"),
            Color::BrightMagenta => Some("95"),
            Color::BrightCyan => Some("96"),
            Color::White => Some("97"),
        }
    }

    /// Return the ANSI background escape code for this color.
    fn bg_code(self) -> Option<&'static str> {
        self.fg_code().map(|c| match c {
            "30" => "40",
            "31" => "41",
            "32" => "42",
            "33" => "43",
            "34" => "44",
            "35" => "45",
            "36" => "46",
            "90" => "100",
            "91" => "101",
            "92" => "102",
            "93" => "103",
            "94" => "104",
            "95" => "105",
            "96" => "106",
            "97" => "107",
            other => other,
        })
    }
}

/// Text style modifiers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Style {
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub dim: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
}

impl Style {
    /// Create a new default style.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the foreground color.
    pub fn fg(mut self, color: Color) -> Self {
        self.fg = color;
        self
    }

    /// Set the background color.
    pub fn bg(mut self, color: Color) -> Self {
        self.bg = color;
        self
    }

    /// Make the text bold.
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    /// Dim the text.
    pub fn dim(mut self) -> Self {
        self.dim = true;
        self
    }

    /// Italicize the text.
    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    /// Underline the text.
    pub fn underline(mut self) -> Self {
        self.underline = true;
        self
    }

    /// Strike through the text.
    pub fn strikethrough(mut self) -> Self {
        self.strikethrough = true;
        self
    }

    /// Wrap a string in this style, producing a `StyledString`.
    pub fn styled(self, text: impl Into<String>) -> StyledString {
        StyledString::new(text, self)
    }

    /// Render the ANSI escape sequence that opens this style.
    fn open(&self) -> String {
        if !use_color() {
            return String::new();
        }
        let mut codes: Vec<&str> = Vec::new();
        if self.bold {
            codes.push("1");
        }
        if self.dim {
            codes.push("2");
        }
        if self.italic {
            codes.push("3");
        }
        if self.underline {
            codes.push("4");
        }
        if self.strikethrough {
            codes.push("9");
        }
        if let Some(c) = self.fg.fg_code() {
            codes.push(c);
        }
        if let Some(c) = self.bg.bg_code() {
            codes.push(c);
        }
        if codes.is_empty() {
            String::new()
        } else {
            format!("\x1b[{}m", codes.join(";"))
        }
    }

    /// Render the ANSI reset escape sequence.
    fn close(&self) -> &'static str {
        if use_color() { "\x1b[0m" } else { "" }
    }
}

/// Check whether color output should be used.
///
/// Colors are enabled when stdout is a TTY and the `NO_COLOR` environment
/// variable is not set. The `CLICOLOR_FORCE` variable overrides the TTY check.
fn use_color() -> bool {
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("CLICOLOR_FORCE").is_ok() {
        return true;
    }
    io::stdout().is_terminal()
}

/// A styled string ready for printing.
#[derive(Debug, Clone)]
pub struct StyledString {
    text: String,
    style: Style,
}

impl StyledString {
    /// Create a new styled string.
    pub fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }

    /// Create a plain string with no styling.
    pub fn plain(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: Style::default(),
        }
    }

    /// Render the styled string with ANSI codes.
    pub fn render(&self) -> String {
        format!("{}{}{}", self.style.open(), self.text, self.style.close())
    }
}

impl Display for StyledString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render())
    }
}

/// Convenience trait to apply styles to any `Display` value.
pub trait Stylize: Sized {
    fn style(self, style: Style) -> StyledString;
    fn fg(self, color: Color) -> StyledString;
    fn bold(self) -> StyledString;
    fn dim(self) -> StyledString;
    fn green(self) -> StyledString;
    fn red(self) -> StyledString;
    fn yellow(self) -> StyledString;
    fn cyan(self) -> StyledString;
    fn blue(self) -> StyledString;
    fn magenta(self) -> StyledString;
}

impl<T: Display> Stylize for T {
    fn style(self, style: Style) -> StyledString {
        StyledString::new(self.to_string(), style)
    }

    fn fg(self, color: Color) -> StyledString {
        StyledString::new(self.to_string(), Style::new().fg(color))
    }

    fn bold(self) -> StyledString {
        StyledString::new(self.to_string(), Style::new().bold())
    }

    fn dim(self) -> StyledString {
        StyledString::new(self.to_string(), Style::new().dim())
    }

    fn green(self) -> StyledString {
        self.fg(Color::Green)
    }

    fn red(self) -> StyledString {
        self.fg(Color::Red)
    }

    fn yellow(self) -> StyledString {
        self.fg(Color::Yellow)
    }

    fn cyan(self) -> StyledString {
        self.fg(Color::Cyan)
    }

    fn blue(self) -> StyledString {
        self.fg(Color::Blue)
    }

    fn magenta(self) -> StyledString {
        self.fg(Color::Magenta)
    }
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

/// Horizontal alignment for a table column.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Alignment {
    #[default]
    Left,
    Center,
    Right,
}

/// A column specification for a [`Table`].
#[derive(Debug, Clone)]
pub struct Column {
    pub header: String,
    pub alignment: Alignment,
    pub min_width: usize,
    pub max_width: Option<usize>,
    pub style: Style,
    pub header_style: Style,
}

impl Column {
    /// Create a new column with a header string.
    pub fn new(header: impl Into<String>) -> Self {
        Self {
            header: header.into(),
            alignment: Alignment::Left,
            min_width: 0,
            max_width: None,
            style: Style::default(),
            header_style: Style::default().bold(),
        }
    }

    /// Set the column alignment.
    pub fn align(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set the minimum column width.
    pub fn min_width(mut self, width: usize) -> Self {
        self.min_width = width;
        self
    }

    /// Set the maximum column width (truncates with `…`).
    pub fn max_width(mut self, width: usize) -> Self {
        self.max_width = Some(width);
        self
    }

    /// Apply a style to all cells in this column.
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Apply a style to the header cell.
    pub fn header_style(mut self, style: Style) -> Self {
        self.header_style = style;
        self
    }
}

/// Border style for a table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TableBorder {
    /// No borders, just whitespace alignment.
    None,
    /// Only a horizontal rule below the header.
    #[default]
    Header,
    /// Full box-drawing borders around all cells.
    Box,
    /// Markdown-style with pipes and dashes.
    Markdown,
}

/// A table builder for rendering aligned tabular data.
pub struct Table {
    columns: Vec<Column>,
    rows: Vec<Vec<String>>,
    border: TableBorder,
    title: Option<String>,
    show_header: bool,
}

impl Table {
    /// Create a new empty table.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            border: TableBorder::Header,
            title: None,
            show_header: true,
        }
    }

    /// Create a table from a list of column headers.
    pub fn from_headers(headers: &[&str]) -> Self {
        let mut t = Self::new();
        for h in headers {
            t.columns.push(Column::new(*h));
        }
        t
    }

    /// Add a column to the table.
    pub fn column(mut self, col: Column) -> Self {
        self.columns.push(col);
        self
    }

    /// Add a row to the table. Cells are coerced to strings.
    pub fn row(mut self, cells: &[&str]) -> Self {
        let cells: Vec<String> = cells.iter().map(|s| s.to_string()).collect();
        self.rows.push(cells);
        self
    }

    /// Add a row from owned strings.
    pub fn row_owned(mut self, cells: Vec<String>) -> Self {
        self.rows.push(cells);
        self
    }

    /// Set the table border style.
    pub fn border(mut self, border: TableBorder) -> Self {
        self.border = border;
        self
    }

    /// Set a title displayed above the table.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Toggle header visibility.
    pub fn show_header(mut self, show: bool) -> Self {
        self.show_header = show;
        self
    }

    /// Compute the display width of each column.
    fn column_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self
            .columns
            .iter()
            .map(|c| display_width(&c.header).max(c.min_width))
            .collect();
        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    let w = display_width(cell);
                    if w > widths[i] {
                        widths[i] = w;
                    }
                }
            }
        }
        // Respect max_width.
        for (i, col) in self.columns.iter().enumerate() {
            if let Some(max) = col.max_width {
                if widths[i] > max {
                    widths[i] = max;
                }
            }
        }
        widths
    }

    /// Render the table to a string.
    pub fn render(&self) -> String {
        let mut out = String::new();

        if let Some(title) = &self.title {
            out.push_str(&format!("{}\n", title.bold()));
        }

        let widths = self.column_widths();
        if widths.is_empty() {
            return out;
        }

        // Header row.
        if self.show_header {
            match self.border {
                TableBorder::Box | TableBorder::Markdown => {
                    out.push_str(&self.horizontal_rule(&widths, BorderPosition::Top));
                }
                _ => {}
            }
            out.push_str(&self.render_header(&widths));
            match self.border {
                TableBorder::Header | TableBorder::Box | TableBorder::Markdown => {
                    out.push_str(&self.horizontal_rule(&widths, BorderPosition::Mid));
                }
                _ => {}
            }
        }

        // Data rows.
        for row in &self.rows {
            out.push_str(&self.render_row(row, &widths));
            if self.border == TableBorder::Box {
                out.push_str(&self.horizontal_rule(&widths, BorderPosition::Mid));
            }
        }

        if self.border == TableBorder::Box {
            out.push_str(&self.horizontal_rule(&widths, BorderPosition::Bottom));
        }

        out
    }

    /// Render the header row.
    fn render_header(&self, widths: &[usize]) -> String {
        let sep = match self.border {
            TableBorder::Markdown => " | ",
            TableBorder::Box => " │ ",
            _ => "  ",
        };
        let mut out = String::new();
        if self.border == TableBorder::Markdown || self.border == TableBorder::Box {
            out.push_str(sep.trim_start());
        }
        for (i, col) in self.columns.iter().enumerate() {
            if i > 0 {
                out.push_str(sep);
            }
            let cell = truncate(&col.header, widths[i]);
            out.push_str(&format!(
                "{}",
                StyledString::new(pad(&cell, widths[i], col.alignment), col.header_style)
            ));
        }
        if self.border == TableBorder::Markdown || self.border == TableBorder::Box {
            out.push_str(sep.trim_end());
        }
        out.push('\n');
        out
    }

    /// Render a single data row.
    fn render_row(&self, row: &[String], widths: &[usize]) -> String {
        let sep = match self.border {
            TableBorder::Markdown => " | ",
            TableBorder::Box => " │ ",
            _ => "  ",
        };
        let mut out = String::new();
        if self.border == TableBorder::Markdown || self.border == TableBorder::Box {
            out.push_str(sep.trim_start());
        }
        for i in 0..self.columns.len() {
            if i > 0 {
                out.push_str(sep);
            }
            let cell = row.get(i).map(|s| s.as_str()).unwrap_or("");
            let truncated = truncate(cell, widths[i]);
            let col = &self.columns[i];
            let padded = pad(&truncated, widths[i], col.alignment);
            out.push_str(&format!("{}", StyledString::new(padded, col.style)));
        }
        if self.border == TableBorder::Markdown || self.border == TableBorder::Box {
            out.push_str(sep.trim_end());
        }
        out.push('\n');
        out
    }

    /// Render a horizontal rule.
    fn horizontal_rule(&self, widths: &[usize], pos: BorderPosition) -> String {
        let (left, mid, right, fill) = match (&self.border, pos) {
            (TableBorder::Markdown, _) => ("|", "|", "|", "-"),
            (TableBorder::Box, BorderPosition::Top) => ("┌", "┬", "┐", "─"),
            (TableBorder::Box, BorderPosition::Mid) => ("├", "┼", "┤", "─"),
            (TableBorder::Box, BorderPosition::Bottom) => ("└", "┴", "┘", "─"),
            _ => ("", "", "", "-"),
        };
        let mut out = String::new();
        out.push_str(left);
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                out.push_str(mid);
            }
            for _ in 0..(*w + 2) {
                out.push_str(fill);
            }
        }
        out.push_str(right);
        out.push('\n');
        out
    }

    /// Print the table to stdout.
    pub fn print(&self) {
        print!("{}", self.render());
    }
}

impl Default for Table {
    fn default() -> Self {
        Self::new()
    }
}

/// Position of a horizontal rule relative to table content.
#[derive(Debug, Clone, Copy)]
enum BorderPosition {
    Top,
    Mid,
    Bottom,
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/// Pad or truncate a string to exactly `width` display columns.
fn pad(s: &str, width: usize, alignment: Alignment) -> String {
    let len = display_width(s);
    if len >= width {
        return s.to_string();
    }
    let pad = width - len;
    match alignment {
        Alignment::Left => format!("{s}{}", " ".repeat(pad)),
        Alignment::Right => format!("{}{s}", " ".repeat(pad)),
        Alignment::Center => {
            let left = pad / 2;
            let right = pad - left;
            format!("{}{s}{}", " ".repeat(left), " ".repeat(right))
        }
    }
}

/// Truncate a string to `width` display columns, appending `…` if truncated.
fn truncate(s: &str, width: usize) -> String {
    let len = display_width(s);
    if len <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_string();
    }
    // Reserve one column for the ellipsis.
    let target = width - 1;
    let mut result = String::new();
    let mut current = 0usize;
    for ch in s.chars() {
        let w = char_width(ch);
        if current + w > target {
            break;
        }
        result.push(ch);
        current += w;
    }
    result.push('…');
    result
}

/// Compute the display width of a string, accounting for wide CJK characters.
fn display_width(s: &str) -> usize {
    s.chars().map(char_width).sum()
}

/// Return the display width of a single character.
fn char_width(c: char) -> usize {
    let code = c as u32;
    // Zero-width characters.
    if code == 0 {
        return 0;
    }
    // Control characters take no width.
    if code < 0x20 || (0x7f..0xa0).contains(&code) {
        return 0;
    }
    // CJK and fullwidth ranges (simplified).
    let wide = (0x1100..=0x115f).contains(&code) // Hangul Jamo
        || (0x2e80..=0x303e).contains(&code) // CJK Radicals
        || (0x3041..=0x33ff).contains(&code) // Hiragana, Katakana, CJK
        || (0x3400..=0x4dbf).contains(&code) // CJK Ext A
        || (0x4e00..=0x9fff).contains(&code) // CJK Unified
        || (0xa000..=0xa4cf).contains(&code) // Yi
        || (0xac00..=0xd7a3).contains(&code) // Hangul Syllables
        || (0xf900..=0xfaff).contains(&code) // CJK Compat
        || (0xfe30..=0xfe4f).contains(&code) // CJK Compat Forms
        || (0xff00..=0xff60).contains(&code) // Fullwidth Forms
        || (0xffe0..=0xffe6).contains(&code)
        || (0x1f300..=0x1faff).contains(&code); // Emoji
    if wide { 2 } else { 1 }
}

// ---------------------------------------------------------------------------
// Panels and rules
// ---------------------------------------------------------------------------

/// Draw a horizontal rule across the terminal width (or 80 columns).
pub fn rule() {
    let width = terminal_width().min(80);
    println!("{}", "─".repeat(width).dim());
}

/// Print a titled section header.
pub fn section(title: &str) {
    println!();
    println!("{}", title.bold());
    rule();
}

/// Print a message inside a simple panel.
pub fn panel(title: &str, body: &str) {
    let width = terminal_width().min(80).max(40);
    let inner = width.saturating_sub(4);
    println!("┌{}┐", "─".repeat(width - 2));
    if !title.is_empty() {
        println!("│ {} │", center(title, inner));
        println!("├{}┤", "─".repeat(width - 2));
    }
    for line in body.lines() {
        for chunk in wrap_text(line, inner) {
            println!("│ {:<inner$} │", chunk, inner = inner);
        }
    }
    println!("└{}┘", "─".repeat(width - 2));
}

/// Center a string in a fixed width.
fn center(s: &str, width: usize) -> String {
    let len = display_width(s);
    if len >= width {
        return s.to_string();
    }
    let left = (width - len) / 2;
    let right = width - len - left;
    format!("{}{s}{}", " ".repeat(left), " ".repeat(right))
}

/// Wrap text to a maximum column width.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }
    let mut result = Vec::new();
    let mut current = String::new();
    let mut current_len = 0usize;
    for word in text.split_whitespace() {
        let word_len = display_width(word);
        if current_len + word_len + 1 > width && !current.is_empty() {
            result.push(current.clone());
            current.clear();
            current_len = 0;
        }
        if !current.is_empty() {
            current.push(' ');
            current_len += 1;
        }
        current.push_str(word);
        current_len += word_len;
    }
    if !current.is_empty() {
        result.push(current);
    }
    if result.is_empty() {
        vec![String::new()]
    } else {
        result
    }
}

/// Get the terminal width, defaulting to 80.
fn terminal_width() -> usize {
    // Use the COLUMNS env var if set; otherwise default to 80.
    std::env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80)
}

// ---------------------------------------------------------------------------
// Key-value pairs
// ---------------------------------------------------------------------------

/// A builder for rendering a list of key-value pairs.
pub struct KeyValue {
    entries: Vec<(String, String, Style)>,
    key_width: usize,
}

impl KeyValue {
    /// Create a new empty key-value list.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            key_width: 0,
        }
    }

    /// Add a key-value pair with the default value style.
    pub fn entry(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        let key = key.into();
        if key.len() > self.key_width {
            self.key_width = key.len();
        }
        self.entries.push((key, value.into(), Style::default()));
        self
    }

    /// Add a key-value pair with a custom value style.
    pub fn entry_styled(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
        style: Style,
    ) -> Self {
        let key = key.into();
        if key.len() > self.key_width {
            self.key_width = key.len();
        }
        self.entries.push((key, value.into(), style));
        self
    }

    /// Render the key-value list to a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for (key, value, style) in &self.entries {
            out.push_str(&format!(
                "  {:>key_w$}: {}{}\n",
                key,
                StyledString::new(value, *style),
                "",
                key_w = self.key_width
            ));
        }
        out
    }

    /// Print the key-value list.
    pub fn print(&self) {
        print!("{}", self.render());
    }
}

impl Default for KeyValue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Status indicators
// ---------------------------------------------------------------------------

/// Render a green check mark.
pub fn ok() -> StyledString {
    StyledString::new("✓", Style::new().fg(Color::Green).bold())
}

/// Render a red cross.
pub fn fail() -> StyledString {
    StyledString::new("✗", Style::new().fg(Color::Red).bold())
}

/// Render a yellow warning.
pub fn warn() -> StyledString {
    StyledString::new("⚠", Style::new().fg(Color::Yellow).bold())
}

/// Render an info bullet.
pub fn info() -> StyledString {
    StyledString::new("•", Style::new().fg(Color::Blue).bold())
}

/// Render a status badge from a boolean.
pub fn status_badge(success: bool) -> StyledString {
    if success { ok() } else { fail() }
}

/// Render a health status as a colored label.
pub fn health_label(status: &str) -> StyledString {
    let style = match status.to_lowercase().as_str() {
        "healthy" | "ok" | "active" | "running" | "connected" => {
            Style::new().fg(Color::Green).bold()
        }
        "degraded" | "paused" | "warning" | "slow" => Style::new().fg(Color::Yellow).bold(),
        "unhealthy" | "error" | "failed" | "killed" | "stopped" => {
            Style::new().fg(Color::Red).bold()
        }
        "disabled" | "idle" | "archived" => Style::new().fg(Color::Gray),
        _ => Style::default(),
    };
    StyledString::new(status, style)
}

// ---------------------------------------------------------------------------
// Progress bar
// ---------------------------------------------------------------------------

/// A simple progress bar that renders to stdout.
pub struct ProgressBar {
    total: u64,
    current: u64,
    width: usize,
    label: String,
}

impl ProgressBar {
    /// Create a new progress bar.
    pub fn new(total: u64, label: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            width: 40,
            label: label.into(),
        }
    }

    /// Set the bar display width.
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Advance the progress by `n` steps.
    pub fn advance(&mut self, n: u64) {
        self.current = (self.current + n).min(self.total);
        self.render_line();
    }

    /// Set the current progress.
    pub fn set(&mut self, current: u64) {
        self.current = current.min(self.total);
        self.render_line();
    }

    /// Finish the progress bar.
    pub fn finish(&mut self) {
        self.current = self.total;
        self.render_line();
        println!();
    }

    fn render_line(&self) {
        let pct = if self.total > 0 {
            self.current as f64 / self.total as f64
        } else {
            0.0
        };
        let filled = (pct * self.width as f64) as usize;
        let bar: String = "█".repeat(filled);
        let empty: String = "░".repeat(self.width - filled);
        let pct_str = format!("{:>5.1}%", pct * 100.0);
        print!(
            "\r{} [{}{}] {}/{} {}",
            self.label, bar, empty, self.current, self.total, pct_str
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// A node in a tree listing.
pub struct TreeNode {
    label: String,
    children: Vec<TreeNode>,
    style: Style,
}

impl TreeNode {
    /// Create a new tree node.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            children: Vec::new(),
            style: Style::default(),
        }
    }

    /// Apply a style to this node's label.
    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    /// Add a child node.
    pub fn child(mut self, node: TreeNode) -> Self {
        self.children.push(node);
        self
    }

    /// Add a simple leaf child.
    pub fn leaf(mut self, label: impl Into<String>) -> Self {
        self.children.push(TreeNode::new(label));
        self
    }

    /// Render the tree to a string.
    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_into(&mut out, "", true);
        out
    }

    fn render_into(&self, out: &mut String, prefix: &str, is_last: bool) {
        let connector = if is_last { "└── " } else { "├── " };
        out.push_str(prefix);
        out.push_str(connector);
        out.push_str(&format!("{}", StyledString::new(&self.label, self.style)));
        out.push('\n');
        let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });
        for (i, child) in self.children.iter().enumerate() {
            let last = i == self.children.len() - 1;
            child.render_into(out, &child_prefix, last);
        }
    }

    /// Print the tree.
    pub fn print(&self) {
        print!("{}", self.render());
    }
}

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------

/// A simple text spinner for indeterminate progress.
pub struct Spinner {
    frames: &'static [&'static str],
    index: usize,
    label: String,
}

impl Spinner {
    /// Create a new spinner with a label.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            frames: &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            index: 0,
            label: label.into(),
        }
    }

    /// Advance the spinner by one frame and print.
    pub fn tick(&mut self) {
        let frame = self.frames[self.index % self.frames.len()];
        self.index += 1;
        print!("\r{} {}", frame, self.label);
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    /// Stop the spinner and print a completion message.
    pub fn finish(&mut self, msg: &str) {
        print!("\r{}  \n", msg);
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pad_left() {
        assert_eq!(pad("hi", 5, Alignment::Left), "hi   ");
    }

    #[test]
    fn test_pad_right() {
        assert_eq!(pad("hi", 5, Alignment::Right), "   hi");
    }

    #[test]
    fn test_pad_center() {
        assert_eq!(pad("hi", 6, Alignment::Center), "  hi  ");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 3), "he…");
        assert_eq!(truncate("hello", 1), "…");
    }

    #[test]
    fn test_display_width() {
        assert_eq!(display_width("abc"), 3);
        assert_eq!(display_width("你好"), 4);
    }

    #[test]
    fn test_table_basic() {
        let table = Table::from_headers(&["Name", "Age"])
            .row(&["Alice", "30"])
            .row(&["Bob", "25"]);
        let rendered = table.render();
        assert!(rendered.contains("Alice"));
        assert!(rendered.contains("Name"));
    }

    #[test]
    fn test_keyvalue() {
        let kv = KeyValue::new().entry("name", "Alice").entry("age", "30");
        let rendered = kv.render();
        assert!(rendered.contains("Alice"));
        assert!(rendered.contains("name"));
    }

    #[test]
    fn test_tree() {
        let tree = TreeNode::new("root").leaf("child1").leaf("child2");
        let rendered = tree.render();
        assert!(rendered.contains("root"));
        assert!(rendered.contains("child1"));
    }

    #[test]
    fn test_styled_string_plain() {
        let s = StyledString::plain("hello");
        // Without color, render is just the text.
        assert_eq!(s.render(), "hello");
    }
}
