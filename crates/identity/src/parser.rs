/// A parsed terminal command line.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ParsedCommand {
    /// The raw input line.
    pub raw: String,
    /// Command name (first whitespace-delimited token).
    pub command: String,
    /// Remaining arguments after the command name.
    pub args: Vec<String>,
    /// Whether the line is a slash command (starts with `/`).
    pub is_slash: bool,
}

impl ParsedCommand {
    /// Parse a raw input line into a command and its arguments.
    pub fn parse(line: &str) -> Self {
        let raw = line.trim().to_string();
        let mut parts = raw.split_whitespace();
        let command = parts.next().unwrap_or("").to_string();
        let args = parts.map(|s| s.to_string()).collect::<Vec<_>>();
        let is_slash = command.starts_with('/');
        Self {
            raw,
            command,
            args,
            is_slash,
        }
    }

    /// The full argument list joined back together.
    pub fn arg_str(&self) -> String {
        self.args.join(" ")
    }

    /// Whether this command is empty (blank line).
    pub fn is_empty(&self) -> bool {
        self.command.is_empty()
    }
}
