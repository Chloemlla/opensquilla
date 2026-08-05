//! # OpenSquilla CLI
//!
//! Command-line interface and TUI for the OpenSquilla agent runtime.
//! Built with clap for argument parsing and ratatui for TUI rendering.

pub mod channels;
pub mod chat;
pub mod commands;
pub mod config;
pub mod doctor;
pub mod gateway;
pub mod memory;
pub mod models;
pub mod providers;
pub mod sandbox;
pub mod scheduler;
pub mod sessions;
pub mod skills;
pub mod tui;
pub mod util;

pub use chat::run_chat;
pub use commands::Cli;
pub use tui::{run_tui, TuiApp};
