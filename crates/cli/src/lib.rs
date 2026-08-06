//! # OpenSquilla CLI
//!
//! Command-line interface and TUI for the OpenSquilla agent runtime.
//! Built with clap for argument parsing and ratatui for TUI rendering.
//!
//! ## Command Groups
//!
//! | Command     | Description                          | Mode |
//! |-------------|--------------------------------------|------|
//! | `chat`      | Interactive chat REPL or one-shot    | A/B  |
//! | `agent`     | Autonomous agent task execution      | B    |
//! | `config`    | Configuration management             | B    |
//! | `providers` | Provider listing and testing         | B    |
//! | `session`   | Session CRUD and transcripts         | A    |
//! | `models`    | Model catalog and capabilities       | B    |
//! | `memory`    | Memory store and dream consolidation | B    |
//! | `skills`    | Skill hub: install, search, list     | B    |
//! | `sandbox`   | Sandbox policy and execution         | B    |
//! | `channels`  | Channel manager status               | B    |
//! | `scheduler` | Cron and interval task scheduling    | B    |
//! | `doctor`    | Health and diagnostics               | B    |
//! | `gateway`   | Gateway lifecycle management         | C    |
//! | `cost`      | Usage and cost reporting             | A    |
//! | `onboard`   | Interactive setup wizard             | B    |
//! | `router`    | Model routing calibration            | B    |
//! | `init`      | Project scaffolding                  | B    |
//! | `tui`       | Terminal UI                          | B    |

pub mod agent;
pub mod channels;
pub mod chat;
pub mod commands;
pub mod config;
pub mod cost;
pub mod doctor;
pub mod gateway;
pub mod init;
pub mod memory;
pub mod models;
pub mod onboard;
pub mod providers;
pub mod router;
pub mod rpc;
pub mod sandbox;
pub mod scheduler;
pub mod search;
pub mod sessions;
pub mod skills;
pub mod status;
pub mod table;
pub mod tools;
pub mod tui;
pub mod util;

pub use chat::run_chat;
pub use commands::Cli;
pub use tui::{run_tui, TuiApp};
