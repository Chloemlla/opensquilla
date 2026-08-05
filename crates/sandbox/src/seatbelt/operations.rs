//! Seatbelt operation reference.
//!
//! A curated catalogue of Seatbelt (SBPL) operation names grouped by
//! category. This is documentation data — it does not affect compilation,
//! but it lets callers introspect what operations the generated profile
//! references, and it powers the profile summary used in audit logs.

use serde::{Deserialize, Serialize};

/// A category of Seatbelt operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatbeltCategory {
    File,
    Network,
    Process,
    Mach,
    Ipc,
    Sysctl,
    Iokit,
    Signal,
    IpcPosix,
}

impl SeatbeltCategory {
    /// Human-readable label.
    pub fn as_str(self) -> &'static str {
        match self {
            SeatbeltCategory::File => "file",
            SeatbeltCategory::Network => "network",
            SeatbeltCategory::Process => "process",
            SeatbeltCategory::Mach => "mach",
            SeatbeltCategory::Ipc => "ipc",
            SeatbeltCategory::Sysctl => "sysctl",
            SeatbeltCategory::Iokit => "iokit",
            SeatbeltCategory::Signal => "signal",
            SeatbeltCategory::IpcPosix => "ipc_posix",
        }
    }
}

/// A Seatbelt operation with its category and a short description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeatbeltOperation {
    /// The operation name as it appears in SBPL (e.g. `"file-read*"`).
    pub name: &'static str,
    /// The category.
    pub category: SeatbeltCategory,
    /// Short description.
    pub description: &'static str,
}

/// The catalogue of known Seatbelt operations.
pub static OPERATIONS: &[SeatbeltOperation] = &[
    SeatbeltOperation {
        name: "file-read*",
        category: SeatbeltCategory::File,
        description: "Read from a file or file descriptor.",
    },
    SeatbeltOperation {
        name: "file-write*",
        category: SeatbeltCategory::File,
        description: "Write to a file or file descriptor.",
    },
    SeatbeltOperation {
        name: "file-read-metadata",
        category: SeatbeltCategory::File,
        description: "Read file metadata (stat, access).",
    },
    SeatbeltOperation {
        name: "file-read-data",
        category: SeatbeltCategory::File,
        description: "Read file data (read, pread).",
    },
    SeatbeltOperation {
        name: "file-write-data",
        category: SeatbeltCategory::File,
        description: "Write file data (write, pwrite).",
    },
    SeatbeltOperation {
        name: "network*",
        category: SeatbeltCategory::Network,
        description: "Any network operation (socket, connect, bind, ...).",
    },
    SeatbeltOperation {
        name: "network-outbound",
        category: SeatbeltCategory::Network,
        description: "Outbound network connection (connect).",
    },
    SeatbeltOperation {
        name: "network-inbound",
        category: SeatbeltCategory::Network,
        description: "Inbound network operation (accept).",
    },
    SeatbeltOperation {
        name: "network-bind",
        category: SeatbeltCategory::Network,
        description: "Bind a socket to a local address.",
    },
    SeatbeltOperation {
        name: "process-fork",
        category: SeatbeltCategory::Process,
        description: "Fork a new process.",
    },
    SeatbeltOperation {
        name: "process-exec",
        category: SeatbeltCategory::Process,
        description: "Execute a new program (execve).",
    },
    SeatbeltOperation {
        name: "process-info*",
        category: SeatbeltCategory::Process,
        description: "Query process information (ps, getrusage).",
    },
    SeatbeltOperation {
        name: "signal",
        category: SeatbeltCategory::Signal,
        description: "Send a signal to a process.",
    },
    SeatbeltOperation {
        name: "mach-task-self",
        category: SeatbeltCategory::Mach,
        description: "Access to the process's own Mach task port.",
    },
    SeatbeltOperation {
        name: "mach-privilege-task-port",
        category: SeatbeltCategory::Mach,
        description: "Access to privileged Mach task ports.",
    },
    SeatbeltOperation {
        name: "mach-lookup",
        category: SeatbeltCategory::Mach,
        description: "Look up a Mach service by name.",
    },
    SeatbeltOperation {
        name: "ipc-posix-semaphore*",
        category: SeatbeltCategory::IpcPosix,
        description: "POSIX semaphore operations.",
    },
    SeatbeltOperation {
        name: "ipc-posix-shm*",
        category: SeatbeltCategory::IpcPosix,
        description: "POSIX shared memory operations.",
    },
    SeatbeltOperation {
        name: "ipc-sysv",
        category: SeatbeltCategory::Ipc,
        description: "System V IPC (msgget, semget, shmget).",
    },
    SeatbeltOperation {
        name: "sysctl-read",
        category: SeatbeltCategory::Sysctl,
        description: "Read a sysctl variable.",
    },
    SeatbeltOperation {
        name: "sysctl*",
        category: SeatbeltCategory::Sysctl,
        description: "Any sysctl operation (read or write).",
    },
    SeatbeltOperation {
        name: "iokit*",
        category: SeatbeltCategory::Iokit,
        description: "Any IOKit operation.",
    },
];

/// Look up an operation by name.
pub fn lookup(name: &str) -> Option<&'static SeatbeltOperation> {
    OPERATIONS.iter().find(|op| op.name == name)
}

/// List all operation names in a category.
pub fn by_category(category: SeatbeltCategory) -> Vec<&'static str> {
    OPERATIONS
        .iter()
        .filter(|op| op.category == category)
        .map(|op| op.name)
        .collect()
}

/// Summarise which operations a compiled profile references.
pub fn summarise_operations(source: &str) -> Vec<&'static SeatbeltOperation> {
    OPERATIONS
        .iter()
        .filter(|op| source.contains(op.name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_works() {
        assert!(lookup("file-read*").is_some());
        assert!(lookup("network*").is_some());
        assert!(lookup("nonexistent").is_none());
    }

    #[test]
    fn by_category_returns_members() {
        let file_ops = by_category(SeatbeltCategory::File);
        assert!(file_ops.contains(&"file-read*"));
        assert!(file_ops.contains(&"file-write*"));
    }
}
