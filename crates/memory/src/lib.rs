//! Memory subsystem: durable, searchable agent memories.
//!
//! Modules:
//! - [`types`]         — canonical memory domain types
//! - [`store`]         — SQLite-backed storage (FTS5 + sqlite-vec) and CRUD
//! - [`embedding`]     — embedding providers (OpenAI / Ollama / ONNX) + cache
//! - [`retrieval`]     — hybrid search (vector + BM25 + time decay + MMR)
//! - [`manager`]       — per-agent memory lifecycle orchestration
//! - [`sync`]          — file-system sync into the memory store
//! - [`dream`]         — periodic memory consolidation into abstractions
//! - [`turn_capture`]  — turn-level incremental persistence
//! - [`session_source`]— session-derived memory documents
//! - [`profile_import`]— external config import (JSON/YAML/TOML + LLM)

pub mod types;
pub mod store;
pub mod embedding;
pub mod retrieval;
pub mod manager;
pub mod sync;
pub mod dream;
pub mod turn_capture;
pub mod session_source;
pub mod profile_import;

pub use store::MemoryStore;
pub use embedding::{EmbeddingProvider, EmbeddingConfig};
pub use retrieval::RetrievalEngine;
pub use manager::MemoryManager;
pub use sync::{SyncManager, SyncConfig, SyncStats, FileWatcher};
pub use dream::{DreamEngine, DreamConfig, DreamEvent, DreamSummary, DreamConsolidator};
pub use turn_capture::{
    TurnCapture, TurnCaptureConfig, TurnSignals, TurnData, TurnCaptureStats,
};
pub use session_source::{
    SessionSource, SessionSourceConfig, SessionMemoryDoc, SessionMemorySource, MemoryDocument,
};
pub use profile_import::{
    ProfileImporter, ProfileImporterConfig, ImportSource, ExtractedMemory, ImportSummary,
    ImportResult, ImportPlan, ConfigType, ProfileDetector,
};

// Re-export the canonical types at the crate root for convenience.
pub use types::{MemoryEntry, MemoryChunk, MemoryQuery, MemorySearchResult, MemoryFilters};
