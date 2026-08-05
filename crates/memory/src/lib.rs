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

pub mod dream;
pub mod embedding;
pub mod manager;
pub mod profile_import;
pub mod retrieval;
pub mod session_source;
pub mod store;
pub mod sync;
pub mod turn_capture;
pub mod types;

pub use dream::{DreamConfig, DreamConsolidator, DreamEngine, DreamEvent, DreamSummary};
pub use embedding::{EmbeddingConfig, EmbeddingProvider};
pub use manager::MemoryManager;
pub use profile_import::{
    ConfigType, ExtractedMemory, ImportPlan, ImportResult, ImportSource, ImportSummary,
    ProfileDetector, ProfileImporter, ProfileImporterConfig,
};
pub use retrieval::RetrievalEngine;
pub use session_source::{
    MemoryDocument, SessionMemoryDoc, SessionMemorySource, SessionSource, SessionSourceConfig,
};
pub use store::MemoryStore;
pub use sync::{FileWatcher, SyncConfig, SyncManager, SyncStats};
pub use turn_capture::{TurnCapture, TurnCaptureConfig, TurnCaptureStats, TurnData, TurnSignals};

// Re-export the canonical types at the crate root for convenience.
pub use types::{MemoryChunk, MemoryEntry, MemoryFilters, MemoryQuery, MemorySearchResult};
