//! Memory management commands.
//!
//! Implements the `memory` subcommand against the memory crate's [`MemoryStore`]
//! (SQLite with FTS5). Supports listing, showing, deleting, full-text search,
//! a consistency check, and dream-consolidation.

use anyhow::{Context, Result};
use opensquilla_core::config::Config;
use opensquilla_core::types::MemoryId;
use opensquilla_memory::DreamEngine;
use opensquilla_memory::store::MemoryEntry;
use opensquilla_memory::store::MemoryStore;
use tracing::info;
use uuid::Uuid;

use crate::util;

/// Open the memory store at the default path, creating it if needed.
fn open_store() -> Result<MemoryStore> {
    let path = util::memory_db_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).ok();
    }
    MemoryStore::new(&path.to_string_lossy())
        .map_err(|e| anyhow::anyhow!("Failed to open memory store at {}: {e}", path.display()))
}

/// List memory entries, optionally filtered by session.
pub async fn list_memory(session: Option<String>) -> Result<()> {
    list_memory_filtered(session, None, 100).await
}

/// List memory entries with session, type, and limit filters.
pub async fn list_memory_filtered(
    session: Option<String>,
    kind: Option<String>,
    limit: u64,
) -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();

    let entries = if let Some(sid) = session {
        store
            .search_fts(&sid, limit, 0)
            .map_err(|e| anyhow::anyhow!("Failed to search memory: {e}"))?
    } else {
        store
            .list_memories(&agent_id, kind.as_deref(), limit, 0)
            .map_err(|e| anyhow::anyhow!("Failed to list memory: {e}"))?
    };

    if entries.is_empty() {
        println!("No memory entries found.");
        return Ok(());
    }

    println!("Memory entries ({}):", entries.len());
    println!("{:-<90}", "");
    for entry in &entries {
        let summary: String = entry.content.chars().take(60).collect();
        println!(
            "  [{}] {}  (type: {}, importance: {:.2})",
            entry.id.0, summary, entry.memory_type, entry.importance
        );
    }
    println!("{:-<90}", "");
    Ok(())
}

/// Show a single memory entry in full.
pub async fn show_memory(id: String) -> Result<()> {
    let store = open_store()?;
    let entry = load_entry(&store, &id)?;

    println!("Memory Entry: {}", entry.id.0);
    println!("  Type:       {}", entry.memory_type);
    println!("  Source:     {}", entry.source);
    println!("  Importance: {:.2}", entry.importance);
    println!("  Created:    {}", entry.created_at.to_rfc3339());
    println!("  Updated:    {}", entry.updated_at.to_rfc3339());
    println!("  Access:     {} time(s)", entry.access_count);
    println!();
    println!("{}", entry.content);
    Ok(())
}

/// Delete a single memory entry.
pub async fn delete_memory(id: String) -> Result<()> {
    let store = open_store()?;
    let memory_id = parse_memory_id(&id)?;
    if store
        .get_memory(&memory_id)
        .map_err(|e| anyhow::anyhow!("Failed to load memory: {e}"))?
        .is_none()
    {
        anyhow::bail!("Memory entry '{id}' not found");
    }
    store
        .delete_memory(&memory_id)
        .map_err(|e| anyhow::anyhow!("Failed to delete memory: {e}"))?;
    println!("Memory entry {id} deleted.");
    Ok(())
}

/// Delete every memory entry for the default agent.
pub async fn clear_memory() -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();
    let entries = store
        .list_memories(&agent_id, None, u64::MAX, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list memory: {e}"))?;
    for entry in &entries {
        store
            .delete_memory(&entry.id)
            .map_err(|e| anyhow::anyhow!("Failed to delete memory {}: {e}", entry.id.0))?;
    }
    println!("Cleared {} memory entries.", entries.len());
    Ok(())
}

/// Full-text search across memory entries.
pub async fn search_memory(query: String) -> Result<()> {
    let store = open_store()?;
    let results = store
        .search_fts(&query, 20, 0)
        .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;

    if results.is_empty() {
        println!("No memory entries matched '{query}'.");
        return Ok(());
    }

    println!("Search results for '{query}':");
    println!("{:-<90}", "");
    for entry in &results {
        let snippet: String = entry.content.chars().take(80).collect();
        println!("  [{}] {}  ({:.2})", entry.id.0, snippet, entry.importance);
    }
    println!("{:-<90}", "");
    println!("{} result(s)", results.len());
    Ok(())
}

/// Run a memory consistency check (counts, embeddings, orphans).
pub async fn memory_check() -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();

    let total = store
        .list_memories(&agent_id, None, u64::MAX, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list memory: {e}"))?
        .len();
    let embeddings = store
        .get_all_embeddings(&agent_id)
        .map_err(|e| anyhow::anyhow!("Failed to list embeddings: {e}"))?;

    let mut orphans = 0u64;
    for (mid, _) in &embeddings {
        if store
            .get_memory(mid)
            .map_err(|e| anyhow::anyhow!("Failed to load memory: {e}"))?
            .is_none()
        {
            orphans += 1;
        }
    }

    println!("Memory check");
    println!("{:-<40}", "");
    println!("  Total memories:   {total}");
    println!("  With embeddings:  {}", embeddings.len());
    println!("  Orphaned embeddings: {orphans}");
    println!(
        "  Status:           {}",
        if orphans == 0 {
            "healthy"
        } else {
            "needs repair"
        }
    );
    println!("{:-<40}", "");
    Ok(())
}

/// Trigger dream consolidation on the memory corpus.
pub async fn memory_dream() -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();

    let dream = DreamEngine::new(store);
    let summary = dream
        .consolidate(&agent_id)
        .map_err(|e| anyhow::anyhow!("Dream consolidation failed: {e}"))?;

    println!("Dream consolidation complete");
    println!("{:-<50}", "");
    println!("  Total memories:    {}", summary.total_memories);
    println!("  Consolidated:      {}", summary.consolidated);
    println!("  Patterns found:    {}", summary.patterns_found);
    println!("  Abstractions:      {}", summary.abstractions_created);
    println!("  At:                {}", summary.timestamp.to_rfc3339());
    println!("{:-<50}", "");
    info!(
        "Dream consolidation complete for agent {}",
        summary.agent_id
    );
    Ok(())
}

fn parse_memory_id(id: &str) -> Result<MemoryId> {
    let uuid = Uuid::parse_str(id).map_err(|_| anyhow::anyhow!("Invalid memory id: {id}"))?;
    Ok(MemoryId(uuid))
}

fn load_entry(store: &MemoryStore, id: &str) -> Result<MemoryEntry> {
    let memory_id = parse_memory_id(id)?;
    store
        .get_memory(&memory_id)
        .map_err(|e| anyhow::anyhow!("Failed to load memory: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Memory entry '{id}' not found"))
}

/// Full-text search across memory entries with a configurable limit.
pub async fn search_memory_limited(query: String, limit: u64) -> Result<()> {
    let store = open_store()?;
    let results = store
        .search_fts(&query, limit, 0)
        .map_err(|e| anyhow::anyhow!("Search failed: {e}"))?;

    if results.is_empty() {
        println!("No memory entries matched '{query}'.");
        return Ok(());
    }

    println!("Search results for '{query}':");
    println!("{:-<90}", "");
    for entry in &results {
        let snippet: String = entry.content.chars().take(80).collect();
        println!("  [{}] {}  ({:.2})", entry.id.0, snippet, entry.importance);
    }
    println!("{:-<90}", "");
    println!("{} result(s)", results.len());
    Ok(())
}

/// Add a memory entry manually.
pub async fn add_memory(
    content: String,
    kind: Option<String>,
    importance: Option<f64>,
    tags: Vec<String>,
) -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();
    let memory_id = MemoryId::new();
    let memory_type = kind.unwrap_or_else(|| "note".to_string());
    let importance = importance.unwrap_or(0.5);

    let entry = MemoryEntry::new(
        memory_id,
        agent_id,
        content.clone(),
        "cli".to_string(),
        memory_type.clone(),
        importance,
        serde_json::Value::Null,
    );
    let entry = entry.with_tags(tags);

    store
        .update_memory(&entry)
        .map_err(|e| anyhow::anyhow!("Failed to add memory: {e}"))?;

    println!("{} Added memory entry: {}", crate::table::ok(), entry.id.0);
    println!("  Type:       {memory_type}");
    println!("  Importance: {importance:.2}");
    println!(
        "  Content:    {}",
        content.chars().take(80).collect::<String>()
    );
    Ok(())
}

/// Flush a session transcript into durable memory.
///
/// Minimal local flush: reads the session's transcript and writes each message
/// as a searchable durable memory entry (source `"flush"`). A full
/// `SessionFlushService` (LLM summarization, segmentation, cost accounting) is
/// a gateway-side concern not yet wired into the CLI; this keeps the command
/// present and runnable for automation workflows.
pub async fn flush_session(key: String, output: Option<String>) -> Result<()> {
    let config = Config::load().context("Failed to load configuration")?;
    let manager = util::build_session_manager(&config)?;
    let uid = Uuid::parse_str(&key)
        .map_err(|_| anyhow::anyhow!("Invalid session id: {key}"))?;
    let session = manager
        .get_session(&uid)
        .map_err(|e| anyhow::anyhow!("Failed to load session: {e}"))?
        .ok_or_else(|| anyhow::anyhow!("Session '{key}' not found"))?;

    let transcript = manager
        .get_transcript(&session.id, 10000, 0)
        .map_err(|e| anyhow::anyhow!("Failed to load transcript: {e}"))?;

    let store = open_store()?;
    let agent_id = util::default_agent_id();
    let mut flushed = 0u64;
    let mut chars = 0usize;
    for entry in &transcript {
        let content = format!("[{}] {}", entry.role, entry.content);
        let memory_id = MemoryId::new();
        let mem = MemoryEntry::new(
            memory_id,
            agent_id,
            content.clone(),
            "flush".to_string(),
            "episodic".to_string(),
            0.5,
            serde_json::Value::Null,
        )
        .with_tags(vec![format!("session:{}", session.id), key.clone()]);
        store
            .update_memory(&mem)
            .map_err(|e| anyhow::anyhow!("Failed to flush message: {e}"))?;
        flushed += 1;
        chars += content.len();
    }

    let receipt = serde_json::json!({
        "ok": true,
        "mode": "raw",
        "agent_id": agent_id.to_string(),
        "session_key": key,
        "flushed_count": flushed,
        "flushed_chars": chars,
        "flushed_paths": [],
    });

    if let Some(path) = output {
        let payload = serde_json::json!({
            "ok": true,
            "key": key,
            "agent_id": agent_id.to_string(),
            "flush_receipt": receipt,
            "flushed_count": flushed,
        });
        let json = util::to_pretty_json(&payload)?;
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(&path, json).with_context(|| format!("Failed to write {path}"))?;
    }

    println!("Session flushed: {key}");
    println!("  Agent:             {}", agent_id);
    println!("  Messages flushed:  {flushed}");
    println!("  Chars:             {chars}");
    println!("  Flush mode:        raw (non-searchable fallback)");
    Ok(())
}

/// Export memory entries to a JSON file.
pub async fn export_memory(output: String, kind: Option<String>) -> Result<()> {
    let store = open_store()?;
    let agent_id = util::default_agent_id();
    let entries = store
        .list_memories(&agent_id, kind.as_deref(), u64::MAX, 0)
        .map_err(|e| anyhow::anyhow!("Failed to list memory: {e}"))?;

    let export = serde_json::json!({
        "agent_id": agent_id.to_string(),
        "exported_at": chrono::Utc::now().to_rfc3339(),
        "count": entries.len(),
        "entries": entries,
    });

    let json = serde_json::to_string_pretty(&export)
        .map_err(|e| anyhow::anyhow!("Failed to serialize: {e}"))?;

    if let Some(dir) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&output, json).with_context(|| format!("Failed to write {output}"))?;
    println!("Exported {} memory entries to {output}", entries.len());
    Ok(())
}
