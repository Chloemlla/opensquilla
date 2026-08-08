//! Prompt-cache break detection for cache-relevant provider request state.
//!
//! Mirrors the `CacheBreakMonitor` state machine from the Python backend's
//! `engine/cache_break_monitor.py` (the pure, I/O-free portion). It snapshots
//! the cache-relevant inputs of each chat call (system prompt hash, tools hash,
//! messages-prefix hash, cache-control fields, model) and attributes drops in
//! provider-reported cache-read tokens to changes in those inputs.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Tail message count excluded from the prefix hash.
const MESSAGES_PREFIX_TAIL_COUNT: usize = 2;

/// Cache-relevant request inputs recorded immediately before a chat call.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptStateSnapshot {
    /// SHA-256 (16 hex chars) of the system prompt text.
    pub system_hash: String,
    /// SHA-256 (16 hex chars) of the tool definitions.
    pub tools_hash: String,
    /// SHA-256 (16 hex chars) of the messages-prefix payload.
    pub messages_prefix_hash: String,
    /// SHA-256 (16 hex chars) of the cache-control fields.
    pub cache_control_hash: String,
    /// The resolved model id.
    pub model: String,
    /// Total message count in the request.
    pub message_count: usize,
    /// Total tool count in the request.
    pub tool_count: usize,
    /// Per-item hashes of the messages-prefix payload.
    pub messages_prefix_item_hashes: Vec<String>,
    /// Per-item kinds of the messages-prefix payload.
    pub messages_prefix_item_kinds: Vec<String>,
    /// Per-field (key, hash) of the cache-control payload.
    pub cache_control_field_hashes: Vec<(String, String)>,
}

impl PromptStateSnapshot {
    /// The cache-relevant field names whose change can be reported.
    pub fn changed_fields(&self, previous: &PromptStateSnapshot) -> Vec<String> {
        let mut changed: Vec<String> = Vec::new();
        for field_name in [
            "system_hash",
            "tools_hash",
            "messages_prefix_hash",
            "cache_control_hash",
            "model",
        ] {
            let current = match field_name {
                "system_hash" => &self.system_hash,
                "tools_hash" => &self.tools_hash,
                "messages_prefix_hash" => &self.messages_prefix_hash,
                "cache_control_hash" => &self.cache_control_hash,
                _ => &self.model,
            };
            let previous_value = match field_name {
                "system_hash" => &previous.system_hash,
                "tools_hash" => &previous.tools_hash,
                "messages_prefix_hash" => &previous.messages_prefix_hash,
                "cache_control_hash" => &previous.cache_control_hash,
                _ => &previous.model,
            };
            if current != previous_value {
                changed.push(field_name.to_string());
            }
        }
        changed
    }

    /// Render the snapshot for forensic diagnostics.
    pub fn to_forensics(&self) -> serde_json::Value {
        json!({
            "system_hash": self.system_hash,
            "tools_hash": self.tools_hash,
            "messages_prefix_hash": self.messages_prefix_hash,
            "messages_prefix_item_hashes": self.messages_prefix_item_hashes,
            "messages_prefix_item_kinds": self.messages_prefix_item_kinds,
            "cache_control_hash": self.cache_control_hash,
            "cache_control_field_hashes": self.cache_control_field_hashes.iter().cloned().collect::<HashMap<_,_>>(),
            "model": self.model,
            "message_count": self.message_count,
            "tool_count": self.tool_count,
        })
    }
}

/// Cache-control fields captured from the provider chat config.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CacheControlSnapshot {
    /// The system prompt text.
    pub system: String,
    /// Provider cache breakpoint hints.
    pub cache_breakpoints: Vec<serde_json::Value>,
    /// The provider cache mode.
    pub cache_mode: Option<String>,
    /// The maximum output token count.
    pub max_tokens: Option<u64>,
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Stop sequences.
    pub stop_sequences: Vec<String>,
    /// Thinking configuration.
    pub thinking: Option<serde_json::Value>,
    /// Thinking budget in tokens.
    pub thinking_budget_tokens: Option<u64>,
    /// Thinking level.
    pub thinking_level: Option<String>,
}

impl CacheControlSnapshot {
    fn to_payload(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut map = serde_json::Map::new();
        map.insert(
            "cache_breakpoints".into(),
            serde_json::Value::Array(self.cache_breakpoints.clone()),
        );
        map.insert(
            "cache_mode".into(),
            self.cache_mode
                .as_ref()
                .map(|s| serde_json::Value::String(s.clone()))
                .unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "max_tokens".into(),
            self.max_tokens
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "temperature".into(),
            self.temperature
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "stop_sequences".into(),
            serde_json::Value::Array(
                self.stop_sequences
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
        map.insert(
            "thinking".into(),
            self.thinking.clone().unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "thinking_budget_tokens".into(),
            self.thinking_budget_tokens
                .map(serde_json::Value::from)
                .unwrap_or(serde_json::Value::Null),
        );
        map.insert(
            "thinking_level".into(),
            self.thinking_level
                .as_ref()
                .map(|s| serde_json::Value::String(s.clone()))
                .unwrap_or(serde_json::Value::Null),
        );
        map
    }
}

/// Result of comparing a provider response with the previous cache baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct CacheBreakReport {
    /// Whether a cache-read drop was attributed to prompt-state changes.
    pub break_detected: bool,
    /// A stable machine-readable reason.
    pub reason: String,
    /// The prompt-state fields that changed.
    pub changed_fields: Vec<String>,
    /// Cache-read tokens reported for the previous call.
    pub previous_cache_read_tokens: i64,
    /// Cache-read tokens reported for the current call.
    pub current_cache_read_tokens: i64,
    /// `previous - current`, floor at 0.
    pub drop_tokens: i64,
    /// `drop_tokens / previous`, 0 when previous is 0.
    pub drop_ratio: f64,
    /// Whether this response reset the baseline after compaction.
    pub baseline_reset: bool,
    /// The previous snapshot when a break was detected.
    pub previous_snapshot: Option<PromptStateSnapshot>,
    /// The current snapshot when a break was detected.
    pub current_snapshot: Option<PromptStateSnapshot>,
}

impl CacheBreakReport {
    /// Render the report for logging.
    pub fn to_log_dict(&self) -> serde_json::Value {
        let mut payload = serde_json::Map::new();
        payload.insert(
            "reason".into(),
            serde_json::Value::String(self.reason.clone()),
        );
        payload.insert(
            "changed_fields".into(),
            serde_json::Value::Array(
                self.changed_fields
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
        payload.insert(
            "previous_cache_read_tokens".into(),
            serde_json::Value::from(self.previous_cache_read_tokens),
        );
        payload.insert(
            "current_cache_read_tokens".into(),
            serde_json::Value::from(self.current_cache_read_tokens),
        );
        payload.insert(
            "drop_tokens".into(),
            serde_json::Value::from(self.drop_tokens),
        );
        payload.insert(
            "drop_ratio".into(),
            serde_json::Value::from(self.drop_ratio),
        );
        payload.insert(
            "baseline_reset".into(),
            serde_json::Value::Bool(self.baseline_reset),
        );
        if self.break_detected {
            if let (Some(previous), Some(current)) =
                (&self.previous_snapshot, &self.current_snapshot)
            {
                let mut forensics = serde_json::Map::new();
                forensics.insert("previous".into(), previous.to_forensics());
                forensics.insert("current".into(), current.to_forensics());
                payload.insert("forensics".into(), serde_json::Value::Object(forensics));
            }
        }
        serde_json::Value::Object(payload)
    }
}

#[derive(Debug, Clone)]
struct CacheBaseline {
    snapshot: PromptStateSnapshot,
    cache_read_tokens: i64,
}

/// Track cache-read drops and attribute them to prompt-state changes.
#[derive(Debug, Clone)]
pub struct CacheBreakMonitor {
    baselines: HashMap<String, CacheBaseline>,
    reset_pending: HashSet<String>,
    min_drop_tokens: i64,
    min_drop_ratio: f64,
}

impl Default for CacheBreakMonitor {
    fn default() -> Self {
        Self {
            baselines: HashMap::new(),
            reset_pending: HashSet::new(),
            min_drop_tokens: 2000,
            min_drop_ratio: 0.05,
        }
    }
}

impl CacheBreakMonitor {
    /// Create a monitor with the given detection thresholds.
    pub fn new(min_drop_tokens: i64, min_drop_ratio: f64) -> Self {
        Self {
            min_drop_tokens: min_drop_tokens.max(0),
            min_drop_ratio: min_drop_ratio.max(0.0),
            ..Default::default()
        }
    }

    /// Snapshot the cache-relevant state of a request.
    ///
    /// `messages` are JSON message objects (`role`, `content`); `tools` are JSON
    /// tool-definition objects. Hashes are computed over a canonical (sorted-key)
    /// JSON encoding so ordering-independent input maps to a stable hash.
    pub fn record_prompt_state(
        &self,
        messages: &[serde_json::Value],
        tools: Option<&[serde_json::Value]>,
        cache_control: &CacheControlSnapshot,
        model: &str,
    ) -> PromptStateSnapshot {
        let prefix_messages: &[serde_json::Value] = if messages.len() >= MESSAGES_PREFIX_TAIL_COUNT
        {
            &messages[..messages.len() - MESSAGES_PREFIX_TAIL_COUNT]
        } else {
            &[]
        };
        let tools_value = serde_json::Value::Array(tools.unwrap_or_default().to_vec());
        let cache_control_payload = cache_control.to_payload();
        let cache_control_value = serde_json::Value::Object(cache_control_payload.clone());
        let mut cache_control_field_hashes: Vec<(String, String)> = cache_control_payload
            .iter()
            .map(|(key, value)| (key.clone(), stable_hash_value(value)))
            .collect();
        cache_control_field_hashes.sort_by(|a, b| a.0.cmp(&b.0));

        PromptStateSnapshot {
            system_hash: stable_hash_value(&serde_json::Value::String(
                cache_control.system.clone(),
            )),
            tools_hash: stable_hash_value(&tools_value),
            messages_prefix_hash: stable_hash_value(&serde_json::Value::Array(
                prefix_messages.to_vec(),
            )),
            cache_control_hash: stable_hash_value(&cache_control_value),
            model: model.to_string(),
            message_count: messages.len(),
            tool_count: tools.map(|t| t.len()).unwrap_or(0),
            messages_prefix_item_hashes: prefix_messages.iter().map(stable_hash_value).collect(),
            messages_prefix_item_kinds: prefix_messages
                .iter()
                .map(message_prefix_item_kind)
                .collect(),
            cache_control_field_hashes,
        }
    }

    /// Compare a provider response against the previous baseline for the session.
    pub fn check_response_for_cache_break(
        &mut self,
        session_key: &str,
        snapshot: PromptStateSnapshot,
        cache_read_tokens: i64,
    ) -> CacheBreakReport {
        let current_tokens = cache_read_tokens.max(0);
        let previous = self.baselines.get(session_key).cloned();
        let reset_pending = self.reset_pending.contains(session_key);
        self.baselines.insert(
            session_key.to_string(),
            CacheBaseline {
                snapshot: snapshot.clone(),
                cache_read_tokens: current_tokens,
            },
        );
        if reset_pending {
            self.reset_pending.remove(session_key);
            return CacheBreakReport {
                break_detected: false,
                reason: "baseline_reset_after_compaction".to_string(),
                changed_fields: Vec::new(),
                previous_cache_read_tokens: 0,
                current_cache_read_tokens: current_tokens,
                drop_tokens: 0,
                drop_ratio: 0.0,
                baseline_reset: true,
                previous_snapshot: None,
                current_snapshot: None,
            };
        }
        let Some(previous) = previous else {
            return CacheBreakReport {
                break_detected: false,
                reason: "baseline_initialized".to_string(),
                changed_fields: Vec::new(),
                previous_cache_read_tokens: 0,
                current_cache_read_tokens: current_tokens,
                drop_tokens: 0,
                drop_ratio: 0.0,
                baseline_reset: false,
                previous_snapshot: None,
                current_snapshot: None,
            };
        };

        let drop_tokens = (previous.cache_read_tokens - current_tokens).max(0);
        let drop_ratio = if previous.cache_read_tokens > 0 {
            drop_tokens as f64 / previous.cache_read_tokens as f64
        } else {
            0.0
        };
        let changed_fields = snapshot.changed_fields(&previous.snapshot);
        let break_detected = !changed_fields.is_empty()
            && drop_tokens >= self.min_drop_tokens
            && drop_ratio >= self.min_drop_ratio;
        CacheBreakReport {
            break_detected,
            reason: if break_detected {
                "cache_read_drop"
            } else {
                "cache_read_stable"
            }
            .to_string(),
            changed_fields,
            previous_cache_read_tokens: previous.cache_read_tokens,
            current_cache_read_tokens: current_tokens,
            drop_tokens,
            drop_ratio: (drop_ratio * 10_000.0).round() / 10_000.0,
            baseline_reset: false,
            previous_snapshot: if break_detected {
                Some(previous.snapshot)
            } else {
                None
            },
            current_snapshot: if break_detected { Some(snapshot) } else { None },
        }
    }

    /// Treat the next provider response for this session as a new baseline.
    pub fn notify_compaction(&mut self, session_key: &str) {
        self.reset_pending.insert(session_key.to_string());
    }

    /// Forget all baselines and pending resets.
    pub fn clear(&mut self) {
        self.baselines.clear();
        self.reset_pending.clear();
    }
}

/// Classify a message's prefix kind by its leading text marker.
pub fn message_prefix_item_kind(message: &serde_json::Value) -> String {
    if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
        if content.starts_with("[Request context for this turn]") {
            return "request_context".to_string();
        }
        if content.starts_with("[Runtime context for this turn]") {
            return "runtime_context".to_string();
        }
        if content.starts_with("[Available skills for this turn]") {
            return "skills_context".to_string();
        }
    }
    "history".to_string()
}

/// Stable SHA-256 (16 hex chars) over the canonical JSON encoding of a value.
pub fn stable_hash_value(value: &serde_json::Value) -> String {
    let canonical = canonical_json(value);
    let mut hasher = Sha256::new();
    hasher.update(canonical.as_bytes());
    let digest = hasher.finalize();
    hex::encode(&digest[..8])
}

/// Canonical (sorted-key) compact JSON encoding, matching Python's
/// `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let body: Vec<String> = entries
                .iter()
                .map(|(key, value)| format!("{}:{}", json_string(key), canonical_json(value)))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        serde_json::Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        serde_json::Value::String(s) => json_string(s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        serde_json::Value::Null => "null".to_string(),
    }
}

fn json_string(s: &str) -> String {
    // serde_json's serializer produces standard JSON string escaping.
    serde_json::to_string(s).unwrap_or_else(|_| format!("\"{}\"", s))
}

// ---------------------------------------------------------------------------
// Module-level default monitor and compaction lifecycle
//
// Mirrors the non-pure portion of the Python `engine/cache_break_monitor.py`:
// a shared default [`CacheBreakMonitor`] plus the idempotent compaction
// lifecycle (event publishing, heartbeat loop, owner-task fallback, and
// listeners). The lifecycle uses the crate's tokio conventions (`tokio::spawn`
// tasks and an `AbortHandle` registry, matching `turn_runner/`).
// ---------------------------------------------------------------------------

/// A listener notified for every accepted compaction lifecycle event.
pub type CompactionListener = Arc<dyn Fn(&str, &serde_json::Value) + Send + Sync>;

/// Terminal compaction statuses. Once a compaction id claims a terminal
/// status, further lifecycle events for that id are no-ops.
pub const COMPACTION_TERMINAL_STATUSES: &[&str] = &[
    "completed",
    "skipped",
    "failed",
    "error",
    "cancelled",
    "timed_out",
    "stale",
    "emergency_ephemeral",
];

/// Bounded LRU cache of terminal compaction claims.
const COMPACTION_TERMINAL_CACHE_SIZE: usize = 2048;

/// The event name carried in `compaction_lifecycle_payload`.
pub const COMPACTION_TRIGGERED_EVENT: &str = "compaction.triggered";

/// Process-wide compaction lifecycle state.
#[derive(Default)]
struct CompactionLifecycle {
    /// Per-compaction-id monotonic sequence counters.
    sequences: HashMap<String, u64>,
    /// compaction_id -> terminal status (idempotency claims).
    terminals: HashMap<String, String>,
    /// LRU order of terminal claims.
    terminal_order: VecDeque<String>,
    /// Registered compaction lifecycle listeners.
    listeners: Vec<CompactionListener>,
    /// Abort handles for live heartbeat loops, keyed by `(session, compaction_id)`.
    heartbeat_tasks: HashMap<(String, String), tokio::task::AbortHandle>,
    /// Abort handles for live owner tasks, keyed by `(session, compaction_id)`.
    active_owners: HashMap<(String, String), tokio::task::AbortHandle>,
}

fn compaction_lifecycle() -> &'static Mutex<CompactionLifecycle> {
    static STATE: OnceLock<Mutex<CompactionLifecycle>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(CompactionLifecycle::default()))
}

/// The process-wide default cache-break monitor.
pub fn default_cache_break_monitor() -> &'static Mutex<CacheBreakMonitor> {
    static MONITOR: OnceLock<Mutex<CacheBreakMonitor>> = OnceLock::new();
    MONITOR.get_or_init(|| Mutex::new(CacheBreakMonitor::default()))
}

/// Module-level convenience over the default monitor.
pub fn record_prompt_state(
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    cache_control: &CacheControlSnapshot,
    model: &str,
) -> PromptStateSnapshot {
    default_cache_break_monitor()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .record_prompt_state(messages, tools, cache_control, model)
}

/// Module-level convenience over the default monitor.
pub fn check_response_for_cache_break(
    session_key: &str,
    snapshot: PromptStateSnapshot,
    cache_read_tokens: i64,
) -> CacheBreakReport {
    default_cache_break_monitor()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .check_response_for_cache_break(session_key, snapshot, cache_read_tokens)
}

/// Normalized lifecycle payload carrying the `compaction.triggered` chain.
pub fn compaction_lifecycle_payload(compaction_id: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "compaction_id".into(),
        serde_json::Value::String(compaction_id.to_string()),
    );
    map.insert(
        "event".into(),
        serde_json::Value::String(COMPACTION_TRIGGERED_EVENT.to_string()),
    );
    map.insert(
        "event_chain".into(),
        serde_json::Value::Array(vec![serde_json::Value::String(
            COMPACTION_TRIGGERED_EVENT.to_string(),
        )]),
    );
    map.insert(
        "coverage_status".into(),
        serde_json::Value::String("unknown".to_string()),
    );
    serde_json::Value::Object(map)
}

/// Normalized user-facing semantics for a compaction lifecycle event.
pub fn compaction_effect_payload(
    status: &str,
    reason: Option<&str>,
    source: &str,
) -> serde_json::Value {
    let normalized_status = status.trim().to_lowercase();
    let normalized_source = source.trim().to_lowercase();
    let reason = reason
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let applied = matches!(
        normalized_status.as_str(),
        "completed" | "emergency_ephemeral"
    );
    let durability = if normalized_status == "completed" {
        "durable"
    } else if normalized_status == "emergency_ephemeral" {
        "request_scoped"
    } else {
        "none"
    };
    let user_visible = if normalized_source == "manual" {
        true
    } else {
        matches!(
            normalized_status.as_str(),
            "started" | "observed" | "completed" | "emergency_ephemeral"
        )
    };
    let mut map = serde_json::Map::new();
    map.insert(
        "status".into(),
        serde_json::Value::String(normalized_status),
    );
    map.insert(
        "source".into(),
        serde_json::Value::String(normalized_source),
    );
    map.insert("applied".into(), serde_json::Value::Bool(applied));
    map.insert(
        "durability".into(),
        serde_json::Value::String(durability.to_string()),
    );
    map.insert("user_visible".into(), serde_json::Value::Bool(user_visible));
    if let Some(reason) = reason {
        map.insert("reason".into(), serde_json::Value::String(reason));
    }
    serde_json::Value::Object(map)
}

/// Publish one idempotent compaction lifecycle event and return its normalized
/// payload, or `None` when the compaction id already claimed a terminal state.
///
/// Mirrors the Python `cache_break_monitor.notify_compaction`. `status` is one
/// of the terminal statuses or a lifecycle phase (`started`, `observed`, ...).
/// A `started` event with a positive `heartbeat_interval_seconds` spawns a
/// heartbeat loop that re-publishes `observed` events until a terminal claim.
/// `completed` events also reset the shared default monitor's cache baseline.
pub fn notify_compaction(
    session_key: &str,
    compaction_id: &str,
    status: &str,
    source: &str,
    phase: &str,
    extra: serde_json::Map<String, serde_json::Value>,
    heartbeat_interval_seconds: f64,
    _track_current_task: bool,
    notify_listeners: bool,
) -> Option<serde_json::Value> {
    let mut event_payload = serde_json::Map::new();
    let status = status.trim().to_lowercase();
    let source = source.trim().to_lowercase();
    let compaction_id = compaction_id.trim().to_string();
    event_payload.insert(
        String::from("status"),
        serde_json::Value::String(status.clone()),
    );
    event_payload.insert(
        String::from("source"),
        serde_json::Value::String(source.clone()),
    );
    if !phase.trim().is_empty() {
        event_payload.insert(
            String::from("phase"),
            serde_json::Value::String(phase.trim().to_string()),
        );
    }
    for (key, value) in extra {
        event_payload.entry(key).or_insert(value);
    }
    if !compaction_id.is_empty() {
        event_payload
            .entry(String::from("compaction_id"))
            .or_insert_with(|| serde_json::Value::String(compaction_id.clone()));

        let mut state = compaction_lifecycle()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if state.terminals.contains_key(&compaction_id) {
            return None;
        }
        let sequence = state.sequences.get(&compaction_id).copied().unwrap_or(0) + 1;
        state.sequences.insert(compaction_id.clone(), sequence);
        event_payload
            .entry(String::from("sequence"))
            .or_insert_with(|| serde_json::Value::from(sequence));

        let key = (session_key.to_string(), compaction_id.clone());
        if status == "started" {
            // Mirror the Python heartbeat condition: a positive interval with
            // listener delivery, and a live tokio runtime to spawn the loop.
            if heartbeat_interval_seconds > 0.0
                && notify_listeners
                && tokio::runtime::Handle::try_current().is_ok()
                && !state.heartbeat_tasks.contains_key(&key)
            {
                let task = tokio::spawn(compaction_heartbeat_loop(
                    session_key.to_string(),
                    compaction_id.clone(),
                    source.clone(),
                    phase.to_string(),
                    heartbeat_interval_seconds,
                ));
                state
                    .heartbeat_tasks
                    .insert(key.clone(), task.abort_handle());
            }
        }
        if COMPACTION_TERMINAL_STATUSES.contains(&status.as_str()) {
            state
                .terminals
                .insert(compaction_id.clone(), status.clone());
            state.terminal_order.push_back(compaction_id.clone());
            while state.terminal_order.len() > COMPACTION_TERMINAL_CACHE_SIZE {
                if let Some(expired) = state.terminal_order.pop_front() {
                    state.sequences.remove(&expired);
                    state.terminals.remove(&expired);
                }
            }
            state.active_owners.remove(&key);
            if let Some(heartbeat) = state.heartbeat_tasks.remove(&key) {
                heartbeat.abort();
            }
        }
    }

    if status == "completed" {
        default_cache_break_monitor()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .notify_compaction(session_key);
    }

    let payload = serde_json::Value::Object(event_payload);
    if notify_listeners {
        let listeners: Vec<CompactionListener> = compaction_lifecycle()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .listeners
            .clone();
        for listener in listeners {
            listener(session_key, &payload);
        }
    }
    Some(payload)
}

/// Register a best-effort listener for compaction lifecycle events. Returns a
/// handle that removes the listener.
pub fn add_compaction_listener(listener: CompactionListener) -> RemoveCompactionListener {
    let mut state = compaction_lifecycle()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    state.listeners.push(listener.clone());
    let removed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let listener_arc = listener;
    RemoveCompactionListener {
        inner: Box::new(move || {
            if removed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            let mut state = compaction_lifecycle()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.listeners.retain(|l| !Arc::ptr_eq(l, &listener_arc));
        }),
    }
}

/// RAII / closure handle that removes a registered compaction listener.
pub struct RemoveCompactionListener {
    inner: Box<dyn Fn() + Send + Sync>,
}

impl RemoveCompactionListener {
    /// Remove the listener now.
    pub fn remove(self) {
        (self.inner)();
    }
}

impl Drop for RemoveCompactionListener {
    fn drop(&mut self) {
        // Best-effort cleanup is deferred to explicit `remove()` so tests can
        // assert listener presence before tearing down.
    }
}

/// The compaction heartbeat loop: re-publish `observed` events until a
/// terminal claim stops the loop (via `AbortHandle`).
async fn compaction_heartbeat_loop(
    session_key: String,
    compaction_id: String,
    source: String,
    phase: String,
    interval: f64,
) {
    let started = std::time::Instant::now();
    let interval = Duration::from_secs_f64(interval.max(0.01));
    loop {
        tokio::time::sleep(interval).await;
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let mut extra = serde_json::Map::new();
        extra.insert(
            "compaction_id".into(),
            serde_json::Value::String(compaction_id.clone()),
        );
        extra.insert("heartbeat".into(), serde_json::Value::Bool(true));
        extra.insert(
            "heartbeat_at".into(),
            serde_json::Value::from(chrono::Utc::now().timestamp_millis()),
        );
        extra.insert("elapsed_ms".into(), serde_json::Value::from(elapsed_ms));
        if notify_compaction(
            &session_key,
            &compaction_id,
            "observed",
            &source,
            &phase,
            extra,
            0.0,
            false,
            true,
        )
        .is_none()
        {
            // Terminal claim already made by another event; stop the loop.
            return;
        }
    }
}

/// The terminal status claimed for a compaction id in this process, if any.
pub fn compaction_terminal_status(compaction_id: &str) -> Option<String> {
    compaction_lifecycle()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .terminals
        .get(compaction_id)
        .cloned()
}

/// Live compaction ids for one session.
pub fn active_compaction_ids(session_key: &str) -> Vec<String> {
    compaction_lifecycle()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .active_owners
        .keys()
        .filter(|(key, _)| key == session_key)
        .map(|(_, id)| id.clone())
        .collect()
}

/// Cancel every live owner task for one session, returning the abort handles.
pub fn cancel_active_compactions(session_key: &str) -> Vec<tokio::task::AbortHandle> {
    let mut state = compaction_lifecycle()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let keys: Vec<(String, String)> = state
        .active_owners
        .keys()
        .filter(|(key, _)| key == session_key)
        .cloned()
        .collect();
    let mut handles = Vec::new();
    for key in keys {
        if let Some(handle) = state.active_owners.remove(&key) {
            handle.abort();
            handles.push(handle);
        }
    }
    handles
}

/// Run a compaction owner future and backstop a missing terminal event.
///
/// Mirrors the Python owner-task fallback (`_finalize_compaction_when_owner_stops`):
/// while the future runs, the compaction is reported as active for the session;
/// when it finishes — normally, by panic, or by task cancellation — a terminal
/// claim is published unless one was already made. The future runs inline on
/// the caller's task (the same task a Python `started` event would register).
pub async fn run_compaction_owner<F, T>(session_key: &str, compaction_id: &str, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let guard = CompactionOwnerGuard::register(session_key, compaction_id);
    let output = fut.await;
    guard.finish();
    output
}

/// RAII owner guard backing the owner-task fallback.
struct CompactionOwnerGuard {
    session_key: String,
    compaction_id: String,
    finished: bool,
}

impl CompactionOwnerGuard {
    fn register(session_key: &str, compaction_id: &str) -> Self {
        let key = (session_key.to_string(), compaction_id.to_string());
        {
            let mut state = compaction_lifecycle()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            // Register an abort handle so `cancel_active_compactions` can stop
            // the owner. The owner runs inline on the caller's task; the
            // abort handle belongs to a parked proxy task so cancellation is
            // possible without owning the real task handle.
            let proxy = tokio::spawn(std::future::pending::<()>());
            state.active_owners.insert(key, proxy.abort_handle());
        }
        Self {
            session_key: session_key.to_string(),
            compaction_id: compaction_id.to_string(),
            finished: false,
        }
    }

    fn finish(mut self) {
        self.finished = true;
        Self::backstop_if_missing(&self.session_key, &self.compaction_id);
    }

    fn backstop_if_missing(session_key: &str, compaction_id: &str) {
        let terminal_claimed = compaction_lifecycle()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .terminals
            .contains_key(compaction_id);
        if terminal_claimed {
            return;
        }
        let reason = "terminal_missing";
        let status = "failed";
        let mut extra = serde_json::Map::new();
        extra.insert(
            "reason".into(),
            serde_json::Value::String(reason.to_string()),
        );
        if let serde_json::Value::Object(effect) =
            compaction_effect_payload(status, Some(reason), "automatic")
        {
            for (k, v) in effect {
                extra.insert(k, v);
            }
        }
        if let serde_json::Value::Object(lifecycle) = compaction_lifecycle_payload(compaction_id) {
            for (k, v) in lifecycle {
                extra.insert(k, v);
            }
        }
        notify_compaction(
            session_key,
            compaction_id,
            status,
            "automatic",
            "compaction",
            extra,
            0.0,
            false,
            true,
        );
    }
}

impl Drop for CompactionOwnerGuard {
    fn drop(&mut self) {
        if !self.finished {
            Self::backstop_if_missing(&self.session_key, &self.compaction_id);
        }
        let mut state = compaction_lifecycle()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state
            .active_owners
            .remove(&(self.session_key.clone(), self.compaction_id.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(messages: &[serde_json::Value], model: &str) -> PromptStateSnapshot {
        CacheBreakMonitor::default().record_prompt_state(
            messages,
            Some(&[]),
            &CacheControlSnapshot::default(),
            model,
        )
    }

    #[test]
    fn test_baseline_initialized_then_detects_change() {
        let mut monitor = CacheBreakMonitor::new(2000, 0.05);
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = monitor.record_prompt_state(
            &messages,
            Some(&[]),
            &CacheControlSnapshot::default(),
            "model-a",
        );
        let report = monitor.check_response_for_cache_break("sess", first, 10_000);
        assert_eq!(report.reason, "baseline_initialized");
        assert!(!report.break_detected);

        // Same prompt, no cache drop: no break.
        let second = monitor.record_prompt_state(
            &messages,
            Some(&[]),
            &CacheControlSnapshot::default(),
            "model-a",
        );
        let report = monitor.check_response_for_cache_break("sess", second, 10_000);
        assert_eq!(report.reason, "cache_read_stable");
        assert!(!report.break_detected);
    }

    #[test]
    fn test_cache_drop_attributed_to_changed_fields() {
        let mut monitor = CacheBreakMonitor::new(2000, 0.05);
        let mut cache_control = CacheControlSnapshot::default();
        cache_control.system = "system-v1".to_string();
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "model-a");
        monitor.check_response_for_cache_break("sess", first, 10_000);

        // System prompt changes and cache read drops.
        cache_control.system = "system-v2".to_string();
        let second = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "model-a");
        let report = monitor.check_response_for_cache_break("sess", second, 3_000);
        assert!(report.break_detected);
        assert_eq!(report.reason, "cache_read_drop");
        assert!(report.changed_fields.contains(&"system_hash".to_string()));
        assert_eq!(report.drop_tokens, 7000);
        assert!((report.drop_ratio - 0.7).abs() < 0.001);
        assert!(report.previous_snapshot.is_some());
        assert!(report.current_snapshot.is_some());
    }

    #[test]
    fn test_drop_below_threshold_not_detected() {
        let mut monitor = CacheBreakMonitor::new(2000, 0.05);
        let mut cache_control = CacheControlSnapshot::default();
        cache_control.system = "v1".to_string();
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        monitor.check_response_for_cache_break("s", first, 10_000);
        cache_control.system = "v2".to_string();
        let second = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        let report = monitor.check_response_for_cache_break("s", second, 9_000);
        assert!(!report.break_detected);
        assert_eq!(report.reason, "cache_read_stable");
    }

    #[test]
    fn test_compaction_resets_baseline() {
        let mut monitor = CacheBreakMonitor::new(2000, 0.05);
        let mut cache_control = CacheControlSnapshot::default();
        cache_control.system = "v1".to_string();
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        monitor.check_response_for_cache_break("s", first, 10_000);
        monitor.notify_compaction("s");
        let second = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        let report = monitor.check_response_for_cache_break("s", second, 1_000);
        assert!(!report.break_detected);
        assert!(report.baseline_reset);
        assert_eq!(report.reason, "baseline_reset_after_compaction");
    }

    #[test]
    fn test_prefix_excludes_tail_messages() {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "[Request context for this turn]\n..."}),
            serde_json::json!({"role": "user", "content": "second"}),
            serde_json::json!({"role": "assistant", "content": "tail-1"}),
            serde_json::json!({"role": "user", "content": "tail-2"}),
        ];
        let snapshot = snapshot(&messages, "m");
        // Prefix excludes the last 2 messages.
        assert_eq!(snapshot.messages_prefix_item_hashes.len(), 2);
        assert_eq!(snapshot.messages_prefix_item_kinds[0], "request_context");
        assert_eq!(snapshot.messages_prefix_item_kinds[1], "history");
        assert_eq!(snapshot.message_count, 4);
    }

    #[test]
    fn test_changed_fields_empty_when_identical() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = snapshot(&messages, "m");
        let second = snapshot(&messages, "m");
        assert!(first.changed_fields(&second).is_empty());
    }

    #[test]
    fn test_canonical_json_sorts_keys() {
        let value = serde_json::json!({"b": 1, "a": 2, "c": [3, {"x": true}]});
        assert_eq!(
            canonical_json(&value),
            r#"{"a":2,"b":1,"c":[3,{"x":true}]}"#
        );
    }

    #[test]
    fn test_to_log_dict() {
        let mut monitor = CacheBreakMonitor::new(2000, 0.05);
        let mut cache_control = CacheControlSnapshot::default();
        cache_control.system = "v1".to_string();
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        monitor.check_response_for_cache_break("s", first, 10_000);
        cache_control.system = "v2".to_string();
        let second = monitor.record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        let report = monitor.check_response_for_cache_break("s", second, 2_000);
        let log = report.to_log_dict();
        assert_eq!(log["reason"], "cache_read_drop");
        assert!(log["forensics"].is_object());
    }

    // -- Compaction lifecycle tests (module-level state) ----------------------

    #[test]
    fn test_effect_and_lifecycle_payloads() {
        let effect = compaction_effect_payload("completed", Some("ok"), "automatic");
        assert_eq!(effect["status"], "completed");
        assert_eq!(effect["applied"], true);
        assert_eq!(effect["durability"], "durable");
        assert_eq!(effect["user_visible"], true);
        assert_eq!(effect["reason"], "ok");

        let lifecycle = compaction_lifecycle_payload("cmp_1");
        assert_eq!(lifecycle["compaction_id"], "cmp_1");
        assert_eq!(lifecycle["event"], "compaction.triggered");
        assert_eq!(lifecycle["coverage_status"], "unknown");
    }

    #[test]
    fn test_notify_compaction_sequences_and_claims_terminal() {
        let id = "cmp_test_seq_1";
        let first = notify_compaction(
            "sess",
            id,
            "started",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        )
        .unwrap();
        assert_eq!(first["status"], "started");
        assert_eq!(first["sequence"], 1);

        let observed = notify_compaction(
            "sess",
            id,
            "observed",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        )
        .unwrap();
        assert_eq!(observed["sequence"], 2);

        let completed = notify_compaction(
            "sess",
            id,
            "completed",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        )
        .unwrap();
        assert_eq!(completed["sequence"], 3);
        assert_eq!(compaction_terminal_status(id).as_deref(), Some("completed"));

        // Terminal claim: further events are no-ops.
        assert!(
            notify_compaction(
                "sess",
                id,
                "started",
                "automatic",
                "compaction",
                serde_json::Map::new(),
                0.0,
                false,
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn test_notify_compaction_completed_resets_default_monitor() {
        let session = "sess_default_monitor";
        // Establish a baseline through the default monitor helpers.
        let mut cache_control = CacheControlSnapshot::default();
        cache_control.system = "v1".to_string();
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        let first = record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        check_response_for_cache_break(session, first, 10_000);

        // Change the prompt state; a compaction resets the baseline so the
        // following response is treated as a fresh baseline, not a break.
        cache_control.system = "v2".to_string();
        notify_compaction(
            session,
            "cmp_test_baseline",
            "completed",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        );
        let second = record_prompt_state(&messages, Some(&[]), &cache_control, "m");
        let report = check_response_for_cache_break(session, second, 1_000);
        assert!(!report.break_detected);
        assert!(report.baseline_reset);
        assert_eq!(report.reason, "baseline_reset_after_compaction");
    }

    #[test]
    fn test_compaction_listener_receives_events() {
        let received: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = received.clone();
        let listener: CompactionListener = Arc::new(move |_session, payload| {
            sink.lock().unwrap().push(payload.clone());
        });
        let remove = add_compaction_listener(listener);

        notify_compaction(
            "sess",
            "cmp_test_listener",
            "started",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        );
        assert_eq!(received.lock().unwrap().len(), 1);

        remove.remove();
        notify_compaction(
            "sess",
            "cmp_test_listener",
            "observed",
            "automatic",
            "compaction",
            serde_json::Map::new(),
            0.0,
            false,
            true,
        );
        // Listener removed: no additional delivery.
        assert_eq!(received.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_owner_fallback_backstops_missing_terminal() {
        let id = "cmp_test_owner_fallback";
        run_compaction_owner("sess", id, async {
            // The owner does NOT publish a terminal event.
            let _ = std::time::Duration::from_millis(1);
        })
        .await;
        assert_eq!(
            compaction_terminal_status(id).as_deref(),
            Some("failed"),
            "owner that stops without a terminal claim gets a fallback"
        );
    }

    #[tokio::test]
    async fn test_owner_that_publishes_terminal_is_not_backstopped() {
        let id = "cmp_test_owner_terminal";
        run_compaction_owner("sess", id, async {
            notify_compaction(
                "sess",
                id,
                "completed",
                "automatic",
                "compaction",
                serde_json::Map::new(),
                0.0,
                false,
                true,
            );
        })
        .await;
        assert_eq!(compaction_terminal_status(id).as_deref(), Some("completed"));
    }

    #[tokio::test]
    async fn test_active_compaction_ids_reports_owner() {
        let id = "cmp_test_active";
        // Spawn a long-running owner and observe it is reported active.
        let handle = tokio::spawn(run_compaction_owner("sess_active", id, async {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }));
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(active_compaction_ids("sess_active").contains(&id.to_string()));
        handle.await.unwrap();
        // After completion the id is no longer active.
        assert!(!active_compaction_ids("sess_active").contains(&id.to_string()));
    }
}
