use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast, mpsc};
use tracing::{debug, info, warn};

/// A notification published whenever the approval queue changes state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernanceEvent {
    pub request_id: String,
    pub kind: GovernanceEventKind,
    pub operation: String,
    pub command: String,
    pub at: DateTime<Utc>,
    pub detail: String,
}

/// The kind of state transition described by a [`GovernanceEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceEventKind {
    Submitted,
    Approved,
    Rejected,
    Expired,
    AutoRejected,
}

/// Pending approval request for a sandbox operation requiring human approval.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApprovalRequest {
    pub id: String,
    pub operation: String,
    pub command: String,
    pub args: Vec<String>,
    pub reason: String,
    pub requested_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub status: ApprovalStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalStatus {
    Pending,
    Approved,
    Rejected,
    Expired,
}

/// A permanently recorded rejection, kept for audit and deduplication.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RejectionEntry {
    pub request_id: String,
    pub operation: String,
    pub command: String,
    pub rejected_at: DateTime<Utc>,
    pub reason: String,
    pub rejected_by: String,
    /// Hash of the operation for deduplication
    pub operation_hash: String,
}

/// Counter snapshot describing cumulative queue activity.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GovernanceMetrics {
    pub total_submitted: u64,
    pub total_approved: u64,
    pub total_rejected: u64,
    pub total_expired: u64,
    pub total_auto_rejected: u64,
}

/// Bookkeeping stored per rejected operation so the post-rejection guard can
/// enforce a cooldown window.
#[derive(Debug, Clone)]
struct RejectionGuardEntry {
    rejected_at: DateTime<Utc>,
    cooldown_secs: u64,
    reason: String,
}

/// Approval queue for sandbox operations requiring human approval.
///
/// Provides:
/// - Pending approval queue with timeouts
/// - Rejection ledger for audit trail, persisted to disk (JSON)
/// - Post-rejection guard: prevents re-execution of rejected operations for a
///   cooldown window
/// - Timeout-based auto-rejection via a background worker
/// - Event notifications (broadcast) and persistence via `tokio::sync::mpsc`
///
/// The struct is cheap to clone — all state lives behind `Arc` — so a shared
/// queue can be handed to RPC handlers, background workers and test harnesses.
#[derive(Clone)]
pub struct ApprovalQueue {
    pending: Arc<Mutex<HashMap<String, ApprovalRequest>>>,
    rejections: Arc<Mutex<Vec<RejectionEntry>>>,
    rejection_guard: Arc<Mutex<HashMap<String, RejectionGuardEntry>>>,
    event_tx: broadcast::Sender<GovernanceEvent>,
    persistence_tx: mpsc::UnboundedSender<RejectionEntry>,
    persistence_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<RejectionEntry>>>>,
    ledger_path: Arc<Mutex<Option<PathBuf>>>,
    default_timeout_secs: u64,
    cooldown_secs: u64,
    metrics: Arc<Mutex<GovernanceMetrics>>,
}

impl ApprovalQueue {
    /// Create a queue with a default 5-minute approval timeout and a 1-hour
    /// post-rejection cooldown.
    pub fn new() -> Self {
        Self::new_internal(300, 3600)
    }

    /// Create a queue with a custom approval timeout (seconds).
    pub fn with_timeout(timeout_secs: u64) -> Self {
        Self::new_internal(timeout_secs, 3600)
    }

    /// Set the post-rejection cooldown window (seconds) during which a
    /// rejected operation cannot be re-submitted.
    pub fn with_cooldown(mut self, cooldown_secs: u64) -> Self {
        self.cooldown_secs = cooldown_secs;
        self
    }

    /// Attach a rejection-ledger file and load any existing entries from it.
    ///
    /// The ledger is written incrementally on every rejection once a
    /// persistence worker is started (see [`ApprovalQueue::start_persistence_worker`]);
    /// if no worker is running, rejections are persisted synchronously.
    pub async fn with_ledger(self, path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        *self.ledger_path.lock().await = Some(path.clone());
        if let Err(e) = self.load_ledger(&path).await {
            warn!(
                "governance: failed to load rejection ledger from {}: {}",
                path.display(),
                e
            );
        }
        self
    }

    fn new_internal(timeout_secs: u64, cooldown_secs: u64) -> Self {
        let (event_tx, _) = broadcast::channel(256);
        let (persistence_tx, persistence_rx) = mpsc::unbounded_channel();
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            rejections: Arc::new(Mutex::new(Vec::new())),
            rejection_guard: Arc::new(Mutex::new(HashMap::new())),
            event_tx,
            persistence_tx,
            persistence_rx: Arc::new(Mutex::new(Some(persistence_rx))),
            ledger_path: Arc::new(Mutex::new(None)),
            default_timeout_secs: timeout_secs,
            cooldown_secs,
            metrics: Arc::new(Mutex::new(GovernanceMetrics::default())),
        }
    }

    /// Start the background persistence worker.
    ///
    /// The worker drains rejection entries from an mpsc channel and writes the
    /// full ledger to the configured ledger path after each entry. Call this
    /// once per queue (subsequent calls return an error).
    pub fn start_persistence_worker(&self) -> Result<(), String> {
        let rx = self
            .persistence_rx
            .try_lock()
            .map_err(|_| "governance: persistence receiver lock busy".to_string())?
            .take()
            .ok_or_else(|| "governance: persistence worker already started".to_string())?;
        let queue = self.clone();
        tokio::spawn(async move {
            let mut rx = rx;
            while let Some(entry) = rx.recv().await {
                if let Some(path) = queue.ledger_path().await {
                    if let Err(e) = queue.persist_ledger(&path).await {
                        warn!(
                            "governance: ledger persistence failed for {}: {}",
                            entry.request_id, e
                        );
                    } else {
                        debug!("governance: persisted rejection {}", entry.request_id);
                    }
                }
            }
        });
        Ok(())
    }

    /// Submit a new approval request.
    ///
    /// Refuses to submit operations that are still inside their post-rejection
    /// cooldown window.
    pub async fn submit(
        &self,
        operation: &str,
        command: &str,
        args: &[String],
        reason: &str,
    ) -> Result<String, String> {
        if let Some(remaining) = self.cooldown_remaining(operation, command, args).await {
            return Err(format!(
                "Operation was recently rejected; blocked for {} more second(s)",
                remaining
            ));
        }

        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let request = ApprovalRequest {
            id: id.clone(),
            operation: operation.to_string(),
            command: command.to_string(),
            args: args.to_vec(),
            reason: reason.to_string(),
            requested_at: now,
            expires_at: now + chrono::Duration::seconds(self.default_timeout_secs as i64),
            status: ApprovalStatus::Pending,
        };

        {
            let mut pending = self.pending.lock().await;
            pending.insert(id.clone(), request.clone());
        }

        self.metrics.lock().await.total_submitted += 1;
        let _ = self.event_tx.send(GovernanceEvent {
            request_id: id.clone(),
            kind: GovernanceEventKind::Submitted,
            operation: request.operation.clone(),
            command: request.command.clone(),
            at: now,
            detail: reason.to_string(),
        });

        info!(
            "Approval request submitted: {} for operation '{}'",
            id, operation
        );
        Ok(id)
    }

    /// Approve a pending request.
    pub async fn approve(&self, request_id: &str) -> Result<(), String> {
        let mut pending = self.pending.lock().await;
        let request = pending
            .get_mut(request_id)
            .ok_or("Approval request not found")?;

        if request.status != ApprovalStatus::Pending {
            return Err(format!(
                "Request {} is not pending (status: {:?})",
                request_id, request.status
            ));
        }

        if Utc::now() > request.expires_at {
            request.status = ApprovalStatus::Expired;
            return Err("Approval request has expired".to_string());
        }

        request.status = ApprovalStatus::Approved;
        let operation = request.operation.clone();
        let command = request.command.clone();

        self.metrics.lock().await.total_approved += 1;
        let _ = self.event_tx.send(GovernanceEvent {
            request_id: request_id.to_string(),
            kind: GovernanceEventKind::Approved,
            operation,
            command,
            at: Utc::now(),
            detail: "approved".to_string(),
        });

        info!("Approval request {} approved", request_id);
        Ok(())
    }

    /// Reject a pending request and record it in the rejection ledger.
    pub async fn reject(
        &self,
        request_id: &str,
        reason: &str,
        rejected_by: &str,
    ) -> Result<(), String> {
        let mut pending = self.pending.lock().await;
        let request = pending
            .get_mut(request_id)
            .ok_or("Approval request not found")?;

        if request.status != ApprovalStatus::Pending {
            return Err(format!("Request {} is not pending", request_id));
        }

        request.status = ApprovalStatus::Rejected;
        let op_hash =
            self.compute_operation_hash(&request.operation, &request.command, &request.args);

        let entry = RejectionEntry {
            request_id: request_id.to_string(),
            operation: request.operation.clone(),
            command: request.command.clone(),
            rejected_at: Utc::now(),
            reason: reason.to_string(),
            rejected_by: rejected_by.to_string(),
            operation_hash: op_hash.clone(),
        };

        self.rejections.lock().await.push(entry.clone());
        self.rejection_guard.lock().await.insert(
            op_hash,
            RejectionGuardEntry {
                rejected_at: entry.rejected_at,
                cooldown_secs: self.cooldown_secs,
                reason: entry.reason.clone(),
            },
        );

        self.metrics.lock().await.total_rejected += 1;
        let _ = self.event_tx.send(GovernanceEvent {
            request_id: request_id.to_string(),
            kind: GovernanceEventKind::Rejected,
            operation: entry.operation.clone(),
            command: entry.command.clone(),
            at: entry.rejected_at,
            detail: format!("rejected by {}: {}", rejected_by, reason),
        });

        // If the persistence worker has taken the receiver, hand it the entry
        // over mpsc; otherwise persist synchronously so the ledger is never
        // silently dropped.
        let worker_started = self.persistence_rx.lock().await.is_none();
        if worker_started {
            let _ = self.persistence_tx.send(entry);
        } else if let Some(path) = self.ledger_path().await {
            if let Err(e) = self.persist_ledger(&path).await {
                warn!("governance: ledger persistence failed: {}", e);
            }
        }

        info!(
            "Approval request {} rejected by {}: {}",
            request_id, rejected_by, reason
        );
        Ok(())
    }

    /// Mark pending requests that have outlived their timeout as expired.
    ///
    /// This is a display-facing transition; see [`ApprovalQueue::auto_reject_expired`]
    /// for the stronger version that moves them into the rejection ledger.
    pub async fn cleanup_expired(&self) -> usize {
        let mut pending = self.pending.lock().await;
        let now = Utc::now();
        let expired: Vec<String> = pending
            .iter()
            .filter(|(_, r)| r.status == ApprovalStatus::Pending && now > r.expires_at)
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            if let Some(request) = pending.get_mut(&id) {
                request.status = ApprovalStatus::Expired;
                self.metrics.lock().await.total_expired += 1;
                let _ = self.event_tx.send(GovernanceEvent {
                    request_id: id.clone(),
                    kind: GovernanceEventKind::Expired,
                    operation: request.operation.clone(),
                    command: request.command.clone(),
                    at: now,
                    detail: "request expired".to_string(),
                });
            }
        }

        count
    }

    /// Auto-reject pending requests that have expired, recording each one in
    /// the rejection ledger and the post-rejection guard.
    ///
    /// Returns the number of requests auto-rejected.
    pub async fn auto_reject_expired(&self) -> usize {
        let now = Utc::now();
        let mut to_reject = Vec::new();
        {
            let mut pending = self.pending.lock().await;
            for (id, r) in pending.iter_mut() {
                if r.status == ApprovalStatus::Pending && now > r.expires_at {
                    r.status = ApprovalStatus::Rejected;
                    to_reject.push(r.clone());
                }
            }
        }

        let mut count = 0usize;
        for request in to_reject {
            let entry = RejectionEntry {
                request_id: request.id.clone(),
                operation: request.operation.clone(),
                command: request.command.clone(),
                rejected_at: now,
                reason: "approval timeout: request expired without a decision".to_string(),
                rejected_by: "system".to_string(),
                operation_hash: self.compute_operation_hash(
                    &request.operation,
                    &request.command,
                    &request.args,
                ),
            };
            self.rejections.lock().await.push(entry.clone());
            self.rejection_guard.lock().await.insert(
                entry.operation_hash.clone(),
                RejectionGuardEntry {
                    rejected_at: now,
                    cooldown_secs: self.cooldown_secs,
                    reason: entry.reason.clone(),
                },
            );
            self.metrics.lock().await.total_auto_rejected += 1;
            let _ = self.event_tx.send(GovernanceEvent {
                request_id: entry.request_id.clone(),
                kind: GovernanceEventKind::AutoRejected,
                operation: entry.operation.clone(),
                command: entry.command.clone(),
                at: now,
                detail: entry.reason.clone(),
            });
            let _ = self.persistence_tx.send(entry);
            count += 1;
        }

        if count > 0 {
            info!("governance: auto-rejected {} expired request(s)", count);
        }
        count
    }

    /// Check if an operation has been rejected and is still within its
    /// cooldown window (post-rejection guard).
    pub async fn is_rejected(&self, operation: &str, command: &str, args: &[String]) -> bool {
        self.cooldown_remaining(operation, command, args)
            .await
            .is_some()
    }

    /// The number of seconds remaining before a rejected operation may be
    /// re-submitted, or `None` if it was never rejected or its cooldown has
    /// lapsed.
    pub async fn cooldown_remaining(
        &self,
        operation: &str,
        command: &str,
        args: &[String],
    ) -> Option<u64> {
        let op_hash = self.compute_operation_hash(operation, command, args);
        let guard = self.rejection_guard.lock().await;
        if let Some(entry) = guard.get(&op_hash) {
            let elapsed = Utc::now()
                .signed_duration_since(entry.rejected_at)
                .num_seconds();
            if elapsed < entry.cooldown_secs as i64 {
                return Some(entry.cooldown_secs.saturating_sub(elapsed.max(0) as u64));
            }
        }
        None
    }

    /// Get all pending requests.
    pub async fn get_pending(&self) -> Vec<ApprovalRequest> {
        let pending = self.pending.lock().await;
        pending
            .values()
            .filter(|r| r.status == ApprovalStatus::Pending)
            .cloned()
            .collect()
    }

    /// Get the rejection ledger.
    pub async fn get_rejection_ledger(&self) -> Vec<RejectionEntry> {
        self.rejections.lock().await.clone()
    }

    /// Number of pending requests.
    pub async fn pending_count(&self) -> usize {
        self.pending.lock().await.values().count()
    }

    /// Number of recorded rejections.
    pub async fn rejection_count(&self) -> usize {
        self.rejections.lock().await.len()
    }

    /// Snapshot of cumulative queue metrics.
    pub async fn metrics(&self) -> GovernanceMetrics {
        self.metrics.lock().await.clone()
    }

    /// Subscribe to governance events.
    pub fn subscribe(&self) -> broadcast::Receiver<GovernanceEvent> {
        self.event_tx.subscribe()
    }

    /// Persist the full rejection ledger to disk as JSON.
    pub async fn persist_ledger(&self, path: &Path) -> Result<(), String> {
        let ledger = self.get_rejection_ledger().await;
        let json = serde_json::to_string_pretty(&ledger)
            .map_err(|e| format!("governance: ledger serialization failed: {e}"))?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("governance: ledger mkdir failed: {e}"))?;
        }
        // Atomic-ish write: write a temp file then rename over the target.
        let tmp = path.with_extension("json.tmp");
        tokio::fs::write(&tmp, json.as_bytes())
            .await
            .map_err(|e| format!("governance: ledger write failed: {e}"))?;
        tokio::fs::rename(&tmp, path)
            .await
            .map_err(|e| format!("governance: ledger rename failed: {e}"))?;
        Ok(())
    }

    /// Load a rejection ledger from disk, replacing the in-memory state and
    /// seeding the post-rejection guard.
    pub async fn load_ledger(&self, path: &Path) -> Result<(), String> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| format!("governance: ledger read failed: {e}"))?;
        let entries: Vec<RejectionEntry> = serde_json::from_str(&content)
            .map_err(|e| format!("governance: ledger parse failed: {e}"))?;

        let mut rejections = self.rejections.lock().await;
        *rejections = entries;
        let mut guard = self.rejection_guard.lock().await;
        for entry in rejections.iter() {
            guard
                .entry(entry.operation_hash.clone())
                .or_insert(RejectionGuardEntry {
                    rejected_at: entry.rejected_at,
                    cooldown_secs: self.cooldown_secs,
                    reason: entry.reason.clone(),
                });
        }
        Ok(())
    }

    async fn ledger_path(&self) -> Option<PathBuf> {
        self.ledger_path.lock().await.clone()
    }

    fn compute_operation_hash(&self, operation: &str, command: &str, args: &[String]) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(operation.as_bytes());
        hasher.update([0u8]);
        hasher.update(command.as_bytes());
        hasher.update([0u8]);
        for arg in args {
            hasher.update(arg.as_bytes());
            hasher.update([0u8]);
        }
        hex::encode(hasher.finalize())
    }
}

impl Default for ApprovalQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a periodic worker that auto-rejects expired approval requests.
///
/// Spawn this once per queue at application startup; it also expires
/// display-facing pending requests on every tick.
pub async fn run_auto_reject_worker(queue: ApprovalQueue, interval_secs: u64) {
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
    loop {
        ticker.tick().await;
        let rejected = queue.auto_reject_expired().await;
        if rejected > 0 {
            debug!("governance: auto-reject worker processed {rejected} request(s)");
        }
        let _ = queue.cleanup_expired().await;
    }
}

/// Escalation policy: decides who must approve an operation based on its
/// classification and the resource it touches.
///
/// The default policy routes credential-access and system-modification
/// operations to a "security" approver channel, package installs to a
/// "release" channel, and everything else to a "default" channel. Operations
/// touching denied paths always escalate to "security".
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Map of operation classification → approver channel.
    routing: std::collections::HashMap<String, ApproverChannel>,
    /// The default channel when no routing matches.
    default_channel: ApproverChannel,
    /// Operations that require dual approval (two distinct approvers).
    dual_approval_operations: std::collections::HashSet<String>,
    /// Paths that, when touched, force escalation to the security channel.
    sensitive_paths: Vec<String>,
}

/// A named approver channel (e.g. "security", "release", "default").
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ApproverChannel(pub String);

impl ApproverChannel {
    /// The default approver channel.
    pub fn default_channel() -> Self {
        Self("default".to_string())
    }

    /// The security approver channel.
    pub fn security() -> Self {
        Self("security".to_string())
    }

    /// The release approver channel.
    pub fn release() -> Self {
        Self("release".to_string())
    }

    /// The channel name.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        let mut routing = std::collections::HashMap::new();
        routing.insert(
            "credential_access".to_string(),
            ApproverChannel::security(),
        );
        routing.insert(
            "system_modification".to_string(),
            ApproverChannel::security(),
        );
        routing.insert(
            "package_install".to_string(),
            ApproverChannel::release(),
        );
        routing.insert("code_execution".to_string(), ApproverChannel::security());
        Self {
            routing,
            default_channel: ApproverChannel::default_channel(),
            dual_approval_operations: ["package_install", "system_modification"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            sensitive_paths: vec![
                "/etc/shadow".to_string(),
                "/etc/sudoers".to_string(),
                "/root/.ssh".to_string(),
                "/etc/ssh".to_string(),
                "/etc/kubernetes".to_string(),
                "/var/run/docker.sock".to_string(),
                "/etc/passwd".to_string(),
            ],
        }
    }
}

impl EscalationPolicy {
    /// Create a new policy with the default routing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Route an operation classification to a specific channel.
    pub fn route(mut self, operation: impl Into<String>, channel: ApproverChannel) -> Self {
        self.routing.insert(operation.into(), channel);
        self
    }

    /// Set the default channel.
    pub fn default_to(mut self, channel: ApproverChannel) -> Self {
        self.default_channel = channel;
        self
    }

    /// Mark an operation as requiring dual approval.
    pub fn require_dual_approval(mut self, operation: impl Into<String>) -> Self {
        self.dual_approval_operations.insert(operation.into());
        self
    }

    /// Add a sensitive path that forces escalation.
    pub fn sensitive_path(mut self, path: impl Into<String>) -> Self {
        self.sensitive_paths.push(path.into());
        self
    }

    /// Resolve the approver channel for an operation and the paths it touches.
    pub fn resolve_channel(&self, operation: &str, touched_paths: &[&str]) -> ApproverChannel {
        // Sensitive paths always escalate to security.
        for path in touched_paths {
            if self.sensitive_paths.iter().any(|s| path.starts_with(s.as_str())) {
                return ApproverChannel::security();
            }
        }
        self.routing
            .get(operation)
            .cloned()
            .unwrap_or_else(|| self.default_channel.clone())
    }

    /// Does this operation require dual approval?
    pub fn requires_dual_approval(&self, operation: &str) -> bool {
        self.dual_approval_operations.contains(operation)
    }

    /// Get all configured sensitive paths.
    pub fn sensitive_paths(&self) -> &[String] {
        &self.sensitive_paths
    }
}

impl Default for ApproverChannel {
    fn default() -> Self {
        Self::default_channel()
    }
}

/// A persistent audit trail of governance decisions.
///
/// Wraps an append-only log with rotation and query support. Each entry is a
/// [`GovernanceAuditEntry`] capturing the who/what/when/decision of every
/// governance event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GovernanceAuditEntry {
    pub timestamp: DateTime<Utc>,
    pub request_id: String,
    pub operation: String,
    pub command: String,
    pub channel: String,
    pub decision: GovernanceDecision,
    pub actor: String,
    pub reason: String,
}

/// The outcome recorded in an audit entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceDecision {
    Submitted,
    Approved,
    Rejected,
    Expired,
    AutoRejected,
    Escalated,
}

/// Append-only audit trail backed by an in-memory ring buffer with optional
/// disk persistence.
#[derive(Clone)]
pub struct GovernanceAuditTrail {
    entries: Arc<Mutex<std::collections::VecDeque<GovernanceAuditEntry>>>,
    capacity: usize,
    path: Arc<Mutex<Option<PathBuf>>>,
}

impl GovernanceAuditTrail {
    /// Create a trail with the given in-memory capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(std::collections::VecDeque::with_capacity(capacity))),
            capacity,
            path: Arc::new(Mutex::new(None)),
        }
    }

    /// Attach a persistence path. The trail is appended to this file on every
    /// `record` call.
    pub async fn with_persistence(self, path: impl Into<PathBuf>) -> Self {
        *self.path.lock().await = Some(path.into());
        self
    }

    /// Record an audit entry.
    pub async fn record(&self, entry: GovernanceAuditEntry) {
        let mut entries = self.entries.lock().await;
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(entry.clone());
        drop(entries);
        if let Some(path) = self.path.lock().await.clone() {
            if let Err(e) = self.append_to_disk(&path, &entry).await {
                warn!("governance audit: append failed: {e}");
            }
        }
    }

    /// Snapshot all entries (oldest first).
    pub async fn entries(&self) -> Vec<GovernanceAuditEntry> {
        self.entries.lock().await.iter().cloned().collect()
    }

    /// Query entries matching a predicate.
    pub async fn query<F>(&self, predicate: F) -> Vec<GovernanceAuditEntry>
    where
        F: Fn(&GovernanceAuditEntry) -> bool,
    {
        self.entries
            .lock()
            .await
            .iter()
            .filter(|e| predicate(e))
            .cloned()
            .collect()
    }

    /// Entries for a specific operation.
    pub async fn for_operation(&self, operation: &str) -> Vec<GovernanceAuditEntry> {
        self.query(|e| e.operation == operation).await
    }

    /// Count of entries by decision.
    pub async fn counts_by_decision(&self) -> std::collections::HashMap<GovernanceDecision, u64> {
        let mut counts = std::collections::HashMap::new();
        for entry in self.entries.lock().await.iter() {
            *counts.entry(entry.decision).or_insert(0) += 1;
        }
        counts
    }

    /// Number of entries currently held.
    pub async fn len(&self) -> usize {
        self.entries.lock().await.len()
    }

    /// Is the trail empty?
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Clear all in-memory entries.
    pub async fn clear(&self) {
        self.entries.lock().await.clear();
    }

    async fn append_to_disk(
        &self,
        path: &Path,
        entry: &GovernanceAuditEntry,
    ) -> Result<(), String> {
        use tokio::io::AsyncWriteExt;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("audit mkdir: {e}"))?;
        }
        let line = serde_json::to_string(entry)
            .map_err(|e| format!("audit serialize: {e}"))?;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
            .map_err(|e| format!("audit open: {e}"))?;
        file.write_all(line.as_bytes())
            .await
            .map_err(|e| format!("audit write: {e}"))?;
        file.write_all(b"\n")
            .await
            .map_err(|e| format!("audit newline: {e}"))?;
        Ok(())
    }
}

/// A governance coordinator that ties together the approval queue, escalation
/// policy and audit trail.
#[derive(Clone)]
pub struct GovernanceCoordinator {
    pub queue: ApprovalQueue,
    pub escalation: Arc<EscalationPolicy>,
    pub audit: GovernanceAuditTrail,
}

impl GovernanceCoordinator {
    /// Create a coordinator with default settings.
    pub fn new() -> Self {
        Self {
            queue: ApprovalQueue::new(),
            escalation: Arc::new(EscalationPolicy::new()),
            audit: GovernanceAuditTrail::new(10_000),
        }
    }

    /// Create a coordinator with custom components.
    pub fn with_components(
        queue: ApprovalQueue,
        escalation: EscalationPolicy,
        audit: GovernanceAuditTrail,
    ) -> Self {
        Self {
            queue,
            escalation: Arc::new(escalation),
            audit,
        }
    }

    /// Submit an operation for approval, recording the escalation channel.
    pub async fn submit(
        &self,
        operation: &str,
        command: &str,
        args: &[String],
        reason: &str,
        touched_paths: &[&str],
    ) -> Result<String, String> {
        let channel = self.escalation.resolve_channel(operation, touched_paths);
        let dual = self.escalation.requires_dual_approval(operation);
        // A non-default channel (security/release) or a dual-approval
        // requirement counts as an escalation.
        let escalated = dual || channel != ApproverChannel::default_channel();
        let request_id = self.queue.submit(operation, command, args, reason).await?;
        self.audit
            .record(GovernanceAuditEntry {
                timestamp: Utc::now(),
                request_id: request_id.clone(),
                operation: operation.to_string(),
                command: command.to_string(),
                channel: channel.name().to_string(),
                decision: if escalated {
                    GovernanceDecision::Escalated
                } else {
                    GovernanceDecision::Submitted
                },
                actor: "system".to_string(),
                reason: reason.to_string(),
            })
            .await;
        Ok(request_id)
    }

    /// Approve a request, recording the approver.
    pub async fn approve(
        &self,
        request_id: &str,
        approver: &str,
    ) -> Result<(), String> {
        self.queue.approve(request_id).await?;
        let pending = self.queue.get_pending().await;
        let req = pending
            .iter()
            .find(|r| r.id == request_id)
            .cloned()
            .or_else(|| {
                // The request may have been removed from pending after
                // approval; reconstruct a minimal entry for the audit.
                Some(ApprovalRequest {
                    id: request_id.to_string(),
                    operation: String::new(),
                    command: String::new(),
                    args: vec![],
                    reason: String::new(),
                    requested_at: Utc::now(),
                    expires_at: Utc::now(),
                    status: ApprovalStatus::Approved,
                })
            });
        if let Some(req) = req {
            self.audit
                .record(GovernanceAuditEntry {
                    timestamp: Utc::now(),
                    request_id: request_id.to_string(),
                    operation: req.operation,
                    command: req.command,
                    channel: String::new(),
                    decision: GovernanceDecision::Approved,
                    actor: approver.to_string(),
                    reason: "approved".to_string(),
                })
                .await;
        }
        Ok(())
    }

    /// Reject a request, recording the rejecter and reason.
    pub async fn reject(
        &self,
        request_id: &str,
        reason: &str,
        rejected_by: &str,
    ) -> Result<(), String> {
        self.queue.reject(request_id, reason, rejected_by).await?;
        self.audit
            .record(GovernanceAuditEntry {
                timestamp: Utc::now(),
                request_id: request_id.to_string(),
                operation: String::new(),
                command: String::new(),
                channel: String::new(),
                decision: GovernanceDecision::Rejected,
                actor: rejected_by.to_string(),
                reason: reason.to_string(),
            })
            .await;
        Ok(())
    }
}

impl Default for GovernanceCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queue_submit_approve_reject() {
        let q = ApprovalQueue::new();
        let id = q
            .submit("code_execution", "python", &["-c".to_string()], "test")
            .await
            .unwrap();
        assert_eq!(q.pending_count().await, 1);
        q.approve(&id).await.unwrap();
        let metrics = q.metrics().await;
        assert_eq!(metrics.total_approved, 1);
    }

    #[tokio::test]
    async fn cooldown_blocks_resubmit() {
        let q = ApprovalQueue::new().with_cooldown(3600);
        let id = q
            .submit("code_execution", "python", &[], "test")
            .await
            .unwrap();
        q.reject(&id, "no", "tester").await.unwrap();
        let again = q
            .submit("code_execution", "python", &[], "test")
            .await;
        assert!(again.is_err());
    }

    #[test]
    fn escalation_routes_credential_access_to_security() {
        let policy = EscalationPolicy::new();
        let channel = policy.resolve_channel("credential_access", &[]);
        assert_eq!(channel, ApproverChannel::security());
    }

    #[test]
    fn escalation_sensitive_path_overrides() {
        let policy = EscalationPolicy::new();
        let channel = policy.resolve_channel("file_read", &["/etc/shadow"]);
        assert_eq!(channel, ApproverChannel::security());
    }

    #[test]
    fn escalation_dual_approval() {
        let policy = EscalationPolicy::new();
        assert!(policy.requires_dual_approval("package_install"));
        assert!(!policy.requires_dual_approval("file_read"));
    }

    #[tokio::test]
    async fn audit_trail_records_and_queries() {
        let trail = GovernanceAuditTrail::new(100);
        trail
            .record(GovernanceAuditEntry {
                timestamp: Utc::now(),
                request_id: "r1".to_string(),
                operation: "code_execution".to_string(),
                command: "python".to_string(),
                channel: "security".to_string(),
                decision: GovernanceDecision::Submitted,
                actor: "system".to_string(),
                reason: "test".to_string(),
            })
            .await;
        assert_eq!(trail.len().await, 1);
        let ops = trail.for_operation("code_execution").await;
        assert_eq!(ops.len(), 1);
        let other = trail.for_operation("other").await;
        assert!(other.is_empty());
    }

    #[tokio::test]
    async fn audit_trail_rotates_at_capacity() {
        let trail = GovernanceAuditTrail::new(3);
        for i in 0..5 {
            trail
                .record(GovernanceAuditEntry {
                    timestamp: Utc::now(),
                    request_id: format!("r{i}"),
                    operation: "op".to_string(),
                    command: "cmd".to_string(),
                    channel: "default".to_string(),
                    decision: GovernanceDecision::Submitted,
                    actor: "system".to_string(),
                    reason: String::new(),
                })
                .await;
        }
        assert_eq!(trail.len().await, 3);
    }

    #[tokio::test]
    async fn coordinator_submit_records_audit() {
        let coord = GovernanceCoordinator::new();
        let id = coord
            .submit("credential_access", "cat", &["/etc/shadow".to_string()], "test", &["/etc/shadow"])
            .await
            .unwrap();
        assert!(!id.is_empty());
        let entries = coord.audit.entries().await;
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].decision, GovernanceDecision::Escalated);
        assert_eq!(entries[0].channel, "security");
    }
}
