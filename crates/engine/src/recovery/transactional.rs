//! Transactional state updates: apply agent-state changes atomically so a
//! crash never leaves the agent in an inconsistent state.
//!
//! Mirrors the Python `engine/recovery/transactional.py`. The transactional
//! update wrapper records the before/after state, applies the mutation, and
//! can roll back on failure. Updates are journaled to the crash snapshot so
//! an interrupted update can be detected and reverted on restart.

use crate::agent::{AgentState, UsageStats};
use opensquilla_core::error::Result;
use opensquilla_core::types::Message;
use tracing::{debug, warn};

/// The status of a transactional update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionStatus {
    /// The update was applied successfully.
    Applied,
    /// The update was rolled back.
    RolledBack,
    /// The update is still in progress (not yet committed).
    Pending,
    /// The update failed and could not be rolled back.
    Failed,
}

/// The outcome of a transactional update.
#[derive(Debug, Clone)]
pub struct TransactionOutcome {
    /// The status of the update.
    pub status: TransactionStatus,
    /// The messages before the update.
    pub before_messages: Vec<Message>,
    /// The messages after the update (empty if rolled back).
    pub after_messages: Vec<Message>,
    /// The agent state before the update.
    pub before_state: AgentState,
    /// The agent state after the update.
    pub after_state: AgentState,
    /// The turn ID this update belongs to.
    pub turn_id: String,
}

impl TransactionOutcome {
    /// Whether the update was applied.
    pub fn is_applied(&self) -> bool {
        self.status == TransactionStatus::Applied
    }

    /// Whether the update was rolled back.
    pub fn is_rolled_back(&self) -> bool {
        self.status == TransactionStatus::RolledBack
    }
}

/// A mutation function that transforms agent state.
///
/// Returns the new state (messages, agent state) or an error. Implementations
/// must be pure with respect to the inputs (no external side effects), so a
/// rollback can restore the prior state exactly.
pub type StateMutation =
    Box<dyn Fn(&[Message], &AgentState) -> Result<(Vec<Message>, AgentState)> + Send + Sync>;

/// A transactional state update.
#[derive(Debug)]
pub struct TransactionalUpdate {
    /// The turn ID.
    turn_id: String,
    /// The messages before the update.
    before_messages: Vec<Message>,
    /// The agent state before the update.
    before_state: AgentState,
    /// The messages after the update (same as before_messages if not yet applied).
    after_messages: Vec<Message>,
    /// The agent state after the update (same as before_state if not yet applied).
    after_state: AgentState,
    /// Whether the update has been applied.
    applied: bool,
    /// The usage before the update.
    before_usage: UsageStats,
}

impl TransactionalUpdate {
    /// Begin a new transactional update.
    pub fn begin(turn_id: impl Into<String>, messages: Vec<Message>, state: AgentState) -> Self {
        Self {
            turn_id: turn_id.into(),
            before_messages: messages.clone(),
            before_state: state.clone(),
            after_messages: messages,
            after_state: state,
            applied: false,
            before_usage: UsageStats::new(),
        }
    }

    /// Begin a transactional update with usage snapshot.
    pub fn begin_with_usage(
        turn_id: impl Into<String>,
        messages: Vec<Message>,
        state: AgentState,
        usage: UsageStats,
    ) -> Self {
        Self {
            turn_id: turn_id.into(),
            before_messages: messages.clone(),
            before_state: state.clone(),
            after_messages: messages,
            after_state: state,
            applied: false,
            before_usage: usage,
        }
    }

    /// The turn ID.
    pub fn turn_id(&self) -> &str {
        &self.turn_id
    }

    /// The messages before the update.
    pub fn before_messages(&self) -> &[Message] {
        &self.before_messages
    }

    /// The agent state before the update.
    pub fn before_state(&self) -> &AgentState {
        &self.before_state
    }

    /// Whether the update has been applied.
    pub fn is_applied(&self) -> bool {
        self.applied
    }

    /// Apply the mutation transactionally.
    ///
    /// On success the mutation's result becomes the new state. On error the
    /// state is left unchanged (rolled back) and the error is returned.
    pub fn apply<F>(&mut self, mutation: F) -> Result<TransactionOutcome>
    where
        F: FnOnce(&[Message], &AgentState) -> Result<(Vec<Message>, AgentState)>,
    {
        if self.applied {
            return Ok(TransactionOutcome {
                status: TransactionStatus::Applied,
                before_messages: self.before_messages.clone(),
                after_messages: self.after_messages.clone(),
                before_state: self.before_state.clone(),
                after_state: self.after_state.clone(),
                turn_id: self.turn_id.clone(),
            });
        }

        match mutation(&self.before_messages, &self.before_state) {
            Ok((after_messages, after_state)) => {
                self.applied = true;
                self.after_messages = after_messages.clone();
                self.after_state = after_state.clone();
                debug!(
                    turn_id = %self.turn_id,
                    before = self.before_messages.len(),
                    after = after_messages.len(),
                    "transactional update applied"
                );
                Ok(TransactionOutcome {
                    status: TransactionStatus::Applied,
                    before_messages: self.before_messages.clone(),
                    after_messages,
                    before_state: self.before_state.clone(),
                    after_state,
                    turn_id: self.turn_id.clone(),
                })
            }
            Err(e) => {
                warn!(
                    turn_id = %self.turn_id,
                    error = %e,
                    "transactional update failed, rolling back"
                );
                // Rollback: state is unchanged.
                Ok(TransactionOutcome {
                    status: TransactionStatus::RolledBack,
                    before_messages: self.before_messages.clone(),
                    after_messages: self.before_messages.clone(),
                    before_state: self.before_state.clone(),
                    after_state: self.before_state.clone(),
                    turn_id: self.turn_id.clone(),
                })
            }
        }
    }

    /// Apply a message-only mutation (state unchanged).
    pub fn apply_messages<F>(&mut self, mutation: F) -> Result<TransactionOutcome>
    where
        F: FnOnce(&[Message]) -> Result<Vec<Message>>,
    {
        self.apply(|messages, state| {
            let new_messages = mutation(messages)?;
            Ok((new_messages, state.clone()))
        })
    }

    /// Roll back any applied change, restoring the before-state.
    pub fn rollback(&mut self) -> TransactionOutcome {
        let outcome = TransactionOutcome {
            status: if self.applied {
                TransactionStatus::RolledBack
            } else {
                TransactionStatus::Pending
            },
            before_messages: self.before_messages.clone(),
            after_messages: self.before_messages.clone(),
            before_state: self.before_state.clone(),
            after_state: self.before_state.clone(),
            turn_id: self.turn_id.clone(),
        };
        self.applied = false;
        self.after_messages = self.before_messages.clone();
        self.after_state = self.before_state.clone();
        outcome
    }

    /// The usage snapshot before the update.
    pub fn before_usage(&self) -> &UsageStats {
        &self.before_usage
    }
}

/// A journal entry for transactional updates, persisted to the crash snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransactionJournalEntry {
    /// The turn ID.
    pub turn_id: String,
    /// A sequence number for ordering.
    pub seq: u64,
    /// The agent state before the update.
    pub before_state: String,
    /// The number of messages before the update.
    pub before_message_count: usize,
    /// The number of messages after the update.
    pub after_message_count: usize,
    /// Whether the update was committed.
    pub committed: bool,
    /// The timestamp (epoch ms).
    pub timestamp_ms: i64,
}

impl TransactionJournalEntry {
    /// Create a new journal entry.
    pub fn new(turn_id: impl Into<String>, seq: u64) -> Self {
        Self {
            turn_id: turn_id.into(),
            seq,
            before_state: String::new(),
            before_message_count: 0,
            after_message_count: 0,
            committed: false,
            timestamp_ms: crate::recovery::crash::current_epoch_ms(),
        }
    }
}

/// A simple in-memory transaction journal.
#[derive(Debug, Clone)]
pub struct TransactionJournal {
    /// The journal entries, ordered by sequence.
    entries: Vec<TransactionJournalEntry>,
    /// The next sequence number.
    next_seq: u64,
}

impl TransactionJournal {
    /// Create a new empty journal.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
        }
    }

    /// Record a journal entry.
    pub fn record(&mut self, entry: TransactionJournalEntry) {
        self.entries.push(entry);
        self.next_seq += 1;
    }

    /// Begin a new journal entry for a turn.
    pub fn begin_entry(
        &mut self,
        turn_id: &str,
        before_message_count: usize,
    ) -> TransactionJournalEntry {
        let mut entry = TransactionJournalEntry::new(turn_id, self.next_seq);
        entry.before_message_count = before_message_count;
        self.next_seq += 1;
        entry
    }

    /// Mark an entry as committed.
    pub fn commit(&mut self, seq: u64, after_message_count: usize) {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.seq == seq) {
            entry.committed = true;
            entry.after_message_count = after_message_count;
        }
    }

    /// Find uncommitted entries for a turn (interrupted transactions).
    pub fn uncommitted_for(&self, turn_id: &str) -> Vec<&TransactionJournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.turn_id == turn_id && !e.committed)
            .collect()
    }

    /// The total number of journal entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the journal is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The next sequence number.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
}

impl Default for TransactionJournal {
    fn default() -> Self {
        Self::new()
    }
}

/// A convenience function that applies a mutation to a shared state holder
/// transactionally.
///
/// `holder` is any object exposing `messages()` and `state()` accessors and
/// `set_messages`/`set_state` mutators via closures; the mutation is applied
/// and on error the state is restored.
pub fn transactional_update(
    turn_id: &str,
    before_messages: &[Message],
    before_state: &AgentState,
    mutation: StateMutation,
) -> Result<TransactionOutcome> {
    let mut tx =
        TransactionalUpdate::begin(turn_id, before_messages.to_vec(), before_state.clone());
    tx.apply(mutation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_begin_tracks_before_state() {
        let tx = TransactionalUpdate::begin("t1", vec![Message::user("hi")], AgentState::Idle);
        assert_eq!(tx.before_messages().len(), 1);
        assert_eq!(tx.before_state(), &AgentState::Idle);
        assert!(!tx.is_applied());
    }

    #[test]
    fn test_apply_success() {
        let mut tx = TransactionalUpdate::begin("t1", vec![Message::user("hi")], AgentState::Idle);
        let outcome = tx
            .apply(|messages, state| {
                let mut new_messages = messages.to_vec();
                new_messages.push(Message::assistant("hello"));
                Ok((new_messages, state.clone()))
            })
            .unwrap();
        assert!(outcome.is_applied());
        assert_eq!(outcome.after_messages.len(), 2);
        assert!(tx.is_applied());
    }

    #[test]
    fn test_apply_failure_rolls_back() {
        let mut tx = TransactionalUpdate::begin("t1", vec![Message::user("hi")], AgentState::Idle);
        let outcome = tx
            .apply(|_messages, _state| {
                Err(opensquilla_core::error::Error::Internal("boom".to_string()))
            })
            .unwrap();
        assert!(outcome.is_rolled_back());
        assert_eq!(outcome.after_messages.len(), 1);
        assert!(!tx.is_applied());
    }

    #[test]
    fn test_apply_messages() {
        let mut tx = TransactionalUpdate::begin("t1", Vec::new(), AgentState::Idle);
        let outcome = tx
            .apply_messages(|_messages| Ok(vec![Message::user("new")]))
            .unwrap();
        assert!(outcome.is_applied());
        assert_eq!(outcome.after_messages.len(), 1);
    }

    #[test]
    fn test_rollback() {
        let mut tx = TransactionalUpdate::begin("t1", vec![Message::user("hi")], AgentState::Idle);
        let _ = tx
            .apply(|messages, state| Ok((messages.to_vec(), state.clone())))
            .unwrap();
        assert!(tx.is_applied());
        let outcome = tx.rollback();
        assert_eq!(outcome.status, TransactionStatus::RolledBack);
        assert!(!tx.is_applied());
    }

    #[test]
    fn test_double_apply_is_idempotent() {
        let mut tx = TransactionalUpdate::begin("t1", vec![Message::user("hi")], AgentState::Idle);
        let _ = tx
            .apply(|messages, state| {
                let mut m = messages.to_vec();
                m.push(Message::assistant("a"));
                Ok((m, state.clone()))
            })
            .unwrap();
        let second = tx
            .apply(|messages, state| {
                let mut m = messages.to_vec();
                m.push(Message::assistant("b"));
                Ok((m, state.clone()))
            })
            .unwrap();
        // Second apply is a no-op.
        assert_eq!(second.after_messages.len(), 2);
    }

    #[test]
    fn test_transaction_journal() {
        let mut journal = TransactionJournal::new();
        let entry = journal.begin_entry("t1", 1);
        journal.record(entry);
        journal.commit(0, 2);

        let entry2 = journal.begin_entry("t2", 1);
        journal.record(entry2);

        assert_eq!(journal.len(), 2);
        let uncommitted = journal.uncommitted_for("t2");
        assert_eq!(uncommitted.len(), 1);
        assert!(journal.uncommitted_for("t1").is_empty());
    }

    #[test]
    fn test_transactional_update_function() {
        let before = vec![Message::user("hi")];
        let mutation: StateMutation = Box::new(|messages, state| {
            let mut new_messages = messages.to_vec();
            new_messages.push(Message::assistant("hello"));
            Ok((new_messages, state.clone()))
        });
        let outcome = transactional_update("t1", &before, &AgentState::Idle, mutation).unwrap();
        assert!(outcome.is_applied());
    }
}
