use std::collections::HashMap;

use chrono::{DateTime, Utc};
use opensquilla_core::config::Config;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// A session message for merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// A session to be merged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeableSession {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub messages: Vec<SessionMessage>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Result of a session merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    pub new_session_id: String,
    pub merged_from: Vec<String>,
    pub total_messages: usize,
    pub deduplicated: usize,
    pub timestamp: DateTime<Utc>,
}

/// Strategy for merging sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Combine all messages in chronological order.
    Chronological,
    /// Keep messages from the most recent session, then append older ones.
    MostRecentFirst,
    /// Keep only unique messages, removing duplicates by content.
    Deduplicate,
    /// Keep messages from the specified primary session only.
    KeepPrimary,
}

/// Session merge manager.
#[derive(Debug, Clone)]
pub struct SessionMerge {
    config: Config,
    strategy: MergeStrategy,
}

impl SessionMerge {
    /// Create a new session merge manager.
    pub fn new(config: &Config) -> Self {
        let strategy = match config
            .get("merge.strategy")
            .unwrap_or_else(|| "chronological".to_string())
            .as_str()
        {
            "chronological" => MergeStrategy::Chronological,
            "most_recent_first" => MergeStrategy::MostRecentFirst,
            "deduplicate" => MergeStrategy::Deduplicate,
            "keep_primary" => MergeStrategy::KeepPrimary,
            _ => MergeStrategy::Chronological,
        };

        info!("Session merge initialized with strategy {:?}", strategy);

        Self {
            config: config.clone(),
            strategy,
        }
    }

    /// Set the merge strategy.
    pub fn set_strategy(&mut self, strategy: MergeStrategy) {
        self.strategy = strategy;
        debug!("Merge strategy set to {:?}", strategy);
    }

    /// Merge multiple sessions into one.
    pub async fn merge_sessions(
        &self,
        sessions: Vec<MergeableSession>,
        primary_session_id: Option<&str>,
    ) -> Result<MergeResult, MergeError> {
        if sessions.is_empty() {
            return Err(MergeError::NoSessions);
        }

        if sessions.len() == 1 {
            return Ok(MergeResult {
                new_session_id: sessions[0].id.clone(),
                merged_from: vec![sessions[0].id.clone()],
                total_messages: sessions[0].messages.len(),
                deduplicated: 0,
                timestamp: Utc::now(),
            });
        }

        let session_ids: Vec<String> = sessions.iter().map(|s| s.id.clone()).collect();
        let mut all_messages: Vec<SessionMessage> = Vec::new();
        let mut deduplicated = 0;

        match self.strategy {
            MergeStrategy::Chronological => {
                let mut all: Vec<&SessionMessage> = sessions
                    .iter()
                    .flat_map(|s| &s.messages)
                    .collect();
                all.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

                // Deduplicate by content
                let mut seen = std::collections::HashSet::new();
                for msg in all {
                    let key = format!("{}{}", msg.role, msg.content);
                    if seen.insert(key) {
                        all_messages.push(msg.clone());
                    } else {
                        deduplicated += 1;
                    }
                }
            }
            MergeStrategy::MostRecentFirst => {
                let mut sessions_sorted = sessions.clone();
                sessions_sorted.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

                let mut seen = std::collections::HashSet::new();
                for session in &sessions_sorted {
                    for msg in &session.messages {
                        let key = format!("{}{}", msg.role, msg.content);
                        if seen.insert(key) {
                            all_messages.push(msg.clone());
                        } else {
                            deduplicated += 1;
                        }
                    }
                }
            }
            MergeStrategy::Deduplicate => {
                let mut seen = std::collections::HashSet::new();
                for session in &sessions {
                    for msg in &session.messages {
                        let key = format!("{}{}", msg.role, msg.content);
                        if seen.insert(key) {
                            all_messages.push(msg.clone());
                        } else {
                            deduplicated += 1;
                        }
                    }
                }
                // Sort by timestamp
                all_messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
            }
            MergeStrategy::KeepPrimary => {
                let primary_id = primary_session_id.ok_or(MergeError::PrimarySessionRequired)?;
                let primary = sessions
                    .iter()
                    .find(|s| s.id == primary_id)
                    .ok_or(MergeError::PrimarySessionNotFound(primary_id.to_string()))?;

                all_messages = primary.messages.clone();

                // Append messages from other sessions that aren't duplicates
                let mut seen: std::collections::HashSet<String> = all_messages
                    .iter()
                    .map(|m| format!("{}{}", m.role, m.content))
                    .collect();

                for session in &sessions {
                    if session.id == primary_id {
                        continue;
                    }
                    for msg in &session.messages {
                        let key = format!("{}{}", msg.role, msg.content);
                        if seen.insert(key) {
                            all_messages.push(msg.clone());
                        } else {
                            deduplicated += 1;
                        }
                    }
                }
            }
        }

        // Sort by timestamp for final output
        all_messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let new_session_id = uuid::Uuid::new_v4().to_string();

        info!(
            "Merged {} sessions into {new_session_id}: {} messages, {deduplicated} deduplicated",
            sessions.len(),
            all_messages.len()
        );

        Ok(MergeResult {
            new_session_id,
            merged_from: session_ids,
            total_messages: all_messages.len(),
            deduplicated,
            timestamp: Utc::now(),
        })
    }

    /// Merge sessions by their IDs.
    pub async fn merge_by_ids(
        &self,
        session_ids: &[String],
        session_loader: impl Fn(&str) -> futures::future::BoxFuture<'_, Result<MergeableSession, MergeError>>,
    ) -> Result<MergeResult, MergeError> {
        if session_ids.is_empty() {
            return Err(MergeError::NoSessions);
        }

        let mut sessions = Vec::new();
        for id in session_ids {
            let session = session_loader(id).await?;
            sessions.push(session);
        }

        let primary = session_ids.first().map(|s| s.as_str());
        self.merge_sessions(sessions, primary).await
    }

    /// Get the current merge strategy.
    pub fn strategy(&self) -> MergeStrategy {
        self.strategy
    }

    /// Get a reference to the underlying config.
    pub fn config(&self) -> &Config {
        &self.config
    }
}

/// A conflicting entry between a source and target session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    /// The message from the source session.
    pub source_message: SessionMessage,
    /// The message from the target session.
    pub target_message: SessionMessage,
    /// A human-readable reason for the conflict.
    pub reason: String,
}

/// A single planned operation in a [`MergePlan`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeOperation {
    /// Keep the message from the source session.
    KeepSource(SessionMessage),
    /// Keep the message from the target session.
    KeepTarget(SessionMessage),
    /// Drop the message as a duplicate.
    Deduplicate(SessionMessage),
    /// The entry requires conflict resolution.
    Conflict(MergeConflict),
}

/// A planned merge of two sessions, prior to execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePlan {
    /// The source session ids (in order).
    pub source_ids: Vec<String>,
    /// The id of the resulting session.
    pub target_id: String,
    /// The strategy used to build the plan.
    pub strategy: MergeStrategy,
    /// The ordered operations of the merge.
    pub operations: Vec<MergeOperation>,
}

/// How a single merge conflict is resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeResolution {
    KeepSource(SessionMessage),
    KeepTarget(SessionMessage),
    KeepBoth,
    Drop,
}

impl SessionMerge {
    /// Merge two sessions into one, deduplicating by `(role, content)`.
    ///
    /// This is the pairwise complement of the batch
    /// [`SessionMerge::merge_sessions`]; the resulting messages are sorted
    /// chronologically.
    pub async fn merge_sessions_pair(
        &self,
        source: &MergeableSession,
        target: &MergeableSession,
    ) -> Result<MergeResult, MergeError> {
        let merged = self.merge_turns(source.messages.clone(), target.messages.clone());
        let deduplicated =
            source.messages.len() + target.messages.len() - merged.len();
        Ok(MergeResult {
            new_session_id: uuid::Uuid::new_v4().to_string(),
            merged_from: vec![source.id.clone(), target.id.clone()],
            total_messages: merged.len(),
            deduplicated,
            timestamp: Utc::now(),
        })
    }

    /// Merge two turn histories into one, deduplicating by `(role, content)`.
    pub fn merge_turns(
        &self,
        source_turns: Vec<SessionMessage>,
        target_turns: Vec<SessionMessage>,
    ) -> Vec<SessionMessage> {
        let mut all = source_turns;
        let mut seen: std::collections::HashSet<String> = all
            .iter()
            .map(|m| format!("{}{}", m.role, m.content))
            .collect();
        for message in target_turns {
            let key = format!("{}{}", message.role, message.content);
            if seen.insert(key) {
                all.push(message);
            }
        }
        all.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        all
    }

    /// Resolve a set of conflicts using the configured strategy.
    pub fn resolve_conflicts(&self, conflicts: Vec<MergeConflict>) -> Vec<MergeResolution> {
        conflicts
            .into_iter()
            .map(|conflict| match self.strategy {
                MergeStrategy::MostRecentFirst => {
                    if conflict.source_message.timestamp >= conflict.target_message.timestamp {
                        MergeResolution::KeepSource(conflict.source_message)
                    } else {
                        MergeResolution::KeepTarget(conflict.target_message)
                    }
                }
                MergeStrategy::KeepPrimary => {
                    MergeResolution::KeepTarget(conflict.target_message)
                }
                _ => MergeResolution::KeepBoth,
            })
            .collect()
    }

    /// Build a merge plan describing the operations a pairwise merge would take,
    /// without executing it.
    pub fn plan_merge(
        &self,
        source: &MergeableSession,
        target: &MergeableSession,
    ) -> MergePlan {
        let mut seen: std::collections::HashSet<String> =
            source.messages.iter().map(msg_key).collect();
        let mut operations: Vec<MergeOperation> = source
            .messages
            .iter()
            .cloned()
            .map(MergeOperation::KeepSource)
            .collect();

        for message in &target.messages {
            let key = msg_key(message);
            if !seen.insert(key) {
                operations.push(MergeOperation::Deduplicate(message.clone()));
            } else if is_conflicting_with(&operations, message) {
                operations.push(MergeOperation::Conflict(MergeConflict {
                    source_message: message.clone(),
                    target_message: message.clone(),
                    reason: "overlapping turn region".to_string(),
                }));
            } else {
                operations.push(MergeOperation::KeepTarget(message.clone()));
            }
        }

        MergePlan {
            source_ids: vec![source.id.clone(), target.id.clone()],
            target_id: uuid::Uuid::new_v4().to_string(),
            strategy: self.strategy,
            operations,
        }
    }
}

/// Dedup key for a session message.
fn msg_key(message: &SessionMessage) -> String {
    format!("{}{}", message.role, message.content)
}

/// Whether a message conflicts with an existing planned operation (same role
/// and overlapping timestamp window).
fn is_conflicting_with(operations: &[MergeOperation], message: &SessionMessage) -> bool {
    operations.iter().any(|op| {
        let existing = match op {
            MergeOperation::KeepSource(m)
            | MergeOperation::KeepTarget(m)
            | MergeOperation::Deduplicate(m) => m,
            MergeOperation::Conflict(c) => &c.target_message,
        };
        existing.role == message.role
            && (existing.timestamp - message.timestamp).num_seconds().abs() <= 1
            && existing.content != message.content
    })
}

#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    #[error("No sessions provided for merge")]
    NoSessions,

    #[error("Primary session ID is required for KeepPrimary strategy")]
    PrimarySessionRequired,

    #[error("Primary session not found: {0}")]
    PrimarySessionNotFound(String),

    #[error("Session load error: {0}")]
    SessionLoadError(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use opensquilla_core::config::Config;

    fn message(role: &str, content: &str, ts: DateTime<Utc>) -> SessionMessage {
        SessionMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: ts,
            metadata: Default::default(),
        }
    }

    fn session(id: &str, messages: Vec<SessionMessage>) -> MergeableSession {
        let now = Utc::now();
        MergeableSession {
            id: id.to_string(),
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
            messages,
            created_at: now,
            updated_at: now,
            metadata: Default::default(),
        }
    }

    fn merge() -> SessionMerge {
        SessionMerge::new(&Config::default())
    }

    #[tokio::test]
    async fn test_merge_sessions_pair_deduplicates() {
        let now = Utc::now();
        let source = session(
            "s1",
            vec![
                message("user", "hello", now),
                message("assistant", "hi there", now),
            ],
        );
        let target = session(
            "s2",
            vec![
                message("user", "hello", now), // duplicate
                message("user", "what is rust?", now),
            ],
        );
        let result = merge().merge_sessions_pair(&source, &target).await.unwrap();
        assert_eq!(result.total_messages, 3);
        assert_eq!(result.deduplicated, 1);
        assert_eq!(result.merged_from, vec!["s1", "s2"]);
    }

    #[test]
    fn test_merge_turns_sorts_and_deduplicates() {
        let t1 = Utc::now();
        let t0 = t1 - chrono::Duration::seconds(10);
        let source = vec![
            message("assistant", "later", t1),
            message("user", "earlier", t0),
        ];
        let target = vec![message("user", "earlier", t0)]; // duplicate
        let merged = merge().merge_turns(source, target);
        assert_eq!(merged.len(), 2);
        // Sorted ascending by timestamp.
        assert!(merged[0].content == "earlier");
        assert!(merged[1].content == "later");
    }

    #[test]
    fn test_resolve_conflicts_most_recent_first() {
        let old = Utc::now() - chrono::Duration::seconds(60);
        let new = Utc::now();
        let conflict = MergeConflict {
            source_message: message("assistant", "old answer", old),
            target_message: message("assistant", "new answer", new),
            reason: "overlap".to_string(),
        };
        let mut sess = merge();
        sess.set_strategy(MergeStrategy::MostRecentFirst);
        let resolutions = sess.resolve_conflicts(vec![conflict]);
        match &resolutions[0] {
            MergeResolution::KeepTarget(m) => assert_eq!(m.content, "new answer"),
            _ => panic!("expected KeepTarget for most recent"),
        }
    }

    #[test]
    fn test_plan_merge_contains_operations() {
        let now = Utc::now();
        let source = session("s1", vec![message("user", "hi", now)]);
        let target = session("s2", vec![message("assistant", "hello", now)]);
        let plan = merge().plan_merge(&source, &target);
        assert_eq!(plan.source_ids, vec!["s1", "s2"]);
        assert!(!plan.operations.is_empty());
        assert!(plan
            .operations
            .iter()
            .any(|op| matches!(op, MergeOperation::KeepTarget(_))));
    }
}