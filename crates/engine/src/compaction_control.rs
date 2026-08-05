//! Compaction continuation decisions.
//!
//! Pure functions that decide whether (and when) to compact a conversation
//! based on token count, message count, and the configured context window.
//! This module performs NO I/O; the runtime applies the returned decision.

/// How aggressively to compact when over budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Summarize old messages into a single summary message.
    Summarize,
    /// Drop the least-relevant older messages without summarizing.
    DropOldest,
    /// Keep only the most recent messages within a hard cap.
    Truncate,
}

/// The outcome of a compaction decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionDecision {
    /// Compaction is not needed; the context fits within the window.
    NoCompaction,
    /// Compaction should run now with the given strategy.
    Compact(CompactionStrategy),
    /// The context is far over budget; compaction is urgent.
    UrgentCompaction(CompactionStrategy),
}

impl CompactionDecision {
    /// Returns true if any compaction should occur.
    pub fn should_compact(self) -> bool {
        !matches!(self, CompactionDecision::NoCompaction)
    }

    /// Returns true if compaction should be treated as urgent.
    pub fn is_urgent(self) -> bool {
        matches!(self, CompactionDecision::UrgentCompaction(_))
    }
}

/// Inputs to the compaction decision function.
#[derive(Debug, Clone)]
pub struct CompactionInput {
    /// The current estimated token count of the conversation.
    pub token_count: u64,
    /// The current number of messages in the conversation.
    pub message_count: usize,
    /// The context window size of the model, in tokens.
    pub context_window_tokens: u64,
    /// The fraction of the context window at which compaction triggers.
    ///
    /// Defaults to 0.75 when zero.
    pub threshold_fraction: f64,
    /// The fraction of the context window at which compaction is urgent.
    ///
    /// Defaults to 0.9 when zero. Must be >= threshold.
    pub urgent_fraction: f64,
    /// The minimum message count before compaction is ever considered.
    ///
    /// Defaults to 10 when zero.
    pub min_messages: usize,
    /// The preferred strategy when no explicit strategy is given.
    pub strategy: CompactionStrategy,
}

impl Default for CompactionInput {
    fn default() -> Self {
        Self {
            token_count: 0,
            message_count: 0,
            context_window_tokens: 128_000,
            threshold_fraction: 0.75,
            urgent_fraction: 0.9,
            min_messages: 10,
            strategy: CompactionStrategy::Summarize,
        }
    }
}

impl CompactionInput {
    /// The number of tokens at which compaction should trigger.
    pub fn threshold_tokens(&self) -> u64 {
        let window = self.context_window_tokens.max(1);
        let frac = if self.threshold_fraction > 0.0 {
            self.threshold_fraction.min(1.0)
        } else {
            0.75
        };
        ((window as f64) * frac) as u64
    }

    /// The number of tokens at which compaction is urgent.
    pub fn urgent_tokens(&self) -> u64 {
        let window = self.context_window_tokens.max(1);
        let frac = if self.urgent_fraction > 0.0 {
            self.urgent_fraction.min(1.0)
        } else {
            0.9
        };
        ((window as f64) * frac) as u64
    }

    /// The fraction of the context window currently consumed.
    pub fn utilization(&self) -> f64 {
        if self.context_window_tokens == 0 {
            return 0.0;
        }
        self.token_count as f64 / self.context_window_tokens as f64
    }
}

/// Decide whether and how to compact.
///
/// # Decision order
///
/// 1. If the token count is at or above the urgent threshold, return
///    `UrgentCompaction`.
/// 2. If the token count is at or above the normal threshold AND the message
///    count is at least `min_messages`, return `Compact`.
/// 3. If the message count alone exceeds a sane bound (e.g. thousands of tiny
///    messages), return `Compact(Truncate)`.
/// 4. Otherwise return `NoCompaction`.
pub fn decide_compaction(input: &CompactionInput) -> CompactionDecision {
    let threshold = input.threshold_tokens();
    let urgent = input.urgent_tokens();

    // 1. Urgent compaction.
    if input.token_count >= urgent {
        return CompactionDecision::UrgentCompaction(input.strategy);
    }

    // 2. Normal threshold compaction.
    if input.token_count >= threshold && input.message_count >= input.min_messages.max(1) {
        return CompactionDecision::Compact(input.strategy);
    }

    // 3. Message-count-only bound: many tiny messages still blow up request
    //    overhead. 1024 messages is an aggressive hard ceiling.
    if input.message_count > 1024 {
        return CompactionDecision::Compact(CompactionStrategy::Truncate);
    }

    // 4. Nothing to do.
    CompactionDecision::NoCompaction
}

/// Decide the number of messages to keep when truncating.
///
/// Keeps the system messages plus the most recent `budget` non-system
/// messages.
pub fn truncation_budget(max_messages: usize) -> usize {
    max_messages.max(4).saturating_sub(1)
}

/// Estimate whether a message budget is sufficient for a tool round trip.
///
/// Tool loops typically need at least: user message + assistant tool_use +
/// tool result + assistant response. Returns the minimum recommended message
/// budget.
pub fn recommended_message_budget(tool_rounds: u32) -> usize {
    // Each round consumes ~3 messages (assistant tool_use, tool result,
    // following assistant text). Include slack for the original user message.
    (tool_rounds as usize) * 3 + 4
}

/// Returns the number of tokens projected to be reclaimed by dropping
/// `messages_to_drop` messages with the given average token size.
pub fn projected_reclaim(dropped_messages: usize, avg_tokens_per_message: u64) -> u64 {
    dropped_messages as u64 * avg_tokens_per_message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compaction_under_threshold() {
        let input = CompactionInput {
            token_count: 50_000,
            message_count: 20,
            context_window_tokens: 128_000,
            ..Default::default()
        };
        assert_eq!(decide_compaction(&input), CompactionDecision::NoCompaction);
    }

    #[test]
    fn test_compact_at_threshold() {
        let input = CompactionInput {
            token_count: 100_000,
            message_count: 20,
            context_window_tokens: 128_000,
            ..Default::default()
        };
        // 100k >= 96k (75% of 128k).
        assert_eq!(
            decide_compaction(&input),
            CompactionDecision::Compact(CompactionStrategy::Summarize)
        );
    }

    #[test]
    fn test_urgent_compaction() {
        let input = CompactionInput {
            token_count: 120_000,
            message_count: 20,
            context_window_tokens: 128_000,
            ..Default::default()
        };
        assert_eq!(
            decide_compaction(&input),
            CompactionDecision::UrgentCompaction(CompactionStrategy::Summarize)
        );
    }

    #[test]
    fn test_min_messages_gate() {
        let input = CompactionInput {
            token_count: 100_000,
            message_count: 3,
            context_window_tokens: 128_000,
            ..Default::default()
        };
        // Token threshold hit but too few messages: no compaction.
        assert_eq!(decide_compaction(&input), CompactionDecision::NoCompaction);
    }

    #[test]
    fn test_message_count_bound() {
        let input = CompactionInput {
            token_count: 10_000,
            message_count: 2_000,
            context_window_tokens: 128_000,
            ..Default::default()
        };
        assert_eq!(
            decide_compaction(&input),
            CompactionDecision::Compact(CompactionStrategy::Truncate)
        );
    }

    #[test]
    fn test_threshold_and_urgent_values() {
        let input = CompactionInput::default();
        assert_eq!(input.threshold_tokens(), 96_000);
        assert_eq!(input.urgent_tokens(), 115_200);
    }
}
