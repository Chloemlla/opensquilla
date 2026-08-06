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

/// A concrete compaction plan for a conversation.
#[derive(Debug, Clone)]
pub struct CompactionPlan {
    /// The strategy to apply.
    pub strategy: CompactionStrategy,
    /// The number of messages to drop.
    pub messages_to_drop: usize,
    /// The estimated tokens to reclaim.
    pub reclaimable_tokens: u64,
    /// The estimated tokens remaining after compaction.
    pub remaining_tokens: u64,
    /// The message count remaining after compaction.
    pub remaining_messages: usize,
    /// The urgency of the compaction.
    pub urgent: bool,
    /// Whether the plan is a no-op (nothing to do).
    pub is_noop: bool,
}

impl CompactionPlan {
    /// Whether the plan drops any messages.
    pub fn reclaims_messages(&self) -> bool {
        self.messages_to_drop > 0
    }
}

/// Plan a compaction for the given conversation profile.
///
/// This is the "planning" half of compaction: given the token count, message
/// count, and context window, it decides the strategy, how many messages to
/// drop, and the projected savings — without mutating the conversation.
///
/// `message_count` and `avg_tokens_per_message` are used to estimate the
/// per-message cost. The plan keeps at least `min_keep` messages.
pub fn plan_compaction(
    token_count: u64,
    message_count: usize,
    context_window_tokens: u64,
    avg_tokens_per_message: u64,
    min_keep: usize,
) -> CompactionPlan {
    let decision = decide_compaction(&CompactionInput {
        token_count,
        message_count,
        context_window_tokens,
        ..Default::default()
    });

    let (strategy, urgent) = match decision {
        CompactionDecision::NoCompaction => {
            return CompactionPlan {
                strategy: CompactionStrategy::Summarize,
                messages_to_drop: 0,
                reclaimable_tokens: 0,
                remaining_tokens: token_count,
                remaining_messages: message_count,
                urgent: false,
                is_noop: true,
            };
        }
        CompactionDecision::Compact(s) => (s, false),
        CompactionDecision::UrgentCompaction(s) => (s, true),
    };

    // Estimate how many messages must be dropped to get under budget.
    let budget = truncation_budget(message_count.max(4).max(min_keep));
    let keep = budget.max(min_keep);
    let messages_to_drop = message_count.saturating_sub(keep);

    let reclaimable = projected_reclaim(messages_to_drop, avg_tokens_per_message.max(1));
    let remaining_tokens = token_count.saturating_sub(reclaimable);
    let remaining_messages = message_count.saturating_sub(messages_to_drop);

    CompactionPlan {
        strategy,
        messages_to_drop,
        reclaimable_tokens: reclaimable,
        remaining_tokens,
        remaining_messages,
        urgent,
        is_noop: false,
    }
}

/// Estimate the average token size of messages given a total and count.
pub fn average_message_tokens(token_count: u64, message_count: usize) -> u64 {
    if message_count == 0 {
        0
    } else {
        token_count / message_count as u64
    }
}

/// Whether a conversation needs compaction before the next generation round.
///
/// This combines the token threshold with a lookahead: if the current round's
/// estimated output would push the total over the urgent threshold, compaction
/// is warranted even when the current tokens are under the trigger.
pub fn should_compact_before_generation(
    token_count: u64,
    context_window_tokens: u64,
    estimated_output_tokens: u64,
) -> bool {
    let input = CompactionInput {
        token_count,
        context_window_tokens,
        ..Default::default()
    };
    let threshold = input.threshold_tokens();
    let urgent = input.urgent_tokens();
    token_count >= threshold
        || token_count + estimated_output_tokens >= urgent
        || token_count.saturating_add(estimated_output_tokens) >= context_window_tokens
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
