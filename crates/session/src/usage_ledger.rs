use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageEntry {
    pub id: Uuid,
    pub session_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_nanodollars: u64,
    pub model: String,
    pub provider: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSummary {
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub total_cost_nanodollars: u64,
    pub total_calls: u64,
    pub by_model: std::collections::HashMap<String, ModelUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cost_nanodollars: u64,
}

pub struct UsageLedger {
    entries: DashMap<Uuid, Vec<UsageEntry>>,
    session_totals: DashMap<Uuid, UsageSummary>,
}

impl UsageLedger {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
            session_totals: DashMap::new(),
        }
    }

    pub fn record(
        &self,
        session_id: Uuid,
        prompt_tokens: u64,
        completion_tokens: u64,
        model: String,
        provider: String,
    ) -> UsageEntry {
        let cost_nanodollars = Self::compute_cost_nanodollars(prompt_tokens, completion_tokens, &model);

        let entry = UsageEntry {
            id: Uuid::new_v4(),
            session_id,
            timestamp: Utc::now(),
            prompt_tokens,
            completion_tokens,
            cost_nanodollars,
            model: model.clone(),
            provider,
        };

        self.entries.entry(session_id).or_default().push(entry.clone());

        self.session_totals
            .entry(session_id)
            .and_modify(|s| {
                s.total_prompt_tokens += prompt_tokens;
                s.total_completion_tokens += completion_tokens;
                s.total_cost_nanodollars += cost_nanodollars;
                s.total_calls += 1;
                s.by_model
                    .entry(model.clone())
                    .and_modify(|m| {
                        m.calls += 1;
                        m.prompt_tokens += prompt_tokens;
                        m.completion_tokens += completion_tokens;
                        m.cost_nanodollars += cost_nanodollars;
                    })
                    .or_insert(ModelUsage {
                        calls: 1,
                        prompt_tokens,
                        completion_tokens,
                        cost_nanodollars,
                    });
            })
            .or_insert_with(|| {
                let mut by_model = std::collections::HashMap::new();
                by_model.insert(
                    model.clone(),
                    ModelUsage {
                        calls: 1,
                        prompt_tokens,
                        completion_tokens,
                        cost_nanodollars,
                    },
                );
                UsageSummary {
                    total_prompt_tokens: prompt_tokens,
                    total_completion_tokens: completion_tokens,
                    total_cost_nanodollars: cost_nanodollars,
                    total_calls: 1,
                    by_model,
                }
            });

        entry
    }

    pub fn get_summary(&self, session_id: &Uuid) -> Option<UsageSummary> {
        self.session_totals.get(session_id).map(|s| s.clone())
    }

    pub fn get_entries(&self, session_id: &Uuid) -> Vec<UsageEntry> {
        self.entries
            .get(session_id)
            .map(|e| e.clone())
            .unwrap_or_default()
    }

    pub fn all_sessions(&self) -> Vec<Uuid> {
        self.entries.iter().map(|e| *e.key()).collect()
    }

    /// Compute cost in nanodollars (1 nanodollar = 1e-9 USD).
    /// Uses approximate pricing per 1K tokens by model family.
    fn compute_cost_nanodollars(prompt_tokens: u64, completion_tokens: u64, model: &str) -> u64 {
        let model_lower = model.to_lowercase();

        let (prompt_per_1k_nano, completion_per_1k_nano) = if model_lower.contains("gpt-4") {
            (30_000_000, 60_000_000) // $30/m tok prompt, $60/m tok completion
        } else if model_lower.contains("gpt-3.5") {
            (1_500_000, 2_000_000)
        } else if model_lower.contains("claude-3-opus") || model_lower.contains("claude-3.5") {
            (15_000_000, 75_000_000)
        } else if model_lower.contains("claude-3-sonnet") {
            (3_000_000, 15_000_000)
        } else if model_lower.contains("claude-3-haiku") {
            (250_000, 1_250_000)
        } else if model_lower.contains("gemini-1.5-pro") {
            (3_500_000, 10_500_000)
        } else if model_lower.contains("gemini-1.5-flash") {
            (75_000, 300_000)
        } else {
            (1_000_000, 2_000_000) // default
        };

        let prompt_cost = (prompt_tokens as u128 * prompt_per_1k_nano as u128) / 1000;
        let completion_cost = (completion_tokens as u128 * completion_per_1k_nano as u128) / 1000;

        (prompt_cost + completion_cost) as u64
    }
}

impl Default for UsageLedger {
    fn default() -> Self {
        Self::new()
    }
}