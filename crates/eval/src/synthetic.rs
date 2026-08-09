//! Offline synthetic provider powering the ensemble benchmark's `--dry-run`.
//!
//! Mirrors `src/opensquilla/eval/synthetic.py`. [`SyntheticProvider`] replays
//! one fixed successful turn (an optional text delta plus a terminal `Done`)
//! with no network or credentials, carrying the simulated token counts, billed
//! cost, model id, and (for the ensemble arm) a synthetic `ensemble_trace`. It
//! lives in `eval` rather than `provider` because it only serves the
//! evaluation harness and is never constructed on a live path.

use async_trait::async_trait;
use futures::Stream;
use futures::StreamExt;
use opensquilla_core::types::{ChatMessage, ToolDefinition, Usage};
use opensquilla_provider::types::{
    ChatConfig, ChatProvider, Provider, ProviderResponse, ProviderResult, StreamEvent,
};

/// Deterministic offline provider that always streams one success turn.
///
/// Every `chat`/`stream_chat` call yields the same shape: an optional text
/// delta followed by a terminal [`StreamEvent::Done`] carrying the configured
/// token counts, billed cost, model id, and (for the ensemble arm) a synthetic
/// `ensemble_trace`. `provider_name` defaults to a registered provider id
/// (`"openai"`) so failures round-trip through the shared failure taxonomy.
#[derive(Debug, Clone)]
pub struct SyntheticProvider {
    model: String,
    provider_name: String,
    input_tokens: u64,
    output_tokens: u64,
    billed_cost: f64,
    cost_source: String,
    text: String,
    ensemble_trace: Option<serde_json::Value>,
}

impl Default for SyntheticProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticProvider {
    /// Create a synthetic provider with the default match-Python settings.
    pub fn new() -> Self {
        Self {
            model: "synthetic-model".to_string(),
            provider_name: "openai".to_string(),
            input_tokens: 1200,
            output_tokens: 400,
            billed_cost: 0.0,
            cost_source: "synthetic".to_string(),
            text: "synthetic answer".to_string(),
            ensemble_trace: None,
        }
    }

    /// Set the model id.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the provider id.
    pub fn provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Set the simulated input token count.
    pub fn input_tokens(mut self, n: u64) -> Self {
        self.input_tokens = n;
        self
    }

    /// Set the simulated output token count.
    pub fn output_tokens(mut self, n: u64) -> Self {
        self.output_tokens = n;
        self
    }

    /// Set the reported billed cost (USD).
    pub fn billed_cost(mut self, cost: f64) -> Self {
        self.billed_cost = cost;
        self
    }

    /// Set the cost source label.
    pub fn cost_source(mut self, source: impl Into<String>) -> Self {
        self.cost_source = source.into();
        self
    }

    /// Set the fixed response text (empty disables the text delta).
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Set the ensemble trace carried on the terminal `Done` event.
    pub fn ensemble_trace(mut self, trace: serde_json::Value) -> Self {
        self.ensemble_trace = Some(trace);
        self
    }

    /// The configured model id.
    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// List supported models — always empty for the synthetic provider.
    pub async fn list_models(&self) -> Vec<String> {
        Vec::new()
    }
}

#[async_trait]
impl Provider for SyntheticProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn supported_models(&self) -> Vec<String> {
        Vec::new()
    }

    async fn send_message(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<ProviderResponse> {
        let stream = self.stream_chat(config, messages, tools).await?;

        let mut text = String::new();
        let mut usage = Usage::default();
        let mut stop_reason = None;
        let mut billed_cost = None;
        let mut cost_source = None;
        let mut ensemble_trace = None;

        futures::pin_mut!(stream);
        while let Some(ev) = stream.next().await {
            match ev? {
                StreamEvent::Text { text: t } => text.push_str(&t),
                StreamEvent::Done {
                    usage: u,
                    stop_reason: s,
                    billed_cost: bc,
                    cost_source: cs,
                    ensemble_trace: et,
                    ..
                } => {
                    if let Some(u) = u {
                        usage = u;
                    }
                    stop_reason = s;
                    billed_cost = bc;
                    cost_source = cs;
                    ensemble_trace = et;
                }
                _ => {}
            }
        }

        Ok(ProviderResponse {
            content: vec![ChatMessage::assistant(text)],
            usage,
            model: self.model.clone(),
            stop_reason,
            billed_cost,
            cost_source,
            ensemble_trace,
        })
    }

    async fn stream_chat(
        &self,
        _config: &ChatConfig,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        let mut events: Vec<ProviderResult<StreamEvent>> = Vec::new();
        if !self.text.is_empty() {
            events.push(Ok(StreamEvent::Text {
                text: self.text.clone(),
            }));
        }
        events.push(Ok(StreamEvent::Done {
            usage: Some(Usage::new(self.input_tokens, self.output_tokens)),
            stop_reason: Some("end_turn".to_string()),
            billed_cost: Some(self.billed_cost),
            cost_source: Some(self.cost_source.clone()),
            ensemble_trace: self.ensemble_trace.clone(),
        }));
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

#[async_trait]
impl ChatProvider for SyntheticProvider {
    async fn chat(
        &self,
        config: &ChatConfig,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> ProviderResult<Box<dyn Stream<Item = ProviderResult<StreamEvent>> + Send + Unpin>> {
        self.stream_chat(config, messages, tools).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ChatConfig {
        ChatConfig {
            model: "synthetic-model".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_stream_emits_text_and_done_with_cost() {
        let provider = SyntheticProvider::new()
            .model("gpt-5.5")
            .input_tokens(2400)
            .output_tokens(600)
            .billed_cost(0.0123)
            .cost_source("synthetic")
            .text("hello world")
            .ensemble_trace(serde_json::json!({
                "mode": "b5_fusion",
                "successful_proposers": 3,
                "total_candidates": 5,
                "fallback_used": false,
            }));

        let stream = provider
            .stream_chat(&config(), &[], &[])
            .await
            .unwrap();
        futures::pin_mut!(stream);

        let mut texts = Vec::new();
        let mut done = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::Text { text } => texts.push(text),
                StreamEvent::Done {
                    usage,
                    stop_reason,
                    billed_cost,
                    cost_source,
                    ensemble_trace,
                    ..
                } => {
                    done = Some((
                        usage.unwrap(),
                        stop_reason,
                        billed_cost,
                        cost_source,
                        ensemble_trace,
                    ));
                }
                _ => {}
            }
        }

        assert_eq!(texts, vec!["hello world".to_string()]);
        let (usage, stop_reason, billed_cost, cost_source, ensemble_trace) =
            done.expect("expected Done event");
        assert_eq!(usage.input_tokens, 2400);
        assert_eq!(usage.output_tokens, 600);
        assert_eq!(stop_reason.as_deref(), Some("end_turn"));
        assert_eq!(billed_cost, Some(0.0123));
        assert_eq!(cost_source.as_deref(), Some("synthetic"));
        let trace = ensemble_trace.expect("expected ensemble_trace");
        assert_eq!(trace["successful_proposers"], 3);
        assert_eq!(trace["total_candidates"], 5);
    }

    #[tokio::test]
    async fn test_send_message_collects_response() {
        let provider = SyntheticProvider::new()
            .input_tokens(10)
            .output_tokens(20)
            .billed_cost(0.5)
            .text("a fixed answer");
        let resp = provider
            .send_message(&config(), &[], &[])
            .await
            .unwrap();
        assert_eq!(resp.content[0].text_content(), "a fixed answer");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 20);
        assert_eq!(resp.billed_cost, Some(0.5));
        assert_eq!(resp.cost_source.as_deref(), Some("synthetic"));
    }

    #[tokio::test]
    async fn test_list_models_empty() {
        let provider = SyntheticProvider::new();
        assert!(provider.list_models().await.is_empty());
        assert!(provider.supported_models().is_empty());
    }

    #[test]
    fn test_stream_event_done_defaults_new_fields() {
        // A `Done` event without the new fields deserializes with `None`
        // defaults, preserving wire compatibility with older payloads.
        let json = r#"{"type":"done","usage":null,"stop_reason":"stop"}"#;
        let event: StreamEvent = serde_json::from_str(json).unwrap();
        match event {
            StreamEvent::Done {
                billed_cost,
                cost_source,
                ensemble_trace,
                ..
            } => {
                assert!(billed_cost.is_none());
                assert!(cost_source.is_none());
                assert!(ensemble_trace.is_none());
            }
            _ => panic!("expected Done event"),
        }
    }

    #[test]
    fn test_provider_response_defaults_new_fields() {
        let json = r#"{
            "content": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "model": "m",
            "stop_reason": null
        }"#;
        let resp: ProviderResponse = serde_json::from_str(json).unwrap();
        assert!(resp.billed_cost.is_none());
        assert!(resp.cost_source.is_none());
        assert!(resp.ensemble_trace.is_none());
    }

    #[test]
    fn test_stream_event_done_serializes_none_as_omitted() {
        let event = StreamEvent::Done {
            usage: None,
            stop_reason: Some("stop".to_string()),
            billed_cost: None,
            cost_source: None,
            ensemble_trace: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("billed_cost"));
        assert!(!json.contains("cost_source"));
        assert!(!json.contains("ensemble_trace"));
    }
}