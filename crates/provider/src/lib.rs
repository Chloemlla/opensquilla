//! # OpenSquilla Provider
//!
//! LLM provider abstraction layer. Defines the `Provider` trait and provides
//! implementations for OpenAI-compatible APIs, Anthropic, and Ollama, as well
//! as the OpenAI Responses and Codex backends, an ensemble provider, a
//! declarative provider registry with 50+ specs, a model selector with
//! fallback, SSE stream assembly, a tool-call normalizer, credential pooling,
//! error classification with a circuit breaker, OpenAI compatibility policy
//! data, model/live catalogs, request pre-flight proofing, and image/audio
//! generation providers.

pub mod anthropic;
pub mod audio;
pub mod compaction;
pub mod compat_policy;
pub mod credentials;
pub mod ensemble;
pub mod failures;
pub mod image_generation;
pub mod live_catalog;
pub mod model_catalog;
pub mod normalizer;
pub mod ollama;
pub mod openai;
pub mod openai_codex;
pub mod openai_responses;
pub mod registry;
pub mod request_proof;
pub mod selector;
pub mod stream;
pub mod stream_assembly;
pub mod text_tool_normalizer;
pub mod types;
pub mod util;

pub use openai::{
    AuthHeader, ChatRequest, ChatResponse, OpenAIProvider, OpenAiCompatProvider, OpenAiConfig,
    ProviderInfo, build_chat_request, build_chat_request_full, build_credential_pool,
    classify_openai_error, extract_error_code, extract_error_message, is_reasoning_model,
    map_http_error, map_model, merge_live_models, parse_chat_response, parse_openai_sse_event,
    provider_from_id, resolve_chat_url, seed_catalog, should_retry, sse_deltas,
};
pub use registry::{
    ANTHROPIC_CONTEXT_PROFILE, AuthScheme, BackendType, ContextProfile, NativeCompactionSupport,
    OPENAI_RESPONSES_CONTEXT_PROFILE, OPENROUTER_CONTEXT_PROFILE, OpenAiCompatPolicy,
    PromptCacheSupport, ProviderRegistry, ProviderSpec, ProviderSpecTable, SelectableModelCatalog,
    TextToolModelRule,
};
pub use selector::ModelSelector;
/// Re-export key types at the crate root.
pub use types::{
    ChatConfig, ChatProvider, Provider, ProviderError, ProviderResponse, ProviderResult,
    StreamEvent,
};
// Note: `OpenAiResponsesProvider` is the provider *kind* enum; the concrete
// provider struct is `OpenAIResponsesProvider`.
pub use anthropic::{
    AnthropicConfig, AnthropicProvider, AnthropicProviderKind, AuthHeaderStyle, CacheBreakpoint,
    ThinkingConfig,
};
pub use audio::{
    AudioConfig, AudioFormat, AudioProvider, AudioProviderType, SttParams, SttResult, TtsParams,
    TtsRequest, TtsResponse, TtsResult, Voice, VoiceSettings, default_openai_voices,
    estimate_duration_seconds, estimate_stt_cost, estimate_tts_cost, pcm_to_wav,
    wav_duration_seconds,
};
pub use compaction::{
    AggregateToolResult, CompactionConfig, aggregate_tool_result_compacted, compact_tool_results,
    is_provider_projection,
};
pub use compat_policy::{CompatPolicy, CompatPolicyRegistry, policy_for};
pub use credentials::CredentialPool;
pub use ensemble::{
    AggregationSpec, AggregationStrategy, AllFailedPolicy, BestOfNStrategy, DebateStrategy,
    EnsembleConfig, EnsembleCost, EnsembleMember, EnsembleOrchestrator, EnsembleOutput,
    EnsembleProposer, EnsembleProvider, EnsembleRequest, EnsembleStrategy, ExecutionMode,
    FallbackSpec, MergeMethod, MergeOutcome, MixtureOfAgentsStrategy, ParserKind, PlainTextParser,
    PromptExample, PromptStrategy, PromptStrategyKind, Proposal, ProposerRole, ProposerSpec,
    ProvenanceEntry, ResponseParser, ScoringStrategy, StandardProposer, VotingStrategy,
};
pub use failures::{CircuitBreaker, ErrorCategory, RecoveryAction, classify as classify_error};
pub use image_generation::{
    GeneratedImage, GenerationAttempt, ImageGenConfig, ImageGenProvider, ImageGenerationParams,
    ImageGenerationProvider, ImageGenerationRequest, ImageGenerationResponse,
    ImageGenerationResult, ImageQuality, ImageResponseFormat, ImageSize,
    build_chat_completions_image_body, build_openai_body, build_prodia_body,
    build_qwen_token_plan_body, build_replicate_input, build_stability_form, decode_base64,
    decode_data_url, download_generated_image, encode_base64, encode_data_url, estimate_image_cost,
    extract_openrouter_image_url, extract_qwen_token_plan_image_url, generate_with_fallbacks,
    image_dimensions, image_mime_type, parse_chat_completions_image, parse_openai_json,
    parse_stability_json, per_image_cost, qwen_wire_size, validate_image_bytes,
    validate_image_mime, validate_image_size,
};
pub use live_catalog::LiveCatalog;
pub use model_catalog::ModelCatalog;
pub use ollama::{OllamaConfig, OllamaModel, OllamaModelDetails, OllamaProvider, PullProgress};
pub use openai_codex::OpenAICodexProvider;
pub use openai_codex::{
    CodeDebugIssue, CodeDebugReport, CodeExecutionRequest, CodeExecutionResult, CodeFile,
    CodexConfig, GeneratedCode, KNOWN_CODEX_MODELS, normalize_base_url,
};
pub use openai_responses::{
    ContentPart, InputItem, OpenAIResponsesProvider, OpenAiResponsesConfig,
    OpenAiResponsesProvider, OutputSchema, ResponsesOutputItem, ResponsesRequest, ResponsesTool,
    ResponsesToolType, ResponsesUsage, build_responses_input_items, parse_responses_sse_event,
    responses_sse_to_delta, responses_sse_value_to_delta, role_str,
};
pub use request_proof::RequestProof;
pub use util::{RateLimiter, RetryConfig, check_status, is_retryable_error, with_retry};
