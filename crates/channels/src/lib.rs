pub mod admission;
pub mod approval;
pub mod command_registry;
pub mod delivery;
pub mod dingtalk;
pub mod discord;
pub mod feishu;
pub mod manager;
pub mod matrix;
pub mod msteams;
pub mod qq;
pub mod slack;
pub mod stream_policy;
pub mod system_messages;
pub mod telegram;
pub mod terminal;
pub mod types;
pub mod webhook;
pub mod websocket;
pub mod wecom;

pub use admission::{AdmissionReason, ChannelAdmissionDecision, decide_channel_admission};
pub use approval::{ApprovalDecision, parse_approval_action, render_approval_prompt_text};
pub use command_registry::{CommandRegistry, build_default};
pub use stream_policy::{ChannelStreamMode, ChannelStreamPolicy, resolve_channel_stream_policy};
pub use system_messages::{MessageKey, Messages, channel_message_locale, render_channel_message};

pub use delivery::{DeliveryStatus, DeliveryStore, OutboxEntry, OutboxWorker, retry_delay};
pub use discord::{
    DiscordChannel, DiscordRateLimiter, DiscordWsStream, INTENT_DIRECT_MESSAGES,
    INTENT_GUILD_MEMBERS, INTENT_GUILD_MESSAGE_REACTIONS, INTENT_GUILD_MESSAGES, INTENT_GUILDS,
    INTENT_MESSAGE_CONTENT,
};
pub use manager::{ChannelManager, ChannelState, ManagedChannel, StartHook, StopHook};
pub use slack::{
    SlackBlockBuilder, SlackChannel, SlackClient, SlackOAuth, SlackOAuthTokens, SlackTokenStore,
    SlackWsStream, parse_socket_event, to_mrkdwn,
};
pub use telegram::{InlineKeyboardBuilder, TelegramChannel};
pub use terminal::{
    ANSI_BLUE, ANSI_BOLD, ANSI_CYAN, ANSI_DIM, ANSI_GRAY, ANSI_GREEN, ANSI_MAGENTA, ANSI_RED,
    ANSI_RESET, ANSI_UNDERLINE, ANSI_YELLOW, EditorAction, LineEditor, TerminalChannel, colorize,
    disable_raw_mode, enable_raw_mode, format_outgoing, styled,
};
pub use types::{Channel, ChannelConfig, IncomingMessage, MessageAttachment, OutgoingMessage};
pub use webhook::{
    FunctionWebhookHandler, SlackWebhookHandler, TelegramWebhookHandler, WeComWebhookHandler,
    WebhookError, WebhookHandler, WebhookMethod, WebhookRegistry, WebhookResponse, WebhookRoute,
    WebhookState, decrypt_wecom_payload, parse_incoming_message, parse_slack_payload,
    parse_telegram_payload, parse_wecom_payload, verify_hmac_sha256, verify_slack_signature,
    verify_telegram_secret_token, verify_webhook_signature, verify_wecom_signature,
};
pub use websocket::{ConnectionInfo, WebSocketChannel, parse_client_frame};
pub use wecom::{
    MessageBuilder, WeComChannel, compute_wecom_signature, decrypt_wecom_encrypted,
    encrypt_wecom_payload,
};
