//! Localized, server-generated notices shared by chat channels.
//!
//! Mirrors the Python `channels/system_messages.py`. The locale is a
//! Gateway-wide operator preference stored in `control_ui.default_locale`. It
//! deliberately does not depend on an inbound provider locale, request
//! headers, or authorization state: wording can vary while admission and
//! approval decisions must not.

use std::collections::HashMap;
use std::sync::OnceLock;

/// A fixed channel system message key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageKey {
    PairingRequired,
    PairingApproved,
    ApprovalPrompt,
    ApprovalPromptAlways,
    ApprovalCardTitle,
    ApprovalCardQuestion,
    ApprovalCardDetails,
    ApprovalCardApprove,
    ApprovalCardAlways,
    ApprovalCardDeny,
    ApprovalCardNote,
    ApprovalCardNoteAlways,
    ApprovalLabelCommand,
    ApprovalLabelNetwork,
    ApprovalLabelNetworkHost,
    ApprovalLabelPath,
    ApprovalLabelCode,
    ApprovalPackages,
    ApprovalUnknownCommand,
    ApprovalProbeThrottled,
    ApprovalNoPending,
    ApprovalOwnerOnly,
    ApprovalAlwaysRequiresAdmin,
    ApprovalInvalidChoice,
    ApprovalAlreadyResolved,
    ApprovalResolutionFailed,
    ApprovalDenied,
    ApprovalApprovedOnce,
    ApprovalApprovedAlways,
    CommandUsageSandbox,
    CommandUnsupported,
    CommandCompleted,
    CommandDenied,
    CommandFailed,
    CommandSandboxDenied,
    CommandSandboxFailed,
    CommandSandboxUpdated,
    CommandSandboxStandard,
    CommandSandboxTrusted,
    CommandSandboxFull,
    CommandSandboxUnknownMode,
    CommandCompactDenied,
    CommandCompactFailed,
    CommandCompactCompleted,
    CommandCompactSkipped,
    CommandMetaDenied,
    CommandMetaFailed,
    CommandMetaEmpty,
    CommandMetaHeading,
    CommandMissingScope,
    CommandNewDenied,
    CommandNewUnavailable,
}

/// The six supported locales, keyed by their variant id.
#[derive(Debug, Clone)]
pub struct Messages {
    pub en: HashMap<MessageKey, &'static str>,
    pub zh_hans: HashMap<MessageKey, &'static str>,
    pub ja: HashMap<MessageKey, &'static str>,
    pub fr: HashMap<MessageKey, &'static str>,
    pub de: HashMap<MessageKey, &'static str>,
    pub es: HashMap<MessageKey, &'static str>,
}

/// Build a `HashMap<MessageKey, &'static str>` from a literal key→text list.
macro_rules! messages_for {
    ($($key:ident => $text:expr),* $(,)?) => {{
        let mut m = std::collections::HashMap::new();
        $(m.insert(MessageKey::$key, $text as &'static str);)*
        m
    }};
}

fn build_messages() -> Messages {
    Messages {
        en: messages_for!(
            PairingRequired => "Access approval is required. Pairing request: {pairing_code}. Ask an OpenSquilla operator to approve it before sending another message.",
            PairingApproved => "Access approved. Send a message to start chatting.",
            ApprovalPrompt => "Approval needed to run a privileged command.\n{label}: {command}\nCode: {code}\nReply /approve {code} to allow or /deny {code} to refuse.",
            ApprovalPromptAlways => "Approval needed to run a privileged command.\n{label}: {command}\nCode: {code}\nReply /approve {code} to allow, /approve {code} always to stop asking for this kind, or /deny {code} to refuse.",
            ApprovalCardTitle => "Approval needed",
            ApprovalCardQuestion => "Run a privileged command?",
            ApprovalCardDetails => "{question}\n**{label}:** `{command}`\n**{code_label}:** `{code}`",
            ApprovalCardApprove => "Approve",
            ApprovalCardAlways => "Always allow",
            ApprovalCardDeny => "Deny",
            ApprovalCardNote => "Or reply /approve {code} or /deny {code}.",
            ApprovalCardNoteAlways => "Or reply /approve {code}, /approve {code} always, or /deny {code}.",
            ApprovalLabelCommand => "Command",
            ApprovalLabelNetwork => "Network",
            ApprovalLabelNetworkHost => "Network host",
            ApprovalLabelPath => "Path",
            ApprovalLabelCode => "Code",
            ApprovalPackages => "packages: {bundle_id}",
            ApprovalUnknownCommand => "(unknown command)",
            ApprovalProbeThrottled => "Too many failed approval attempts — wait a minute and try again.",
            ApprovalNoPending => "No pending approval {code}.",
            ApprovalOwnerOnly => "Only the session owner can resolve this. Ask them to reply /approve {code}.",
            ApprovalAlwaysRequiresAdmin => "'Always' needs a channel admin. Reply /approve {code} to allow just this once.",
            ApprovalInvalidChoice => "Could not apply approval {code} — it is still pending. Resolve it from the console.",
            ApprovalAlreadyResolved => "Approval {code} was already resolved.",
            ApprovalResolutionFailed => "Could not apply approval {code} — it is still pending, please try again.",
            ApprovalDenied => "Denied {code}.",
            ApprovalApprovedOnce => "Approved {code} — running …",
            ApprovalApprovedAlways => "Approved {code} — this kind won't ask again this session.",
            CommandUsageSandbox => "Usage: /sandbox standard | trusted | full",
            CommandUnsupported => "Unsupported command: {command}. Try /help.",
            CommandCompleted => "/{name} completed",
            CommandDenied => "/{name} denied{reason}",
            CommandFailed => "/{name} failed{reason}",
            CommandSandboxDenied => "Sandbox mode denied: {reason}",
            CommandSandboxFailed => "Sandbox mode failed: {reason}",
            CommandSandboxUpdated => "Sandbox mode set to {mode}.",
            CommandSandboxStandard => "Standard-Sandbox",
            CommandSandboxTrusted => "Managed Execution",
            CommandSandboxFull => "Full Host Access",
            CommandSandboxUnknownMode => "updated",
            CommandCompactDenied => "Compact denied: {reason}",
            CommandCompactFailed => "Compact failed: {reason}",
            CommandCompactCompleted => "Context compacted.",
            CommandCompactSkipped => "Already within context budget; no compact was applied.",
            CommandMetaDenied => "/meta denied: {reason}",
            CommandMetaFailed => "/meta failed: {reason}",
            CommandMetaEmpty => "No meta-skills available.",
            CommandMetaHeading => "Available meta-skills:",
            CommandMissingScope => ": missing {missing}",
            CommandNewDenied => "/new denied: Insufficient scope for method: {method}{detail}",
            CommandNewUnavailable => "/new failed: command unavailable",
        ),
        zh_hans: messages_for!(
            PairingRequired => "需要访问审批。配对申请：{pairing_code}。请联系 OpenSquilla 操作员批准后再发送消息。",
            PairingApproved => "访问已获批准。请发送一条消息以开始对话。",
            ApprovalPrompt => "需要批准才能运行特权命令。\n{label}：{command}\n代码：{code}\n回复 /approve {code} 以允许，或回复 /deny {code} 以拒绝。",
            ApprovalPromptAlways => "需要批准才能运行特权命令。\n{label}：{command}\n代码：{code}\n回复 /approve {code} 以允许，回复 /approve {code} always 以不再询问此类操作，或回复 /deny {code} 以拒绝。",
            ApprovalCardTitle => "需要批准",
            ApprovalCardQuestion => "要运行特权命令吗？",
            ApprovalCardDetails => "{question}\n**{label}：** `{command}`\n**{code_label}：** `{code}`",
            ApprovalCardApprove => "批准",
            ApprovalCardAlways => "始终允许",
            ApprovalCardDeny => "拒绝",
            ApprovalCardNote => "或回复 /approve {code} 或 /deny {code}。",
            ApprovalCardNoteAlways => "或回复 /approve {code}、/approve {code} always 或 /deny {code}。",
            ApprovalLabelCommand => "命令",
            ApprovalLabelNetwork => "网络",
            ApprovalLabelNetworkHost => "网络主机",
            ApprovalLabelPath => "路径",
            ApprovalLabelCode => "代码",
            ApprovalPackages => "软件包：{bundle_id}",
            ApprovalUnknownCommand => "（未知命令）",
            ApprovalProbeThrottled => "失败的批准尝试过多，请等待一分钟后重试。",
            ApprovalNoPending => "没有待处理的批准 {code}。",
            ApprovalOwnerOnly => "只有会话所有者可以处理此批准。请让其回复 /approve {code}。",
            ApprovalAlwaysRequiresAdmin => "“始终允许”需要频道管理员权限。回复 /approve {code} 仅允许这一次。",
            ApprovalInvalidChoice => "无法应用批准 {code}，它仍在等待处理。请在控制台中处理。",
            ApprovalAlreadyResolved => "批准 {code} 已被处理。",
            ApprovalResolutionFailed => "无法应用批准 {code}，它仍在等待处理，请重试。",
            ApprovalDenied => "已拒绝 {code}。",
            ApprovalApprovedOnce => "已批准 {code}，正在运行……",
            ApprovalApprovedAlways => "已批准 {code}，本次会话中将不再询问此类操作。",
            CommandUsageSandbox => "用法：/sandbox standard | trusted | full",
            CommandUnsupported => "不支持的命令：{command}。请尝试 /help。",
            CommandCompleted => "/{name} 已完成",
            CommandDenied => "/{name} 已被拒绝{reason}",
            CommandFailed => "/{name} 失败{reason}",
            CommandSandboxDenied => "沙箱模式被拒绝：{reason}",
            CommandSandboxFailed => "沙箱模式设置失败：{reason}",
            CommandSandboxUpdated => "沙箱模式已设为 {mode}。",
            CommandSandboxStandard => "标准沙箱",
            CommandSandboxTrusted => "受管执行",
            CommandSandboxFull => "完全主机访问",
            CommandSandboxUnknownMode => "已更新",
            CommandCompactDenied => "压缩被拒绝：{reason}",
            CommandCompactFailed => "压缩失败：{reason}",
            CommandCompactCompleted => "上下文已压缩。",
            CommandCompactSkipped => "上下文仍在预算范围内，未执行压缩。",
            CommandMetaDenied => "/meta 被拒绝：{reason}",
            CommandMetaFailed => "/meta 失败：{reason}",
            CommandMetaEmpty => "没有可用的元技能。",
            CommandMetaHeading => "可用的元技能：",
            CommandMissingScope => "：缺少 {missing}",
            CommandNewDenied => "/new 被拒绝：方法权限不足：{method}{detail}",
            CommandNewUnavailable => "/new 失败：命令不可用",
        ),
        ja: messages_for!(
            PairingRequired => "アクセスの承認が必要です。ペアリング リクエスト: {pairing_code}。別のメッセージを送る前に、OpenSquilla のオペレーターに承認を依頼してください。",
            PairingApproved => "アクセスが承認されました。メッセージを送信して会話を開始してください。",
            ApprovalPrompt => "特権コマンドを実行するには承認が必要です。\n{label}: {command}\nコード: {code}\n/approve {code} で許可するか、/deny {code} で拒否してください。",
            ApprovalPromptAlways => "特権コマンドを実行するには承認が必要です。\n{label}: {command}\nコード: {code}\n/approve {code} で許可、/approve {code} always でこの種類の確認を今後表示しない、または /deny {code} で拒否してください。",
            ApprovalCardTitle => "承認が必要です",
            ApprovalCardQuestion => "特権コマンドを実行しますか？",
            ApprovalCardDetails => "{question}\n**{label}：** `{command}`\n**{code_label}：** `{code}`",
            ApprovalCardApprove => "承認",
            ApprovalCardAlways => "常に許可",
            ApprovalCardDeny => "拒否",
            ApprovalCardNote => "または /approve {code} か /deny {code} と返信してください。",
            ApprovalCardNoteAlways => "または /approve {code}、/approve {code} always、/deny {code} と返信してください。",
            ApprovalLabelCommand => "コマンド",
            ApprovalLabelNetwork => "ネットワーク",
            ApprovalLabelNetworkHost => "ネットワーク ホスト",
            ApprovalLabelPath => "パス",
            ApprovalLabelCode => "コード",
            ApprovalPackages => "パッケージ: {bundle_id}",
            ApprovalUnknownCommand => "(不明なコマンド)",
            ApprovalProbeThrottled => "承認コードの試行回数が多すぎます。1 分待ってからもう一度試してください。",
            ApprovalNoPending => "保留中の承認 {code} はありません。",
            ApprovalOwnerOnly => "この承認を解決できるのはセッション所有者だけです。/approve {code} と返信するよう依頼してください。",
            ApprovalAlwaysRequiresAdmin => "「常に許可」にはチャンネル管理者が必要です。/approve {code} で今回だけ許可してください。",
            ApprovalInvalidChoice => "承認 {code} を適用できません。まだ保留中です。コンソールから解決してください。",
            ApprovalAlreadyResolved => "承認 {code} はすでに解決されています。",
            ApprovalResolutionFailed => "承認 {code} を適用できません。まだ保留中です。もう一度試してください。",
            ApprovalDenied => "{code} を拒否しました。",
            ApprovalApprovedOnce => "{code} を承認しました。実行中です ...",
            ApprovalApprovedAlways => "{code} を承認しました。このセッションではこの種類を今後確認しません。",
            CommandUsageSandbox => "使い方: /sandbox standard | trusted | full",
            CommandUnsupported => "未対応のコマンドです: {command}。/help を試してください。",
            CommandCompleted => "/{name} が完了しました",
            CommandDenied => "/{name} は拒否されました{reason}",
            CommandFailed => "/{name} は失敗しました{reason}",
            CommandSandboxDenied => "サンドボックス モードは拒否されました: {reason}",
            CommandSandboxFailed => "サンドボックス モードの設定に失敗しました: {reason}",
            CommandSandboxUpdated => "サンドボックス モードを {mode} に設定しました。",
            CommandSandboxStandard => "標準サンドボックス",
            CommandSandboxTrusted => "管理された実行",
            CommandSandboxFull => "完全なホストアクセス",
            CommandSandboxUnknownMode => "更新済み",
            CommandCompactDenied => "コンテキスト圧縮は拒否されました: {reason}",
            CommandCompactFailed => "コンテキスト圧縮に失敗しました: {reason}",
            CommandCompactCompleted => "コンテキストを圧縮しました。",
            CommandCompactSkipped => "コンテキストは予算内のため、圧縮しませんでした。",
            CommandMetaDenied => "/meta は拒否されました: {reason}",
            CommandMetaFailed => "/meta は失敗しました: {reason}",
            CommandMetaEmpty => "利用可能なメタスキルはありません。",
            CommandMetaHeading => "利用可能なメタスキル:",
            CommandMissingScope => ": 不足している権限 {missing}",
            CommandNewDenied => "/new は拒否されました: メソッドの権限が不足しています: {method}{detail}",
            CommandNewUnavailable => "/new は失敗しました: コマンドを利用できません",
        ),
        fr: messages_for!(
            PairingRequired => "Une approbation d'accès est requise. Demande d'appairage : {pairing_code}. Demandez à un opérateur OpenSquilla de l'approuver avant d'envoyer un autre message.",
            PairingApproved => "Accès approuvé. Envoyez un message pour commencer à discuter.",
            ApprovalPrompt => "Une approbation est requise pour exécuter une commande privilégiée.\n{label} : {command}\nCode : {code}\nRépondez /approve {code} pour autoriser ou /deny {code} pour refuser.",
            ApprovalPromptAlways => "Une approbation est requise pour exécuter une commande privilégiée.\n{label} : {command}\nCode : {code}\nRépondez /approve {code} pour autoriser, /approve {code} always pour ne plus demander pour ce type, ou /deny {code} pour refuser.",
            ApprovalCardTitle => "Approbation requise",
            ApprovalCardQuestion => "Exécuter une commande privilégiée ?",
            ApprovalCardDetails => "{question}\n**{label} :** `{command}`\n**{code_label} :** `{code}`",
            ApprovalCardApprove => "Approuver",
            ApprovalCardAlways => "Toujours autoriser",
            ApprovalCardDeny => "Refuser",
            ApprovalCardNote => "Ou répondez /approve {code} ou /deny {code}.",
            ApprovalCardNoteAlways => "Ou répondez /approve {code}, /approve {code} always ou /deny {code}.",
            ApprovalLabelCommand => "Commande",
            ApprovalLabelNetwork => "Réseau",
            ApprovalLabelNetworkHost => "Hôte réseau",
            ApprovalLabelPath => "Chemin",
            ApprovalLabelCode => "Code",
            ApprovalPackages => "paquets : {bundle_id}",
            ApprovalUnknownCommand => "(commande inconnue)",
            ApprovalProbeThrottled => "Trop de tentatives d'approbation ont échoué. Attendez une minute et réessayez.",
            ApprovalNoPending => "Aucune approbation en attente pour {code}.",
            ApprovalOwnerOnly => "Seul le propriétaire de la session peut résoudre ceci. Demandez-lui de répondre /approve {code}.",
            ApprovalAlwaysRequiresAdmin => "« Toujours autoriser » nécessite un administrateur du canal. Répondez /approve {code} pour autoriser cette fois seulement.",
            ApprovalInvalidChoice => "Impossible d'appliquer l'approbation {code} ; elle est toujours en attente. Résolvez-la depuis la console.",
            ApprovalAlreadyResolved => "L'approbation {code} a déjà été résolue.",
            ApprovalResolutionFailed => "Impossible d'appliquer l'approbation {code} ; elle est toujours en attente. Réessayez.",
            ApprovalDenied => "Approbation {code} refusée.",
            ApprovalApprovedOnce => "Approbation {code} accordée - exécution en cours ...",
            ApprovalApprovedAlways => "Approbation {code} accordée - ce type ne demandera plus cette session.",
            CommandUsageSandbox => "Utilisation : /sandbox standard | trusted | full",
            CommandUnsupported => "Commande non prise en charge : {command}. Essayez /help.",
            CommandCompleted => "/{name} terminé",
            CommandDenied => "/{name} refusé{reason}",
            CommandFailed => "/{name} a échoué{reason}",
            CommandSandboxDenied => "Mode bac à sable refusé : {reason}",
            CommandSandboxFailed => "Échec du mode bac à sable : {reason}",
            CommandSandboxUpdated => "Mode bac à sable défini sur {mode}.",
            CommandSandboxStandard => "Bac à sable standard",
            CommandSandboxTrusted => "Exécution gérée",
            CommandSandboxFull => "Accès complet à l'hôte",
            CommandSandboxUnknownMode => "mis à jour",
            CommandCompactDenied => "Compactage refusé : {reason}",
            CommandCompactFailed => "Échec du compactage : {reason}",
            CommandCompactCompleted => "Contexte compacté.",
            CommandCompactSkipped => "Le contexte est déjà dans le budget ; aucun compactage appliqué.",
            CommandMetaDenied => "/meta refusé : {reason}",
            CommandMetaFailed => "/meta a échoué : {reason}",
            CommandMetaEmpty => "Aucune méta-compétence disponible.",
            CommandMetaHeading => "Méta-compétences disponibles :",
            CommandMissingScope => " : autorisation manquante {missing}",
            CommandNewDenied => "/new refusé : autorisation insuffisante pour la méthode : {method}{detail}",
            CommandNewUnavailable => "/new a échoué : commande indisponible",
        ),
        de: messages_for!(
            PairingRequired => "Eine Zugriffsfreigabe ist erforderlich. Kopplungsanfrage: {pairing_code}. Bitten Sie einen OpenSquilla-Operator, sie zu genehmigen, bevor Sie eine weitere Nachricht senden.",
            PairingApproved => "Zugriff genehmigt. Senden Sie eine Nachricht, um den Chat zu beginnen.",
            ApprovalPrompt => "Zum Ausführen eines privilegierten Befehls ist eine Freigabe erforderlich.\n{label}: {command}\nCode: {code}\nAntworten Sie mit /approve {code} zum Erlauben oder /deny {code} zum Ablehnen.",
            ApprovalPromptAlways => "Zum Ausführen eines privilegierten Befehls ist eine Freigabe erforderlich.\n{label}: {command}\nCode: {code}\nAntworten Sie mit /approve {code} zum Erlauben, /approve {code} always, um für diese Art nicht mehr gefragt zu werden, oder /deny {code} zum Ablehnen.",
            ApprovalCardTitle => "Freigabe erforderlich",
            ApprovalCardQuestion => "Privilegierten Befehl ausführen?",
            ApprovalCardDetails => "{question}\n**{label}:** `{command}`\n**{code_label}:** `{code}`",
            ApprovalCardApprove => "Genehmigen",
            ApprovalCardAlways => "Immer erlauben",
            ApprovalCardDeny => "Ablehnen",
            ApprovalCardNote => "Oder antworten Sie mit /approve {code} oder /deny {code}.",
            ApprovalCardNoteAlways => "Oder antworten Sie mit /approve {code}, /approve {code} always oder /deny {code}.",
            ApprovalLabelCommand => "Befehl",
            ApprovalLabelNetwork => "Netzwerk",
            ApprovalLabelNetworkHost => "Netzwerkhost",
            ApprovalLabelPath => "Pfad",
            ApprovalLabelCode => "Code",
            ApprovalPackages => "Pakete: {bundle_id}",
            ApprovalUnknownCommand => "(unbekannter Befehl)",
            ApprovalProbeThrottled => "Zu viele fehlgeschlagene Freigabeversuche. Warten Sie eine Minute und versuchen Sie es erneut.",
            ApprovalNoPending => "Keine ausstehende Freigabe für {code}.",
            ApprovalOwnerOnly => "Nur der Sitzungsinhaber kann dies auflösen. Bitten Sie ihn, mit /approve {code} zu antworten.",
            ApprovalAlwaysRequiresAdmin => "„Immer erlauben“ benötigt einen Kanaladministrator. Antworten Sie mit /approve {code}, um dies nur einmal zu erlauben.",
            ApprovalInvalidChoice => "Freigabe {code} konnte nicht angewendet werden; sie ist weiterhin ausstehend. Lösen Sie sie in der Konsole auf.",
            ApprovalAlreadyResolved => "Freigabe {code} wurde bereits aufgelöst.",
            ApprovalResolutionFailed => "Freigabe {code} konnte nicht angewendet werden; sie ist weiterhin ausstehend. Versuchen Sie es erneut.",
            ApprovalDenied => "Freigabe {code} abgelehnt.",
            ApprovalApprovedOnce => "Freigabe {code} genehmigt - wird ausgeführt ...",
            ApprovalApprovedAlways => "Freigabe {code} genehmigt - diese Art wird in dieser Sitzung nicht erneut fragen.",
            CommandUsageSandbox => "Verwendung: /sandbox standard | trusted | full",
            CommandUnsupported => "Nicht unterstützter Befehl: {command}. Versuchen Sie /help.",
            CommandCompleted => "/{name} abgeschlossen",
            CommandDenied => "/{name} abgelehnt{reason}",
            CommandFailed => "/{name} fehlgeschlagen{reason}",
            CommandSandboxDenied => "Sandbox-Modus abgelehnt: {reason}",
            CommandSandboxFailed => "Sandbox-Modus fehlgeschlagen: {reason}",
            CommandSandboxUpdated => "Sandbox-Modus auf {mode} gesetzt.",
            CommandSandboxStandard => "Standard-Sandbox",
            CommandSandboxTrusted => "Verwaltete Ausführung",
            CommandSandboxFull => "Vollständiger Hostzugriff",
            CommandSandboxUnknownMode => "aktualisiert",
            CommandCompactDenied => "Komprimierung abgelehnt: {reason}",
            CommandCompactFailed => "Komprimierung fehlgeschlagen: {reason}",
            CommandCompactCompleted => "Kontext komprimiert.",
            CommandCompactSkipped => "Der Kontext liegt bereits im Budget; keine Komprimierung wurde angewendet.",
            CommandMetaDenied => "/meta abgelehnt: {reason}",
            CommandMetaFailed => "/meta fehlgeschlagen: {reason}",
            CommandMetaEmpty => "Keine Meta-Skills verfügbar.",
            CommandMetaHeading => "Verfügbare Meta-Skills:",
            CommandMissingScope => ": fehlende Berechtigung {missing}",
            CommandNewDenied => "/new abgelehnt: Unzureichender Umfang für Methode: {method}{detail}",
            CommandNewUnavailable => "/new fehlgeschlagen: Befehl nicht verfügbar",
        ),
        es: messages_for!(
            PairingRequired => "Se requiere aprobación de acceso. Solicitud de emparejamiento: {pairing_code}. Pide a un operador de OpenSquilla que la apruebe antes de enviar otro mensaje.",
            PairingApproved => "Acceso aprobado. Envía un mensaje para empezar a chatear.",
            ApprovalPrompt => "Se requiere aprobación para ejecutar un comando con privilegios.\n{label}: {command}\nCódigo: {code}\nResponde /approve {code} para permitir o /deny {code} para rechazar.",
            ApprovalPromptAlways => "Se requiere aprobación para ejecutar un comando con privilegios.\n{label}: {command}\nCódigo: {code}\nResponde /approve {code} para permitir, /approve {code} always para no volver a preguntar por este tipo, o /deny {code} para rechazar.",
            ApprovalCardTitle => "Se requiere aprobación",
            ApprovalCardQuestion => "¿Ejecutar un comando con privilegios?",
            ApprovalCardDetails => "{question}\n**{label}:** `{command}`\n**{code_label}:** `{code}`",
            ApprovalCardApprove => "Aprobar",
            ApprovalCardAlways => "Permitir siempre",
            ApprovalCardDeny => "Rechazar",
            ApprovalCardNote => "O responde /approve {code} o /deny {code}.",
            ApprovalCardNoteAlways => "O responde /approve {code}, /approve {code} always o /deny {code}.",
            ApprovalLabelCommand => "Comando",
            ApprovalLabelNetwork => "Red",
            ApprovalLabelNetworkHost => "Host de red",
            ApprovalLabelPath => "Ruta",
            ApprovalLabelCode => "Código",
            ApprovalPackages => "paquetes: {bundle_id}",
            ApprovalUnknownCommand => "(comando desconocido)",
            ApprovalProbeThrottled => "Demasiados intentos de aprobación fallidos. Espera un minuto e inténtalo de nuevo.",
            ApprovalNoPending => "No hay ninguna aprobación pendiente para {code}.",
            ApprovalOwnerOnly => "Solo el propietario de la sesión puede resolver esto. Pídele que responda /approve {code}.",
            ApprovalAlwaysRequiresAdmin => "«Permitir siempre» requiere un administrador del canal. Responde /approve {code} para permitir solo esta vez.",
            ApprovalInvalidChoice => "No se pudo aplicar la aprobación {code}; sigue pendiente. Resuélvela desde la consola.",
            ApprovalAlreadyResolved => "La aprobación {code} ya se resolvió.",
            ApprovalResolutionFailed => "No se pudo aplicar la aprobación {code}; sigue pendiente. Inténtalo de nuevo.",
            ApprovalDenied => "Aprobación {code} rechazada.",
            ApprovalApprovedOnce => "Aprobación {code} concedida - ejecutando ...",
            ApprovalApprovedAlways => "Aprobación {code} concedida - este tipo no volverá a preguntar en esta sesión.",
            CommandUsageSandbox => "Uso: /sandbox standard | trusted | full",
            CommandUnsupported => "Comando no compatible: {command}. Prueba /help.",
            CommandCompleted => "/{name} completado",
            CommandDenied => "/{name} denegado{reason}",
            CommandFailed => "/{name} falló{reason}",
            CommandSandboxDenied => "Modo aislado denegado: {reason}",
            CommandSandboxFailed => "El modo aislado falló: {reason}",
            CommandSandboxUpdated => "Modo aislado establecido en {mode}.",
            CommandSandboxStandard => "Entorno aislado estándar",
            CommandSandboxTrusted => "Ejecución administrada",
            CommandSandboxFull => "Acceso completo al host",
            CommandSandboxUnknownMode => "actualizado",
            CommandCompactDenied => "Compactación denegada: {reason}",
            CommandCompactFailed => "La compactación falló: {reason}",
            CommandCompactCompleted => "Contexto compactado.",
            CommandCompactSkipped => "El contexto ya está dentro del presupuesto; no se aplicó compactación.",
            CommandMetaDenied => "/meta denegado: {reason}",
            CommandMetaFailed => "/meta falló: {reason}",
            CommandMetaEmpty => "No hay meta-habilidades disponibles.",
            CommandMetaHeading => "Meta-habilidades disponibles:",
            CommandMissingScope => ": falta el permiso {missing}",
            CommandNewDenied => "/new denegado: alcance insuficiente para el método: {method}{detail}",
            CommandNewUnavailable => "/new falló: comando no disponible",
        ),
    }
}

/// Return the (cached) localized message tables.
fn messages() -> &'static Messages {
    static MSG: OnceLock<Messages> = OnceLock::new();
    MSG.get_or_init(build_messages)
}

/// The canonical locale id for a possibly-prefixed gateway default locale.
///
/// `zh*` → `"zh-Hans"`, `ja`/`fr`/`de`/`es` prefix match their variant, and
/// anything else (including `None`) falls back to `"en"`.
pub fn channel_message_locale(default_locale: Option<&str>) -> &'static str {
    let Some(locale) = default_locale else {
        return "en";
    };
    let locale = locale.trim().to_lowercase();
    if locale.starts_with("zh") {
        "zh-Hans"
    } else if locale.starts_with("ja") {
        "ja"
    } else if locale.starts_with("fr") {
        "fr"
    } else if locale.starts_with("de") {
        "de"
    } else if locale.starts_with("es") {
        "es"
    } else {
        "en"
    }
}

/// Render a fixed channel message from the resolved locale, substituting
/// `{name}`-style placeholders from `values`.
pub fn render_channel_message(
    key: MessageKey,
    locale: &str,
    values: &HashMap<&str, String>,
) -> String {
    let lang = channel_message_locale(if locale.is_empty() { None } else { Some(locale) });
    let map = match lang {
        "zh-Hans" => &messages().zh_hans,
        "ja" => &messages().ja,
        "fr" => &messages().fr,
        "de" => &messages().de,
        "es" => &messages().es,
        _ => &messages().en,
    };
    let Some(template) = map.get(&key) else {
        return String::new();
    };
    let mut out = String::with_capacity(template.len());
    let bytes = template.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if let Some(close) = template[i + 1..].find('}') {
                let name = &template[i + 1..i + 1 + close];
                if let Some(replacement) = values.get(name) {
                    out.push_str(replacement);
                    i += close + 2;
                    continue;
                }
            }
        }
        // Copy one full char (handles multi-byte UTF-8 in localized text).
        let ch = template[i..].chars().next().unwrap_or('\u{fffd}');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(pairs: &[(&str, &str)]) -> HashMap<&str, String> {
        pairs
            .iter()
            .map(|(k, v)| (*k, v.to_string()))
            .collect()
    }

    #[test]
    fn locale_prefix_matching() {
        assert_eq!(channel_message_locale(None), "en");
        assert_eq!(channel_message_locale(Some("zh-CN")), "zh-Hans");
        assert_eq!(channel_message_locale(Some("zh-TW")), "zh-Hans");
        assert_eq!(channel_message_locale(Some("ja")), "ja");
        assert_eq!(channel_message_locale(Some("fr-FR")), "fr");
        assert_eq!(channel_message_locale(Some("de-DE")), "de");
        assert_eq!(channel_message_locale(Some("es-ES")), "es");
        assert_eq!(channel_message_locale(Some("ko")), "en");
    }

    #[test]
    fn english_placeholder_rendering() {
        let out = render_channel_message(
            MessageKey::CommandCompleted,
            "en",
            &values(&[("name", "run")]),
        );
        assert_eq!(out, "/run completed");
    }

    #[test]
    fn zh_rendering_uses_zhhans() {
        let out = render_channel_message(
            MessageKey::CommandCompleted,
            "zh-CN",
            &values(&[("name", "run")]),
        );
        assert_eq!(out, "/run 已完成");
    }

    #[test]
    fn missing_values_leave_placeholder_unchanged() {
        let out = render_channel_message(MessageKey::ApprovalDenied, "en", &HashMap::new());
        assert_eq!(out, "Denied {code}.");
    }

    #[test]
    fn all_locales_have_all_keys() {
        let m = messages();
        let keys = m.en.keys().count();
        assert!(keys >= 50, "expected ~53 keys, got {keys}");
        for map in [&m.zh_hans, &m.ja, &m.fr, &m.de, &m.es] {
            assert_eq!(map.len(), keys, "locale table size mismatch");
            for k in m.en.keys() {
                assert!(map.contains_key(k), "missing key {k:?}");
            }
        }
    }
}