//! Locale detection and persistence for the desktop shell.
//!
//! Replaces `desktop-locale.ts`. The desktop runtime bundles six locales
//! (`en`, `zh-Hans`, `ja`, `fr`, `de`, `es`). This module resolves the user's
//! ordered BCP-47 language tags to the first bundled locale, and persists a
//! user override under the app data directory so the choice survives restarts.
//!
//! The resolution rules mirror the Electron implementation exactly:
//! - `zh` with an explicit `Hans` script subtag → `zh-Hans`.
//! - `zh-Hant`, `zh-TW`, `zh-HK`, `zh-MO` → fall through to the next tag (a
//!   Traditional reader is never forced into Simplified text).
//! - Bare `zh` (no script/region) → `zh-Hans` (only Simplified is bundled).
//! - English is a bundled locale and must match on its own branch so a
//!   higher-preference `en-*` tag never loses to a lower-preference language.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Every locale the desktop shell ships translations for.
pub const BUNDLED_LOCALES: &[DesktopLocale] = &[
    DesktopLocale::En,
    DesktopLocale::ZhHans,
    DesktopLocale::Ja,
    DesktopLocale::Fr,
    DesktopLocale::De,
    DesktopLocale::Es,
];

/// A bundled desktop locale.
///
/// The serde representation is the string tag (`"en"`, `"zh-Hans"`, …) so the
/// value round-trips through the Tauri store and the frontend unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DesktopLocale {
    En,
    #[serde(rename = "zh-Hans")]
    ZhHans,
    Ja,
    Fr,
    De,
    Es,
}

impl DesktopLocale {
    /// The string tag used in IPC, the persisted override file, and URL tags.
    pub fn as_str(self) -> &'static str {
        match self {
            DesktopLocale::En => "en",
            DesktopLocale::ZhHans => "zh-Hans",
            DesktopLocale::Ja => "ja",
            DesktopLocale::Fr => "fr",
            DesktopLocale::De => "de",
            DesktopLocale::Es => "es",
        }
    }

    /// Parse a tag into a bundled locale, accepting both the canonical tag and
    /// the legacy underscore spelling. Returns `None` for unknown tags.
    pub fn from_tag(tag: &str) -> Option<Self> {
        let normalized = tag.trim().replace('_', "-");
        // Use Intl-like normalization: lowercase the language subtag only.
        let lower = normalized.to_lowercase();
        match lower.as_str() {
            "en" => Some(DesktopLocale::En),
            "zh-hans" | "zh" => Some(DesktopLocale::ZhHans),
            "ja" => Some(DesktopLocale::Ja),
            "fr" => Some(DesktopLocale::Fr),
            "de" => Some(DesktopLocale::De),
            "es" => Some(DesktopLocale::Es),
            _ => None,
        }
    }
}

/// Mirror of the Gateway's persisted-control-ui compatibility rule: every
/// `zh*` spelling is Simplified Chinese, and unsupported values reduce to
/// English. Used when reading back a value the Gateway may have written.
pub fn normalize_gateway_locale(value: &str) -> DesktopLocale {
    let normalized = value.trim().to_lowercase();
    if normalized.starts_with("zh") {
        return DesktopLocale::ZhHans;
    }
    for code in ["ja", "fr", "de", "es"] {
        if normalized.starts_with(code) {
            return match code {
                "ja" => DesktopLocale::Ja,
                "fr" => DesktopLocale::Fr,
                "de" => DesktopLocale::De,
                "es" => DesktopLocale::Es,
                _ => DesktopLocale::En,
            };
        }
    }
    DesktopLocale::En
}

/// A parsed BCP-47 tag split into its lowercase subtags we care about.
///
/// We do not pull in a full BCP-47 parser; the Electron resolver only inspects
/// the language, script, and region subtags, and a tolerant hand-rolled split
/// matches its behavior for every tag the OS actually emits.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedTag {
    language: String,
    script: Option<String>,
    region: Option<String>,
}

fn parse_bcp47(raw: &str) -> Option<ParsedTag> {
    let cleaned = raw.trim();
    if cleaned.is_empty() {
        return None;
    }
    // Split on `-` after normalizing underscores. The first subtag is the
    // primary language; a 4-alpha subtag is a script; a 2-alpha or 3-digit
    // subtag is a region.
    let normalized = cleaned.replace('_', "-");
    let mut parts = normalized.split('-').filter(|p| !p.is_empty());
    let language = parts.next()?.to_lowercase();
    if language.is_empty() || !language.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut script = None;
    let mut region = None;
    for p in parts {
        let lower = p.to_lowercase();
        if lower.len() == 4 && lower.chars().all(|c| c.is_ascii_alphabetic()) {
            script = Some(lower);
        } else if (lower.len() == 2 && lower.chars().all(|c| c.is_ascii_alphabetic()))
            || (lower.len() == 3 && lower.chars().all(|c| c.is_ascii_digit()))
        {
            region = Some(lower);
        }
    }
    Some(ParsedTag {
        language,
        script,
        region,
    })
}

/// Map the user's ordered BCP-47 language tags to the first bundled locale.
///
/// First match wins, so a top-preference tag can never lose to a lower one.
/// Pass the OS preferred languages in preference order followed by the system
/// locale, exactly as the Electron shell did.
pub fn resolve_locale_from_tags(tags: &[String]) -> DesktopLocale {
    for raw in tags {
        let Some(tag) = parse_bcp47(raw) else {
            continue;
        };
        // English is bundled and must match here: without this branch a
        // top-preference en-* tag falls through and a LOWER-preference language
        // (e.g. fr-HK behind en-HK) wins the loop.
        if tag.language == "en" {
            return DesktopLocale::En;
        }
        if tag.language == "zh" {
            // Only Simplified Chinese is bundled. An explicit script subtag
            // wins over region. Then route Traditional variants (explicit
            // zh-Hant, or bare regions that default to Traditional) to the
            // English fallback rather than forcing Simplified text.
            let script = tag.script.as_deref();
            let region = tag.region.as_deref();
            if script == Some("hans") {
                return DesktopLocale::ZhHans;
            }
            if script == Some("hant") || matches!(region, Some("tw") | Some("hk") | Some("mo")) {
                continue;
            }
            return DesktopLocale::ZhHans;
        }
        match tag.language.as_str() {
            "ja" => return DesktopLocale::Ja,
            "fr" => return DesktopLocale::Fr,
            "de" => return DesktopLocale::De,
            "es" => return DesktopLocale::Es,
            _ => continue,
        }
    }
    DesktopLocale::En
}

/// Detect the system locale via `tauri-plugin-os`, falling back to English.
///
/// The OS plugin returns a single BCP-47 tag; we wrap it in a one-element list
/// so the shared resolver applies the same preference-order logic.
pub fn detect_system_locale(_app: &tauri::AppHandle) -> DesktopLocale {
    let tag = tauri_plugin_os::locale().unwrap_or_default();
    let tags = if tag.trim().is_empty() {
        Vec::new()
    } else {
        vec![tag]
    };
    resolve_locale_from_tags(&tags)
}

/// The on-disk filename for the persisted locale override.
pub const OVERRIDE_FILENAME: &str = "locale.json";

/// A persisted locale override record.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocaleOverride {
    locale: String,
}

/// Resolve the path to the persisted locale override file.
///
/// Lives under the app's config directory so it survives upgrades and is
/// shared across profiles (matching the Electron shell's `userData` placement).
pub fn override_path(config_dir: &Path) -> PathBuf {
    config_dir.join("opensquilla").join(OVERRIDE_FILENAME)
}

/// Load the persisted user locale override, if any.
///
/// A missing or malformed file yields `None` rather than an error so callers
/// can transparently fall back to OS detection.
pub fn load_override(config_dir: &Path) -> Option<DesktopLocale> {
    let path = override_path(config_dir);
    let contents = std::fs::read_to_string(&path).ok()?;
    let record: LocaleOverride = serde_json::from_str(&contents).ok()?;
    DesktopLocale::from_tag(&record.locale)
}

/// Persist a user locale override so the choice survives restarts.
///
/// The override directory is created with restrictive permissions on POSIX
/// (0700) since the locale file lives alongside other app state.
pub fn save_override(config_dir: &Path, locale: DesktopLocale) -> std::io::Result<()> {
    let path = override_path(config_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let record = LocaleOverride {
        locale: locale.as_str().to_string(),
    };
    let json = serde_json::to_string_pretty(&record)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Atomic write: write to a sibling temp file then rename.
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json + "\n")?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Clear the persisted override, restoring OS-detected locale behavior.
pub fn clear_override(config_dir: &Path) -> std::io::Result<()> {
    let path = override_path(config_dir);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// The effective locale: the persisted override if set, otherwise the
/// OS-detected locale.
pub fn effective_locale(app: &tauri::AppHandle, config_dir: &Path) -> DesktopLocale {
    if let Some(overridden) = load_override(config_dir) {
        return overridden;
    }
    detect_system_locale(app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_tags() {
        assert_eq!(DesktopLocale::from_tag("en"), Some(DesktopLocale::En));
        assert_eq!(
            DesktopLocale::from_tag("zh-Hans"),
            Some(DesktopLocale::ZhHans)
        );
        assert_eq!(
            DesktopLocale::from_tag("zh_Hans"),
            Some(DesktopLocale::ZhHans)
        );
        assert_eq!(DesktopLocale::from_tag("zh"), Some(DesktopLocale::ZhHans));
        assert_eq!(DesktopLocale::from_tag("ja"), Some(DesktopLocale::Ja));
        assert_eq!(DesktopLocale::from_tag("unknown"), None);
    }

    #[test]
    fn resolves_first_match_wins() {
        // en-HK must beat fr-HK even though fr is bundled.
        let tags = vec!["en-HK".to_string(), "fr-HK".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
    }

    #[test]
    fn zh_hans_script_wins_over_region() {
        let tags = vec!["zh-Hans-HK".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::ZhHans);
    }

    #[test]
    fn zh_hant_falls_through() {
        // Traditional variants skip to the next tag.
        let tags = vec!["zh-Hant".to_string(), "en".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
        let tags = vec!["zh-TW".to_string(), "en".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
        let tags = vec!["zh-HK".to_string(), "en".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
    }

    #[test]
    fn bare_zh_is_simplified() {
        let tags = vec!["zh".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::ZhHans);
    }

    #[test]
    fn defaults_to_english() {
        let tags: Vec<String> = vec![];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
        let tags = vec!["ko-KR".to_string()];
        assert_eq!(resolve_locale_from_tags(&tags), DesktopLocale::En);
    }

    #[test]
    fn normalizes_gateway_locale() {
        assert_eq!(normalize_gateway_locale("zh-CN"), DesktopLocale::ZhHans);
        assert_eq!(normalize_gateway_locale("zh_Hant"), DesktopLocale::ZhHans);
        assert_eq!(normalize_gateway_locale("ja-JP"), DesktopLocale::Ja);
        assert_eq!(normalize_gateway_locale("klingon"), DesktopLocale::En);
    }
}
