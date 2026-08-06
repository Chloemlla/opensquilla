//! Auto-update: feed resolver, version comparison, scheduled checks, and
//! download/install via `tauri-plugin-updater`.
//!
//! Replaces four Electron modules:
//! - `update-feed-resolver.ts` → [`parse_release_tag`] and
//!   [`select_prerelease_candidate`].
//! - `update-channel.ts` → [`UpdateChannelManifest`] validation,
//!   [`candidate_from_channel`], and [`ordered_update_sources`].
//! - `update-check-scheduler.ts` → [`UpdateCheckScheduler`] (recursive timeout,
//!   single-flight, manual promotion).
//! - `update-verification.ts` → [`parse_sha256_sums_for_asset`] and
//!   [`stream_response_to_verified_file`] (SHA-256 verified download).
//!
//! The Tauri v2 updater plugin performs the actual download + install for the
//! current platform's native updater; this module owns the *discovery* layer
//! (which release is the channel head) and the *verification* layer for any
//! side-loaded asset the plugin does not natively cover.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex as AsyncMutex;

// ---------------------------------------------------------------------------
// Release-tag parsing (update-feed-resolver.ts)
// ---------------------------------------------------------------------------

/// A parsed OpenSquilla release tag: a `base` semver and an optional rc number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedReleaseTag {
    pub base: String,
    pub rc: Option<u32>,
}

/// The GitHub owner/repo used for update feeds.
pub const GITHUB_UPDATE_OWNER: &str = "opensquilla";
/// The GitHub repo used for update feeds.
pub const GITHUB_UPDATE_REPO: &str = "opensquilla";
/// The macOS electron-updater feed asset name.
pub const MAC_UPDATE_FEED_ASSET: &str = "latest-mac.yml";
/// The OSS (China mirror) release root.
pub const UPDATE_OSS_RELEASE_ROOT: &str =
    "https://opensquilla-releases.oss-cn-beijing.aliyuncs.com/releases";
/// The GitHub release download root.
pub const UPDATE_GITHUB_RELEASE_ROOT: &str =
    "https://github.com/opensquilla/opensquilla/releases/download";
/// The GitHub release *page* root (used for `releaseUrl`).
pub const UPDATE_GITHUB_RELEASE_PAGE_ROOT: &str =
    "https://github.com/opensquilla/opensquilla/releases/tag";
/// The GitHub releases API listing URL.
pub const UPDATE_GITHUB_RELEASES_API_URL: &str =
    "https://api.github.com/repos/opensquilla/opensquilla/releases?per_page=100";

/// Parse an OpenSquilla release tag. Accepts the PEP440 rc spelling
/// (`v0.5.0rc2`), the semver rc spelling (`v0.5.0-rc2` / `v0.5.0-rc.2`), and a
/// plain stable tag (`v0.5.0`). Returns `None` for anything else.
pub fn parse_release_tag(tag: &str) -> Option<ParsedReleaseTag> {
    let re: &Regex = static_regex(r"^[vV]?(\d+)\.(\d+)\.(\d+)(?:-?rc\.?(\d+))?$");
    let caps = re.captures(tag.trim())?;
    let base = format!("{}.{}.{}", &caps[1], &caps[2], &caps[3]);
    let rc = caps.get(4).map(|m| m.as_str().parse::<u32>().unwrap());
    Some(ParsedReleaseTag { base, rc })
}

/// The canonical Git tag for a parsed version (`v0.5.0` or `v0.5.0rc2`).
pub fn canonical_tag(version: &ParsedReleaseTag) -> String {
    match version.rc {
        None => format!("v{}", version.base),
        Some(rc) => format!("v{}rc{}", version.base, rc),
    }
}

/// The canonical app (semver) version string (`0.5.0` or `0.5.0-rc2`).
pub fn canonical_app_version(version: &ParsedReleaseTag) -> String {
    match version.rc {
        None => version.base.clone(),
        Some(rc) => format!("{}-rc{}", version.base, rc),
    }
}

/// A minimal release summary used by candidate selection.
#[derive(Debug, Clone, Default)]
pub struct ReleaseSummary {
    pub tag_name: Option<String>,
    pub draft: bool,
    pub assets: Vec<String>,
}

/// Given the running prerelease and the repo's releases, pick the highest
/// same-base release that is newer than the current rc and ships the macOS
/// feed. Returns `None` when nothing newer is publishable.
pub fn select_prerelease_candidate(
    current: &ParsedReleaseTag,
    releases: &[ReleaseSummary],
) -> Option<MacPrereleaseCandidate> {
    let current_rc = current.rc?;
    let mut best: Option<(ParsedReleaseTag, String, u64)> = None;
    for release in releases {
        if release.draft {
            continue;
        }
        let tag = release.tag_name.as_deref().unwrap_or("");
        let Some(parsed) = parse_release_tag(tag) else {
            continue;
        };
        if parsed.base != current.base {
            continue;
        }
        let is_stable = parsed.rc.is_none();
        let is_higher_rc = parsed.rc.map(|rc| rc > current_rc).unwrap_or(false);
        if !is_stable && !is_higher_rc {
            continue;
        }
        if !release.assets.iter().any(|a| a == MAC_UPDATE_FEED_ASSET) {
            continue;
        }
        let rank = match parsed.rc {
            None => u64::MAX,
            Some(rc) => rc as u64,
        };
        if best
            .as_ref()
            .map(|(_, _, best_rank)| rank > *best_rank)
            .unwrap_or(true)
        {
            best = Some((parsed, tag.to_string(), rank));
        }
    }
    let (parsed, tag, _) = best?;
    Some(MacPrereleaseCandidate {
        tag,
        version: canonical_app_version(&parsed),
        feed_url: format!(
            "https://github.com/{GITHUB_UPDATE_OWNER}/{GITHUB_UPDATE_REPO}/releases/download/{}",
            canonical_tag(&parsed)
        ),
    })
}

/// A resolved macOS prerelease update candidate.
#[derive(Debug, Clone)]
pub struct MacPrereleaseCandidate {
    pub tag: String,
    pub version: String,
    pub feed_url: String,
}

// ---------------------------------------------------------------------------
// Channel manifest (update-channel.ts)
// ---------------------------------------------------------------------------

/// The desktop update platform identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DesktopUpdatePlatform {
    DarwinArm64,
    Win32X64,
}

impl DesktopUpdatePlatform {
    /// Detect the current platform's update identifier.
    pub fn current() -> Self {
        match (std::env::consts::OS, std::env::consts::ARCH) {
            ("macos", "aarch64") => Self::DarwinArm64,
            ("macos", _) => Self::DarwinArm64,
            ("windows", _) => Self::Win32X64,
            // Linux desktop builds are not auto-updated via this channel; fall
            // back to the Windows identifier so discovery still produces a
            // candidate shape the UI can render.
            _ => Self::Win32X64,
        }
    }
}

/// The update source: OSS mirror (China) or GitHub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DesktopUpdateSource {
    Oss,
    Github,
}

/// A per-platform entry in the channel manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateChannelPlatformEntry {
    pub feed: String,
    pub installer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<String>,
}

/// The validated channel manifest published by the release mirror.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateChannelManifest {
    pub schema_version: u32,
    pub tag: String,
    pub version: String,
    pub base_version: String,
    pub prerelease: bool,
    pub published_at: String,
    pub release_url: String,
    pub sha256sums: String,
    pub platforms: std::collections::HashMap<String, UpdateChannelPlatformEntry>,
}

/// A resolved update candidate ready to offer the user.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesktopUpdateCandidate {
    pub tag: String,
    pub version: String,
    pub base_version: String,
    pub prerelease: bool,
    pub release_url: String,
    pub feed: String,
    pub installer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub archive: Option<String>,
}

/// Errors produced while resolving or validating an update channel.
#[derive(Debug, thiserror::Error)]
pub enum UpdateChannelError {
    #[error("manifest invalid: {0}")]
    ManifestInvalid(String),
    #[error("current version invalid: {0}")]
    CurrentVersionInvalid(String),
    #[error("download failed: {0}")]
    DownloadFailed(String),
    #[error("integrity failure: {0}")]
    IntegrityFailed(String),
}

/// The channel path for a given current version (`stable.json` or
/// `preview/{base}.json`).
pub fn channel_path_for_version(current_version: &str) -> Option<String> {
    let parsed = parse_release_tag(current_version)?;
    match parsed.rc {
        None => Some("stable.json".to_string()),
        Some(_) => Some(format!("preview/{}.json", parsed.base)),
    }
}

/// The full channel manifest URL for the current version.
pub fn channel_manifest_url(current_version: &str) -> Option<String> {
    let path = channel_path_for_version(current_version)?;
    let root = UPDATE_OSS_RELEASE_ROOT.trim_end_matches('/');
    Some(format!("{root}/channels/{path}"))
}

fn base_tuple(base: &str) -> Result<[u32; 3], UpdateChannelError> {
    let parts: Vec<&str> = base.split('.').collect();
    if parts.len() != 3 || !parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
        return Err(UpdateChannelError::ManifestInvalid(format!(
            "invalid base version: {base}"
        )));
    }
    let mut tuple = [0u32; 3];
    for (i, p) in parts.iter().enumerate() {
        tuple[i] = p.parse::<u32>().map_err(|_| {
            UpdateChannelError::ManifestInvalid(format!("invalid base version: {base}"))
        })?;
    }
    Ok(tuple)
}

fn compare_base(left: &str, right: &str) -> Result<std::cmp::Ordering, UpdateChannelError> {
    let a = base_tuple(left)?;
    let b = base_tuple(right)?;
    Ok(a.cmp(&b))
}

fn release_outranks(
    candidate: &ParsedReleaseTag,
    incumbent: &ParsedReleaseTag,
) -> Result<bool, UpdateChannelError> {
    let by_base = compare_base(&candidate.base, &incumbent.base)?;
    if by_base != std::cmp::Ordering::Equal {
        return Ok(by_base == std::cmp::Ordering::Greater);
    }
    Ok(match (candidate.rc, incumbent.rc) {
        (None, Some(_)) => true,
        (Some(_), None) => false,
        (Some(c), Some(i)) => c > i,
        (None, None) => false,
    })
}

fn required_release_assets(version: &ParsedReleaseTag) -> Vec<String> {
    let app_version = canonical_app_version(version);
    vec![
        "SHA256SUMS".to_string(),
        "latest-mac.yml".to_string(),
        "latest.yml".to_string(),
        format!("OpenSquilla-{app_version}-mac-arm64.zip"),
        format!("OpenSquilla-{app_version}-mac-arm64.dmg"),
        format!("OpenSquilla-{app_version}-win-x64.exe"),
    ]
}

/// Build a channel manifest from the GitHub release inventory (the second
/// discovery source). Returns `None` when the current version has no channel
/// or no published release is eligible for it.
pub fn channel_manifest_from_release_inventory(
    current_version: &str,
    inventory: &serde_json::Value,
) -> Result<Option<UpdateChannelManifest>, UpdateChannelError> {
    let current = parse_release_tag(current_version).ok_or_else(|| {
        UpdateChannelError::CurrentVersionInvalid("current app version is unsupported".into())
    })?;
    let releases = inventory.as_array().ok_or_else(|| {
        UpdateChannelError::ManifestInvalid("The GitHub release inventory must be an array.".into())
    })?;

    let mut best: Option<(ParsedReleaseTag, String, String)> = None;
    for raw in releases {
        let release = match raw.as_object() {
            Some(o) => o,
            None => continue,
        };
        if release.get("draft").and_then(|v| v.as_bool()) == Some(true) {
            continue;
        }
        let tag = release
            .get("tag_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let Some(parsed) = parse_release_tag(&tag) else {
            continue;
        };
        if canonical_tag(&parsed) != tag {
            continue;
        }
        let prerelease_flag = release.get("prerelease").and_then(|v| v.as_bool());
        if let Some(flag) = prerelease_flag {
            if flag != (parsed.rc.is_some()) {
                continue;
            }
        }
        // Channel gating.
        if current.rc.is_none() {
            if parsed.rc.is_some() {
                continue;
            }
        } else if parsed.base != current.base {
            continue;
        }
        let published_at = release
            .get("published_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if published_at.is_empty() || !valid_rfc3339(&published_at) {
            continue;
        }
        let asset_names: std::collections::HashSet<String> = release
            .get("assets")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| a.get("name").and_then(|n| n.as_str()).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        if required_release_assets(&parsed)
            .iter()
            .any(|n| !asset_names.contains(n))
        {
            continue;
        }
        if best
            .as_ref()
            .map(|(incumbent, _, _)| release_outranks(&parsed, incumbent))
            .transpose()?
            .unwrap_or(true)
        {
            best = Some((parsed, tag, published_at));
        }
    }
    let (parsed, tag, published_at) = match best {
        Some(b) => b,
        None => return Ok(None),
    };
    let version = canonical_app_version(&parsed);
    let manifest = serde_json::json!({
        "schemaVersion": 1,
        "tag": tag,
        "version": version,
        "baseVersion": parsed.base,
        "prerelease": parsed.rc.is_some(),
        "publishedAt": published_at,
        "releaseUrl": format!("{UPDATE_GITHUB_RELEASE_PAGE_ROOT}/{tag}"),
        "sha256sums": "SHA256SUMS",
        "platforms": {
            "darwin-arm64": {
                "feed": "latest-mac.yml",
                "archive": format!("OpenSquilla-{version}-mac-arm64.zip"),
                "installer": format!("OpenSquilla-{version}-mac-arm64.dmg"),
            },
            "win32-x64": {
                "feed": "latest.yml",
                "installer": format!("OpenSquilla-{version}-win-x64.exe"),
            },
        },
    });
    Ok(Some(validate_channel_manifest(&manifest)?))
}

/// Validate an untrusted channel manifest payload.
pub fn validate_channel_manifest(
    payload: &serde_json::Value,
) -> Result<UpdateChannelManifest, UpdateChannelError> {
    let obj = payload.as_object().ok_or_else(|| {
        UpdateChannelError::ManifestInvalid("channel manifest must be an object".into())
    })?;
    if obj.get("schemaVersion").and_then(|v| v.as_u64()) != Some(1) {
        return Err(UpdateChannelError::ManifestInvalid(
            "unsupported channel manifest schemaVersion".into(),
        ));
    }
    let tag = obj
        .get("tag")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let version = obj
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let base_version = obj
        .get("baseVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let parsed_tag = parse_release_tag(&tag).ok_or_else(|| {
        UpdateChannelError::ManifestInvalid("channel manifest tag and version disagree".into())
    })?;
    let parsed_version = parse_release_tag(&version).ok_or_else(|| {
        UpdateChannelError::ManifestInvalid("channel manifest tag and version disagree".into())
    })?;
    if parsed_tag != parsed_version {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest tag and version disagree".into(),
        ));
    }
    if canonical_tag(&parsed_tag) != tag || canonical_app_version(&parsed_tag) != version {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest tag and version are not canonical".into(),
        ));
    }
    if base_version != parsed_tag.base {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest baseVersion disagrees with tag".into(),
        ));
    }
    if obj.get("prerelease").and_then(|v| v.as_bool()) != Some(parsed_tag.rc.is_some()) {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest prerelease disagrees with tag".into(),
        ));
    }
    let published_at = obj
        .get("publishedAt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if published_at.is_empty() || !valid_rfc3339(&published_at) {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest publishedAt is invalid".into(),
        ));
    }
    let release_url = format!("{UPDATE_GITHUB_RELEASE_PAGE_ROOT}/{tag}");
    if obj.get("releaseUrl").and_then(|v| v.as_str()) != Some(release_url.as_str()) {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest releaseUrl is not canonical".into(),
        ));
    }
    let sha256sums = safe_filename(
        obj.get("sha256sums").and_then(|v| v.as_str()).unwrap_or(""),
        "sha256sums",
    )?;
    if sha256sums != "SHA256SUMS" {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest sha256sums is invalid".into(),
        ));
    }
    let platforms = obj
        .get("platforms")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            UpdateChannelError::ManifestInvalid(
                "channel manifest platforms must be an object".into(),
            )
        })?;
    let mut parsed_platforms = std::collections::HashMap::new();
    for platform in ["darwin-arm64", "win32-x64"] {
        let entry = platforms
            .get(platform)
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                UpdateChannelError::ManifestInvalid(format!(
                    "channel manifest is missing {platform}"
                ))
            })?;
        let feed = safe_filename(
            entry.get("feed").and_then(|v| v.as_str()).unwrap_or(""),
            &format!("{platform}.feed"),
        )?;
        let installer = safe_filename(
            entry
                .get("installer")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
            &format!("{platform}.installer"),
        )?;
        let archive = match entry.get("archive") {
            Some(serde_json::Value::Null) | None => None,
            Some(v) => Some(safe_filename(
                v.as_str().unwrap_or(""),
                &format!("{platform}.archive"),
            )?),
        };
        parsed_platforms.insert(
            platform.to_string(),
            UpdateChannelPlatformEntry {
                feed,
                installer,
                archive,
            },
        );
    }
    let expected_mac_archive = format!("OpenSquilla-{version}-mac-arm64.zip");
    let expected_mac_installer = format!("OpenSquilla-{version}-mac-arm64.dmg");
    let expected_windows_installer = format!("OpenSquilla-{version}-win-x64.exe");
    let mac = parsed_platforms.get("darwin-arm64").unwrap();
    let win = parsed_platforms.get("win32-x64").unwrap();
    if mac.feed != "latest-mac.yml"
        || mac.archive.as_deref() != Some(expected_mac_archive.as_str())
        || mac.installer != expected_mac_installer
        || win.feed != "latest.yml"
        || win.installer != expected_windows_installer
    {
        return Err(UpdateChannelError::ManifestInvalid(
            "channel manifest platform assets do not match the release version".into(),
        ));
    }
    Ok(UpdateChannelManifest {
        schema_version: 1,
        tag,
        version,
        base_version,
        prerelease: parsed_tag.rc.is_some(),
        published_at,
        release_url,
        sha256sums,
        platforms: parsed_platforms,
    })
}

/// Build a [`DesktopUpdateCandidate`] from a validated channel manifest, if the
/// manifest's version is newer than the running version.
pub fn candidate_from_channel(
    current_version: &str,
    payload: &serde_json::Value,
    platform: DesktopUpdatePlatform,
) -> Result<Option<DesktopUpdateCandidate>, UpdateChannelError> {
    let current = parse_release_tag(current_version).ok_or_else(|| {
        UpdateChannelError::CurrentVersionInvalid("current app version is unsupported".into())
    })?;
    let manifest = validate_channel_manifest(payload)?;
    let candidate = parse_release_tag(&manifest.version).ok_or_else(|| {
        UpdateChannelError::ManifestInvalid("validated manifest version could not be parsed".into())
    })?;
    if !candidate_is_newer(&current, &candidate)? {
        return Ok(None);
    }
    let key = match platform {
        DesktopUpdatePlatform::DarwinArm64 => "darwin-arm64",
        DesktopUpdatePlatform::Win32X64 => "win32-x64",
    };
    let entry = manifest.platforms.get(key).ok_or_else(|| {
        UpdateChannelError::ManifestInvalid(format!("channel manifest is missing {key}"))
    })?;
    Ok(Some(DesktopUpdateCandidate {
        tag: manifest.tag,
        version: manifest.version,
        base_version: manifest.base_version,
        prerelease: manifest.prerelease,
        release_url: manifest.release_url,
        feed: entry.feed.clone(),
        installer: entry.installer.clone(),
        archive: entry.archive.clone(),
    }))
}

fn candidate_is_newer(
    current: &ParsedReleaseTag,
    candidate: &ParsedReleaseTag,
) -> Result<bool, UpdateChannelError> {
    if current.rc.is_some() {
        if candidate.base != current.base {
            return Ok(false);
        }
        return Ok(candidate.rc.is_none() || candidate.rc > current.rc);
    }
    Ok(candidate.rc.is_none()
        && compare_base(&candidate.base, &current.base)? == std::cmp::Ordering::Greater)
}

/// The base download URL for a candidate on a given source.
pub fn feed_base_url(candidate: &DesktopUpdateCandidate, source: DesktopUpdateSource) -> String {
    let root = match source {
        DesktopUpdateSource::Oss => UPDATE_OSS_RELEASE_ROOT,
        DesktopUpdateSource::Github => UPDATE_GITHUB_RELEASE_ROOT,
    };
    format!("{}/{}", root.trim_end_matches('/'), candidate.tag)
}

/// The full asset URL for a candidate on a given source.
pub fn asset_url(
    candidate: &DesktopUpdateCandidate,
    source: DesktopUpdateSource,
    asset: &str,
) -> String {
    format!(
        "{}/{}",
        feed_base_url(candidate, source),
        urlencoding(asset)
    )
}

/// Order the update sources for the user's locale. Mainland-CN hints prefer
/// the OSS mirror; otherwise GitHub first.
pub fn ordered_update_sources(
    locale_tags: &[String],
    last_successful: Option<DesktopUpdateSource>,
    override_source: Option<&str>,
) -> Vec<DesktopUpdateSource> {
    let normalized = override_source.unwrap_or("").trim().to_lowercase();
    if normalized == "oss" || normalized == "china" {
        return vec![DesktopUpdateSource::Oss, DesktopUpdateSource::Github];
    }
    if normalized == "github" || normalized == "global" {
        return vec![DesktopUpdateSource::Github, DesktopUpdateSource::Oss];
    }
    if let Some(last) = last_successful {
        return match last {
            DesktopUpdateSource::Oss => vec![DesktopUpdateSource::Oss, DesktopUpdateSource::Github],
            DesktopUpdateSource::Github => {
                vec![DesktopUpdateSource::Github, DesktopUpdateSource::Oss]
            }
        };
    }
    let mainland_hint = locale_tags.iter().any(|tag| is_mainland_hint(tag));
    if mainland_hint {
        vec![DesktopUpdateSource::Oss, DesktopUpdateSource::Github]
    } else {
        vec![DesktopUpdateSource::Github, DesktopUpdateSource::Oss]
    }
}

fn is_mainland_hint(tag: &str) -> bool {
    // A region of CN is a strong mainland hint. A language-only zh tag (no
    // region, no Hant script) is a weak hint; explicit non-mainland regions
    // (e.g. zh-SG) stay on the global order.
    let normalized = tag.trim().replace('_', "-");
    let mut parts = normalized.split('-').filter(|p| !p.is_empty());
    let language = parts.next().unwrap_or("").to_lowercase();
    if language != "zh" {
        return false;
    }
    let mut script = None;
    let mut region = None;
    for p in parts {
        let lower = p.to_lowercase();
        if lower.len() == 4 && lower.chars().all(|c| c.is_ascii_alphabetic()) {
            script = Some(lower);
        } else if lower.len() == 2 && lower.chars().all(|c| c.is_ascii_alphabetic()) {
            region = Some(lower);
        }
    }
    if region.as_deref() == Some("cn") {
        return true;
    }
    region.is_none() && script.as_deref() != Some("hant")
}

// ---------------------------------------------------------------------------
// SHA-256 verification (update-verification.ts)
// ---------------------------------------------------------------------------

/// Parse a canonical SHA256SUMS file and return the digest for `asset`.
pub fn parse_sha256_sums_for_asset(
    contents: &str,
    asset: &str,
) -> Result<String, UpdateChannelError> {
    if asset.is_empty()
        || asset == "."
        || asset == ".."
        || asset.contains('/')
        || asset.contains('\\')
        || asset.contains('\0')
    {
        return Err(UpdateChannelError::IntegrityFailed(
            "The checksum target must be one canonical asset filename.".into(),
        ));
    }
    let re: &Regex = static_regex(r"^[0-9a-fA-F]{64}[ \t]+\*?([^\r\n]+)$");
    let mut matched: Option<String> = None;
    for raw_line in contents.lines() {
        if raw_line.trim().is_empty() {
            continue;
        }
        let caps = match re.captures(raw_line) {
            Some(c) => c,
            None => {
                return Err(UpdateChannelError::IntegrityFailed(
                    "The canonical SHA256SUMS file is malformed.".into(),
                ));
            }
        };
        let digest = caps[0].split_whitespace().next().unwrap().to_lowercase();
        let filename = &caps[1];
        if filename != asset {
            continue;
        }
        if matched.is_some() {
            return Err(UpdateChannelError::IntegrityFailed(format!(
                "The canonical SHA256SUMS file lists {asset} more than once."
            )));
        }
        matched = Some(digest);
    }
    matched.ok_or_else(|| {
        UpdateChannelError::IntegrityFailed(format!(
            "The canonical SHA256SUMS file does not list {asset}."
        ))
    })
}

/// The result of a verified download.
#[derive(Debug, Clone)]
pub struct VerifiedDownloadResult {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

/// Options for a verified download.
pub struct VerifiedDownloadOptions {
    pub max_bytes: u64,
    pub on_progress: Option<Arc<dyn Fn(u64, Option<u64>) + Send + Sync>>,
}

impl std::fmt::Debug for VerifiedDownloadOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedDownloadOptions")
            .field("max_bytes", &self.max_bytes)
            .field("on_progress", &self.on_progress.is_some())
            .finish()
    }
}

/// Stream an HTTP response to a file, hashing as it goes, and verify the
/// SHA-256 against `expected_sha256` before atomically renaming into place.
///
/// The temporary `.part` file is created with restrictive permissions and
/// removed on any failure (size limit, truncation, hash mismatch, IO error).
pub async fn stream_response_to_verified_file(
    response: reqwest::Response,
    destination: &Path,
    expected_sha256: &str,
    options: &VerifiedDownloadOptions,
) -> Result<VerifiedDownloadResult, UpdateChannelError> {
    let expected = expected_sha256.trim().to_lowercase();
    if !is_sha256_hex(&expected) {
        return Err(UpdateChannelError::IntegrityFailed(
            "The expected installer SHA256 is invalid.".into(),
        ));
    }
    if options.max_bytes == 0 {
        return Err(UpdateChannelError::DownloadFailed(
            "The installer size limit is invalid.".into(),
        ));
    }

    let total_bytes: Option<u64> = response.content_length().filter(|&l| l > 0).or_else(|| {
        // No content-length → unknown total.
        None
    });
    if let Some(total) = total_bytes {
        if total > options.max_bytes {
            return Err(UpdateChannelError::DownloadFailed(
                "The installer is larger than the allowed download size.".into(),
            ));
        }
    }

    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|e| {
            UpdateChannelError::DownloadFailed(format!("could not create destination dir: {e}"))
        })?;
    }
    let tmp = destination.with_extension(format!("{}.part", uuid::Uuid::new_v4()));

    let result =
        write_verified_stream(response, &tmp, destination, &expected, options, total_bytes).await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

async fn write_verified_stream(
    response: reqwest::Response,
    tmp: &Path,
    destination: &Path,
    expected: &str,
    options: &VerifiedDownloadOptions,
    total_bytes: Option<u64>,
) -> Result<VerifiedDownloadResult, UpdateChannelError> {
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(tmp)
        .await
        .map_err(|e| {
            UpdateChannelError::DownloadFailed(format!("could not open temp installer file: {e}"))
        })?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = file
            .metadata()
            .await
            .map(|m| std::fs::set_permissions(tmp, std::fs::Permissions::from_mode(0o600)));
    }

    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            UpdateChannelError::DownloadFailed(format!("installer stream error: {e}"))
        })?;
        received += chunk.len() as u64;
        if received > options.max_bytes {
            return Err(UpdateChannelError::DownloadFailed(
                "The installer exceeded the allowed download size.".into(),
            ));
        }
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| {
            UpdateChannelError::DownloadFailed(format!(
                "The installer could not be written completely: {e}"
            ))
        })?;
        if let Some(progress) = &options.on_progress {
            progress(received, total_bytes);
        }
    }
    if let Some(total) = total_bytes {
        if received != total {
            return Err(UpdateChannelError::DownloadFailed(format!(
                "The installer response was truncated ({received} of {total} bytes)."
            )));
        }
    }
    file.flush().await.ok();
    drop(file);

    let actual = hex::encode(hasher.finalize());
    if actual != expected {
        return Err(UpdateChannelError::IntegrityFailed(
            "The installer SHA256 did not match the canonical checksum.".into(),
        ));
    }

    // Remove any previously verified file before the rename (Windows cannot
    // rename over an existing destination).
    let _ = tokio::fs::remove_file(destination).await;
    tokio::fs::rename(tmp, destination).await.map_err(|e| {
        UpdateChannelError::DownloadFailed(format!(
            "The verified installer could not be finalized: {e}"
        ))
    })?;
    Ok(VerifiedDownloadResult {
        path: destination.to_path_buf(),
        bytes: received,
        sha256: actual,
    })
}

// ---------------------------------------------------------------------------
// Update-check scheduler (update-check-scheduler.ts)
// ---------------------------------------------------------------------------

/// The current update activity that gates new checks.
#[derive(Debug, Clone, Copy, Default)]
pub struct UpdateCheckActivity {
    pub downloading: bool,
    pub applying: bool,
    pub downloaded: bool,
}

impl UpdateCheckActivity {
    /// True when no download or apply is in flight, so a new check is allowed.
    pub fn allows_check(self) -> bool {
        !self.downloading && !self.applying && !self.downloaded
    }
}

/// Recursive-timeout update-check scheduler.
///
/// Runs checks measured from *completion* rather than start. The timer is
/// rescheduled only after a completed check (or when `request` re-arms it),
/// matching the Electron `UpdateCheckScheduler` contract. Manual callers join
/// the current request and can promote a silent automatic request to manual
/// notification semantics without a second fetch.
pub struct UpdateCheckScheduler {
    inner: Arc<AsyncMutex<SchedulerInner>>,
}

struct SchedulerInner {
    started: bool,
    stopped: bool,
    manual_requested: bool,
    in_flight: Option<tokio::task::JoinHandle<()>>,
    timer_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Type-erased check callback: a pinned, boxed future returned from a
/// callable.  Using `Arc<dyn …>` avoids the generic type recursion that
/// occurs when `request` and `schedule` call each other with fresh closure
/// types.
type RunCheck =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>;

impl UpdateCheckScheduler {
    /// Create a new scheduler.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(AsyncMutex::new(SchedulerInner {
                started: false,
                stopped: false,
                manual_requested: false,
                in_flight: None,
                timer_handle: None,
            })),
        }
    }

    /// Whether a manual request is pending promotion.
    pub async fn manual_request_pending(&self) -> bool {
        self.inner.lock().await.manual_requested
    }

    /// Consume and return whether a manual request was pending.
    pub async fn consume_manual_request(&self) -> bool {
        let mut inner = self.inner.lock().await;
        let requested = inner.manual_requested;
        inner.manual_requested = false;
        requested
    }

    /// Start the scheduler with an initial delay before the first check.
    pub async fn start(self: &Arc<Self>, initial_delay: Duration, run_check: RunCheck) {
        let mut inner = self.inner.lock().await;
        if inner.started || inner.stopped {
            return;
        }
        inner.started = true;
        drop(inner);
        self.schedule(initial_delay, run_check);
    }

    /// Stop the scheduler; no further checks will run.
    pub async fn stop(&self) {
        let mut inner = self.inner.lock().await;
        inner.stopped = true;
        if let Some(handle) = inner.timer_handle.take() {
            handle.abort();
        }
    }

    /// Request a check now, joining an in-flight check if one is running.
    ///
    /// `can_check` gates whether a new fetch is permitted (e.g. no download or
    /// install is already in progress). `repeat_delay` is used to re-arm the
    /// recursive timer after the check completes.
    pub async fn request(
        self: &Arc<Self>,
        manual: bool,
        can_check: impl Fn() -> bool + Send + Sync + 'static,
        repeat_delay: Duration,
        run_check: RunCheck,
    ) {
        let mut inner = self.inner.lock().await;
        if inner.stopped {
            return;
        }
        if manual {
            inner.manual_requested = true;
        }
        if inner.in_flight.is_some() {
            return;
        }
        if !can_check() {
            // Keep the recursive schedule alive while a download/install is in
            // progress, but do not disturb a pending timer.
            if inner.started && inner.timer_handle.is_none() {
                drop(inner);
                let this = self.clone();
                this.schedule(repeat_delay, run_check);
            }
            return;
        }
        // Clear any pending timer; the check is starting now.
        if let Some(handle) = inner.timer_handle.take() {
            handle.abort();
        }
        let this = self.clone();
        let handle = tokio::spawn(async move {
            run_check().await;
            let mut inner = this.inner.lock().await;
            inner.in_flight = None;
            inner.manual_requested = false;
            let started = inner.started;
            let stopped = inner.stopped;
            drop(inner);
            if started && !stopped {
                this.schedule(repeat_delay, run_check);
            }
        });
        inner.in_flight = Some(handle);
    }

    /// Set up a timer to fire after `delay` and then call [`request`].
    ///
    /// This is deliberately **not** `async` — returning a concrete
    /// [`JoinHandle`] instead of an opaque future breaks the type-level
    /// recursion cycle with [`request`].
    fn schedule(self: &Arc<Self>, delay: Duration, run_check: RunCheck) {
        let this = self.clone();
        tokio::spawn(async move {
            let mut inner = this.inner.lock().await;
            if inner.stopped {
                return;
            }
            if let Some(handle) = inner.timer_handle.take() {
                handle.abort();
            }
            drop(inner);
            let this_inner = this.clone();
            let handle = tokio::spawn(async move {
                tokio::time::sleep(delay).await;
                this_inner.request(false, || true, delay, run_check).await;
            });
            let mut inner = this.inner.lock().await;
            inner.timer_handle = Some(handle);
        });
    }
}

impl Default for UpdateCheckScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Update state (frontend notification)
// ---------------------------------------------------------------------------

/// The current update state, emitted to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateState {
    pub available: bool,
    pub version: Option<String>,
    pub release_url: Option<String>,
    pub downloading: bool,
    pub downloaded: bool,
    pub applying: bool,
    pub error: Option<String>,
    pub managed: bool,
}

impl Default for UpdateState {
    fn default() -> Self {
        Self {
            available: false,
            version: None,
            release_url: None,
            downloading: false,
            downloaded: false,
            applying: false,
            error: None,
            managed: true,
        }
    }
}

/// The shared update state, accessible to command handlers.
#[derive(Debug, Default)]
pub struct UpdateStateHandle {
    state: Mutex<UpdateState>,
}

impl UpdateStateHandle {
    /// Create a fresh handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current state.
    pub fn snapshot(&self) -> UpdateState {
        self.state.lock().clone()
    }

    /// Update the state in place.
    pub fn set(&self, state: UpdateState) {
        *self.state.lock() = state;
    }

    /// Mutate the state under the lock.
    pub fn with_mut<F: FnOnce(&mut UpdateState)>(&self, f: F) {
        f(&mut self.state.lock());
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn static_regex(pattern: &str) -> &'static Regex {
    // A small cache keyed by pattern. Patterns in this module are compile-time
    // literals, so the cache fills once and subsequent lookups are cheap. The
    // returned Regex is leaked (lives for the program's lifetime), so the
    // 'static lifetime is sound even when the input pattern is not.
    static CACHE: once_cell::sync::Lazy<
        parking_lot::Mutex<std::collections::HashMap<String, &'static Regex>>,
    > = once_cell::sync::Lazy::new(|| parking_lot::Mutex::new(std::collections::HashMap::new()));
    let mut cache = CACHE.lock();
    if let Some(re) = cache.get(pattern) {
        return *re;
    }
    let leaked: &'static Regex = Box::leak(Box::new(Regex::new(pattern).expect("invalid regex")));
    cache.insert(pattern.to_string(), leaked);
    leaked
}

fn safe_filename(value: &str, field: &str) -> Result<String, UpdateChannelError> {
    if value.is_empty() || value == "." || value == ".." {
        return Err(UpdateChannelError::ManifestInvalid(format!(
            "{field} must be a non-empty filename"
        )));
    }
    if value.contains('/') || value.contains('\\') || value.contains('\0') {
        return Err(UpdateChannelError::ManifestInvalid(format!(
            "{field} must be a single filename"
        )));
    }
    Ok(value.to_string())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn valid_rfc3339(value: &str) -> bool {
    let re: &Regex = static_regex(
        r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$",
    );
    let Some(caps) = re.captures(value) else {
        return false;
    };
    let year: u32 = caps[1].parse().unwrap_or(0);
    let month: u32 = caps[2].parse().unwrap_or(0);
    let day: u32 = caps[3].parse().unwrap_or(0);
    let hour: u32 = caps[4].parse().unwrap_or(0);
    let minute: u32 = caps[5].parse().unwrap_or(0);
    let second: u32 = caps[6].parse().unwrap_or(0);
    let offset_hour: u32 = caps
        .get(7)
        .map(|m| m.as_str().parse().unwrap_or(0))
        .unwrap_or(0);
    let offset_minute: u32 = caps
        .get(8)
        .map(|m| m.as_str().parse().unwrap_or(0))
        .unwrap_or(0);
    let days_in_month = days_in_month(year, month);
    year >= 1
        && (1..=12).contains(&month)
        && (1..=days_in_month).contains(&day)
        && hour <= 23
        && minute <= 59
        && second <= 59
        && offset_hour <= 23
        && offset_minute <= 59
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn urlencoding(s: &str) -> String {
    // Minimal percent-encoding for the path characters we expect in asset
    // filenames (none in practice, but be safe).
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{:02X}", b));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_release_tags() {
        assert_eq!(
            parse_release_tag("v0.5.0"),
            Some(ParsedReleaseTag {
                base: "0.5.0".into(),
                rc: None,
            })
        );
        assert_eq!(
            parse_release_tag("v0.5.0rc2"),
            Some(ParsedReleaseTag {
                base: "0.5.0".into(),
                rc: Some(2),
            })
        );
        assert_eq!(
            parse_release_tag("v0.5.0-rc2"),
            Some(ParsedReleaseTag {
                base: "0.5.0".into(),
                rc: Some(2),
            })
        );
        assert_eq!(parse_release_tag("not-a-tag"), None);
    }

    #[test]
    fn canonical_forms() {
        let stable = ParsedReleaseTag {
            base: "0.5.0".into(),
            rc: None,
        };
        assert_eq!(canonical_tag(&stable), "v0.5.0");
        assert_eq!(canonical_app_version(&stable), "0.5.0");
        let rc = ParsedReleaseTag {
            base: "0.5.0".into(),
            rc: Some(3),
        };
        assert_eq!(canonical_tag(&rc), "v0.5.0rc3");
        assert_eq!(canonical_app_version(&rc), "0.5.0-rc3");
    }

    #[test]
    fn channel_path_for_stable_and_preview() {
        assert_eq!(
            channel_path_for_version("0.5.0"),
            Some("stable.json".into())
        );
        assert_eq!(
            channel_path_for_version("0.5.0-rc2"),
            Some("preview/0.5.0.json".into())
        );
        assert_eq!(channel_path_for_version("bogus"), None);
    }

    #[test]
    fn ordered_sources_mainland_hint() {
        let tags = vec!["zh-CN".to_string()];
        let order = ordered_update_sources(&tags, None, None);
        assert_eq!(
            order,
            vec![DesktopUpdateSource::Oss, DesktopUpdateSource::Github]
        );
        let tags = vec!["en-US".to_string()];
        let order = ordered_update_sources(&tags, None, None);
        assert_eq!(
            order,
            vec![DesktopUpdateSource::Github, DesktopUpdateSource::Oss]
        );
    }

    #[test]
    fn ordered_sources_override() {
        let order = ordered_update_sources(&[], None, Some("oss"));
        assert_eq!(
            order,
            vec![DesktopUpdateSource::Oss, DesktopUpdateSource::Github]
        );
        let order = ordered_update_sources(&[], None, Some("github"));
        assert_eq!(
            order,
            vec![DesktopUpdateSource::Github, DesktopUpdateSource::Oss]
        );
    }

    #[test]
    fn sha256sums_parsing() {
        let contents = format!(
            "abcdef0123456789{}  *OpenSquilla-0.5.0-win-x64.exe\n1111  other.exe",
            "a".repeat(48)
        );
        let digest =
            parse_sha256_sums_for_asset(&contents, "OpenSquilla-0.5.0-win-x64.exe").unwrap();
        assert_eq!(digest, format!("abcdef0123456789{}", "a".repeat(48)));
        assert!(parse_sha256_sums_for_asset(&contents, "missing.exe").is_err());
    }

    #[test]
    fn validate_minimal_manifest() {
        let manifest = serde_json::json!({
            "schemaVersion": 1,
            "tag": "v0.6.0",
            "version": "0.6.0",
            "baseVersion": "0.6.0",
            "prerelease": false,
            "publishedAt": "2026-01-02T03:04:05Z",
            "releaseUrl": format!("{UPDATE_GITHUB_RELEASE_PAGE_ROOT}/v0.6.0"),
            "sha256sums": "SHA256SUMS",
            "platforms": {
                "darwin-arm64": {
                    "feed": "latest-mac.yml",
                    "archive": "OpenSquilla-0.6.0-mac-arm64.zip",
                    "installer": "OpenSquilla-0.6.0-mac-arm64.dmg",
                },
                "win32-x64": {
                    "feed": "latest.yml",
                    "installer": "OpenSquilla-0.6.0-win-x64.exe",
                },
            },
        });
        let validated = validate_channel_manifest(&manifest).unwrap();
        assert_eq!(validated.version, "0.6.0");
        assert!(!validated.prerelease);
    }

    #[test]
    fn candidate_is_newer_for_stable() {
        let current = ParsedReleaseTag {
            base: "0.5.0".into(),
            rc: None,
        };
        let candidate = ParsedReleaseTag {
            base: "0.6.0".into(),
            rc: None,
        };
        assert!(candidate_is_newer(&current, &candidate).unwrap());
        let same = ParsedReleaseTag {
            base: "0.5.0".into(),
            rc: None,
        };
        assert!(!candidate_is_newer(&current, &same).unwrap());
    }

    #[test]
    fn rfc3339_validation() {
        assert!(valid_rfc3339("2026-01-02T03:04:05Z"));
        assert!(valid_rfc3339("2026-01-02T03:04:05+08:00"));
        assert!(!valid_rfc3339("not-a-date"));
        assert!(!valid_rfc3339("2026-13-02T03:04:05Z")); // bad month
    }

    #[test]
    fn activity_gates_checks() {
        let a = UpdateCheckActivity::default();
        assert!(a.allows_check());
        let a = UpdateCheckActivity {
            downloading: true,
            ..Default::default()
        };
        assert!(!a.allows_check());
        let a = UpdateCheckActivity {
            downloaded: true,
            ..Default::default()
        };
        assert!(!a.allows_check());
    }
}
