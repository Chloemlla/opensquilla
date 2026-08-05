use serde::{Deserialize, Serialize};

/// Information about the current build.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BuildInfo {
    /// Semver version of the build.
    pub version: String,
    /// Git commit hash, if available at build time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Target triple (e.g. `x86_64-pc-windows-msvc`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Build profile (`debug`/`release`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Feature flags enabled at build time.
    #[serde(default)]
    pub features: Vec<String>,
}

impl BuildInfo {
    /// Build info derived from compile-time environment variables.
    ///
    /// Populated via `CARGO_PKG_VERSION` and the optional `GIT_HASH`,
    /// `TARGET_TRIPLE`, `BUILD_PROFILE`, and `BUILD_FEATURES` env vars set at
    /// compile time.
    pub fn local() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: option_env!("GIT_HASH").map(|s| s.to_string()),
            target: option_env!("TARGET_TRIPLE").map(|s| s.to_string()),
            profile: option_env!("BUILD_PROFILE").map(|s| s.to_string()),
            features: option_env!("BUILD_FEATURES")
                .map(|s| s.split(',').map(|f| f.trim().to_string()).collect())
                .unwrap_or_default(),
        }
    }

    /// Compare two build versions, returning a mismatch error if they differ.
    pub fn ensure_version(&self, expected: &str) -> crate::Result<()> {
        if self.version == expected {
            Ok(())
        } else {
            Err(crate::Error::VersionMismatch {
                expected: expected.to_string(),
                found: self.version.clone(),
            })
        }
    }
}
