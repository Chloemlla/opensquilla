//! Managed network proxy environment shared by sandbox backends.
//!
//! Port of `src/opensquilla/sandbox/managed_proxy_env.py`. Produces the
//! environment-variable set that points every common package manager and HTTP
//! client at the managed proxy, plus the control variables that disable
//! bypasses and local binding.

/// Environment key signalling to the sandbox host that the proxy is active.
pub const PROXY_ACTIVE_ENV_KEY: &str = "CODEX_NETWORK_PROXY_ACTIVE";
/// Environment key signalling that local binding is disallowed.
pub const ALLOW_LOCAL_BINDING_ENV_KEY: &str = "CODEX_NETWORK_ALLOW_LOCAL_BINDING";
/// OpenSquilla network-mode key.
pub const OPENSQUILLA_NETWORK_ENV_KEY: &str = "OPENSQUILLA_SANDBOX_NETWORK";

/// Every proxy URL variable set to `http://<host>:<port>`.
pub const PROXY_ENV_KEYS: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "YARN_HTTP_PROXY",
    "YARN_HTTPS_PROXY",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "npm_config_proxy",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
    "NPM_CONFIG_PROXY",
    "BUNDLE_HTTP_PROXY",
    "BUNDLE_HTTPS_PROXY",
    "PIP_PROXY",
    "DOCKER_HTTP_PROXY",
    "DOCKER_HTTPS_PROXY",
    "WS_PROXY",
    "WSS_PROXY",
    "ws_proxy",
    "wss_proxy",
    "ALL_PROXY",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
];

/// No-proxy variables set to the default loopback/private-range value.
pub const NO_PROXY_ENV_KEYS: &[&str] = &[
    "NO_PROXY",
    "no_proxy",
    "npm_config_noproxy",
    "NPM_CONFIG_NOPROXY",
    "YARN_NO_PROXY",
    "BUNDLE_NO_PROXY",
];

/// Control variables (fixed values) injected into the managed-proxy env.
pub const PROXY_CONTROL_ENV: &[(&str, &str)] = &[
    (PROXY_ACTIVE_ENV_KEY, "1"),
    (ALLOW_LOCAL_BINDING_ENV_KEY, "0"),
    ("NODE_USE_ENV_PROXY", "1"),
    ("ELECTRON_GET_USE_PROXY", "true"),
    (OPENSQUILLA_NETWORK_ENV_KEY, "proxy_allowlist"),
];

/// Windows git SSL backend override (needed because git for Windows bundles a
/// different TLS backend that ignores proxy env vars).
pub const WINDOWS_GIT_SSL_ENV: &[(&str, &str)] = &[
    ("GIT_CONFIG_COUNT", "1"),
    ("GIT_CONFIG_KEY_0", "http.sslBackend"),
    ("GIT_CONFIG_VALUE_0", "openssl"),
];

/// The default no-proxy value: localhost plus the private/link-local ranges.
pub const DEFAULT_NO_PROXY_VALUE: &str =
    "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16";

/// Build the managed-proxy environment for `host:port`.
pub fn managed_proxy_env(
    host: &str,
    port: u16,
    windows_git_ssl_backend: bool,
) -> std::collections::HashMap<String, String> {
    let proxy_url = format!("http://{host}:{port}");
    let mut env = std::collections::HashMap::new();
    for key in PROXY_ENV_KEYS {
        env.insert((*key).to_string(), proxy_url.clone());
    }
    for key in NO_PROXY_ENV_KEYS {
        env.insert((*key).to_string(), DEFAULT_NO_PROXY_VALUE.to_string());
    }
    for (key, value) in PROXY_CONTROL_ENV {
        env.insert((*key).to_string(), (*value).to_string());
    }
    if windows_git_ssl_backend {
        for (key, value) in WINDOWS_GIT_SSL_ENV {
            env.insert((*key).to_string(), (*value).to_string());
        }
    }
    env
}

/// Build the managed-proxy environment for a backend name.
///
/// Mirrors `integration.py::_managed_proxy_env`: Windows backends also inject
/// the git SSL backend override because git-for-Windows ignores proxy env vars
/// with its default TLS backend.
pub fn managed_proxy_env_for_backend(
    backend_name: Option<&str>,
    host: &str,
    port: u16,
) -> std::collections::HashMap<String, String> {
    let windows_git = backend_name
        .map(|name| name.trim().to_lowercase().starts_with("windows_"))
        .unwrap_or(false);
    managed_proxy_env(host, port, windows_git)
}

/// Extend a policy environment allowlist with every key the managed-proxy env
/// touches, so backend env filtering keeps the proxy variables.
///
/// Returns `true` when any key was added (callers can use that to decide
/// whether the policy needs re-validating).
pub fn extend_env_allowlist_with_proxy_vars(
    allowlist: &mut Vec<String>,
    include_windows_git: bool,
) -> bool {
    let mut added = false;
    for key in managed_proxy_env_allowlist(include_windows_git) {
        if !allowlist.contains(&key) {
            allowlist.push(key);
            added = true;
        }
    }
    added
}

/// Every environment variable the managed-proxy env touches.
pub fn managed_proxy_env_allowlist(include_windows_git: bool) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for key in PROXY_ENV_KEYS {
        keys.push((*key).to_string());
    }
    for key in NO_PROXY_ENV_KEYS {
        keys.push((*key).to_string());
    }
    for (key, _) in PROXY_CONTROL_ENV {
        keys.push((*key).to_string());
    }
    if include_windows_git {
        for (key, _) in WINDOWS_GIT_SSL_ENV {
            keys.push((*key).to_string());
        }
    }
    keys
}

/// The allowlist keys, uppercased (for case-insensitive matching).
pub fn managed_proxy_env_names_upper(include_windows_git: bool) -> Vec<String> {
    managed_proxy_env_allowlist(include_windows_git)
        .into_iter()
        .map(|k| k.to_uppercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_env_points_everything_at_proxy() {
        let env = managed_proxy_env("127.0.0.1", 8765, false);
        assert_eq!(env.get("HTTP_PROXY"), Some(&"http://127.0.0.1:8765".to_string()));
        assert_eq!(env.get("HTTPS_PROXY"), Some(&"http://127.0.0.1:8765".to_string()));
        assert_eq!(env.get("http_proxy"), Some(&"http://127.0.0.1:8765".to_string()));
        assert_eq!(env.get("PIP_PROXY"), Some(&"http://127.0.0.1:8765".to_string()));
        assert_eq!(env.get("ALL_PROXY"), Some(&"http://127.0.0.1:8765".to_string()));
    }

    #[test]
    fn no_proxy_and_control_vars() {
        let env = managed_proxy_env("127.0.0.1", 8765, false);
        assert_eq!(env.get("NO_PROXY"), Some(&DEFAULT_NO_PROXY_VALUE.to_string()));
        assert_eq!(env.get(PROXY_ACTIVE_ENV_KEY), Some(&"1".to_string()));
        assert_eq!(env.get(ALLOW_LOCAL_BINDING_ENV_KEY), Some(&"0".to_string()));
        assert_eq!(env.get(OPENSQUILLA_NETWORK_ENV_KEY), Some(&"proxy_allowlist".to_string()));
    }

    #[test]
    fn windows_git_ssl_optional() {
        let env = managed_proxy_env("127.0.0.1", 8765, true);
        assert_eq!(env.get("GIT_CONFIG_COUNT"), Some(&"1".to_string()));
        assert_eq!(env.get("GIT_CONFIG_KEY_0"), Some(&"http.sslBackend".to_string()));
        assert!(!managed_proxy_env("127.0.0.1", 8765, false).contains_key("GIT_CONFIG_COUNT"));
    }

    #[test]
    fn allowlist_covers_all_keys() {
        let allowlist = managed_proxy_env_allowlist(true);
        let env = managed_proxy_env("127.0.0.1", 8765, true);
        for key in allowlist {
            assert!(env.contains_key(&key), "missing key {key}");
        }
        let upper = managed_proxy_env_names_upper(true);
        assert!(upper.contains(&"HTTP_PROXY".to_string()));
        assert!(upper.contains(&"NO_PROXY".to_string()));
    }

    #[test]
    fn backend_aware_proxy_env() {
        let windows = managed_proxy_env_for_backend(Some("windows_default"), "127.0.0.1", 8765);
        assert_eq!(
            windows.get("GIT_CONFIG_COUNT"),
            Some(&"1".to_string()),
            "windows backends get the git ssl override"
        );
        let bwrap = managed_proxy_env_for_backend(Some("bwrap"), "127.0.0.1", 8765);
        assert!(!bwrap.contains_key("GIT_CONFIG_COUNT"));
        assert_eq!(
            bwrap.get("HTTP_PROXY"),
            Some(&"http://127.0.0.1:8765".to_string())
        );
        assert_eq!(
            managed_proxy_env_for_backend(None, "127.0.0.1", 8765),
            managed_proxy_env("127.0.0.1", 8765, false)
        );
    }

    #[test]
    fn env_allowlist_extension_adds_keys() {
        let mut allowlist = vec!["PATH".to_string(), "HTTP_PROXY".to_string()];
        assert!(extend_env_allowlist_with_proxy_vars(&mut allowlist, true));
        assert!(allowlist.contains(&"HTTPS_PROXY".to_string()));
        assert!(allowlist.contains(&"GIT_CONFIG_COUNT".to_string()));
        // Re-running adds nothing new.
        assert!(!extend_env_allowlist_with_proxy_vars(&mut allowlist, true));
    }
}
