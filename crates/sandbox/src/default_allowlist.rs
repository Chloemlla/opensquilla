//! Built-in managed-network allowlist entries.
//!
//! Port of `src/opensquilla/sandbox/default_allowlist.py`. Groups common
//! developer destinations (source hosts, search engines, docs) into named
//! allowlist groups. The groups are read-only: they cannot be extended by
//! callers and always match through [`crate::domain_validation::domain_matches`].

use crate::domain_validation::domain_matches;

/// A named group of allowlisted domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DefaultAllowlistGroup {
    /// Group id, e.g. `"github"`.
    pub group: String,
    /// The domains in this group.
    pub domains: Vec<String>,
    /// Groups are always read-only.
    pub read_only: bool,
}

/// The built-in allowlist groups.
pub fn default_allowlist() -> Vec<DefaultAllowlistGroup> {
    vec![
        DefaultAllowlistGroup {
            group: "github".to_string(),
            domains: vec![
                "github.com".to_string(),
                "api.github.com".to_string(),
                "raw.githubusercontent.com".to_string(),
                "objects.githubusercontent.com".to_string(),
                "codeload.github.com".to_string(),
                "github.githubassets.com".to_string(),
                "avatars.githubusercontent.com".to_string(),
                "uploads.github.com".to_string(),
                "release-assets.githubusercontent.com".to_string(),
                "ghcr.io".to_string(),
                "pkg-containers.githubusercontent.com".to_string(),
            ],
            read_only: true,
        },
        DefaultAllowlistGroup {
            group: "search".to_string(),
            domains: vec![
                "api.search.brave.com".to_string(),
                "html.duckduckgo.com".to_string(),
                "duckduckgo.com".to_string(),
                "api.bochaai.com".to_string(),
                "api.exa.ai".to_string(),
                "api.tavily.com".to_string(),
                "cloud-iqs.aliyuncs.com".to_string(),
                "www.google.com".to_string(),
                "www.bing.com".to_string(),
            ],
            read_only: true,
        },
        DefaultAllowlistGroup {
            group: "developer-docs".to_string(),
            domains: vec![
                "developer.mozilla.org".to_string(),
                "docs.python.org".to_string(),
                "docs.npmjs.com".to_string(),
                "doc.rust-lang.org".to_string(),
                "go.dev".to_string(),
            ],
            read_only: true,
        },
    ]
}

/// Return `"default:<group>"` when `host` matches a domain in a default
/// allowlist group, or `None`.
pub fn default_allowlist_source(host: &str) -> Option<String> {
    for group in default_allowlist() {
        if group
            .domains
            .iter()
            .any(|domain| domain_matches(domain, host))
        {
            return Some(format!("default:{}", group.group));
        }
    }
    None
}

/// Every domain across all default groups, deduplicated and in group order.
///
/// Callers that seed a proxy allowlist or build a
/// [`crate::policy::NetworkPolicy::ProxyAllowlist`] use this to include the
/// built-in developer destinations without duplicating the catalog.
pub fn default_allowlist_domains() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for group in default_allowlist() {
        for domain in group.domains {
            if seen.insert(domain.clone()) {
                out.push(domain);
            }
        }
    }
    out
}

/// A proxy-allowlist network policy seeded with the built-in developer
/// destinations plus every package-manager bundle domain.
///
/// This is the "default network posture" source the operator layer can use to
/// construct a [`crate::policy::NetworkPolicy`] for a fresh install: broad
/// enough for git/pip/npm/cargo/go to work, without host networking.
pub fn default_allowlist_network_policy() -> crate::policy::NetworkPolicy {
    let mut domains = default_allowlist_domains();
    for id in crate::package_bundles::default_package_bundle_ids() {
        for domain in crate::package_bundles::expand_package_bundle(&id) {
            if !domains.contains(&domain) {
                domains.push(domain);
            }
        }
    }
    crate::policy::NetworkPolicy::ProxyAllowlist(domains)
}

/// The serializable payload shape consumed by status endpoints: one entry per
/// group with its domains and a `read_only` flag.
pub fn default_allowlist_payload() -> Vec<serde_json::Value> {
    default_allowlist()
        .into_iter()
        .map(|group| {
            serde_json::json!({
                "group": group.group,
                "domains": group.domains,
                "read_only": group.read_only,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_source_detected() {
        assert_eq!(
            default_allowlist_source("github.com"),
            Some("default:github".to_string())
        );
        assert_eq!(
            default_allowlist_source("raw.githubusercontent.com"),
            Some("default:github".to_string())
        );
    }

    #[test]
    fn search_and_docs() {
        assert_eq!(
            default_allowlist_source("api.tavily.com"),
            Some("default:search".to_string())
        );
        assert_eq!(
            default_allowlist_source("docs.python.org"),
            Some("default:developer-docs".to_string())
        );
    }

    #[test]
    fn unknown_host_has_no_source() {
        assert_eq!(default_allowlist_source("evil.example.com"), None);
        assert_eq!(default_allowlist_source(""), None);
        // A suffix of an allowlisted domain is not a match.
        assert_eq!(default_allowlist_source("notgithub.com"), None);
    }

    #[test]
    fn payload_shape() {
        let payload = default_allowlist_payload();
        assert_eq!(payload.len(), 3);
        for entry in &payload {
            assert_eq!(entry["read_only"], serde_json::Value::Bool(true));
            assert!(entry["domains"].is_array());
        }
    }

    #[test]
    fn domains_collector_is_deduped_and_complete() {
        let domains = default_allowlist_domains();
        let mut dedup = domains.clone();
        dedup.sort();
        dedup.dedup();
        assert_eq!(dedup.len(), domains.len(), "domains must be unique");
        assert!(domains.contains(&"github.com".to_string()));
        assert!(domains.contains(&"docs.python.org".to_string()));
        assert!(domains.contains(&"api.tavily.com".to_string()));
    }

    #[test]
    fn default_network_policy_seeds_bundles() {
        let policy = default_allowlist_network_policy();
        let domains = policy.allowlist().expect("proxy allowlist");
        assert!(domains.contains(&"github.com".to_string()));
        assert!(domains.contains(&"pypi.org".to_string()));
        assert!(domains.contains(&"crates.io".to_string()));
        // All entries pass domain validation (bundle catalogs are valid).
        assert!(policy.invalid_allowlist_entries().is_empty());
    }
}
