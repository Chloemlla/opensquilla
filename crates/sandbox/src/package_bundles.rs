//! Package-manager domain bundles for sandbox managed network.
//!
//! Port of `src/opensquilla/sandbox/package_bundles.py`. Each bundle maps a
//! package manager to the domains its toolchain contacts while installing or
//! resolving packages. The network policy layer expands a bundle id into the
//! allowlist entries the managed proxy should permit.
//!
//! The module is a static catalog; the ordering of bundle ids is preserved for
//! deterministic serialization.

/// A static map of bundle id -> domains.
///
/// The domains are intentionally lowercase and unvalidated here: validation
/// happens at ingestion time via [`crate::domain_validation::validate_domain_pattern`].
pub const PACKAGE_BUNDLES: &[(&str, &[&str])] = &[
    (
        "python-package-install",
        &[
            "pypi.org",
            "files.pythonhosted.org",
            "pypi.python.org",
            "bootstrap.pypa.io",
            "python-poetry.org",
            "install.python-poetry.org",
        ],
    ),
    (
        "node-package-install",
        &[
            "registry.npmjs.org",
            "registry.yarnpkg.com",
            "yarnpkg.com",
            "nodejs.org",
            "unpkg.com",
            "cdn.jsdelivr.net",
        ],
    ),
    (
        "rust-package-install",
        &[
            "crates.io",
            "static.crates.io",
            "index.crates.io",
            "github.com",
            "objects.githubusercontent.com",
        ],
    ),
    (
        "go-package-install",
        &[
            "proxy.golang.org",
            "sum.golang.org",
            "go.dev",
            "golang.org",
            "storage.googleapis.com",
        ],
    ),
    (
        "java-package-install",
        &[
            "repo.maven.apache.org",
            "repo1.maven.org",
            "plugins.gradle.org",
            "services.gradle.org",
        ],
    ),
    (
        "php-package-install",
        &["packagist.org", "repo.packagist.org", "getcomposer.org"],
    ),
    (
        "github-default",
        &[
            "github.com",
            "api.github.com",
            "raw.githubusercontent.com",
            "objects.githubusercontent.com",
            "codeload.github.com",
            "github.githubassets.com",
            "avatars.githubusercontent.com",
            "actions.githubusercontent.com",
            "pipelines.actions.githubusercontent.com",
            "results-receiver.actions.githubusercontent.com",
            "uploads.github.com",
            "release-assets.githubusercontent.com",
            "ghcr.io",
            "pkg-containers.githubusercontent.com",
        ],
    ),
];

/// All bundle ids, in catalog order.
pub fn default_package_bundle_ids() -> Vec<String> {
    PACKAGE_BUNDLES
        .iter()
        .map(|(id, _)| (*id).to_string())
        .collect()
}

/// Expand a bundle id into its domains. Unknown or empty ids yield an empty
/// list (fail closed).
pub fn expand_package_bundle(bundle_id: &str) -> Vec<String> {
    match PACKAGE_BUNDLES
        .iter()
        .find(|(id, _)| *id == bundle_id.trim())
    {
        Some((_, domains)) => domains.iter().map(|d| (*d).to_string()).collect(),
        None => Vec::new(),
    }
}

/// The domains for a bundle id, as static string slices.
pub fn package_bundle_domains(bundle_id: &str) -> &'static [&'static str] {
    match PACKAGE_BUNDLES
        .iter()
        .find(|(id, _)| *id == bundle_id.trim())
    {
        Some((_, domains)) => domains,
        None => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundles_are_unique_and_ordered() {
        let ids = default_package_bundle_ids();
        assert!(ids.contains(&"python-package-install".to_string()));
        assert!(ids.contains(&"github-default".to_string()));
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), ids.len(), "bundle ids must be unique");
    }

    #[test]
    fn expand_known_bundle() {
        let domains = expand_package_bundle("python-package-install");
        assert!(domains.contains(&"pypi.org".to_string()));
        assert!(domains.contains(&"files.pythonhosted.org".to_string()));
    }

    #[test]
    fn expand_unknown_bundle_fails_closed() {
        assert!(expand_package_bundle("not-a-bundle").is_empty());
        assert!(expand_package_bundle("").is_empty());
    }

    #[test]
    fn static_domains_match_expand() {
        assert_eq!(
            package_bundle_domains("node-package-install"),
            &[
                "registry.npmjs.org",
                "registry.yarnpkg.com",
                "yarnpkg.com",
                "nodejs.org",
                "unpkg.com",
                "cdn.jsdelivr.net",
            ]
        );
    }
}
