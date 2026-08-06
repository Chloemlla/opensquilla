//! Shared path-intent checks for tool-facing filesystem inputs.
//!
//! Mirrors the Python `opensquilla.tools.path_policy` module: detects absolute
//! paths that belong to a different host OS (a Windows drive path supplied on
//! a POSIX host, or a POSIX absolute path supplied on Windows). Rejecting
//! foreign-host paths early prevents tools from escaping the workspace via a
//! path spelling the active OS would resolve differently than intended.

use crate::registry::ToolError;

/// Windows drive-letter prefixes, e.g. `C:\...` or `C:/...`.
fn is_windows_drive_path(text: &str) -> bool {
    let bytes = text.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

/// Roots that identify an absolute POSIX path as belonging to a typical
/// macOS/Linux host rather than the active Windows workspace.
const FOREIGN_POSIX_ROOTS: &[&str] = &[
    "Applications",
    "Library",
    "System",
    "Users",
    "Volumes",
    "bin",
    "etc",
    "home",
    "cygdrive",
    "mnt",
    "opt",
    "private",
    "tmp",
    "usr",
    "var",
];

/// Return `true` when `path` is an absolute path for a different host OS.
///
/// `platform` is the *active* platform: `"nt"` (Windows) or anything else
/// (POSIX). On Windows, a `"/Users/..."`-style absolute path is foreign; on
/// POSIX hosts, a `"C:\\..."` / `"C:/..."` drive path is foreign.
pub fn is_foreign_host_path(path: &str, platform: &str) -> bool {
    let text = path.trim();
    if text.is_empty() {
        return false;
    }
    if text.to_ascii_lowercase().starts_with("file://") {
        return true;
    }

    if platform.eq_ignore_ascii_case("nt") {
        let normalized = text.replace('\\', "/");
        if !normalized.starts_with('/') || normalized.starts_with("//") {
            return false;
        }
        let parts: Vec<&str> = normalized.split('/').collect();
        if parts.len() < 2 {
            return false;
        }
        let second = parts[1];
        return FOREIGN_POSIX_ROOTS.contains(&second)
            || (second.len() == 1 && second.chars().next().is_some_and(|c| c.is_ascii_alphabetic()));
    }

    is_windows_drive_path(text)
}

/// Extract a display filename from a path that may use either separator.
fn path_filename(text: &str) -> String {
    let normalized = text.replace('\\', "/");
    normalized
        .rsplit('/')
        .find(|part| !part.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Build the `ToolError` for a rejected foreign-host path.
pub fn foreign_host_path_error(path: &str, workspace: Option<&str>) -> ToolError {
    let filename = path_filename(path);
    let name = if filename.is_empty() {
        "<filename>"
    } else {
        filename.as_str()
    };
    let mut details = vec![
        "foreign_host_path: requested path is from another host/platform".to_string(),
        format!("requested path: {path}"),
    ];
    if let Some(workspace) = workspace {
        details.push(format!("active workspace: {workspace}"));
    }
    details.push(format!(
        "Use a workspace-relative path after creating the file inside the active workspace \
         (for example: {name})."
    ));
    ToolError::new("FOREIGN_HOST_PATH", details.join(". "))
}

/// Raise a `ToolError` when `path` is an absolute path for another host OS.
pub fn reject_foreign_host_path(
    path: &str,
    platform: &str,
    workspace: Option<&str>,
) -> Result<(), ToolError> {
    if is_foreign_host_path(path, platform) {
        return Err(foreign_host_path_error(path, workspace));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_drive_path_is_foreign_on_posix() {
        assert!(is_foreign_host_path("C:\\Users\\me\\file.txt", "posix"));
        assert!(is_foreign_host_path("C:/Users/me/file.txt", "linux"));
        assert!(is_foreign_host_path("d:/x", "posix"));
        assert!(!is_foreign_host_path("relative/path.txt", "posix"));
        assert!(!is_foreign_host_path("/home/me/file.txt", "posix"));
        // Drive letter without a slash separator is not a foreign absolute path.
        assert!(!is_foreign_host_path("C:relative", "posix"));
    }

    #[test]
    fn posix_root_is_foreign_on_windows() {
        assert!(is_foreign_host_path("/Users/me/file.txt", "nt"));
        assert!(is_foreign_host_path("/home/me/file.txt", "nt"));
        assert!(is_foreign_host_path("/etc/hosts", "nt"));
        assert!(is_foreign_host_path("\\\\server\\share", "nt") == false);
        assert!(!is_foreign_host_path("C:/Users/me/file.txt", "nt"));
        assert!(!is_foreign_host_path("relative/path.txt", "nt"));
        assert!(!is_foreign_host_path("/workspace/file.txt", "nt"));
    }

    #[test]
    fn file_uri_is_always_foreign() {
        assert!(is_foreign_host_path("file:///etc/hosts", "nt"));
        assert!(is_foreign_host_path("file://C:/x", "posix"));
    }

    #[test]
    fn empty_path_is_not_foreign() {
        assert!(!is_foreign_host_path("", "nt"));
        assert!(!is_foreign_host_path("  ", "posix"));
    }

    #[test]
    fn error_message_names_path_and_workspace() {
        let error = foreign_host_path_error("/Users/me/x.py", Some("/workspace"));
        assert!(error.message.contains("another host/platform"));
        assert!(error.message.contains("active workspace: /workspace"));
        assert!(error.message.contains("x.py"));
    }

    #[test]
    fn reject_returns_error_for_foreign_path() {
        let result = reject_foreign_host_path("C:\\x\\y.py", "posix", None);
        assert!(result.is_err());
        let result = reject_foreign_host_path("workspace/ok.py", "posix", None);
        assert!(result.is_ok());
    }
}
