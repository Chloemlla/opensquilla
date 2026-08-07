//! Shared per-entry formatting for resilient directory listings.
//!
//! Port of `src/opensquilla/sandbox/directory_listing.py`. Formats a directory
//! entry as a display line without failing when a child's metadata is
//! unavailable (broken symlink, permission error, races). Callers use the
//! returned `is_directory` flag to drive recursion; `follow_target = false` is
//! the fail-closed form for callers whose path-policy resolution failed.

use std::path::Path;

/// Error numbers indicating a symlink target that cannot be resolved.
/// Covers `ENOENT` (2), `ENOTDIR` (20), `ELOOP` (40) and the Windows
/// `ERROR_CANT_RESOLVE_FILENAME` (1921) code.
fn is_broken_symlink_target_error(err: &std::io::Error) -> bool {
    match err.raw_os_error() {
        Some(code) => matches!(code, 2 | 3 | 20 | 40 | 1921),
        None => false,
    }
}

/// Format a directory entry as `(is_directory, display_line)`.
///
/// `follow_target = false` uses only `lstat` metadata and never follows a
/// link. `symlink_target_is_broken` preserves an already-confirmed symlink
/// loop result without following the target again.
pub fn format_directory_entry(
    path: &Path,
    follow_target: bool,
    symlink_target_is_broken: bool,
) -> (bool, String) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned());

    let metadata = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => return (false, format!("[file] {name} (metadata unavailable)")),
    };
    let file_type = metadata.file_type();

    if file_type.is_symlink() {
        if symlink_target_is_broken {
            return (false, format!("[link] {name} (broken symlink)"));
        }
        if !follow_target {
            return (
                false,
                format!("[link] {name} (target metadata unavailable)"),
            );
        }
        return match std::fs::metadata(path) {
            Ok(target) => (
                false,
                format!("[link] {name} ({} bytes target)", target.len()),
            ),
            Err(e) => {
                if is_broken_symlink_target_error(&e) {
                    (false, format!("[link] {name} (broken symlink)"))
                } else {
                    (
                        false,
                        format!("[link] {name} (target metadata unavailable)"),
                    )
                }
            }
        };
    }

    if file_type.is_dir() {
        return (true, format!("[dir]  {name}/"));
    }

    if !follow_target {
        return (false, format!("[file] {name} ({} bytes)", metadata.len()));
    }
    match std::fs::metadata(path) {
        Ok(m) => (false, format!("[file] {name} ({} bytes)", m.len())),
        Err(_) => (false, format!("[file] {name} (size unavailable)")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_entry_is_fail_closed() {
        let p = Path::new("/definitely/not/here/xyz");
        let (is_dir, line) = format_directory_entry(p, true, false);
        assert!(!is_dir);
        assert!(line.contains("metadata unavailable"));
    }

    #[test]
    fn regular_file_prints_size() {
        let dir = std::env::temp_dir().join("osq_dir_listing_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("a.txt");
        std::fs::write(&file, b"hello world").unwrap();
        let (is_dir, line) = format_directory_entry(&file, true, false);
        assert!(!is_dir);
        assert!(line.contains("[file] a.txt (11 bytes)"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_marks_is_dir() {
        let dir = std::env::temp_dir().join("osq_dir_listing_dir_test");
        std::fs::create_dir_all(&dir).unwrap();
        let (is_dir, line) = format_directory_entry(&dir, true, false);
        assert!(is_dir);
        assert!(line.starts_with("[dir]  "));
        assert!(line.ends_with('/'));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn symlink_handling() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let dir = std::env::temp_dir().join("osq_dir_listing_link_test");
            std::fs::create_dir_all(&dir).unwrap();
            let target = dir.join("target.txt");
            std::fs::write(&target, b"x").unwrap();
            let link = dir.join("link.txt");
            let _ = std::fs::remove_file(&link);
            symlink(&target, &link).unwrap();

            let (is_dir, line) = format_directory_entry(&link, true, false);
            assert!(!is_dir);
            assert!(line.contains("[link] link.txt"));

            // A broken symlink is reported as such when follow_target is on.
            let broken = dir.join("broken.txt");
            let _ = std::fs::remove_file(&broken);
            symlink(dir.join("missing-target"), &broken).unwrap();
            let (is_dir, line) = format_directory_entry(&broken, true, false);
            assert!(!is_dir);
            assert!(line.contains("broken symlink"));

            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
