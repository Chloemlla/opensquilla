//! Content classification for unified-diff patches.
//!
//! Mirrors the Python `opensquilla.tools.patch_classification` module. Used
//! by endgame policies that must distinguish diagnostic instrumentation
//! (added print/log statements) from substantive changes. Classification is
//! deliberately conservative — anything it cannot positively identify as
//! instrumentation counts as a substantive change, so misreads fail toward
//! keeping protections active.

use regex::Regex;

/// Matches one line of diagnostic output in the common ecosystems: stdout/stderr
/// print calls, logging-framework calls, and debugger statements. Matched
/// against added lines with leading whitespace stripped. Multi-line calls only
/// match on their first line; continuation lines fail the match and classify
/// the patch as substantive — the conservative direction.
fn instrumentation_line_re() -> &'static Regex {
    static RE: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
        Regex::new(
            r#"^(?:print\s*\(|pprint\s*\(|sys\.(?:stdout|stderr)\.write\s*\(|traceback\.print[a-z_]*\s*\(|(?:logging|logger|log)\.(?:debug|info|warn|warning|error|exception|critical|trace|log)\s*\(|console\.(?:log|error|warn|info|debug|trace|dir)\s*\(|process\.(?:stdout|stderr)\.write\s*\(|fmt\.[A-Za-z]*[Pp]rint[A-Za-z]*\s*\(|log\.(?:Print|Println|Printf)\s*\(|puts\s+["']|println!\s*\(|eprintln!\s*\(|print!\s*\(|eprint!\s*\(|dbg!\s*\(|System\.(?:out|err)\.print[A-Za-z]*\s*\(|[A-Za-z_][A-Za-z0-9_]*\.printStackTrace\s*\(|(?:std::)?(?:cout|cerr)\s*<<|printf\s*\(|fprintf\s*\(|puts\s*\(|perror\s*\(|var_dump\s*\(|print_r\s*\(|error_log\s*\(|\$stderr\.puts\b|\$stdout\.puts\b|debugger\b)"#,
        )
        .expect("valid instrumentation regex")
    });
    &RE
}

/// Split a unified diff into `(added, removed)` content lines.
///
/// File headers (`+++`/`---`), hunk headers, and context lines are excluded;
/// the leading `+`/`-` marker is stripped from the returned lines.
pub fn iter_patch_line_changes(patch: &str) -> (Vec<String>, Vec<String>) {
    let mut added: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();
    for line in patch.lines() {
        if line.starts_with("+++") || line.starts_with("---") {
            continue;
        }
        if line.starts_with('+') {
            added.push(line[1..].to_string());
        } else if line.starts_with('-') {
            removed.push(line[1..].to_string());
        }
    }
    (added, removed)
}

/// Whether a unified diff only adds diagnostic print/log lines.
///
/// True requires: at least one added non-blank line, every added non-blank
/// line matching the instrumentation patterns, and no removed lines at all —
/// any deletion means existing behavior changed, which is never
/// instrumentation-only.
pub fn is_instrumentation_only_patch(patch: &str) -> bool {
    if patch.is_empty() || patch.trim().is_empty() {
        return false;
    }
    let (added, removed) = iter_patch_line_changes(patch);
    if !removed.is_empty() {
        return false;
    }
    let content_lines: Vec<&str> = added
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect();
    if content_lines.is_empty() {
        return false;
    }
    let re = instrumentation_line_re();
    content_lines
        .iter()
        .all(|line| re.is_match(line))
}

/// Whether a single added content line looks like instrumentation.
pub fn is_instrumentation_line(line: &str) -> bool {
    instrumentation_line_re().is_match(line.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_print_patch_is_instrumentation() {
        let patch = "\
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,3 @@
 def run():
     value = compute()
+    print(f\"value={value}\")
+    logging.info(\"done\")
";
        assert!(is_instrumentation_only_patch(patch));
    }

    #[test]
    fn removal_means_substantive() {
        let patch = "\
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,2 @@
-    return old_value
+    return new_value
+    print(\"changed\")
";
        assert!(!is_instrumentation_only_patch(patch));
    }

    #[test]
    fn substantive_additions_are_not_instrumentation() {
        let patch = "\
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,3 @@
 def run():
+    result = expensive_compute()
     print(\"done\")
";
        assert!(!is_instrumentation_only_patch(patch));
    }

    #[test]
    fn empty_and_blank_patches_fail() {
        assert!(!is_instrumentation_only_patch(""));
        assert!(!is_instrumentation_only_patch("   \n  "));
    }

    #[test]
    fn iter_patch_line_changes_splits_and_strips_markers() {
        let patch = "\
--- a/x
+++ b/x
@@ -1 +1 @@
 context
+added line
-removed line
 unchanged
";
        let (added, removed) = iter_patch_line_changes(patch);
        assert_eq!(added, vec!["added line"]);
        assert_eq!(removed, vec!["removed line"]);
    }

    #[test]
    fn rust_ecosystem_instrumentation() {
        let patch = "\
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1 +1,2 @@
 pub fn f() {}
+    dbg!(x);
+    eprintln!(\"debug: {}\", x);
";
        assert!(is_instrumentation_only_patch(patch));
    }

    #[test]
    fn continuation_lines_are_conservative() {
        // A multi-line print whose continuation line is substantive fails the
        // classification (conservative direction).
        let patch = "\
--- a/x
+++ b/x
@@ -1 +1,3 @@
 foo
+    print(
+        compute_expensive_thing())
";
        assert!(!is_instrumentation_only_patch(patch));
    }
}
