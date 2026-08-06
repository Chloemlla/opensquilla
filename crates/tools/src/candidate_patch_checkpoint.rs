//! Content snapshots for reversible candidate-patch trial edits.
//!
//! Mirrors the Python `opensquilla.tools.candidate_patch_checkpoint` module:
//! captures the current dirty-file state of a workspace before trying a
//! candidate patch, and restores it later. The checkpoint stores file content
//! directly and only uses git for read-only status/blob queries — it never
//! runs destructive git commands.

use serde_json::{Value, json};
use sha2::Digest;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A workspace-relative file snapshot.
#[derive(Debug, Clone)]
pub struct CandidatePatchFileSnapshot {
    pub relative_path: String,
    pub exists: bool,
    pub content: Option<Vec<u8>>,
    pub sha256: Option<String>,
}

/// Snapshot of the dirty workspace state before trying a candidate patch.
#[derive(Debug, Clone)]
pub struct CandidatePatchCheckpoint {
    pub workspace: PathBuf,
    pub label: Option<String>,
    pub created_at: f64,
    pub head: Option<String>,
    pub files: BTreeMap<String, CandidatePatchFileSnapshot>,
}

impl CandidatePatchCheckpoint {
    /// Sorted changed paths captured in this checkpoint.
    pub fn changed_paths(&self) -> Vec<String> {
        self.files.keys().cloned().collect()
    }
}

/// Capture current dirty files so a later candidate can be reverted.
pub fn create_candidate_patch_checkpoint(
    workspace: &Path,
    label: Option<&str>,
) -> CandidatePatchCheckpoint {
    let root = workspace.to_path_buf();
    let paths = git_dirty_paths(&root);
    CandidatePatchCheckpoint {
        workspace: root.clone(),
        label: label.map(|s| s.to_string()),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        head: git_head(&root),
        files: paths
            .iter()
            .map(|path| (path.clone(), snapshot_path(&root, path)))
            .collect(),
    }
}

/// Restore the workspace to the checkpoint's dirty-file state.
pub fn restore_candidate_patch_checkpoint(checkpoint: &CandidatePatchCheckpoint) -> Value {
    let root = checkpoint.workspace.clone();
    let current_paths: std::collections::BTreeSet<String> =
        git_dirty_paths(&root).into_iter().collect();
    let checkpoint_paths: std::collections::BTreeSet<String> =
        checkpoint.files.keys().cloned().collect();
    let touched_paths: Vec<String> = current_paths
        .union(&checkpoint_paths)
        .cloned()
        .collect();
    let mut restored: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();

    for relative_path in touched_paths {
        if let Some(snapshot) = checkpoint.files.get(&relative_path) {
            let target = root.join(&relative_path);
            if snapshot.exists {
                if let Some(content) = &snapshot.content {
                    if let Some(parent) = target.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let _ = std::fs::write(&target, content);
                    restored.push(relative_path);
                }
            } else {
                remove_file_if_present(&target);
                removed.push(relative_path);
            }
            continue;
        }

        let head_content = git_show_head_path(&root, &relative_path);
        let target = root.join(&relative_path);
        match head_content {
            Some(content) => {
                if let Some(parent) = target.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&target, content);
                restored.push(relative_path);
            }
            None => {
                remove_file_if_present(&target);
                removed.push(relative_path);
            }
        }
    }

    json!({
        "status": "restored",
        "label": checkpoint.label,
        "path_count": restored.len() + removed.len(),
        "restored_paths": restored,
        "removed_paths": removed,
    })
}

fn snapshot_path(root: &Path, relative_path: &str) -> CandidatePatchFileSnapshot {
    let target = root.join(relative_path);
    if !target.exists() || !target.is_file() {
        return CandidatePatchFileSnapshot {
            relative_path: relative_path.to_string(),
            exists: false,
            content: None,
            sha256: None,
        };
    }
    match std::fs::read(&target) {
        Ok(content) => {
            let mut hasher = sha2::Sha256::new();
            hasher.update(&content);
            CandidatePatchFileSnapshot {
                relative_path: relative_path.to_string(),
                exists: true,
                content: Some(content),
                sha256: Some(format!("{:x}", hasher.finalize())),
            }
        }
        Err(_) => CandidatePatchFileSnapshot {
            relative_path: relative_path.to_string(),
            exists: false,
            content: None,
            sha256: None,
        },
    }
}

fn git_dirty_paths(root: &Path) -> Vec<String> {
    let Ok(output) = std::process::Command::new("git")
        .args(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        .current_dir(root)
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    parse_git_status_z(&String::from_utf8_lossy(&output.stdout))
}

fn parse_git_status_z(output: &str) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    let entries: Vec<&str> = output.split('\0').collect();
    let mut index = 0;
    while index < entries.len() {
        let entry = entries[index];
        index += 1;
        if entry.is_empty() {
            continue;
        }
        let status = &entry[..entry.len().min(2)];
        let mut relative_path = if entry.len() > 3 { &entry[3..] } else { "" };
        if status.as_bytes().first().is_some_and(|b| *b == b'R' || *b == b'C')
            && index < entries.len()
        {
            relative_path = entries[index];
            index += 1;
        }
        if !relative_path.is_empty() {
            paths.push(relative_path.replace('\\', "/"));
        }
    }
    paths.sort();
    paths.dedup();
    paths
}

fn git_head(root: &Path) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let head = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if head.is_empty() {
        None
    } else {
        Some(head)
    }
}

fn git_show_head_path(root: &Path, relative_path: &str) -> Option<Vec<u8>> {
    let output = std::process::Command::new("git")
        .args(["show", "--end-of-options", &format!("HEAD:{relative_path}")])
        .current_dir(root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(output.stdout)
}

fn remove_file_if_present(path: &Path) {
    if path.exists() && path.is_file() {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_git_status_z_renames() {
        let output = " M src/a.rs\0?? new.py\0R  old.rs\0new.rs\0";
        let paths = parse_git_status_z(output);
        assert_eq!(paths, vec!["new.py", "new.rs", "src/a.rs"]);
    }

    #[test]
    fn checkpoint_restore_in_non_git_dir_is_graceful() {
        let temp = tempfile::tempdir().expect("tempdir");
        let checkpoint = create_candidate_patch_checkpoint(temp.path(), Some("trial-1"));
        // Not a git repo: no dirty paths, head is None.
        assert!(checkpoint.files.is_empty());
        assert!(checkpoint.head.is_none());
        assert_eq!(checkpoint.label.as_deref(), Some("trial-1"));

        let result = restore_candidate_patch_checkpoint(&checkpoint);
        assert_eq!(result["status"], "restored");
    }

    #[test]
    fn snapshot_path_captures_existing_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let file = temp.path().join("a.txt");
        std::fs::write(&file, b"hello").expect("write");
        let snapshot = snapshot_path(temp.path(), "a.txt");
        assert!(snapshot.exists);
        assert_eq!(snapshot.content.as_deref(), Some(&b"hello"[..]));
        assert_eq!(snapshot.sha256.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn snapshot_path_missing_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let snapshot = snapshot_path(temp.path(), "missing.txt");
        assert!(!snapshot.exists);
        assert!(snapshot.content.is_none());
    }
}
