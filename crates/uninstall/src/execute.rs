use crate::plan::{UninstallAction, UninstallActionKind, UninstallPlan};

/// Perform a dry run, returning the actions that would be executed without
/// touching the filesystem.
pub fn dry_run(plan: &UninstallPlan) -> Vec<&UninstallAction> {
    plan.actions.iter().collect()
}

/// Execute the plan, deleting each listed item.
///
/// Missing paths are skipped silently (idempotent), while failures to remove
/// an existing path abort with an error.
pub fn execute(plan: &UninstallPlan) -> crate::Result<()> {
    for action in &plan.actions {
        match action.kind {
            UninstallActionKind::Keep => {}
            UninstallActionKind::DeleteFile => {
                if action.path.exists() {
                    std::fs::remove_file(&action.path).map_err(|e| {
                        crate::Error::Execution(format!(
                            "Failed to remove {}: {e}",
                            action.path.display()
                        ))
                    })?;
                }
            }
            UninstallActionKind::DeleteDirectory => {
                if action.path.exists() {
                    std::fs::remove_dir_all(&action.path).map_err(|e| {
                        crate::Error::Execution(format!(
                            "Failed to remove {}: {e}",
                            action.path.display()
                        ))
                    })?;
                }
            }
        }
    }
    Ok(())
}
