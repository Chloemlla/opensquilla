//! Bundled skill loader.
//!
//! Bundled skills are the built-in skills that ship with OpenSquilla. Rather
//! than reading `SKILL.md` files off disk at runtime, they are defined
//! programmatically here as Rust data structures and compiled into the binary.
//!
//! [`BUNDLED_SKILLS`] is a static array of [`BundledSkillDef`] records — one
//! per built-in skill — that are converted to full [`SkillSpec`] values by
//! [`load_bundled_skills`].

use std::collections::HashMap;

use crate::types::{SkillKind, SkillLayer, SkillRequires, SkillSpec};

/// A compile-time definition of a bundled skill.
///
/// The struct is deliberately lightweight (all fields `&'static`) so it can
/// live in a `static` array. [`BundledSkillDef::to_spec`] expands it into a
/// full [`SkillSpec`] at load time.
#[derive(Debug, Clone)]
pub struct BundledSkillDef {
    /// Unique skill identifier.
    pub id: &'static str,
    /// Human-readable name.
    pub name: &'static str,
    /// One-line description.
    pub description: &'static str,
    /// Semantic version string.
    pub version: &'static str,
    /// Author of the skill.
    pub author: &'static str,
    /// The skill kind (always [`SkillKind::Skill`] for bundled skills).
    pub kind: SkillKind,
    /// Required operating systems (`"any"`, `"linux"`, `"macos"`, `"windows"`).
    pub requires_os: &'static [&'static str],
    /// Required binaries, checked via `which`/`where`.
    pub requires_bins: &'static [&'static str],
    /// Categorization tags.
    pub tags: &'static [&'static str],
    /// The instruction body of the skill, embedded at compile time.
    pub body: &'static str,
}

impl BundledSkillDef {
    /// Expand this static definition into a full [`SkillSpec`].
    pub fn to_spec(&self) -> SkillSpec {
        let requires = SkillRequires {
            os: (!self.requires_os.is_empty()).then(|| {
                self.requires_os.iter().map(|s| s.to_string()).collect()
            }),
            binaries: (!self.requires_bins.is_empty()).then(|| {
                self.requires_bins.iter().map(|s| s.to_string()).collect()
            }),
            env_vars: None,
            capabilities: None,
            min_version: None,
            ..SkillRequires::default()
        };

        SkillSpec {
            id: self.id.to_string(),
            name: self.name.to_string(),
            kind: self.kind.clone(),
            description: self.description.to_string(),
            version: Some(self.version.to_string()),
            author: Some(self.author.to_string()),
            layer: SkillLayer::Bundled,
            requires,
            tags: self.tags.iter().map(|s| s.to_string()).collect(),
            steps: Vec::new(),
            outputs: HashMap::new(),
            raw_frontmatter: String::new(),
            source_path: Some(format!("bundled:{}", self.id)),
            body: self.body.to_string(),
            visibility: crate::types::SkillVisibility::Shared,
            scope: crate::types::SkillScope::Global,
            license: None,
            homepage: None,
            metadata: None,
            allowed_tools: Vec::new(),
            disable_model_invocation: false,
            contexts: Vec::new(),
            args: Vec::new(),
            dependencies: Vec::new(),
            disabled: false,
            loaded_at: None,
            mtime_ns: None,
        }
    }
}

/// The complete catalog of bundled skills.
///
/// Each entry is defined entirely in code — no `SKILL.md` files are read at
/// runtime. Order matters only for documentation; the loader returns skills in
/// this order.
pub static BUNDLED_SKILLS: &[BundledSkillDef] = &[
    BundledSkillDef {
        id: "code-review",
        name: "Code Review",
        description: "Systematically review code changes for correctness, security, style, and maintainability, then report findings by severity.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["review", "code", "quality", "security"],
        body: r#"# Code Review

Use when the user asks to review code, a diff, a PR, or "check my code".

## Procedure

1. **Scope the review.** Identify the exact changes: files, hunks, or the full
   tree. If the user provided a diff or PR, review only the changed lines unless
   asked for a broader look.
2. **Passes.** Run these in order:
   - Correctness: logic errors, off-by-one, race conditions, panic/exception paths.
   - Security: injection, unsafe deserialization, secret leakage, authz gaps,
     path traversal, SSRF. Flag anything in the "HIGH" bucket.
   - Performance: accidental O(n^2), unbounded memory, redundant I/O.
   - Style & maintainability: naming, duplication, dead code, clarity.
3. **Report by severity** using this ordering:
   - `HIGH` — bug or security issue that will misbehave in production.
   - `MEDIUM` — likely bug or significant readability/maintainability risk.
   - `LOW` — nit, style, or optional improvement.
4. For every finding, cite the exact file and line/region, explain the impact,
   and suggest a concrete fix.
5. If you found no HIGH/MEDIUM issues, say so plainly. Do not invent problems.

## Constraints

- Do not edit files during a review unless the user explicitly asks for fixes.
- Prefer `git diff` or the provided diff as the source of truth over re-reading
  whole files.
- Summarize: total lines reviewed, N HIGH / N MEDIUM / N LOW findings, and an
  overall readiness verdict (ship / needs work / do not merge).
"#,
    },
    BundledSkillDef {
        id: "debug",
        name: "Debugging",
        description: "Diagnose failing code systematically: reproduce, isolate, form a hypothesis, verify, and fix without guessing.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["debug", "troubleshooting", "diagnosis"],
        body: r#"# Debugging

Use when the user reports a bug, error, crash, or unexpected behavior.

## Procedure

1. **Reproduce.** Get the smallest reliable reproduction. Ask for the exact
   command, input, environment (OS, versions, feature flags), and the full
   error text — not a paraphrase.
2. **Read the error first.** Parse stack traces / logs top-down. Note the first
   frame that is in the user's own code, not library code.
3. **Isolate.** Bisect the surface: recent changes (`git diff`/`git log`), the
   failing input, and the failing subsystem. Reduce the input until the bug
   disappears or becomes minimal.
4. **Hypothesize.** State one falsifiable hypothesis before changing anything,
   e.g. "the tokenizer is receiving a zero-length string". Pick the cheapest
   experiment that can disprove it (add a trace, inspect a value, run a query).
5. **Verify.** After a fix, run the reproduction case plus a regression test.
   Show the before/after evidence.
6. **Root cause, not symptom.** If the fix is a workaround, say so and name the
   underlying root cause.

## Constraints

- Never apply speculative "drive-by" fixes while debugging; change one thing at
  a time and re-run.
- If the environment (sandbox, permissions, network) prevents reproduction,
  produce a diagnostic plan instead and list exactly what the user should run.
"#,
    },
    BundledSkillDef {
        id: "explain",
        name: "Code Explanation",
        description: "Explain unfamiliar code clearly: purpose, architecture, data flow, and key decisions, tuned to the reader's level.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["explain", "education", "documentation"],
        body: r#"# Code Explanation

Use when the user asks "what does this do", "explain this code", or "walk me
through this".

## Procedure

1. **Determine the reader's level.** Ask or infer: newcomer, working engineer,
   or expert. Adjust depth and jargon accordingly.
2. **Start with the big picture.** One or two sentences on what the module
   does and where it sits in the larger system.
3. **Explain the data flow.** Trace the main entry point to the main output:
   inputs, transformations, side effects, and outputs. Use a short ASCII
   sequence when it helps.
4. **Highlight key decisions.** Call out non-obvious choices (why a lock, why a
   cache, why recursion) and trade-offs. Reference the surrounding files/functions
   that make it click.
5. **Define terms.** Any domain or framework concept the reader may not know
   gets a one-line definition on first use.
6. **End with a summary** of what to change or extend if the user wants to
   modify the code.

## Constraints

- Do not dump line-by-line commentary unless the user explicitly asks for it.
- Keep explanations focused on intent and behavior, not a full restatement of
  the syntax.
"#,
    },
    BundledSkillDef {
        id: "refactor",
        name: "Refactoring",
        description: "Restructure code for clarity and maintainability while preserving behavior, with safe, incremental steps.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["refactor", "clean-code", "maintainability"],
        body: r#"# Refactoring

Use when the user asks to clean up, improve, restructure, or simplify code.

## Procedure

1. **Define the goal.** Ask what quality to improve: naming, duplication,
   coupling, dead code, or structure. "Make it better" is not a spec.
2. **Establish a safety net.** Check for existing tests; if there are none and
   the change is non-trivial, suggest adding a characterization test first.
3. **Refactor in small, behavior-preserving steps.** Prefer many tiny commits
   to one large rewrite. Each step should compile and pass tests.
4. **Apply the refactorings in order of value:**
   - Extract functions/structs where responsibilities have drifted.
   - Remove dead code and unused parameters.
   - Rename identifiers to match intent.
   - Collapse duplication while avoiding premature abstraction.
   - Flatten excessive nesting / early-return.
5. **Do not mix behavior changes into a refactor.** If you discover a bug,
   note it separately and fix it in its own step.

## Constraints

- Keep public API and external contract unchanged unless the user opts in.
- After each step, re-run the relevant tests/build and report the result.
"#,
    },
    BundledSkillDef {
        id: "test",
        name: "Test Generation",
        description: "Write targeted unit and integration tests covering the happy path, edge cases, and error paths.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["test", "testing", "qa"],
        body: r#"# Test Generation

Use when the user asks to write tests, increase coverage, or verify behavior.

## Procedure

1. **Identify the contract.** Read the function/module signature and its
   documented behavior. List: inputs, outputs, invariants, and error conditions.
2. **Design test cases:**
   - Happy path: the primary use case.
   - Boundaries: empty input, maximum/minimum values, off-by-one thresholds.
   - Edge cases: nulls, negative values, overflow, Unicode, large inputs.
   - Error paths: each documented failure mode and what the caller receives.
3. **Match the project's test style.** Use the existing framework and naming
   convention (e.g. `test_*` or `#[test]`). Prefer behavior assertions over
   implementation details.
4. **Write focused tests.** One logical scenario per test, with a clear arrange
   / act / assert structure.
5. **Run the tests** and iterate until green. Report the pass/fail summary and
   any uncovered branches worth noting.

## Constraints

- Do not test private implementation details unless the team convention does.
- Avoid flaky dependencies (network, wall-clock time) by injecting fakes or
  using deterministic inputs.
"#,
    },
    BundledSkillDef {
        id: "search",
        name: "Web Search",
        description: "Find authoritative, up-to-date information on the web and synthesize answers with cited sources.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["search", "web", "research"],
        body: r#"# Web Search

Use when the user asks a question that requires current or external
information, or when you do not have reliable knowledge of a topic.

## Procedure

1. **Formulate queries.** Break the question into 2-3 concrete search queries.
   Use site-restricted queries (`site:docs.rs`, `site:developer.mozilla.org`)
   when the target is known.
2. **Prefer primary sources.** Official docs, vendor pages, standards bodies,
   and the original repository over blog summaries and forums.
3. **Evaluate freshness and authority.** Note publication dates and check
   multiple independent sources for claims that look surprising.
4. **Synthesize, don't dump.** Answer the user's question directly, then support
   it with the key facts and their sources.
5. **Cite sources.** For each material claim, list the URL and, where useful,
   the title. If the sources conflict, say so.

## Constraints

- Do not fabricate URLs or quote search snippets as if you read the page.
- If a search returns nothing relevant after a couple of attempts, tell the
  user rather than padding the answer with near-misses.
"#,
    },
    BundledSkillDef {
        id: "shell",
        name: "Shell Command",
        description: "Construct and explain safe shell commands for common system, file, and DevOps tasks across platforms.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["shell", "cli", "system"],
        body: r#"# Shell Command

Use when the user asks for a shell command to accomplish a system, file, or
DevOps task, or asks to explain/repair a command.

## Procedure

1. **Clarify the target platform** (Windows PowerShell/cmd, Linux/macOS sh/bash)
   and the exact goal. Command syntax differs materially between them.
2. **Prefer the simplest built-in** that does the job. Avoid pipelines that
   depend on tools that may not be installed.
3. **Quote and escape correctly.** Quote paths with spaces; use `--` before
   filenames that start with `-`; prefer array/argv invocation over string
   concatenation where possible.
4. **Guard destructive commands.** For `rm`, `del`, `mv`, `truncate`, and any
   overwrite, show exactly what will be affected and suggest a dry-run or a
   backup first.
5. **Explain the command** in one sentence per part, so the user can verify it
   before running.

## Constraints

- Never suggest running `curl | sh` or piping remote content into a privileged
  shell without warning about the supply-chain risk.
- When a task is inherently dangerous (privilege escalation, mass delete,
  firewall changes), stop and confirm with the user before providing the command.
"#,
    },
    BundledSkillDef {
        id: "git",
        name: "Git Operations",
        description: "Diagnose and repair Git state, and construct safe multi-step Git workflows without rewriting shared history.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &["git"],
        tags: &["git", "version-control", "vcs"],
        body: r#"# Git Operations

Use when the user asks for Git help: status, history, branches, stashes,
rebasing, merge conflicts, or undoing mistakes.

## Procedure

1. **Inspect before acting.** Run `git status`, `git log --oneline -5`, and
   `git branch --show-current` first. Never guess repository state.
2. **Prefer non-destructive operations.** Use `git restore` over `git checkout
   --`, `git switch` over `git checkout`, and `--dry-run` where available.
3. **Rewriting shared history.** Do not rebase, reset, or force-push branches
   that others may have based work on, unless the user explicitly confirms.
4. **Merge conflicts.** Show both sides, explain the intent, and propose a
   resolution that preserves both changes where possible.
5. **Explain each command** with its effect so the user can verify before
   running.

## Constraints

- Never run `git push --force` (or `--force-with-lease` without checking) on a
  shared branch without explicit confirmation.
- If a command would discard uncommitted work (`git reset --hard`, `git clean`),
  warn about exactly which files are affected and suggest a backup first.
"#,
    },
    BundledSkillDef {
        id: "memory",
        name: "Memory Management",
        description: "Read, search, and update OpenSquilla's durable memory files (USER.md, MEMORY.md, memory/**/*.md) safely.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["memory", "recall", "notes"],
        body: r#"# Memory Management

Use when the user asks to remember, recall, forget, update, search, or inspect
durable OpenSquilla memory.

## Source Files

- `USER.md`: stable user profile facts (name, preferences, timezone).
  Edit it with filesystem tools, not memory_save.
- `MEMORY.md`: curated long-term facts, decisions, and constraints.
- `memory/YYYY-MM-DD.md` and `memory/**/*.md`: daily/session notes.
- `turns/**/*.md`: private auto-captured turn state — never indexed by
  memory_search and not for ordinary recall.

## Recall

- Prefer injected `USER.md` for current identity/profile questions.
- Use `memory_search` for historical or non-profile recall.
- Use `memory_get` after a search when exact lines or more context are needed.

## Remember / Update

- For profile facts, edit `USER.md` with visible filesystem tools.
- For daily/session notes, write `memory/YYYY-MM-DD.md` or another
  `memory/**/*.md` source.
- For curated facts in `MEMORY.md`, read the current file first and write the
  full updated content with `mode='replace'` — never append.

## Forget / Correct

- Search first, then read the relevant lines before removing anything.
- Remove or correct a single fact by editing the source file directly.
- If no write/delete tool is available, report the exact path and lines that
  should change instead of claiming the memory was updated.

## Boundaries

- Do not store deliverables (reports, JSON, results) in memory source files.
- Do not save secrets, tokens, private keys, or full credential contents.
- Only confirm a memory change after the write or delete actually succeeds.
"#,
    },
    BundledSkillDef {
        id: "planner",
        name: "Planning",
        description: "Break ambiguous goals into an ordered, verifiable execution plan with owners, dependencies, and checkpoints.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["planning", "project-management", "tasks"],
        body: r#"# Planning

Use when the user asks to plan a project, break down a large goal, or sequence
a multi-step task.

## Procedure

1. **Define the outcome.** Write one measurable success criterion. If the goal
   is vague, ask one or two clarifying questions rather than guessing.
2. **Decompose into steps.** List concrete steps that each produce a
   verifiable artifact or state. Prefer small steps that can be checked off.
3. **Order and link dependencies.** Sequence steps so each depends only on
   already-satisfied prerequisites. Note steps that can run in parallel.
4. **Assign owners and effort.** For each step, note who/what executes it and a
   rough effort estimate (S/M/L).
5. **Add checkpoints.** Identify decision points or reviews where progress is
   confirmed and the plan may be revised.
6. **Define done.** For each step, state how you will verify completion.

## Constraints

- Keep the plan realistic; avoid infinite milestone chains. If a step is
  genuinely research-like, mark it as a spike with a time-box.
- When the user only wants a quick sequence, keep it to 3-5 steps and don't
  over-engineer the format.
"#,
    },
];

/// Load all bundled skills as full [`SkillSpec`] values.
///
/// This is the entry point used by the [`crate::loader::SkillLoader`] to
/// register the built-in catalog. It is cheap: it merely expands the static
/// [`BUNDLED_SKILLS`] array.
pub fn load_bundled_skills() -> Vec<SkillSpec> {
    BUNDLED_SKILLS.iter().map(BundledSkillDef::to_spec).collect()
}

/// Look up a single bundled skill by id.
pub fn get_bundled_skill(id: &str) -> Option<SkillSpec> {
    BUNDLED_SKILLS
        .iter()
        .find(|def| def.id == id)
        .map(BundledSkillDef::to_spec)
}

/// The number of bundled skills defined.
pub fn bundled_skill_count() -> usize {
    BUNDLED_SKILLS.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_all_bundled_skills() {
        let specs = load_bundled_skills();
        assert!(specs.len() >= 10, "expected at least 10 bundled skills");
    }

    #[test]
    fn bundled_skills_have_unique_ids() {
        let specs = load_bundled_skills();
        let mut ids: Vec<&str> = specs.iter().map(|s| s.id.as_str()).collect();
        ids.sort_unstable();
        let original_len = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "bundled skill ids must be unique");
    }

    #[test]
    fn bundled_skills_are_on_bundled_layer() {
        let specs = load_bundled_skills();
        assert!(specs.iter().all(|s| s.layer == SkillLayer::Bundled));
    }

    #[test]
    fn lookup_by_id() {
        let spec = get_bundled_skill("git").expect("git skill should exist");
        assert_eq!(spec.name, "Git Operations");
        assert_eq!(
            spec.requires.binaries,
            Some(vec!["git".to_string()]),
            "git skill must require the git binary"
        );
        assert!(get_bundled_skill("does-not-exist").is_none());
    }

    #[test]
    fn every_skill_has_required_fields() {
        for spec in load_bundled_skills() {
            assert!(!spec.id.is_empty(), "skill id must not be empty");
            assert!(!spec.name.is_empty(), "skill name must not be empty");
            assert!(!spec.description.is_empty(), "skill description must not be empty");
            assert!(
                spec.version.as_deref().is_some_and(|v| !v.is_empty()),
                "skill {} must have a version",
                spec.id
            );
        }
    }
}
