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
            os: (!self.requires_os.is_empty())
                .then(|| self.requires_os.iter().map(|s| s.to_string()).collect()),
            binaries: (!self.requires_bins.is_empty())
                .then(|| self.requires_bins.iter().map(|s| s.to_string()).collect()),
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
    BundledSkillDef {
        id: "github",
        name: "GitHub Operations",
        description: "Interact with GitHub repositories via the `gh` CLI: issues, PRs, CI runs, code review, and API queries.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &["gh"],
        tags: &["github", "git", "ci", "issues", "pr"],
        body: r#"# GitHub Operations

Use when the user asks to interact with GitHub: list/view/create issues or PRs,
check CI status, view run logs, or query the GitHub API.

## Setup

Verify authentication first: `gh auth status`. If not authenticated, tell the
user to run `gh auth login` before proceeding.

## Procedure

1. **Determine the target repo.** Use `--repo owner/repo` when not in a git
   directory, or pass a GitHub URL directly (`gh pr view https://...`).
2. **Pick the right command for the task:**
   - PRs: `gh pr list`, `gh pr view <N>`, `gh pr checks <N>`, `gh pr create`,
     `gh pr merge <N> --squash`.
   - Issues: `gh issue list`, `gh issue create`, `gh issue close <N>`.
   - CI: `gh run list --limit 10`, `gh run view <id>`,
     `gh run view <id> --log-failed`, `gh run rerun <id> --failed`.
3. **Use JSON output for parsing.** Most commands support `--json` with `--jq`
   filtering: `gh pr list --json number,title,state --jq '.[] | "\(.number):
   \(.title)"'`.
4. **For complex queries**, drop to `gh api` with `--jq`:
   `gh api repos/owner/repo/pulls/55 --jq '.title, .state'`.

## Constraints

- Always specify `--repo owner/repo` when not inside the repo's working tree.
- Rate limits apply; use `gh api --cache 1h` for repeated identical queries.
- Do not create, merge, or close PRs/issues without explicit user confirmation.
"#,
    },
    BundledSkillDef {
        id: "git-diff",
        name: "Git Diff Capture",
        description: "Capture the current git diff (staged, working-tree, or staged file list) as text for review or workflow consumption.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &["git"],
        tags: &["git", "diff", "vcs", "review"],
        body: r#"# Git Diff Capture

Use when the user needs the current git diff as text — for review, for feeding
into another workflow, or to check what changed.

## Procedure

1. **Determine which diff to capture:**
   - Staged changes: `git diff --cached HEAD`.
   - Working-tree changes: `git diff HEAD`.
   - Staged file list only: `git diff --cached --name-only`.
   - Staged with worktree fallback (default): try `git diff --cached HEAD` first;
     if empty, fall back to `git diff HEAD`.
2. **Scope to a path** if the user only cares about part of the tree:
   `git diff --cached HEAD -- src/`.
3. **Report the output** as-is. If there are no changes, say `NO_DIFF` so
   downstream consumers can short-circuit.
4. **For review context**, also show the stat: `git diff --stat`.

## Constraints

- Prefer `--cached` (staged) over working-tree when the user asks "what will be
  committed".
- Do not modify any files — this skill is read-only.
"#,
    },
    BundledSkillDef {
        id: "code-task",
        name: "Code Task",
        description: "Solve a real-repository coding task end to end: clone, run an agent on a task branch, and independently verify with a red-green-regression test loop.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &["git"],
        tags: &["code", "task", "verification", "testing"],
        body: r#"# Code Task

Use when the user asks to fix/add/implement/change code in a real repository
they name by path or URL. Route through the code-task runner rather than
hand-editing files — the runner provides isolation, a task branch, and
runner-verified red-green-regression proof.

## Procedure

1. **Translate the request.** Map the user's natural-language request to:
   ```
   code-task solve --repo <url-or-path> (--issue N | --task "<text>" | --task-file <path>) [--yes]
   ```
   - A GitHub issue → `--issue N` (needs `gh`).
   - A short request → `--task "<their request>"`.
   - A long spec → save to a file and use `--task-file <path>`.
2. **Two pre-flight checks:**
   - Trusted repo: the runner executes the repo's code on the host — only run
     against repositories the user trusts.
   - Enough information: you must be able to state the expected behavior change.
     If the request is too vague to write an acceptance test for, ask the user
     to clarify before running.
3. **Pass `--yes`** to skip the interactive trusted-host confirmation (you are
   acting on the user's behalf), but only after the safety check.
4. **Watch the run dir, not the source repo.** The runner clones into an
   isolated run directory; the source repo stays empty until a run finishes and
   verifies. Let it finish — do not kill or relaunch.
5. **Read the result.** Key fields:
   - `state`: `verified` (red->green, no regressions), `already_satisfied`,
     `not_testable`, `environment_blocked`, `failed`.
   - `acceptance`: each test with `before`->`after`.
   - `regression`: existing-suite result and `new_failures`.
   - `assumptions`: surface these to the user.
   - `retry_exhausted`: the runner retries internally — do NOT relaunch on
     `failed`; the retries are already exhausted.

## Verification modes

- `red-green` (default): agent writes acceptance tests, runner proves red on
  base and green on the change, then runs regression.
- `build`: for building an app from scratch — runner owns a fixed checklist
  (`npm ci` -> `npm run build` -> package). `state=verified` means it builds.
- `scratch`: for self-contained testable code with no repo — runner scaffolds
  an empty git repo, writes code plus tests, verifies green-only.

## Constraints

- Runs on the gateway host — git, the toolchain, and disk all come from there.
- v1 is host-only and always clones fresh (no `--in-place`). For untrusted
  repositories, a Docker-isolated backend is planned but not in v1.
- Do NOT relaunch the same task on a `failed` result — the internal retries are
  already exhausted. Surface the failure to the user.
"#,
    },
    BundledSkillDef {
        id: "sub-agent",
        name: "Sub-Agent Delegation",
        description: "Delegate a self-contained task to a sub-agent (Codex, Claude Code, OpenCode, or Pi) via background process for coding, reviewing, refactoring, or any LLM-driven sub-task.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["sub-agent", "delegation", "coding", "background"],
        body: r#"# Sub-Agent Delegation

Use when you need to delegate a self-contained task to a coding agent (Codex,
Claude Code, OpenCode, or Pi) running in a background process. The wrapped CLIs
are coding-oriented, but this skill is the generic "spawn a sub-agent with full
tool surface" slot used by meta-skill DAGs for any LLM-driven sub-task.

## Procedure

1. **Choose the agent and execution mode.** Respect the user's choice if they
   name one. Prefer non-interactive modes:
   - Codex/Pi/OpenCode: `codex exec "prompt"`, `pi -p "prompt"`,
     `opencode run "prompt"`.
   - Claude Code: `claude --permission-mode bypassPermissions --print "prompt"`.
2. **Set the workdir.** Agent wakes up in a focused directory. For scratch
   work, create a temp git repo first: `mktemp -d && cd $dir && git init`
   (Codex refuses to run outside a git directory).
3. **For long tasks, use background_process:**
   ```
   background_process(workdir="~/project", command="codex exec --full-auto 'Build feature X'")
   process(action="wait", session_id="XXX")   # blocks until done
   process(action="log", session_id="XXX")     # peek at output
   ```
   Prefer `wait` over polling in a loop — a looped `poll` burns a full turn +
   tokens each time.
4. **For PR reviews**, clone to a temp directory — never review PRs inside the
   live runtime state/workspace directories. Use `git worktree` to keep main
   intact.
5. **Parallel work is OK.** Run multiple agents in parallel using git worktrees:
   ```
   git worktree add -b fix/issue-78 /tmp/issue-78 main
   background_process(workdir="/tmp/issue-78", command="codex exec --full-auto 'Fix issue #78'")
   ```

## Progress updates

- Send 1 short message when you start (what's running + where).
- Update again only when something changes: a milestone completes, the agent
  asks a question, you hit an error, or the agent finishes.
- If you kill a session, immediately say you killed it and why.

## Constraints

- Use the right execution mode per agent: non-interactive command modes for
  Codex/Pi/OpenCode; `--print --permission-mode bypassPermissions` for Claude
  Code.
- Do NOT hand-code patches yourself in orchestrator mode — if the agent fails or
  hangs, respawn it or ask the user for direction; don't silently take over.
- Be patient — don't kill sessions because they're "slow". Monitor with
  `process(action="log")` without interfering.
- NEVER start an agent inside the OpenSquilla state directory or live workspace
  directories — use an explicit project worktree.
"#,
    },
    BundledSkillDef {
        id: "deep-research",
        name: "Deep Research",
        description: "Multi-round research with explicit methodology, evidence tracking, and citation-tagged synthesis across many sources.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["research", "investigation", "citations", "report"],
        body: r#"# Deep Research

Use when the user asks for a "research report", "literature review", "deep dive",
or "investigate X across sources" — tasks that need multi-round investigation
with evidence tracking and citation-tagged synthesis. Distinct from `summarize`
(single-pass condensation of one document).

## Procedure

1. **Decide if this is the right tool.**
   - One-line summary of one article → `summarize`.
   - Multi-round investigation with citations → this skill.
   - Quick lookup, single source → direct web search.
2. **Stage 1 — Plan.** Scope the question into sub-questions. Choose a depth:
   - `overview` — 3-5 sub-questions, 1 source each.
   - `thorough` — 6-10 sub-questions, 2-3 sources each.
   - `exhaustive` — 12-20 sub-questions, 5+ sources each.
3. **Stage 2 — Iterate.** Each round: decide which sub-questions need
   attention, print the fetch list for the host agent to execute via its web
   tools, then record the evidence back. Apply a 5-axis source evaluation:
   Authority, Recency, Evidence, Bias, Corroboration. When all sub-questions
   reach the depth-target coverage, the iteration loop terminates.
4. **Stage 3 — Compile.** Produce a markdown report with:
   - Executive summary (5-8 lines).
   - Methodology block (depth, rounds, source count).
   - Per-sub-question section with embedded citations `[^N]`.
   - References block listing every source with URL + fetched_at + relevance.
   - "What this report does not cover" — explicit gaps from low-coverage
     sub-questions.

## Constraints

- This skill does not fetch the web itself — it is a methodology + state
  manager. Pair it with the host agent's web search/fetch tools.
- The compile step never invents sources — every `[^N]` must correspond to an
  entry recorded in stage 2.
- It does not resolve contradictions among sources automatically; the compile
  step notes conflicting evidence and the user decides which side wins.
- For ongoing monitoring (daily digests, RSS-style updates) build a cron skill
  that calls this one with a fresh question each cycle.
"#,
    },
    BundledSkillDef {
        id: "summarize",
        name: "Summarize",
        description: "Summarize, condense, or digest content into key points, details, and action items.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["summarize", "condense", "digest", "tldr"],
        body: r#"# Summarize

Use when the user asks to summarize, condense, digest, or get a TL;DR of
content.

## Procedure

1. **Read or obtain the full content** before summarizing. Do not summarize
   from a title or abstract alone.
2. **Produce a structured summary:**
   - **Key Points** — 3-5 bullet points of the most important information.
   - **Details** — Brief expansion on each key point if needed for context.
   - **Action Items** — Any tasks or follow-ups identified (if applicable).
3. **Keep it concise.** Match the length to the input: a 1-page document gets
   3-5 bullets; a 20-page document gets a fuller treatment but still focused on
   what matters most.
4. **Preserve numbers, names, and dates** — these are the load-bearing facts
   that make a summary useful.

## Constraints

- Do not inject opinions or analysis beyond what the source says.
- If the content is too long to read in full, say so and summarize the portion
  you did read, noting the gap.
- For multi-round investigation across many sources, use `deep-research`
  instead.
"#,
    },
    BundledSkillDef {
        id: "filesystem",
        name: "Filesystem Operations",
        description: "Advanced filesystem operations: listing, searching, batch processing, and directory analysis with safety checks.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["filesystem", "files", "directory", "search"],
        body: r#"# Filesystem Operations

Use when the user asks to list, search, batch-process, or analyze files and
directories. Use the host-provided filesystem tools in the current workspace.

## Procedure

1. **Smart listing.** List files with filtering by pattern, type, size, or
   date. Use recursive traversal with depth control when needed. Sort by name,
   size, date, or type as the user requests.
2. **Content search.** Search file contents by glob pattern or regex. Show
   matching lines with context. Combine filename and content searches with
   include/exclude filters.
3. **Batch operations.** Copy or move files by pattern with safety checks:
   - Always do a dry-run first to preview what will be affected.
   - Validate paths to prevent directory traversal.
   - Check read/write permissions before operating.
   - Suggest a backup before overwrites.
4. **Directory analysis.** Generate statistics: file counts, size distribution,
   type breakdown, largest files. Show a tree visualization with depth control
   for structure overview.

## Constraints

- Respect `.gitignore` patterns when listing or searching in a git repository.
- For destructive operations (mass delete, overwrite), show exactly what will be
  affected and confirm with the user before proceeding.
- Keep all paths inside the current workspace or the user-specified directory.
"#,
    },
    BundledSkillDef {
        id: "http-fetch",
        name: "HTTP Fetch",
        description: "Fetch a URL via HTTP/HTTPS and return the response body as text. Lightweight single-request entrypoint with no LLM loop.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["http", "fetch", "url", "network"],
        body: r#"# HTTP Fetch

Use when you need to make a single HTTP GET/POST/PUT/DELETE request and return
the response body as text — a lightweight replacement for spawning a sub-agent
just to fetch a URL.

## Procedure

1. **Determine the request parameters:**
   - `url` (required) — absolute http(s) URL.
   - `method` (default `GET`) — `GET` / `POST` / `PUT` / `DELETE`.
   - `body` (optional) — request body, piped via stdin for POST/PUT.
   - `timeout` (default 30s) — request timeout in seconds.
   - `max_bytes` (default 2,000,000) — response body cap; larger payloads are
     truncated.
2. **Execute the request** and handle the result:
   - Success (2xx): response body on stdout (UTF-8 decoded, truncated to
     `max_bytes` if larger).
   - Non-2xx: exit 1, stderr `HTTP <code>: <reason> <body[:200]>`; stdout still
     carries the body for inspection.
   - Network/DNS/timeout failure: exit 2, stderr carries the cause.
3. **Report the result** to the user, noting the status code and any
   truncation.

## When NOT to use

- Crawling multiple pages → use a sub-agent with a scraping library.
- JS-rendered pages → use a sub-agent with browser tools.
- OAuth dance / multi-step auth → use a sub-agent.
- Streaming responses → not supported (we buffer + return).

## Constraints

- No custom-header injection — the request goes out with default headers.
- Do not fetch URLs from untrusted sources without checking for SSRF risk
  (internal IPs, localhost, metadata endpoints).
"#,
    },
    BundledSkillDef {
        id: "cron",
        name: "Cron Scheduling",
        description: "Schedule recurring tasks, one-off reminders, timers, and cron-style jobs through the OpenSquilla cron tool.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["cron", "schedule", "reminder", "timer"],
        body: r#"# Cron Scheduling

Use when the user asks to schedule something, set up a recurring task, create a
timer, or create a reminder.

## Procedure

1. **Translate the natural-language request** into a structured schedule object
   before calling the cron tool. The `schedule` argument is a structured object,
   not a string — the tool rejects flat strings.
2. **Choose the right schedule shape:**
   - **cron** (calendar pattern): `{"kind": "cron", "expr": "<5-field POSIX
     cron>", "tz": "<optional IANA timezone>"}`
     Example: `{"kind": "cron", "expr": "0 9 * * 1-5", "tz": "Asia/Shanghai"}`
     for weekdays at 09:00 Shanghai time.
   - **every** (fixed interval): `{"kind": "every", "every_seconds": <int >= 1>}`
     Example: `{"kind": "every", "every_seconds": 30}` for every 30 seconds.
   - **at** (one-shot absolute): `{"kind": "at", "at": "<ISO-8601 with tz>"}`
     The timestamp must include a timezone offset.
3. **Call the cron tool:**
   - Add: `cron(action="add", schedule={...}, task="...", job_kind="...",
     session_target="...")`
   - List: `cron(action="list")`
   - Trigger now: `cron(action="run", job_id="<id>")`
   - Cancel: `cron(action="remove", job_id="<id>")`
4. **Cron expression format:** `minute hour day month weekday`
   (e.g. `0 9 * * 1-5` = weekdays at 9am).

## Translation examples

- "every 5 minutes, remind me to drink water" ->
  `{"kind": "cron", "expr": "*/5 * * * *"}`
- "every 30 seconds, print once" ->
  `{"kind": "every", "every_seconds": 30}`
- "tomorrow morning at 9am" -> compute the absolute ISO-8601 string with
  timezone, then `{"kind": "at", "at": "<that ISO-8601>"}`
- "every weekday at 9am Los Angeles time" ->
  `{"kind": "cron", "expr": "0 9 * * 1-5", "tz": "America/Los_Angeles"}`

## Constraints

- Do the translation in your own reasoning before calling the tool — the tool
  will not parse free-form text.
- Confirm destructive actions (cancelling jobs) with the user before proceeding.
"#,
    },
    BundledSkillDef {
        id: "skill-creator",
        name: "Skill Creator",
        description: "Create, edit, improve, or audit AgentSkills / SKILL.md files with proper frontmatter, progressive disclosure, and packaging.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["skill", "creator", "authoring", "meta"],
        body: r#"# Skill Creator

Use when creating a new skill from scratch, or when asked to improve, review,
audit, tidy up, or clean up an existing skill or SKILL.md file. Also use when
editing or restructuring a skill directory.

## Core principles

1. **Concise is key.** The context window is a shared resource. Only add
   information the agent doesn't already have. Challenge each paragraph: "Does
   this justify its token cost?"
2. **Set appropriate degrees of freedom.** Match specificity to the task's
   fragility: low freedom (specific scripts) for error-prone operations; high
   freedom (text instructions) when multiple approaches are valid.
3. **Progressive disclosure.** Three levels: metadata (always in context),
   SKILL.md body (loaded on trigger), bundled resources (loaded as needed). Keep
   SKILL.md under 500 lines; split into `references/` when approaching the limit.

## Skill anatomy

```
skill-name/
├── SKILL.md          (required: frontmatter + body)
├── scripts/          (optional: executable code for deterministic tasks)
├── references/       (optional: docs loaded as needed)
└── assets/           (optional: files used in output — templates, images)
```

Do NOT include README.md, CHANGELOG.md, or other auxiliary documentation files.

## Procedure

1. **Understand the skill.** Gather concrete usage examples. Ask: what
   functionality should it support? What triggers it? What would a user say?
2. **Plan reusable contents.** For each example, identify what scripts,
   references, or assets would help when executing repeatedly.
3. **Initialize the skill.** Create the directory with `SKILL.md` and any
   resource subdirectories needed.
4. **Edit the skill.**
   - **Frontmatter:** `name` (lowercase, hyphens, <64 chars) and `description`
     (the primary trigger mechanism — include what it does AND when to use it).
     Do not include other fields.
   - **Body:** Imperative/infinitive form. Instructions for using the skill and
     its bundled resources. Reference `references/` and `scripts/` files
     clearly so the reader knows they exist and when to use them.
   - **Scripts:** Test added scripts by actually running them.
5. **Package the skill** into a `.skill` file (zip with `.skill` extension).
   Packaging validates: frontmatter format, naming conventions, description
   quality, file organization. Symlinks are rejected.
6. **Iterate.** Use the skill on real tasks, notice struggles, update.

## Constraints

- Name skills with lowercase letters, digits, and hyphens only.
- Keep references one level deep from SKILL.md — no deeply nested references.
- For files longer than 100 lines, include a table of contents at the top.
- Avoid duplication: information should live in either SKILL.md or references,
  not both.
"#,
    },
    BundledSkillDef {
        id: "tmux",
        name: "Tmux Session Control",
        description: "Remote-control tmux sessions by sending keystrokes and scraping pane output for interactive CLIs and long-running processes.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["linux", "macos"],
        requires_bins: &["tmux"],
        tags: &["tmux", "terminal", "session", "interactive"],
        body: r#"# Tmux Session Control

Use when you need to remote-control tmux sessions: monitor interactive CLIs
(Claude Code, Codex), send input to terminal applications, scrape output from
long-running processes, or navigate panes/windows programmatically.

## Procedure

1. **List sessions** to see what's running: `tmux list-sessions` (alias `tmux
   ls`).
2. **Capture output** from a pane:
   - Last N lines: `tmux capture-pane -t <session> -p | tail -20`
   - Entire scrollback: `tmux capture-pane -t <session> -p -S -`
   - Specific pane: `tmux capture-pane -t <session>:0.0 -p`
3. **Send keys** to a pane:
   - Text + Enter: `tmux send-keys -t <session> "y" Enter`
   - Special keys: `Enter`, `Escape`, `C-c` (Ctrl+C), `C-d` (EOF), `C-z`
   - Text without Enter: `tmux send-keys -t <session> -l -- "your text"`
4. **For interactive TUIs** (Claude Code, Codex), split text and Enter into
   separate sends to avoid paste/multiline edge cases:
   ```
   tmux send-keys -t shared -l -- "Please apply the patch in src/foo.ts"
   sleep 0.1
   tmux send-keys -t shared Enter
   ```
5. **Session management:**
   - Create: `tmux new-session -d -s <name>`
   - Kill: `tmux kill-session -t <name>`
   - Rename: `tmux rename-session -t <old> <new>`
6. **Window/pane navigation:**
   - `tmux select-window -t <session>:0`
   - `tmux select-pane -t <session>:0.1`
   - `tmux list-windows -t <session>`

## Checking if a session needs input

Look for prompts in the last few lines:
```
tmux capture-pane -t worker-3 -p | tail -10 | grep -E "prompt|Yes.*No|proceed|permission"
```

## Constraints

- Target format: `session:window.pane` (e.g. `shared:0.0`).
- Use `capture-pane -p` to print to stdout (essential for scripting).
- Sessions persist across SSH disconnects.
- Not for one-off shell commands (use `exec_command`) or starting new background
  processes (use `background_process`).
"#,
    },
    BundledSkillDef {
        id: "web-search",
        name: "Web Search",
        description: "Search the web for information, news, images, or videos and return results in text, markdown, or JSON format.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["search", "web", "research", "news"],
        body: r#"# Web Search

Use when the user needs to search the web for information, find current content,
look up news articles, search for images or videos, or fact-check claims.

## Procedure

1. **Identify search intent.** What type of content (web, news, images, videos)?
   How recent should results be? How many are needed? Any filtering requirements?
2. **Configure search parameters:**
   - Search type: web (default), news, images, videos.
   - Max results (default 10).
   - Time range: `d` (day), `w` (week), `m` (month), `y` (year).
   - Region code (e.g. `us-en`, `uk-en`, `wt-wt` for worldwide).
   - Safe search: `on`, `moderate` (default), `off`.
3. **Select output format:** text (default, clean readable), markdown (with
   headers and links), or JSON (for programmatic processing).
4. **Execute the search** and save to file if results need to be preserved
   (`--output <path>`).
5. **Process results:** extract URLs or specific information, combine results
   from multiple searches.

## Image-specific filters

- Size: `Small`, `Medium`, `Large`, `Wallpaper`
- Color: `Monochrome`, `Red`, `Orange`, `Yellow`, `Green`, `Blue`, `Purple`,
  `Pink`, `Brown`, `Black`, `Gray`, `Teal`, `White`
- Type: `photo`, `clipart`, `gif`, `transparent`, `line`
- Layout: `Square`, `Tall`, `Wide`

## Video-specific filters

- Duration: `short`, `medium`, `long`
- Resolution: `high`, `standard`

## Constraints

- Be specific — clear, focused queries produce better results.
- Apply time filters when currency matters.
- Respect usage — don't hammer the API with rapid repeated searches.
- If no results are found, try broader terms or remove time filters.
- Rate limiting may occur; space out searches if making many requests.
"#,
    },
    BundledSkillDef {
        id: "security-audit",
        name: "Security Audit",
        description: "Audit code for security vulnerabilities: injection, auth bypass, secret leakage, path traversal, SSRF, and OWASP top 10 issues.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["security", "audit", "vulnerability", "owasp"],
        body: r#"# Security Audit

Use when the user asks to audit, scan, or review code for security
vulnerabilities. Also use proactively before deploying a new service or exposing
an endpoint.

## Procedure

1. **Scope the audit.** Identify the target: a diff, a directory, a service, or
   a full project. Note the language, framework, and entry points (HTTP routes,
   CLI args, file inputs, network listeners).
2. **Run these passes in order:**
   - **Injection:** SQL/NoSQL injection (unsanitized queries), command injection
     (shell concatenation, `eval`, `exec` with user input), template injection,
     LDAP/XPath injection.
   - **Authentication & authorization:** missing auth checks, broken session
     management, IDOR (insecure direct object references), privilege escalation,
     JWT issues (alg=none, weak secrets, missing expiry).
   - **Secret leakage:** hardcoded credentials/tokens/keys, secrets in logs or
     error messages, secrets in version control, missing env-var indirection.
   - **Path traversal & SSRF:** file path manipulation (`../`, absolute paths),
     URL fetching without internal-IP/localhost blocking, open redirect.
   - **Deserialization:** unsafe `pickle`/`unserialize`/`serde` of untrusted
     input, XXE in XML parsing.
   - **XSS & CSRF:** unescaped output, missing CSRF tokens on state-changing
     requests, permissive CSP.
   - **Crypto:** weak algorithms (MD5, DES, ECB mode), hardcoded IVs, custom
     crypto, weak randomness for tokens.
3. **Report by severity** using CVSS-like buckets:
   - `CRITICAL` — remotely exploitable, leads to RCE or data breach.
   - `HIGH` — exploitable with some access, serious impact.
   - `MEDIUM` — requires specific conditions or has limited impact.
   - `LOW` — defense-in-depth improvement, hardening.
4. **For each finding:** cite the exact file and line, explain the attack
   scenario, and suggest a concrete fix with a code snippet.

## Constraints

- Do not exploit or attempt active penetration — this is a static audit.
- If a finding is framework-specific, verify the framework version before
  reporting (the vulnerability may already be patched).
- Distinguish between "this is a vulnerability" and "this is a code smell" —
  only report confirmed or high-likelihood issues as findings.
- For dependencies, check for known CVEs but note that transitive dependencies
  may not be fully resolvable without a lockfile.
"#,
    },
    BundledSkillDef {
        id: "performance",
        name: "Performance Analysis",
        description: "Analyze code for performance issues: algorithmic complexity, memory leaks, redundant I/O, and bottleneck identification.",
        version: "1.0.0",
        author: "OpenSquilla",
        kind: SkillKind::Skill,
        requires_os: &["any"],
        requires_bins: &[],
        tags: &["performance", "optimization", "profiling", "complexity"],
        body: r#"# Performance Analysis

Use when the user asks to optimize, speed up, or analyze the performance of
code, or when investigating slow operations, high memory usage, or scalability
concerns.

## Procedure

1. **Establish a baseline.** Before optimizing, measure the current performance.
   Identify the metric that matters: latency, throughput, memory, CPU, or I/O.
   If no benchmark exists, suggest creating one.
2. **Profile to find the bottleneck.** Do not guess — use the right tool:
   - CPU: `perf`, `flamegraph`, language profilers (`py-spy`, `pprof`).
   - Memory: heap profilers, allocation tracking.
   - I/O: `strace`, `iotop`, query logs with `EXPLAIN ANALYZE`.
   - Network: `tcpdump`, `wireshark`, latency histograms.
3. **Focus on the hot path.** 80/20 rule: the top 1-2 bottlenecks typically
   account for most of the cost. Optimize those first.
4. **Common issues to check:**
   - **Algorithmic complexity:** accidental O(n^2) or O(n!) inside a loop,
     repeated linear scans where a hash map would do.
   - **Redundant I/O:** N+1 queries, reading the same file repeatedly, missing
     batch/caching.
   - **Memory:** unbounded growth (leaking collections), large allocations in a
     hot loop, unnecessary cloning/copying.
   - **Concurrency:** lock contention, sequential I/O that could be parallel,
     excessive thread spawning.
   - **Allocation pressure:** frequent small allocations in hot paths, missing
     object reuse/pooling.
5. **Propose a fix.** For each bottleneck, explain the root cause and suggest a
   concrete optimization. Prefer algorithmic improvements over micro-tricks.
6. **Verify the improvement.** Re-run the benchmark after the change. Report
   before/after numbers. If the gain is negligible, revert.

## Constraints

- Never optimize without measuring first — "premature optimization is the root
  of all evil."
- Do not sacrifice correctness or readability for marginal gains.
- Prefer asymptotic improvements (O(n^2) -> O(n log n)) over constant-factor
  tweaks.
- Note when an optimization trades memory for speed or vice versa.
"#,
    },
];

/// Load all bundled skills as full [`SkillSpec`] values.
///
/// This is the entry point used by the [`crate::loader::SkillLoader`] to
/// register the built-in catalog. It is cheap: it merely expands the static
/// [`BUNDLED_SKILLS`] array.
pub fn load_bundled_skills() -> Vec<SkillSpec> {
    BUNDLED_SKILLS
        .iter()
        .map(BundledSkillDef::to_spec)
        .collect()
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
        assert!(specs.len() >= 20, "expected at least 20 bundled skills");
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
            assert!(
                !spec.description.is_empty(),
                "skill description must not be empty"
            );
            assert!(
                spec.version.as_deref().is_some_and(|v| !v.is_empty()),
                "skill {} must have a version",
                spec.id
            );
        }
    }
}
