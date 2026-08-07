//! Bundled (built-in) skills.
//!
//! OpenSquilla ships a catalog of built-in skills that are compiled into the
//! binary as Rust data structures (see [`crate::bundled::BUNDLED_SKILLS`]). No
//! `SKILL.md` files are read from disk at runtime; everything lives in code.
//!
//! The bundled layer is the second-lowest of the six skill layers (after
//! `EXTRA`), so user, project, and workspace skills override bundled ones.

pub mod skill_loader;

pub use skill_loader::{
    BUNDLED_SKILLS, BundledSkillDef, bundled_skill_count, get_bundled_skill, load_bundled_skills,
};
