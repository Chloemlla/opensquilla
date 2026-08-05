//! Preset Seatbelt profiles per security level.
//!
//! Each preset returns a [`SeatbeltProfile`] built from a
//! [`SandboxPolicy`](crate::policy::SandboxPolicy) at the corresponding level,
//! with any level-specific tweaks applied.

use crate::policy::SandboxLevel;
use crate::seatbelt::{SeatbeltProfile, compile_policy};

/// The preset profile for a level.
pub fn preset_profile(level: SandboxLevel) -> SeatbeltProfile {
    let policy = crate::policy::SandboxPolicy::build_policy(level, None);
    let source = compile_policy(&policy);
    SeatbeltProfile {
        source,
        name: format!("opensquilla_{}", level_tag(level)),
        level,
        deny_default: true,
    }
    .named(format!("preset_{}", level_tag(level)))
}

/// The STANDARD preset.
pub fn standard() -> SeatbeltProfile {
    preset_profile(SandboxLevel::Standard)
}

/// The STRICT preset.
pub fn strict() -> SeatbeltProfile {
    preset_profile(SandboxLevel::Strict)
}

/// The LOCKED preset.
pub fn locked() -> SeatbeltProfile {
    preset_profile(SandboxLevel::Locked)
}

fn level_tag(level: SandboxLevel) -> &'static str {
    match level {
        SandboxLevel::Standard => "standard",
        SandboxLevel::Strict => "strict",
        SandboxLevel::Locked => "locked",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_compile_and_validate() {
        for p in [standard(), strict(), locked()] {
            p.validate_syntax().unwrap();
        }
    }

    #[test]
    fn locked_denies_network() {
        let p = locked();
        assert!(p.source.contains("(deny network*)"));
    }
}
