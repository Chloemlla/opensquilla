//! # OpenSquilla Safety
//!
//! Prompt injection detection, permission management, and secret redaction.

pub mod injection;
pub mod permissions;
pub mod secrets;

pub use injection::{InjectionDetector, InjectionGuard, InjectionResult, InjectionSeverity};
pub use permissions::{
    PermissionAction, PermissionCheck, PermissionContext, PermissionMatrix, PermissionScope,
    RiskLevel,
};
pub use secrets::{SecretMatch, SecretRedactor, SecretSanitizer, SecretType};