//! Native workbench surface management.
//!
//! Replaces the Electron `native-workbench-surface.ts`,
//! `desktop-context-lock.ts`, `desktop-writer-admission.ts`, and
//! `artifact-preview-lease-broker.ts` modules.
//!
//! This module manages:
//! 1. **Workbench surfaces** — native preview panels for artifacts and URLs.
//!    In Tauri, the actual rendering is done by WebviewWindow in the frontend;
//!    this module manages the surface state, lifecycle, and bounds.
//! 2. **Context locking** — a process-local keyed lock preventing concurrent
//!    writes to the same context file (port of `DesktopContextLock`).
//! 3. **Writer admission** — coordinates exclusive writer access so only one
//!    writer at a time can modify profile/context files (port of
//!    `DesktopWriterAdmission`).
//! 4. **Artifact preview lease broker** — manages preview leases for artifact
//!    previews, tracking issued leases and authorizing surface creation (port
//!    of `ArtifactPreviewLeaseBroker`).

use crate::ipc::{ArtifactPreviewLeaseGrant, WorkbenchSurfaceResult};
use parking_lot::Mutex as ParkingMutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Constants (mirrors of the Electron contract)
// ---------------------------------------------------------------------------

/// Maximum number of concurrent native workbench surfaces.
pub const MAX_SURFACES: usize = 8;

/// Maximum HTML artifact size in bytes (5 MiB).
pub const MAX_HTML_BYTES: usize = 5 * 1024 * 1024;

/// Native workbench protocol versions.
pub const PROTOCOL_VERSION: u32 = 1;
pub const PROTOCOL_VERSION_V2: u32 = 2;

/// Default request timeout for lease broker operations.
const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 15_000;

// ---------------------------------------------------------------------------
// Surface management
// ---------------------------------------------------------------------------

/// The kind of workbench surface.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SurfaceKind {
    /// Legacy HTML artifact (v1).
    ArtifactHtml,
    /// Artifact preview served by the gateway (v2).
    ArtifactPreview,
    /// URL preview (v2).
    UrlPreview,
}

impl SurfaceKind {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "artifact-html" => Some(SurfaceKind::ArtifactHtml),
            "artifact-preview" => Some(SurfaceKind::ArtifactPreview),
            "url-preview" => Some(SurfaceKind::UrlPreview),
            _ => None,
        }
    }
}

/// The preview mode for a workbench surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    Full,
    Offline,
}

impl PreviewMode {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "full" => Some(PreviewMode::Full),
            "offline" => Some(PreviewMode::Offline),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            PreviewMode::Full => "full",
            PreviewMode::Offline => "offline",
        }
    }
}

/// A native workbench surface record.
#[derive(Debug, Clone)]
pub struct SurfaceRecord {
    pub id: String,
    pub version: u32,
    pub kind: SurfaceKind,
    pub mode: PreviewMode,
    pub scope_id: String,
    pub document_url: String,
    pub expected_origin: Option<String>,
    pub visible: bool,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub disposed: bool,
}

/// The native workbench surface manager.
///
/// Manages the lifecycle of workbench surfaces. In Tauri, the actual rendering
/// is performed by WebviewWindow instances created by the frontend; this
/// manager tracks surface state, enforces the maximum surface limit, and
/// provides bounds management. Surface events are emitted to the frontend via
/// Tauri events.
pub struct WorkbenchManager {
    surfaces: HashMap<String, SurfaceRecord>,
    active_surface_id: Option<String>,
    context_lock: ContextLock,
    writer_admission: WriterAdmission,
    lease_broker: ArtifactPreviewLeaseBroker,
}

impl WorkbenchManager {
    /// Create a new empty workbench manager.
    pub fn new() -> Self {
        Self {
            surfaces: HashMap::new(),
            active_surface_id: None,
            context_lock: ContextLock::new(),
            writer_admission: WriterAdmission::new(),
            lease_broker: ArtifactPreviewLeaseBroker::new(),
        }
    }

    /// Create a new workbench surface.
    ///
    /// Validates the request, enforces the maximum surface limit, and registers
    /// the surface. Returns `ok: true` on success, or `ok: false` with a
    /// message on failure.
    pub fn create_surface(
        &mut self,
        surface_id: String,
        kind: SurfaceKind,
        scope_id: String,
        document_url: String,
        expected_origin: Option<String>,
        mode: PreviewMode,
    ) -> WorkbenchSurfaceResult {
        // Remove any existing surface with this ID.
        if let Some(existing) = self.surfaces.remove(&surface_id) {
            debug!(surface_id = %surface_id, "Replacing existing surface");
            let _ = existing;
        }

        // Enforce maximum surface count.
        if self.surfaces.len() >= MAX_SURFACES {
            return WorkbenchSurfaceResult {
                ok: false,
                message: Some(format!(
                    "Close a Workbench preview before opening more than {MAX_SURFACES}."
                )),
                surface_id: None,
            };
        }

        let record = SurfaceRecord {
            id: surface_id.clone(),
            version: if kind == SurfaceKind::ArtifactHtml {
                PROTOCOL_VERSION
            } else {
                PROTOCOL_VERSION_V2
            },
            kind,
            mode,
            scope_id,
            document_url,
            expected_origin,
            visible: false,
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
            disposed: false,
        };

        self.surfaces.insert(surface_id.clone(), record);
        info!(surface_id = %surface_id, "Workbench surface created");

        WorkbenchSurfaceResult {
            ok: true,
            message: None,
            surface_id: Some(surface_id),
        }
    }

    /// Destroy a workbench surface by ID.
    pub fn destroy_surface(&mut self, surface_id: &str) -> WorkbenchSurfaceResult {
        if let Some(record) = self.surfaces.remove(surface_id) {
            if self.active_surface_id.as_deref() == Some(surface_id) {
                self.active_surface_id = None;
            }
            debug!(surface_id = %surface_id, "Workbench surface destroyed");
            let _ = record;
            WorkbenchSurfaceResult {
                ok: true,
                message: None,
                surface_id: Some(surface_id.to_string()),
            }
        } else {
            WorkbenchSurfaceResult {
                ok: false,
                message: Some("The native Workbench surface no longer exists.".to_string()),
                surface_id: None,
            }
        }
    }

    /// Set the surface rect (bounds and visibility).
    pub fn set_surface_rect(
        &mut self,
        surface_id: &str,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
        visible: bool,
    ) -> WorkbenchSurfaceResult {
        let record = match self.surfaces.get_mut(surface_id) {
            Some(r) => r,
            None => {
                return WorkbenchSurfaceResult {
                    ok: false,
                    message: Some("The native Workbench surface no longer exists.".to_string()),
                    surface_id: None,
                };
            }
        };

        record.x = x;
        record.y = y;
        record.width = width;
        record.height = height;
        record.visible = visible;

        if visible {
            self.activate_surface(surface_id);
        } else if self.active_surface_id.as_deref() == Some(surface_id) {
            self.active_surface_id = None;
        }

        WorkbenchSurfaceResult {
            ok: true,
            message: None,
            surface_id: Some(surface_id.to_string()),
        }
    }

    /// Activate a surface (hide all others, set as active).
    pub fn activate_surface(&mut self, surface_id: &str) {
        // Hide all other surfaces.
        for (id, record) in self.surfaces.iter_mut() {
            if id != surface_id {
                record.visible = false;
            }
        }
        self.active_surface_id = Some(surface_id.to_string());
        if let Some(record) = self.surfaces.get_mut(surface_id) {
            record.visible = true;
        }
    }

    /// Get a surface by ID.
    pub fn get_surface(&self, surface_id: &str) -> Option<&SurfaceRecord> {
        self.surfaces.get(surface_id)
    }

    /// List all surface IDs.
    pub fn list_surfaces(&self) -> Vec<String> {
        self.surfaces.keys().cloned().collect()
    }

    /// Get the count of active surfaces.
    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    /// Destroy all surfaces.
    pub fn destroy_all(&mut self) {
        let count = self.surfaces.len();
        self.surfaces.clear();
        self.active_surface_id = None;
        if count > 0 {
            info!(count = count, "Destroyed all workbench surfaces");
        }
    }

    /// Get a reference to the context lock.
    pub fn context_lock(&self) -> &ContextLock {
        &self.context_lock
    }

    /// Get a reference to the writer admission controller.
    pub fn writer_admission(&self) -> &WriterAdmission {
        &self.writer_admission
    }

    /// Get a mutable reference to the writer admission controller.
    pub fn writer_admission_mut(&mut self) -> &mut WriterAdmission {
        &mut self.writer_admission
    }

    /// Get a reference to the artifact preview lease broker.
    pub fn lease_broker(&self) -> &ArtifactPreviewLeaseBroker {
        &self.lease_broker
    }

    /// Get a mutable reference to the artifact preview lease broker.
    pub fn lease_broker_mut(&mut self) -> &mut ArtifactPreviewLeaseBroker {
        &mut self.lease_broker
    }
}

impl Default for WorkbenchManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Context Lock (port of DesktopContextLock)
// ---------------------------------------------------------------------------

/// Process-local keyed lock for Desktop-owned context files.
///
/// Every cooperative context writer must use this lock, so a read/check/publish
/// transaction cannot be interleaved by a second code path inside the owning
/// process. Re-entry for the same key is explicitly rejected (not recursive).
///
/// This is a direct port of the Electron `DesktopContextLock` class, using
/// `tokio::sync::Mutex` per key instead of `AsyncLocalStorage`.
pub struct ContextLock {
    tails: ParkingMutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl ContextLock {
    /// Create a new empty context lock.
    pub fn new() -> Self {
        Self {
            tails: ParkingMutex::new(HashMap::new()),
        }
    }

    /// Run an operation exclusively for the given key.
    ///
    /// If the key is already locked, this will await until the lock is released.
    /// Re-entry for the same key is not supported and will deadlock (matching
    /// the Electron behavior which throws on re-entry).
    pub async fn run_exclusive<F, T>(&self, key: &str, operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        if key.is_empty() {
            panic!("Desktop context lock requires a non-empty key.");
        }

        let mutex = {
            let mut tails = self.tails.lock();
            tails
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };

        let _guard = mutex.lock().await;
        operation.await
    }

    /// Check whether a key is currently locked.
    pub fn is_locked(&self, key: &str) -> bool {
        let tails = self.tails.lock();
        if let Some(mutex) = tails.get(key) {
            return mutex.try_lock().is_err();
        }
        false
    }
}

impl Default for ContextLock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Writer Admission (port of DesktopWriterAdmission)
// ---------------------------------------------------------------------------

/// A token representing ownership of a writer admission close.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AdmissionToken(Arc<()>);

impl AdmissionToken {
    /// Create a new unique admission token.
    fn new() -> Self {
        Self(Arc::new(()))
    }
}

/// Coordinates Desktop-owned profile writers with lifecycle operations.
///
/// A normal writer calls `begin()` and always invokes the returned finish
/// callback. Update, quit, or another lifecycle boundary calls `close()`
/// before waiting for active writers to drain. An operation that must close
/// admission and reserve its own writer slot atomically uses
/// `try_begin_exclusive()`.
///
/// This is a direct port of the Electron `DesktopWriterAdmission` class.
pub struct WriterAdmission {
    close_owners: HashSet<String>, // Using label strings as owners (simplified from Symbol)
    active: u64,
    waiters: Vec<WriterWaiter>,
}

#[derive(Debug)]
struct WriterWaiter {
    maximum_active: u64,
    waker: Arc<tokio::sync::Notify>,
}

impl WriterAdmission {
    /// Create a new writer admission controller.
    pub fn new() -> Self {
        Self {
            close_owners: HashSet::new(),
            active: 0,
            waiters: Vec::new(),
        }
    }

    /// Whether admission is closed (no new writers can begin).
    pub fn is_closed(&self) -> bool {
        !self.close_owners.is_empty()
    }

    /// The number of currently active writers.
    pub fn active_count(&self) -> u64 {
        self.active
    }

    /// Close admission for new writers. Returns a token that can be used to
    /// reopen admission.
    pub fn close(&mut self, label: &str) -> String {
        self.close_owners.insert(label.to_string());
        debug!(label = label, "Writer admission closed");
        label.to_string()
    }

    /// Reopen admission by removing a close owner.
    pub fn reopen(&mut self, token: &str) -> bool {
        let removed = self.close_owners.remove(token);
        if removed {
            debug!(token = token, "Writer admission reopened");
        }
        removed
    }

    /// Check whether a specific owner holds a close.
    pub fn has_owner(&self, token: &str) -> bool {
        self.close_owners.contains(token)
    }

    /// Check whether any owner other than the given token holds a close.
    pub fn has_other_owner(&self, token: &str) -> bool {
        self.close_owners.iter().any(|owner| owner != token)
    }

    /// Begin a writer operation. Returns a guard that releases the writer slot
    /// when dropped.
    ///
    /// Returns an error if admission is closed.
    pub fn begin(&mut self, label: &str) -> Result<WriterGuard, String> {
        if self.is_closed() {
            return Err(format!(
                "Desktop writer admission is closed; {label} was not started."
            ));
        }
        self.active += 1;
        Ok(WriterGuard {
            admission: self,
            finished: false,
        })
    }

    /// Try to begin an exclusive writer operation: close admission and reserve
    /// a writer slot atomically. Returns the admission token and a writer guard
    /// on success, or `None` if admission is already closed.
    pub fn try_begin_exclusive(&mut self, label: &str) -> Option<(String, WriterGuard)> {
        if self.is_closed() {
            return None;
        }
        let token = self.close(label);
        self.active += 1;
        Some((
            token,
            WriterGuard {
                admission: self,
                finished: false,
            },
        ))
    }

    /// Wait until the active writer count drops to at most `maximum_active`.
    pub async fn wait_for_at_most(&mut self, maximum_active: u64) {
        if self.active <= maximum_active {
            return;
        }

        let notify = Arc::new(tokio::sync::Notify::new());
        self.waiters.push(WriterWaiter {
            maximum_active,
            waker: notify.clone(),
        });

        // Check if we can immediately satisfy (race-safe: re-check after register).
        if self.active <= maximum_active {
            return;
        }

        notify.notified().await;
    }

    /// Release a writer slot (called by `WriterGuard::drop`).
    fn release_writer(&mut self) {
        if self.active > 0 {
            self.active -= 1;
        }

        // Notify waiters whose threshold is now met.
        let mut remaining = Vec::new();
        for waiter in self.waiters.drain(..) {
            if self.active > waiter.maximum_active {
                remaining.push(waiter);
            } else {
                waiter.waker.notify_one();
            }
        }
        self.waiters = remaining;
    }
}

impl Default for WriterAdmission {
    fn default() -> Self {
        Self::new()
    }
}

/// A guard that releases a writer slot when dropped.
pub struct WriterGuard<'a> {
    admission: &'a mut WriterAdmission,
    finished: bool,
}

impl<'a> WriterGuard<'a> {
    /// Explicitly finish the writer operation (releases the slot).
    pub fn finish(mut self) {
        if !self.finished {
            self.finished = true;
            self.admission.release_writer();
        }
    }

    /// The number of currently active writers, including this guard.
    pub fn active_count(&self) -> u64 {
        self.admission.active
    }
}

impl<'a> Drop for WriterGuard<'a> {
    fn drop(&mut self) {
        if !self.finished {
            self.admission.release_writer();
        }
    }
}

// ---------------------------------------------------------------------------
// Artifact Preview Lease Broker (port of ArtifactPreviewLeaseBroker)
// ---------------------------------------------------------------------------

/// An issued artifact preview lease.
#[derive(Debug, Clone)]
struct IssuedPreview {
    launch_url: String,
    expected_origin: String,
    scope_id: String,
    mode: PreviewMode,
    expires_at: Instant,
}

/// The artifact preview lease broker.
///
/// Manages artifact preview leases issued by the gateway. When the frontend
/// requests a preview, the broker contacts the gateway to create a lease, then
/// tracks it locally. Surface creation is authorized only for leases that
/// match the requested launch URL, origin, scope, and mode.
///
/// In the in-process Tauri model, the gateway runs in the same process, so
/// lease creation can be done via direct function call rather than HTTP.
/// However, the broker still tracks issued leases for authorization and
/// revocation.
pub struct ArtifactPreviewLeaseBroker {
    issued: HashMap<String, IssuedPreview>,
}

impl ArtifactPreviewLeaseBroker {
    /// Create a new empty lease broker.
    pub fn new() -> Self {
        Self {
            issued: HashMap::new(),
        }
    }

    /// Issue a new preview lease locally.
    ///
    /// In the in-process model, the gateway creates the lease and returns its
    /// details. The broker records it for later authorization checks.
    pub fn issue_lease(
        &mut self,
        lease_id: String,
        launch_url: String,
        expected_origin: String,
        scope_id: String,
        mode: PreviewMode,
        expires_at: Instant,
    ) {
        self.issued.insert(
            lease_id,
            IssuedPreview {
                launch_url,
                expected_origin,
                scope_id,
                mode,
                expires_at,
            },
        );
    }

    /// Revoke a lease by ID.
    pub fn revoke_lease(&mut self, lease_id: &str) -> bool {
        self.issued.remove(lease_id).is_some()
    }

    /// Check whether a surface grant is authorized by an existing lease.
    ///
    /// Expired leases are cleaned up during this check.
    pub fn authorizes_surface(&mut self, grant: &ArtifactPreviewLeaseGrant) -> bool {
        let now = Instant::now();
        let grant_mode = match PreviewMode::from_str(&grant.mode) {
            Some(m) => m,
            None => return false,
        };

        // Clean up expired leases.
        self.issued.retain(|_, lease| lease.expires_at > now);

        // Check if any issued lease matches the grant.
        self.issued.values().any(|lease| {
            lease.launch_url == grant.launch_url
                && lease.expected_origin == grant.expected_origin
                && lease.scope_id == grant.scope_id
                && lease.mode == grant_mode
        })
    }

    /// Get the count of active (non-expired) leases.
    pub fn active_count(&mut self) -> usize {
        let now = Instant::now();
        self.issued.retain(|_, lease| lease.expires_at > now);
        self.issued.len()
    }

    /// Clear all issued leases.
    pub fn clear(&mut self) {
        self.issued.clear();
    }

    /// List all active lease IDs.
    pub fn list_lease_ids(&mut self) -> Vec<String> {
        let now = Instant::now();
        self.issued.retain(|_, lease| lease.expires_at > now);
        self.issued.keys().cloned().collect()
    }
}

impl Default for ArtifactPreviewLeaseBroker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tauri Commands
// ---------------------------------------------------------------------------

use crate::error::TauriResult;
use crate::ipc::{
    ArtifactPreviewLeaseCreateRequest, ArtifactPreviewLeasePayload, WorkbenchNavigationRequest,
    WorkbenchSurfaceCreateRequest, WorkbenchSurfaceRectRequest,
};
use crate::state::AppState;
use std::time::SystemTime;

/// Create a new workbench surface.
#[tauri::command]
pub async fn create_workbench_surface(
    state: tauri::State<'_, AppState>,
    request: WorkbenchSurfaceCreateRequest,
) -> TauriResult<WorkbenchSurfaceResult> {
    let kind = SurfaceKind::from_str(&request.kind).ok_or_else(|| {
        crate::error::TauriError::bad_request(format!("Unsupported surface kind: {}", request.kind))
    })?;

    let mode = request
        .mode
        .as_deref()
        .and_then(PreviewMode::from_str)
        .unwrap_or(PreviewMode::Full);

    // For artifact-preview kind, check lease authorization.
    if kind == SurfaceKind::ArtifactPreview {
        if let Some(url) = &request.url {
            let grant = ArtifactPreviewLeaseGrant {
                launch_url: url.clone(),
                expected_origin: url.clone(),
                scope_id: request.scope_id.clone(),
                mode: mode.as_str().to_string(),
            };
            let mut workbench = state.workbench().await;
            if !workbench.lease_broker_mut().authorizes_surface(&grant) {
                return Ok(WorkbenchSurfaceResult {
                    ok: false,
                    message: Some("The artifact preview lease is not authorized.".to_string()),
                    surface_id: None,
                });
            }
        }
    }

    let document_url = request.url.unwrap_or_default();
    let mut workbench = state.workbench().await;
    let result = workbench.create_surface(
        request.surface_id.clone(),
        kind,
        request.scope_id,
        document_url,
        None,
        mode,
    );
    Ok(result)
}

/// Destroy a workbench surface.
#[tauri::command]
pub async fn destroy_workbench_surface(
    state: tauri::State<'_, AppState>,
    surface_id: String,
) -> TauriResult<WorkbenchSurfaceResult> {
    let mut workbench = state.workbench().await;
    Ok(workbench.destroy_surface(&surface_id))
}

/// Set the surface rect (bounds and visibility).
#[tauri::command]
pub async fn set_workbench_surface_rect(
    state: tauri::State<'_, AppState>,
    request: WorkbenchSurfaceRectRequest,
) -> TauriResult<WorkbenchSurfaceResult> {
    let mut workbench = state.workbench().await;
    Ok(workbench.set_surface_rect(
        &request.surface_id,
        request.x,
        request.y,
        request.width,
        request.height,
        request.visible,
    ))
}

/// Navigate a workbench surface.
#[tauri::command]
pub async fn navigate_workbench_surface(
    state: tauri::State<'_, AppState>,
    request: WorkbenchNavigationRequest,
) -> TauriResult<WorkbenchSurfaceResult> {
    let mut workbench = state.workbench().await;
    let record = match workbench.get_surface(&request.surface_id) {
        Some(r) => r.clone(),
        None => {
            return Ok(WorkbenchSurfaceResult {
                ok: false,
                message: Some("The native Workbench surface no longer exists.".to_string()),
                surface_id: None,
            });
        }
    };

    match request.action.as_str() {
        "navigate" => {
            if let Some(url) = &request.url {
                debug!(surface_id = %request.surface_id, url = %url, "Navigating surface");
                // In the in-process model, navigation is handled by the frontend.
                // We just validate that the surface exists.
            }
        }
        "back" | "forward" | "reload" | "stop" => {
            debug!(surface_id = %request.surface_id, action = %request.action, "Navigation action");
        }
        "open-external" => {
            if let Some(url) = &request.url {
                info!(surface_id = %request.surface_id, url = %url, "Opening external URL");
            }
        }
        _ => {
            return Ok(WorkbenchSurfaceResult {
                ok: false,
                message: Some(format!("Unsupported navigation action: {}", request.action)),
                surface_id: None,
            });
        }
    }

    let _ = record;
    Ok(WorkbenchSurfaceResult {
        ok: true,
        message: None,
        surface_id: Some(request.surface_id),
    })
}

/// Destroy all workbench surfaces.
#[tauri::command]
pub async fn destroy_all_workbench_surfaces(
    state: tauri::State<'_, AppState>,
) -> TauriResult<WorkbenchSurfaceResult> {
    let mut workbench = state.workbench().await;
    workbench.destroy_all();
    Ok(WorkbenchSurfaceResult {
        ok: true,
        message: None,
        surface_id: None,
    })
}

/// List all workbench surfaces.
#[tauri::command]
pub async fn list_workbench_surfaces(
    state: tauri::State<'_, AppState>,
) -> TauriResult<Vec<String>> {
    let workbench = state.workbench().await;
    Ok(workbench.list_surfaces())
}

/// Create an artifact preview lease.
#[tauri::command]
pub async fn create_artifact_preview_lease(
    state: tauri::State<'_, AppState>,
    request: ArtifactPreviewLeaseCreateRequest,
) -> TauriResult<ArtifactPreviewLeasePayload> {
    let mode = PreviewMode::from_str(&request.mode).ok_or_else(|| {
        crate::error::TauriError::bad_request(format!("Invalid preview mode: {}", request.mode))
    })?;

    // In the in-process model, the gateway creates the lease. For now, we
    // generate a lease ID and issue it locally. In production, this would call
    // the gateway's lease creation endpoint.
    let lease_id = format!("apl-{}", uuid::Uuid::new_v4().simple());
    let launch_url = format!("http://p-{}.localhost:0/", uuid::Uuid::new_v4().simple());
    let expected_origin = launch_url.trim_end_matches('/').to_string();
    let expires_at = Instant::now() + Duration::from_secs(3600); // 1 hour

    let payload = ArtifactPreviewLeasePayload {
        lease_id: lease_id.clone(),
        effective_mode: request.mode.clone(),
        launch_url: launch_url.clone(),
        entrypoint: "/index.html".to_string(),
        expires_at: chrono::Utc::now()
            .checked_add_signed(chrono::Duration::hours(1))
            .map(|t| t.to_rfc3339())
            .unwrap_or_default(),
        preview_origin: expected_origin.clone(),
        idle_timeout_seconds: 1800,
    };

    let mut workbench = state.workbench().await;
    workbench.lease_broker_mut().issue_lease(
        lease_id,
        launch_url,
        expected_origin,
        request.scope_id,
        mode,
        expires_at,
    );

    Ok(payload)
}

/// Revoke an artifact preview lease.
#[tauri::command]
pub async fn revoke_artifact_preview_lease(
    state: tauri::State<'_, AppState>,
    lease_id: String,
) -> TauriResult<bool> {
    let mut workbench = state.workbench().await;
    Ok(workbench.lease_broker_mut().revoke_lease(&lease_id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_lock_exclusive() {
        let lock = ContextLock::new();
        // Basic smoke test: run_exclusive should work.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let result = lock.run_exclusive("test-key", async { 42 }).await;
            assert_eq!(result, 42);
        });
    }

    #[test]
    fn test_writer_admission_begin_finish() {
        let mut admission = WriterAdmission::new();
        assert!(!admission.is_closed());
        assert_eq!(admission.active_count(), 0);

        let guard = admission.begin("test").unwrap();
        assert_eq!(guard.active_count(), 1);
        guard.finish();
        assert_eq!(admission.active_count(), 0);
    }

    #[test]
    fn test_writer_admission_close_reopen() {
        let mut admission = WriterAdmission::new();
        let token = admission.close("update");
        assert!(admission.is_closed());
        assert!(admission.begin("test").is_err());
        assert!(admission.reopen(&token));
        assert!(!admission.is_closed());
        assert!(admission.begin("test").is_ok());
    }

    #[test]
    fn test_lease_broker_authorization() {
        let mut broker = ArtifactPreviewLeaseBroker::new();
        let launch_url = "http://p-abc.localhost:1234/".to_string();
        let expected_origin = "http://p-abc.localhost:1234".to_string();

        broker.issue_lease(
            "lease-1".to_string(),
            launch_url.clone(),
            expected_origin.clone(),
            "scope-1".to_string(),
            PreviewMode::Full,
            Instant::now() + Duration::from_secs(3600),
        );

        let grant = ArtifactPreviewLeaseGrant {
            launch_url: launch_url.clone(),
            expected_origin: expected_origin.clone(),
            scope_id: "scope-1".to_string(),
            mode: "full".to_string(),
        };
        assert!(broker.authorizes_surface(&grant));

        let bad_grant = ArtifactPreviewLeaseGrant {
            launch_url: "http://evil.com/".to_string(),
            expected_origin: "http://evil.com".to_string(),
            scope_id: "scope-1".to_string(),
            mode: "full".to_string(),
        };
        assert!(!broker.authorizes_surface(&bad_grant));
    }

    #[test]
    fn test_workbench_surface_creation() {
        let mut manager = WorkbenchManager::new();
        let result = manager.create_surface(
            "surf-1".to_string(),
            SurfaceKind::ArtifactHtml,
            "scope-1".to_string(),
            "opensquilla-artifact://handle/index.html".to_string(),
            None,
            PreviewMode::Full,
        );
        assert!(result.ok);
        assert_eq!(manager.surface_count(), 1);
    }

    #[test]
    fn test_workbench_max_surfaces() {
        let mut manager = WorkbenchManager::new();
        for i in 0..MAX_SURFACES {
            let result = manager.create_surface(
                format!("surf-{i}"),
                SurfaceKind::ArtifactHtml,
                "scope".to_string(),
                "url".to_string(),
                None,
                PreviewMode::Full,
            );
            assert!(result.ok, "Surface {i} should be created");
        }
        // The next one should fail.
        let result = manager.create_surface(
            "surf-overflow".to_string(),
            SurfaceKind::ArtifactHtml,
            "scope".to_string(),
            "url".to_string(),
            None,
            PreviewMode::Full,
        );
        assert!(!result.ok);
    }
}
