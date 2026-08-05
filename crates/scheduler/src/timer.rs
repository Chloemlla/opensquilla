use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration, MissedTickBehavior};
use tracing::{info, warn};

/// A precise tick loop that fires at a configurable interval.
///
/// Uses 	okio::time::interval with MissedTickBehavior::Skip to ensure
/// ticks are not backlogged if the handler takes longer than the interval.
pub struct TickLoop {
    running: Arc<AtomicBool>,
    interval_secs: u64,
    name: String,
}

impl TickLoop {
    /// Create a new tick loop with the given interval.
    pub fn new(name: impl Into<String>, interval_secs: u64) -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            interval_secs,
            name: name.into(),
        }
    }

    /// Start the tick loop. Calls the provided handler on each tick.
    pub fn start<F>(&self, mut handler: F) -> TickHandle
    where F: FnMut(u64) + Send + 'static {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let interval_secs = self.interval_secs;
        let name = self.name.clone();
        let handle = TickHandle { running: running.clone() };

        tokio::spawn(async move {
            let mut tick_count = 0u64;
            let mut ticker = interval(Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            info!("Tick loop '{}' started (interval: {}s)", name, interval_secs);

            while running.load(Ordering::SeqCst) {
                ticker.tick().await;
                tick_count += 1;
                let tick_start = std::time::Instant::now();
                handler(tick_count);
                let elapsed = tick_start.elapsed();
                if elapsed > Duration::from_secs(interval_secs) {
                    warn!("Tick loop '{}' tick #{} took {:?} (exceeded interval)", name, tick_count, elapsed);
                }
            }
            info!("Tick loop '{}' stopped ({} ticks)", name, tick_count);
        });
        handle
    }

    /// Start a tick loop with an async handler.
    pub fn start_async<F, Fut>(&self, handler: F) -> TickHandle
    where F: Fn(u64) -> Fut + Send + 'static, Fut: std::future::Future<Output = ()> + Send {
        self.running.store(true, Ordering::SeqCst);
        let running = self.running.clone();
        let interval_secs = self.interval_secs;
        let name = self.name.clone();
        let handle = TickHandle { running: running.clone() };

        tokio::spawn(async move {
            let mut tick_count = 0u64;
            let mut ticker = interval(Duration::from_secs(interval_secs));
            ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
            info!("Async tick loop '{}' started (interval: {}s)", name, interval_secs);

            while running.load(Ordering::SeqCst) {
                ticker.tick().await;
                tick_count += 1;
                let tick_start = std::time::Instant::now();
                handler(tick_count).await;
                let elapsed = tick_start.elapsed();
                if elapsed > Duration::from_secs(interval_secs) {
                    warn!("Tick loop '{}' tick #{} took {:?} (exceeded interval)", name, tick_count, elapsed);
                }
            }
            info!("Async tick loop '{}' stopped ({} ticks)", name, tick_count);
        });
        handle
    }

    pub fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    pub fn interval_secs(&self) -> u64 { self.interval_secs }
}

/// Handle to control a running tick loop.
#[derive(Clone)]
pub struct TickHandle {
    running: Arc<AtomicBool>,
}

impl TickHandle {
    pub fn stop(&self) { self.running.store(false, Ordering::SeqCst); }
    pub fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
}

/// A timer that fires once at a specific future time.
pub struct OneShotTimer;

impl OneShotTimer {
    pub fn at<F>(target: chrono::DateTime<chrono::Utc>, handler: F) -> tokio::task::JoinHandle<()>
    where F: FnOnce() + Send + 'static {
        let now = chrono::Utc::now();
        let delay = if target > now { (target - now).num_milliseconds().max(0) as u64 } else { 0 };
        tokio::spawn(async move {
            if delay > 0 { tokio::time::sleep(Duration::from_millis(delay)).await; }
            handler();
        })
    }

    pub fn after<F>(duration: Duration, handler: F) -> tokio::task::JoinHandle<()>
    where F: FnOnce() + Send + 'static {
        tokio::spawn(async move {
            tokio::time::sleep(duration).await;
            handler();
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[tokio::test]
    async fn test_tick_loop_fires() {
        let counter = Arc::new(AtomicU64::new(0));
        let cc = counter.clone();
        let loop_ = TickLoop::new("test", 1);
        let handle = loop_.start(move |_| { cc.fetch_add(1, Ordering::SeqCst); });
        tokio::time::sleep(Duration::from_millis(1100)).await;
        handle.stop();
        assert!(counter.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_tick_handle_stops() {
        // Use a longer interval so no tick fires before we stop.
        let counter = Arc::new(AtomicU64::new(0));
        let cc = counter.clone();
        let loop_ = TickLoop::new("stop_test", 10);
        let handle = loop_.start(move |_| { cc.fetch_add(1, Ordering::SeqCst); });
        tokio::time::sleep(Duration::from_millis(100)).await;
        handle.stop();
        let before = counter.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(before, counter.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_one_shot() {
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        OneShotTimer::after(Duration::from_millis(100), move || { f.store(true, Ordering::SeqCst); });
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(fired.load(Ordering::SeqCst));
    }
}
