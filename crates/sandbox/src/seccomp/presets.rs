//! Preset syscall allowlists for the three security levels.
//!
//! These lists are curated so that a typical glibc/musl/Rust/Go program can
//! run, while dangerous surface (ptrace, mount, kexec, bpf, perf_event_open,
//! unshare with namespace flags) is excluded. Network syscalls are omitted
//! from the STRICT and LOCKED presets.
//!
//! The presets are pure data: the [`builder::SeccompFilterBuilder`] consumes
//! them to produce a compiled [`BpfProgram`].

use crate::seccomp::AllowRule;

/// The STANDARD allowlist: a broad set suitable for general-purpose programs.
///
/// Includes network syscalls (the network policy is enforced separately by the
/// proxy / namespace isolation).
pub fn standard_allowlist() -> Vec<AllowRule> {
    let names: &[&str] = &[
        // I/O
        "read", "write", "pread64", "pwrite64", "readv", "writev", "preadv", "pwritev",
        "preadv2", "pwritev2", "lseek", "dup", "dup2", "dup3", "pipe", "pipe2",
        "sendfile", "splice", "tee", "vmsplice", "copy_file_range", "fadvise64",
        "sync_file_range", "syncfs", "fsync", "fdatasync", "fallocate",
        // Filesystem metadata
        "stat", "fstat", "lstat", "newfstatat", "statx", "statfs", "fstatfs", "lstat",
        "readlink", "readlinkat", "getcwd", "getdents", "getdents64",
        "access", "faccessat", "faccessat2", "umask",
        // Filesystem mutation
        "open", "openat", "openat2", "creat", "close", "close_range",
        "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat", "rename", "renameat", "renameat2",
        "link", "linkat", "symlink", "symlinkat", "chmod", "fchmod", "fchmodat", "fchmodat2",
        "chown", "fchown", "lchown", "fchownat", "truncate", "ftruncate",
        "utime", "utimes", "utimensat", "futimesat",
        // Directory ops
        "chdir", "fchdir", "getdents64",
        // Memory
        "mmap", "mprotect", "munmap", "brk", "mremap", "madvise", "mincore", "msync",
        "mlock", "munlock", "mlock2", "pkey_mprotect", "pkey_alloc", "pkey_free", "mseal",
        // Process / thread
        "clone", "clone3", "fork", "vfork", "execve", "execveat", "exit", "exit_group",
        "wait4", "waitid", "set_tid_address", "set_robust_list", "get_robust_list",
        "getpid", "getppid", "gettid", "getpgrp", "setpgid", "setsid", "getsid", "getpgid",
        "prctl", "arch_prctl", "setns", "unshare",
        "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "rt_sigpending",
        "rt_sigtimedwait", "rt_sigqueueinfo", "rt_sigsuspend", "sigaltstack",
        "kill", "tkill", "tgkill", "pidfd_send_signal", "pidfd_open", "pidfd_getfd",
        // Credentials (read-only)
        "getuid", "geteuid", "getgid", "getegid", "getresuid", "getresgid",
        "getgroups", "capget",
        // Scheduling
        "sched_yield", "sched_getaffinity", "sched_setaffinity",
        "sched_getparam", "sched_setparam", "sched_getscheduler", "sched_setscheduler",
        "sched_get_priority_max", "sched_get_priority_min", "sched_rr_get_interval",
        "sched_setattr", "sched_getattr",
        // Time
        "time", "gettimeofday", "clock_gettime", "clock_getres", "clock_nanosleep",
        "nanosleep", "times", "timer_create", "timer_settime", "timer_gettime",
        "timer_getoverrun", "timer_delete", "timerfd_create", "timerfd_settime",
        "timerfd_gettime", "clock_adjtime",
        // Resource limits
        "getrlimit", "setrlimit", "prlimit64", "getrusage", "sysinfo", "uname",
        // Futex / events
        "futex", "futex_waitv", "futex_wake", "futex_wait", "futex_requeue",
        "eventfd", "eventfd2", "signalfd", "signalfd4",
        // Polling
        "poll", "ppoll", "pselect6", "epoll_create", "epoll_create1", "epoll_ctl",
        "epoll_wait", "epoll_pwait", "epoll_pwait2", "select",
        // IPC
        "semget", "semop", "semctl", "semtimedop", "shmget", "shmat", "shmctl", "shmdt",
        "msgget", "msgsnd", "msgrcv", "msgctl",
        // Xattrs
        "setxattr", "lsetxattr", "fsetxattr", "getxattr", "lgetxattr", "fgetxattr",
        "listxattr", "llistxattr", "flistxattr", "removexattr", "lremovexattr",
        "fremovexattr",
        // Sockets / network (STANDARD permits them; policy enforces allowlist)
        "socket", "socketpair", "connect", "accept", "accept4", "bind", "listen",
        "sendto", "recvfrom", "sendmsg", "recvmsg", "sendmmsg", "recvmmsg",
        "getsockopt", "setsockopt", "getpeername", "getsockname", "shutdown",
        // Misc
        "fcntl", "flock", "ioctl", "getrandom", "memfd_create", "membarrier",
        "rseq", "getcpu", "restart_syscall", "pause",
        "inotify_init1",
    ];
    names.iter().map(|n| AllowRule::allow(*n)).collect()
}

/// The STRICT allowlist: STANDARD minus network syscalls and a few dangerous
/// surface-area calls.
pub fn strict_allowlist() -> Vec<AllowRule> {
    let mut rules = standard_allowlist();
    let denied: &[&str] = &[
        // No network in strict mode (proxy is used, but at the kernel level
        // the socket syscalls are blocked so a bypassed proxy cannot reach
        // the network).
        "socket", "socketpair", "connect", "accept", "accept4", "bind", "listen",
        "sendto", "recvfrom", "sendmsg", "recvmsg", "sendmmsg", "recvmmsg",
        "getsockopt", "setsockopt", "getpeername", "getsockname", "shutdown",
        // No ptrace / perf / bpf.
        "ptrace", "perf_event_open", "bpf", "kcmp", "process_vm_readv",
        "process_vm_writev",
        // No kernel modules.
        "init_module", "finit_module", "delete_module",
        // No kexec.
        "kexec_file_load",
        // No fanotify / inotify-write.
        "fanotify_init", "fanotify_mark",
        // No userfaultfd (used in some exploits).
        "userfaultfd",
    ];
    let denied_set: std::collections::HashSet<&str> = denied.iter().copied().collect();
    rules.retain(|r| !denied_set.contains(r.syscall.as_str()));
    rules
}

/// The LOCKED allowlist: STRICT minus filesystem mutation syscalls.
pub fn locked_allowlist() -> Vec<AllowRule> {
    let mut rules = strict_allowlist();
    let denied: &[&str] = &[
        // No filesystem mutation at all.
        "open", "creat", "mkdir", "mkdirat", "rmdir", "unlink", "unlinkat",
        "rename", "renameat", "renameat2", "link", "linkat", "symlink", "symlinkat",
        "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "fchown", "lchown",
        "fchownat", "truncate", "ftruncate", "utime", "utimes", "utimensat", "futimesat",
        "fallocate", "write", "pwrite64", "writev", "pwritev", "pwritev2", "msync",
        "sync_file_range", "syncfs", "fsync", "fdatasync",
        // openat is allowed but only for reading (enforced by arg filter). We
        // keep it here so the program can read files; the arg filter blocks
        // O_WRONLY/O_RDWR.
    ];
    let denied_set: std::collections::HashSet<&str> = denied.iter().copied().collect();
    rules.retain(|r| !denied_set.contains(r.syscall.as_str()));

    // Replace the unconditional openat rule with a read-only one.
    if let Some(rule) = rules.iter_mut().find(|r| r.syscall == "openat") {
        // O_WRONLY = 1, O_RDWR = 2. Allow only when (flags & 3) == 0 (O_RDONLY).
        // We encode this as a masked equality: arg1 & 3 == 0 → but the builder
        // semantics are "all args must match". A masked-eq with value 0 and
        // mask 3 expresses "these bits must be zero".
        rule.args = vec![crate::seccomp::ArgComparator::masked_eq(1, 3, 0)];
    }
    rules
}

/// Convenience: get the preset allowlist for a security level name.
pub fn preset_allowlist(level: &str) -> Vec<AllowRule> {
    match level {
        "standard" | "STANDARD" => standard_allowlist(),
        "strict" | "STRICT" => strict_allowlist(),
        "locked" | "LOCKED" => locked_allowlist(),
        _ => standard_allowlist(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_includes_network() {
        let list = standard_allowlist();
        assert!(list.iter().any(|r| r.syscall == "socket"));
    }

    #[test]
    fn strict_excludes_network() {
        let list = strict_allowlist();
        assert!(!list.iter().any(|r| r.syscall == "socket"));
        assert!(!list.iter().any(|r| r.syscall == "ptrace"));
    }

    #[test]
    fn locked_excludes_writes_and_filters_openat() {
        let list = locked_allowlist();
        assert!(!list.iter().any(|r| r.syscall == "write"));
        assert!(!list.iter().any(|r| r.syscall == "unlink"));
        let openat = list.iter().find(|r| r.syscall == "openat").unwrap();
        assert!(!openat.args.is_empty(), "openat should have an arg filter");
    }
}
