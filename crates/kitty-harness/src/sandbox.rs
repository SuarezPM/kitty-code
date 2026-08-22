//! Sandbox — real Linux containment for child processes.
//!
//! Replaces the former print-to-stdout stubs ("security theater", audit P0)
//! with kernel enforcement, porting patterns from the user's verified Apohara
//! Catalyst `apohara-sandbox`:
//!
//! | Layer | Mechanism | Origin |
//! |---|---|---|
//! | FS writes outside `fs_allow` | Landlock ruleset: read/execute everywhere, write only under allow-roots | plan (`AccessFs::from_all(ABI)`) |
//! | Syscalls | seccomp-bpf, default `EPERM`, conditional `openat`/`fcntl`/`ioctl`/`clone` rules | Catalyst `profile/linux.rs:110-187` |
//! | PID/mount/user isolation | `unshare(CLONE_NEWUSER\|CLONE_NEWNS\|CLONE_NEWPID)` + uid/gid map writes | Catalyst `namespace.rs:52` |
//! | Memory | `RLIMIT_AS = mem_limit_mb * 1 MiB` | Catalyst `runner/imp.rs` |
//! | Output flood | 16 MiB per-stream capture cap ([`MAX_OUTPUT_BYTES`]) | Catalyst `runner/imp.rs:110` |
//! | Exec failures | propagate to parent as a `spawn()` error | std implements the same CLOEXEC exec-error pipe pattern as Catalyst `runner/imp.rs` |
//!
//! Child-side ordering (each step is irreversible; least-restrictive first,
//! seccomp LAST because it fail-closes everything not allowlisted):
//! `NO_NEW_PRIVS` → `RLIMIT_AS` → namespaces → Landlock → seccomp.
//!
//! Honest scope note: `unshare(CLONE_NEWPID)` moves only *descendants* of the
//! exec'd program into the new PID namespace — the exec'd process itself stays
//! in the host PID namespace (kernel semantics; Catalyst solved this with a
//! double-fork runner that would break `Child::kill()` timeout semantics
//! here). Direct cross-process signaling is covered by the seccomp allowlist
//! instead: `kill`/`tkill`/`ptrace` are not allowlisted, so they fail with
//! `EPERM`.
//!
//! Fail-closed policy: [`Sandbox::apply`] returns an error instead of arming
//! a partial sandbox when the kernel lacks Landlock or unprivileged user
//! namespaces. Tests skip with a clear message rather than fake-pass.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Hard cap on bytes buffered per output stream (stdout / stderr). A runaway
/// or hostile child can no longer OOM the harness by spewing gigabytes into
/// a pipe. Ported from Catalyst `runner/imp.rs::MAX_OUTPUT_BYTES`.
pub const MAX_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("sandbox denied: {path} — {reason}")]
    Denied { path: String, reason: String },
    #[error("sandbox IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Containment could not be established (namespace/seccomp/Landlock failure).
    #[error("sandbox setup failed: {0}")]
    Setup(String),
    /// The kernel cannot provide the requested containment at all.
    #[error("sandbox unsupported on this kernel: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub fs_allow: Vec<PathBuf>,
    pub net_deny: bool,
    pub mem_limit_mb: u64,
    pub cpu_timeout_secs: u64,
}

impl SandboxConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        let ws = workspace.into();
        Self {
            fs_allow: vec![ws, PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 512,
            cpu_timeout_secs: 30,
        }
    }
    pub fn with_fs_allow(mut self, allow: Vec<PathBuf>) -> Self {
        self.fs_allow = allow;
        self
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        let ws = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        Self::new(ws)
    }
}

/// Outcome of a contained run via [`Sandbox::run`].
#[derive(Debug, Clone)]
pub struct SandboxOutcome {
    pub success: bool,
    pub exit_code: Option<i32>,
    #[cfg(unix)]
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// True when stdout or stderr hit [`MAX_OUTPUT_BYTES`] and were truncated.
    pub truncated: bool,
    /// True when the wall-clock budget ([`SandboxConfig::cpu_timeout_secs`])
    /// expired and the child was killed.
    pub timed_out: bool,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct Sandbox {
    config: SandboxConfig,
}

impl Sandbox {
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }
    pub fn is_path_allowed(&self, path: impl AsRef<Path>) -> bool {
        let n = normalize(path.as_ref());
        self.config
            .fs_allow
            .iter()
            .any(|root| n.starts_with(normalize(root)))
    }
    pub fn check_path(&self, path: impl AsRef<Path>) -> Result<(), SandboxError> {
        let p = path.as_ref();
        if self.is_path_allowed(p) {
            Ok(())
        } else {
            Err(SandboxError::Denied {
                path: p.display().to_string(),
                reason: format!("outside sandbox fs_allow={:?}", self.config.fs_allow),
            })
        }
    }

    /// Arm containment on any `tokio` command. The kernel enforcement chain
    /// runs in the forked child just before exec (`pre_exec`); a failure to
    /// establish containment fails the spawn instead of degrading silently.
    ///
    /// Keep calling `Guard::check_chain` BEFORE this — guard policy first,
    /// kernel containment second.
    pub fn apply(&self, cmd: &mut tokio::process::Command) -> Result<(), SandboxError> {
        cmd.env("KITTY_SANDBOX_MEM_MB", self.config.mem_limit_mb.to_string());
        cmd.env(
            "KITTY_SANDBOX_CPU_SECS",
            self.config.cpu_timeout_secs.to_string(),
        );
        cmd.env(
            "KITTY_SANDBOX_NET_DENY",
            if self.config.net_deny { "1" } else { "0" },
        );
        #[cfg(target_os = "linux")]
        {
            // Fail-closed preflight: refuse to spawn a half-sandboxed child.
            let abi = containment::landlock_abi_version();
            if abi == 0 {
                return Err(SandboxError::Unsupported(
                    "Landlock ABI unavailable (needs Linux >= 5.13)".into(),
                ));
            }
            if !containment::unprivileged_userns_available() {
                return Err(SandboxError::Unsupported(
                    "unprivileged user namespaces disabled \
                     (sysctl user.max_user_namespaces / kernel.unprivileged_userns_clone)"
                        .into(),
                ));
            }
            // Resolve allow-roots in the parent so symlink loops/dangling
            // roots surface as errors before any child exists.
            let roots = containment::resolve_allow_roots(&self.config.fs_allow)?;
            let mem_bytes = self.config.mem_limit_mb.saturating_mul(1024 * 1024);
            let net_deny = self.config.net_deny;
            tracing::debug!(
                landlock_abi = abi,
                ?roots,
                net_deny,
                mem_bytes,
                "sandbox containment armed"
            );
            // SAFETY: the closure executes only inside the forked child after
            // `fork(2)` and before `execve(2)`. It mutates nothing shared with
            // the parent (the captured values are owned clones) and never
            // returns into the async runtime: on success std immediately
            // execs, on failure we return Err and std `_exit`s the child.
            // Post-fork allocation here follows the same pragmatic trade-off
            // as std's own child-side env handling.
            unsafe {
                cmd.pre_exec(move || {
                    containment::contain_child(mem_bytes, &roots, net_deny)
                        .map_err(std::io::Error::other)
                });
            }
        }
        #[cfg(not(target_os = "linux"))]
        tracing::debug!(
            "sandbox stub (non-linux): fs_allow={:?} net_deny={}",
            self.config.fs_allow,
            self.config.net_deny
        );
        Ok(())
    }

    /// Spawn `program args` under full containment and collect capped output.
    ///
    /// Exec failures propagate to the parent as [`SandboxError::Io`] from the
    /// spawn itself — std reports them over its internal CLOEXEC exec-error
    /// pipe (same pattern as Catalyst's hand-rolled runner).
    pub async fn run(
        &self,
        program: impl AsRef<Path>,
        args: &[&str],
        workdir: Option<&Path>,
    ) -> Result<SandboxOutcome, SandboxError> {
        let started = Instant::now();
        let mut cmd = tokio::process::Command::new(program.as_ref());
        cmd.args(args);
        if let Some(dir) = workdir {
            cmd.current_dir(dir);
        }
        use std::process::Stdio;
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        self.apply(&mut cmd)?;

        let mut child = cmd.spawn().map_err(SandboxError::Io)?;
        let out_pipe = child
            .stdout
            .take()
            .ok_or_else(|| SandboxError::Setup("child stdout not piped".into()))?;
        let err_pipe = child
            .stderr
            .take()
            .ok_or_else(|| SandboxError::Setup("child stderr not piped".into()))?;
        let drain = async { tokio::join!(read_capped(out_pipe), read_capped(err_pipe)) };
        let budget = Duration::from_secs(self.config.cpu_timeout_secs);
        let mut timed_out = false;
        let ((stdout, out_trunc), (stderr, err_trunc)) =
            match tokio::time::timeout(budget, drain).await {
                Ok(pair) => pair,
                Err(_elapsed) => {
                    timed_out = true;
                    ((String::new(), false), (String::new(), false))
                }
            };
        if timed_out {
            let _ = child.start_kill();
        }
        let status = child.wait().await?;
        Ok(SandboxOutcome {
            success: !timed_out && status.success(),
            exit_code: status.code(),
            #[cfg(unix)]
            signal: {
                use std::os::unix::process::ExitStatusExt;
                status.signal()
            },
            stdout,
            stderr,
            truncated: out_trunc || err_trunc,
            timed_out,
            duration_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// Drain a stream to EOF, keeping at most [`MAX_OUTPUT_BYTES`]. Anything
/// beyond the cap is discarded and reported via the returned flag.
async fn read_capped<R>(stream: R) -> (String, bool)
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let mut reader = stream;
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let mut chunk = [0u8; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                if buf.len() + n <= MAX_OUTPUT_BYTES {
                    buf.extend_from_slice(&chunk[..n]);
                } else if buf.len() < MAX_OUTPUT_BYTES {
                    buf.extend_from_slice(&chunk[..MAX_OUTPUT_BYTES - buf.len()]);
                    truncated = true;
                } else {
                    truncated = true;
                }
            }
            // Treat a mid-stream read error as EOF-with-truncation: the
            // caller still gets whatever prefix was captured.
            Err(_) => {
                truncated = true;
                break;
            }
        }
    }
    (String::from_utf8_lossy(&buf).into_owned(), truncated)
}

fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        if let std::path::Component::CurDir = comp {
            continue;
        }
        out.push(comp.as_os_str());
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    }
}

#[cfg(target_os = "linux")]
mod containment {
    use super::SandboxError;
    use nix::sched::{unshare, CloneFlags};
    use nix::unistd::{getgid, getuid};
    use serde_json::{json, Map, Value};
    use std::ffi::OsString;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    /// Maximum symlink hops before declaring a loop. Matches Catalyst
    /// apohara-pathsafety `MAX_SYMLINK_HOPS` (POSIX minimum).
    pub(super) const MAX_SYMLINK_HOPS: u32 = 32;

    /// Why an allow-root could not be resolved for Landlock binding.
    enum ResolveError {
        /// Hostile config — hard error, refuse to arm.
        Loop(PathBuf),
        /// Missing/dangling root — warn and skip (kernel needs existing paths).
        Skip(String),
    }

    /// Port of Catalyst apohara-pathsafety `canonicalize_recursive`: resolve
    /// symlinks hop-by-hop with a hard hop cap, then canonicalize.
    fn canonicalize_recursive(path: &Path) -> Result<PathBuf, ResolveError> {
        let mut current = path.to_path_buf();
        let mut hops: u32 = 0;
        loop {
            let meta = std::fs::symlink_metadata(&current).map_err(|e| {
                if hops > 0 {
                    ResolveError::Skip(format!(
                        "dangling symlink {} -> {}: {e}",
                        path.display(),
                        current.display()
                    ))
                } else {
                    ResolveError::Skip(format!("stat {}: {e}", path.display()))
                }
            })?;
            if !meta.file_type().is_symlink() {
                break;
            }
            if hops >= MAX_SYMLINK_HOPS {
                return Err(ResolveError::Loop(path.to_path_buf()));
            }
            let target = std::fs::read_link(&current)
                .map_err(|e| ResolveError::Skip(format!("read_link {}: {e}", current.display())))?;
            current = if target.is_absolute() {
                target
            } else {
                current
                    .parent()
                    .map(|p| p.join(&target))
                    .unwrap_or_else(|| PathBuf::from(OsString::from(target)))
            };
            hops += 1;
        }
        std::fs::canonicalize(path)
            .map_err(|e| ResolveError::Skip(format!("canonicalize {}: {e}", path.display())))
    }

    /// Canonicalize every `fs_allow` root for Landlock binding. Symlinked
    /// allow-roots bind their REAL target hierarchy (stricter than lexical
    /// matching). A symlink loop is a hostile config and aborts arming.
    pub(super) fn resolve_allow_roots(allow: &[PathBuf]) -> Result<Vec<PathBuf>, SandboxError> {
        let mut out = Vec::with_capacity(allow.len());
        for root in allow {
            match canonicalize_recursive(root) {
                Ok(p) => out.push(p),
                Err(ResolveError::Loop(p)) => {
                    return Err(SandboxError::Setup(format!(
                        "fs_allow root symlink loop past {MAX_SYMLINK_HOPS} hops: {}",
                        p.display()
                    )));
                }
                Err(ResolveError::Skip(msg)) => {
                    tracing::warn!(root = %root.display(), %msg,
                        "sandbox fs_allow root skipped (no Landlock rule added)");
                }
            }
        }
        Ok(out)
    }

    /// Full child-side containment chain. Runs inside the forked child right
    /// before exec. Order matters: seccomp LAST (fail-closed from then on).
    pub(super) fn contain_child(
        mem_bytes: u64,
        allow_roots: &[PathBuf],
        net_deny: bool,
    ) -> Result<(), String> {
        set_no_new_privs()?;
        apply_rlimit_as(mem_bytes)?;
        enter_isolated_namespaces()?;
        apply_landlock(allow_roots)?;
        install_seccomp(net_deny)?;
        Ok(())
    }

    fn set_no_new_privs() -> Result<(), String> {
        // SAFETY: prctl with a constant option; affects only this process.
        let rc = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        if rc != 0 {
            return Err(format!(
                "prctl(PR_SET_NO_NEW_PRIVS): {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    fn apply_rlimit_as(mem_bytes: u64) -> Result<(), String> {
        let limit = libc::rlimit {
            rlim_cur: mem_bytes,
            rlim_max: mem_bytes,
        };
        // SAFETY: setrlimit with a stack-allocated valid rlimit struct.
        let rc = unsafe { libc::setrlimit(libc::RLIMIT_AS, &limit) };
        if rc != 0 {
            return Err(format!(
                "setrlimit(RLIMIT_AS, {mem_bytes}): {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    /// Enter user+mount+PID namespaces without root. Ported from Catalyst
    /// `namespace.rs::enter_isolated_namespaces`. The PID namespace applies
    /// to future children (see module docs for the honest scope note).
    fn enter_isolated_namespaces() -> Result<(), String> {
        let host_uid = getuid().as_raw();
        let host_gid = getgid().as_raw();

        let flags = CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWNS | CloneFlags::CLONE_NEWPID;

        unshare(flags).map_err(|e| {
            format!(
                "unshare(CLONE_NEWUSER|CLONE_NEWNS|CLONE_NEWPID) failed: {e}. \
                 Hint: requires kernel.unprivileged_userns_clone=1 and \
                 user.max_user_namespaces > 0 (sysctl)."
            )
        })?;

        write_proc_self("setgroups", "deny")?;
        write_proc_self("uid_map", &format!("0 {host_uid} 1"))?;
        write_proc_self("gid_map", &format!("0 {host_gid} 1"))?;
        Ok(())
    }

    /// `/proc/self/<name>` mapping files accept exactly one write; use
    /// truncate-not-append (Catalyst `namespace.rs::write_proc_self`).
    fn write_proc_self(name: &str, contents: &str) -> Result<(), String> {
        let path = format!("/proc/self/{name}");
        let mut f = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("open {path}: {e}"))?;
        f.write_all(contents.as_bytes())
            .map_err(|e| format!("write {path}: {e}"))
    }

    /// Landlock ruleset: handle ALL V1 FS accesses, grant read/execute on `/`
    /// and read-write only under each resolved allow-root. Fails closed when
    /// the result would not be fully enforced.
    fn apply_landlock(allow_roots: &[PathBuf]) -> Result<(), String> {
        use landlock::{
            Access, AccessFs, PathBeneath, PathFd, Ruleset, RulesetAttr, RulesetCreatedAttr,
            RulesetStatus, ABI,
        };
        let abi = ABI::V1;
        let all = AccessFs::from_all(abi);
        let read_only = AccessFs::from_read(abi);

        let created = Ruleset::default()
            .handle_access(all)
            .map_err(|e| format!("landlock handle_access: {e}"))?
            .create()
            .map_err(|e| format!("landlock create ruleset: {e}"))?;

        let mut created = created
            .add_rule(PathBeneath::new(
                PathFd::new("/").map_err(|e| format!("landlock PathFd(/): {e}"))?,
                read_only,
            ))
            .map_err(|e| format!("landlock add_rule(/): {e}"))?;

        for root in allow_roots {
            let fd = PathFd::new(root)
                .map_err(|e| format!("landlock PathFd({}): {e}", root.display()))?;
            created = created
                .add_rule(PathBeneath::new(fd, all))
                .map_err(|e| format!("landlock add_rule({}): {e}", root.display()))?;
        }

        let status = created
            .restrict_self()
            .map_err(|e| format!("landlock restrict_self: {e}"))?;
        match status.ruleset {
            RulesetStatus::FullyEnforced => Ok(()),
            other => Err(format!("landlock not fully enforced: {other:?}")),
        }
    }

    /// Raw probe of the kernel Landlock ABI version
    /// (`landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION)`).
    /// Returns 0 when unsupported.
    pub(super) fn landlock_abi_version() -> u32 {
        const LANDLOCK_CREATE_RULESET_VERSION: libc::c_ulong = 1;
        // SAFETY: probe syscall with NULL/zero arguments; the kernel only
        // returns the supported ABI version and touches no user memory.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_landlock_create_ruleset,
                std::ptr::null::<libc::c_void>(),
                0usize,
                LANDLOCK_CREATE_RULESET_VERSION,
            )
        };
        if rc >= 1 {
            rc as u32
        } else {
            0
        }
    }

    /// Cheap sysctl check for unprivileged user-namespace availability.
    pub(super) fn unprivileged_userns_available() -> bool {
        let max = std::fs::read_to_string("/proc/sys/user/max_user_namespaces")
            .ok()
            .and_then(|s| s.trim().parse::<i64>().ok())
            .unwrap_or(0);
        if max <= 0 {
            return false;
        }
        // Debian-family hard switch; file absent means the distro allows it.
        match std::fs::read_to_string("/proc/sys/kernel/unprivileged_userns_clone") {
            Ok(s) => s.trim() != "0",
            Err(_) => true,
        }
    }

    /// Pure-allow syscall list: Catalyst WorkspaceWrite tier
    /// (`profile/syscalls.rs` READONLY ∪ WORKSPACE_WRITE additions) plus a
    /// small documented availability set (`futex`, `clock_getres`,
    /// `sched_getaffinity`, `restart_syscall`, `access`, `getpgrp`,
    /// `setpgid`, `getgroups`, `setsid`, `umask`) — read-only or
    /// strictly-self-scoped calls modern glibc/dash/python expect to
    /// succeed; blocking them buys no privilege containment and risks
    /// spurious EPERM failures (dash aborts on `getpgrp` EPERM).
    /// Network syscalls are NOT here; they are appended only when
    /// `net_deny == false`. Everything unlisted fails with EPERM.
    const SECCOMP_PURE_ALLOW: &[&str] = &[
        // Read I/O
        "read",
        "pread64",
        "readv",
        "preadv2",
        // FD management
        "close",
        "dup",
        "dup2",
        "dup3",
        "lseek",
        // Memory
        "mmap",
        "munmap",
        "mremap",
        "brk",
        "mprotect",
        "madvise",
        "futex",
        // Process info
        "getpid",
        "getppid",
        "gettid",
        "getuid",
        "getgid",
        "geteuid",
        "getegid",
        "getcwd",
        "prlimit64",
        "sched_getaffinity",
        "tgkill",
        // Process group / session info (self-scoped; dash needs getpgrp)
        "getpgrp",
        "setpgid",
        "getgroups",
        "setsid",
        "umask",
        // Signals
        "rt_sigprocmask",
        "rt_sigaction",
        "rt_sigreturn",
        "sigaltstack",
        // Time
        "clock_gettime",
        "clock_getres",
        "gettimeofday",
        "nanosleep",
        "clock_nanosleep",
        // Entropy
        "getrandom",
        // Process control (see Catalyst notes: prctl stays unconstrained;
        // NO_NEW_PRIVS is sticky and disables ptrace escalation vectors)
        "prctl",
        "arch_prctl",
        // Exit
        "exit",
        "exit_group",
        "restart_syscall",
        // Stat family
        "newfstatat",
        "statx",
        "fstat",
        "lstat",
        "faccessat",
        "faccessat2",
        "access",
        "readlinkat",
        "fstatfs",
        "statfs",
        // Process startup + spawn (clone handled conditionally below;
        // clone3 denied outright — flags live behind a pointer seccomp
        // cannot dereference, glibc falls back to clone(2))
        "execve",
        "execveat",
        "wait4",
        "waitid",
        "set_robust_list",
        "rseq",
        "set_tid_address",
        "uname",
        // Write I/O
        "write",
        "pwrite64",
        "writev",
        "pwritev2",
        // Path mutation (path-level policy comes from Landlock above)
        "creat",
        "mkdirat",
        "unlinkat",
        "renameat2",
        "linkat",
        "symlinkat",
        // Truncation
        "ftruncate",
        "truncate",
        // Metadata
        "fchmodat",
        "chmod",
        "utimensat",
        // Pipes
        "pipe2",
        // Working directory
        "fchdir",
        "chdir",
        // Directory iteration
        "getdents64",
        "getdents",
        // File copy primitives
        "copy_file_range",
        "sendfile",
        // Ownership
        "fchown",
        "fchownat",
        "lchown",
        // Storage hints / sync
        "fallocate",
        "fsync",
        "fdatasync",
        "sync_file_range",
    ];

    /// Socket-family syscalls, allowed ONLY when `net_deny == false`.
    const NET_SYSCALLS: &[&str] = &[
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "recvfrom",
        "sendmsg",
        "recvmsg",
        "shutdown",
        "getsockname",
        "getpeername",
        "setsockopt",
        "getsockopt",
    ];

    /// Build the seccompiler JSON spec (Catalyst `profile/linux.rs`
    /// `build_json_spec` shape): mismatch = `errno(EPERM)` fail-closed,
    /// match = allow, with argument-constrained conditional rules.
    pub(super) fn build_seccomp_json_spec(net_deny: bool) -> Value {
        let mut filter_rules: Vec<Value> = Vec::new();

        for syscall in SECCOMP_PURE_ALLOW {
            filter_rules.push(json!({ "syscall": syscall }));
        }
        if net_deny {
            // Socket family deliberately absent -> default EPERM.
        } else {
            for syscall in NET_SYSCALLS {
                filter_rules.push(json!({ "syscall": syscall }));
            }
        }

        // openat unconditional: FS write policy is enforced by Landlock.
        filter_rules.push(json!({ "syscall": "openat" }));

        // fcntl narrowed to safe cmds (+F_SETFD: own-fd flag flips only).
        for cmd in [
            libc::F_GETFL,
            libc::F_SETFL,
            libc::F_DUPFD,
            libc::F_DUPFD_CLOEXEC,
            libc::F_SETFD,
        ] {
            filter_rules.push(json!({
                "syscall": "fcntl",
                "args": [{
                    "index": 1,
                    "type": "dword",
                    "op": "eq",
                    "val": cmd as u64,
                }]
            }));
        }

        // ioctl narrowed to safe requests.
        for req in [libc::TIOCGWINSZ, libc::FIOCLEX, libc::FIONCLEX] {
            filter_rules.push(json!({
                "syscall": "ioctl",
                "args": [{
                    "index": 1,
                    "type": "dword",
                    "op": "eq",
                    "val": req,
                }]
            }));
        }

        // clone(flags, ...) allowed only WITHOUT any CLONE_NEW* bit — blocks
        // nested-userns escape vectors. Mask mirrors Catalyst linux.rs:
        //   NEWNS|NEWCGROUP|NEWUTS|NEWIPC|NEWUSER|NEWPID|NEWNET = 0x7E02_0000
        let clone_ns_mask: u64 = 0x7E02_0000;
        filter_rules.push(json!({
            "syscall": "clone",
            "args": [{
                "index": 0,
                "type": "dword",
                "op": { "masked_eq": clone_ns_mask },
                "val": 0u64,
            }]
        }));

        let mut main_thread = Map::new();
        main_thread.insert(
            "mismatch_action".into(),
            json!({ "errno": libc::EPERM as u32 }),
        );
        main_thread.insert("match_action".into(), Value::String("allow".into()));
        main_thread.insert("filter".into(), Value::Array(filter_rules));

        let mut root = Map::new();
        root.insert("main_thread".into(), Value::Object(main_thread));
        Value::Object(root)
    }

    /// Compile the spec for this arch and apply it to the calling thread.
    /// Must be the LAST containment step (Catalyst ordering).
    fn install_seccomp(net_deny: bool) -> Result<(), String> {
        let arch: seccompiler::TargetArch = std::env::consts::ARCH
            .try_into()
            .map_err(|e| format!("seccomp target arch: {e}"))?;
        let spec = build_seccomp_json_spec(net_deny);
        let bytes =
            serde_json::to_vec(&spec).map_err(|e| format!("seccomp spec serialize: {e}"))?;
        let map = seccompiler::compile_from_json(bytes.as_slice(), arch)
            .map_err(|e| format!("seccomp compile_from_json: {e}"))?;
        let prog = map
            .get("main_thread")
            .ok_or_else(|| "seccomp compiled map missing 'main_thread'".to_string())?;
        seccompiler::apply_filter(prog).map_err(|e| format!("seccomp apply_filter: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws_config() -> SandboxConfig {
        SandboxConfig {
            fs_allow: vec![PathBuf::from("/tmp/ws"), PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 512,
            cpu_timeout_secs: 30,
        }
    }

    #[test]
    fn allow_workspace_write() {
        let sb = Sandbox::new(ws_config());
        assert!(sb.is_path_allowed("/tmp/ws/file.txt"));
        assert!(sb.check_path("/tmp/ws/file.txt").is_ok());
        assert!(sb.is_path_allowed("/tmp/ws/sub/inner.rs"));
        assert!(sb.is_path_allowed("/tmp/other.txt"));
    }

    #[test]
    fn deny_outside() {
        let sb = Sandbox::new(ws_config());
        assert!(!sb.is_path_allowed("/etc/passwd"));
        assert!(matches!(
            sb.check_path("/etc/passwd").unwrap_err(),
            SandboxError::Denied { .. }
        ));
        assert!(!sb.is_path_allowed("/home/user/evil.sh"));
        let strict = Sandbox::new(
            SandboxConfig::new("/tmp/ws").with_fs_allow(vec![PathBuf::from("/tmp/ws")]),
        );
        assert!(!strict.is_path_allowed("/tmp/ws-evil/file"));
        assert!(strict.check_path("/tmp/ws-evil/file").is_err());
    }

    #[tokio::test]
    async fn apply_wraps_command() {
        let cfg = SandboxConfig {
            fs_allow: vec![PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 256,
            cpu_timeout_secs: 10,
        };
        let sb = Sandbox::new(cfg);
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("echo hi");
        sb.apply(&mut cmd).expect("apply must succeed");
        let out = cmd.output().await.expect("command should run");
        assert!(
            out.status.success(),
            "sandboxed sh failed: {:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[tokio::test]
    async fn sandbox_env_hints_set() {
        let sb = Sandbox::new(SandboxConfig {
            fs_allow: vec![PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 256,
            cpu_timeout_secs: 10,
        });
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg("echo $KITTY_SANDBOX_MEM_MB:$KITTY_SANDBOX_CPU_SECS:$KITTY_SANDBOX_NET_DENY");
        sb.apply(&mut cmd).expect("apply must succeed");
        let out = cmd.output().await.expect("command should run");
        assert!(
            out.status.success(),
            "sandboxed sh failed: {:?} stderr={}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr)
        );
        let txt = String::from_utf8_lossy(&out.stdout);
        assert!(txt.contains("256:10:1"), "got: {txt}");
    }

    /// SAFETY-contract witness (runs OUTSIDE Miri): sandbox types cross
    /// await points and threads in the agent loop, so they must stay
    /// Send + Sync. The unsafe pre_exec closure captures only owned,
    /// Send data (u64 / bool / Vec<PathBuf>).
    #[test]
    fn sandbox_types_are_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        _assert_send_sync::<Sandbox>();
        _assert_send_sync::<SandboxConfig>();
        _assert_send_sync::<SandboxOutcome>();
        _assert_send_sync::<SandboxError>();
    }
}

/// Pure-Rust unit tests — the ONLY scope Miri executes
/// (`cargo miri test -p kitty-harness --lib sandbox::pure`). Fork/exec/
/// seccomp/landlock paths are excluded because Miri cannot emulate those
/// syscalls; they are covered by the behavioral tests instead.
#[cfg(all(test, target_os = "linux"))]
mod pure {
    use super::containment::build_seccomp_json_spec;
    use super::{normalize, MAX_OUTPUT_BYTES};
    use std::path::Path;

    #[test]
    fn seccomp_spec_is_fail_closed_eperm() {
        let spec = build_seccomp_json_spec(true);
        let main = spec.get("main_thread").expect("main_thread present");
        assert_eq!(
            main.get("match_action").and_then(|v| v.as_str()),
            Some("allow")
        );
        let errno = main
            .get("mismatch_action")
            .and_then(|v| v.get("errno"))
            .and_then(|v| v.as_u64())
            .expect("mismatch must be errno");
        assert_eq!(errno, libc::EPERM as u64);
    }

    #[test]
    fn seccomp_net_deny_toggles_socket_rules() {
        let names = |net_deny: bool| {
            let spec = build_seccomp_json_spec(net_deny);
            let filter = spec
                .pointer("/main_thread/filter")
                .and_then(|v| v.as_array())
                .expect("filter array");
            filter
                .iter()
                .filter_map(|r| r.get("syscall").and_then(|s| s.as_str()))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let denied = names(true);
        assert!(
            !denied.iter().any(|s| s == "socket"),
            "net_deny must omit socket"
        );
        assert!(
            !denied.iter().any(|s| s == "connect"),
            "net_deny must omit connect"
        );
        assert!(
            !denied.iter().any(|s| s == "kill"),
            "kill must never be allowlisted"
        );
        assert!(
            !denied.iter().any(|s| s == "ptrace"),
            "ptrace must never be allowlisted"
        );
        assert!(denied.iter().any(|s| s == "openat"));
        let allowed = names(false);
        assert!(allowed.iter().any(|s| s == "socket"));
        assert!(allowed.iter().any(|s| s == "connect"));
    }

    #[test]
    fn seccomp_clone_blocks_namespace_bits() {
        let spec = build_seccomp_json_spec(true);
        let filter = spec
            .pointer("/main_thread/filter")
            .unwrap()
            .as_array()
            .unwrap();
        let clone = filter
            .iter()
            .find(|r| r.get("syscall").and_then(|s| s.as_str()) == Some("clone"))
            .expect("conditional clone rule present");
        let op = clone
            .pointer("/args/0/op/masked_eq")
            .and_then(|v| v.as_u64());
        assert_eq!(
            op,
            Some(0x7E02_0000),
            "(flags & CLONE_NEW*) == 0 constraint"
        );
        assert_eq!(
            clone.pointer("/args/0/val").and_then(|v| v.as_u64()),
            Some(0)
        );
    }

    #[test]
    fn output_cap_constant_is_16mib() {
        assert_eq!(MAX_OUTPUT_BYTES, 16 * 1024 * 1024);
    }

    #[test]
    fn normalize_ignores_curdir_components() {
        assert_eq!(normalize(Path::new("./a/./b")), Path::new("a/b"));
        assert_eq!(normalize(Path::new(".")), Path::new("."));
    }
}
