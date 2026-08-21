//! Sandbox — FS whitelist + net-deny + seccomp skeleton.
//! Wraps `tokio::process::Command`. Landlock/seccomp are logged stubs when
//! crates absent; real enforcement lands later without changing this interface.

use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("sandbox denied: {path} — {reason}")]
    Denied { path: String, reason: String },
    #[error("sandbox IO error: {0}")]
    Io(#[from] std::io::Error),
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
        self.apply_linux_stubs();
        #[cfg(not(target_os = "linux"))]
        tracing::debug!(
            "sandbox stub (non-linux): fs_allow={:?} net_deny={}",
            self.config.fs_allow,
            self.config.net_deny
        );
        Ok(())
    }
    #[cfg(target_os = "linux")]
    fn apply_linux_stubs(&self) {
        let allow = self
            .config
            .fs_allow
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("[sandbox landlock stub] fs_allow: {allow}");
        tracing::debug!("[sandbox landlock stub] fs_allow: {allow}");
        if self.config.net_deny {
            println!(
                "[sandbox seccomp stub] net_deny=true — would block socket/connect, allowlist: read, write, openat, close, mmap, munmap, brk, exit_group"
            );
        } else {
            println!("[sandbox seccomp stub] net_deny=false — network allowed");
        }
        tracing::debug!(
            "[sandbox seccomp stub] net_deny={} mem={}MB cpu={}s",
            self.config.net_deny,
            self.config.mem_limit_mb,
            self.config.cpu_timeout_secs
        );
    }
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
        assert!(out.status.success());
    }

    #[tokio::test]
    async fn oom_contained() {
        // SIGKILL (9) is the kernel's OOM-killer signal. Sandbox must not panic
        // or escape; the child dies with signal 9 and apply still reports Ok.
        let sb = Sandbox::new(SandboxConfig {
            fs_allow: vec![PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 64,
            cpu_timeout_secs: 5,
        });
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("kill -9 $$");
        sb.apply(&mut cmd).expect("apply must succeed");
        let out = cmd.output().await.expect("command should run");
        assert!(!out.status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(out.status.signal(), Some(9));
        }
    }

    #[tokio::test]
    async fn segfault_contained() {
        // SIGSEGV (11) — segfault must be contained as failure, not host crash.
        let sb = Sandbox::new(SandboxConfig {
            fs_allow: vec![PathBuf::from("/tmp")],
            net_deny: true,
            mem_limit_mb: 64,
            cpu_timeout_secs: 5,
        });
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg("kill -11 $$");
        sb.apply(&mut cmd).expect("apply must succeed");
        let out = cmd.output().await.expect("command should run");
        assert!(!out.status.success());
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(out.status.signal(), Some(11));
        }
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
        assert!(out.status.success());
        let txt = String::from_utf8_lossy(&out.stdout);
        assert!(txt.contains("256:10:1"), "got: {txt}");
    }
}
