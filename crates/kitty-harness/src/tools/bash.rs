//! Bash tool + containment stack (W2 split of `tools.rs`, por símbolo).
//!
//! `Guard::check_chain` policy, `Sandbox::apply` containment, hard timeout.
use super::{ToolError, ToolRegistry, ToolResult};
use crate::guard::{Guard, GuardVerdict};
use crate::redact::redact_secrets;
use crate::sandbox::{Sandbox, SandboxError};
use std::path::Path;
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

impl ToolRegistry {
    /// Execute a Bash tool call.
    ///
    /// The command runs under `sh -c` with a hard timeout (F1.3): a command
    /// that outlives `bash_timeout_secs` is killed and reported as a failed
    /// tool result (`success: false`), never as a fatal error propagating out
    /// of the agent loop.
    ///
    /// Containment is layered: `Guard::check_chain` (policy) runs first, then
    /// `Sandbox::apply` arms Landlock/seccomp/namespaces/RLIMIT_AS on the
    /// child just before exec. Sandbox setup failures are fail-closed: the
    /// command is NOT executed and a [`ToolError::Sandbox`] with an actionable
    /// message is returned. Setting `KITTY_SANDBOX=off` in the environment
    /// disables containment explicitly — loudly logged on every activation,
    /// never silent.
    pub(crate) async fn execute_bash(
        &self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        // BenchPolicy primero (cat bench/... denegado aunque Guard lo
        // consideraría SAFE_OUTPUT Allow).
        if let Some(reason) = crate::guard::bench_policy_check("Bash", args) {
            return Err(ToolError::BenchDenied(reason));
        }
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Bash".to_string(),
                reason: "missing 'command'".to_string(),
            })?;
        self.run_bash_command(command, Duration::from_secs(self.bash_timeout_secs))
            .await
    }

    /// Run a shell command under the same containment stack as the Bash tool
    /// (`Guard::check_chain` policy → `Sandbox::apply` kernel containment →
    /// hard timeout), inheriting the process working directory. Used by the
    /// Bash tool itself.
    pub async fn run_bash_command(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<ToolResult, ToolError> {
        self.run_bash_command_in(command, timeout, None).await
    }

    /// Like [`Self::run_bash_command`] but optionally executing `command` in
    /// `dir`. The verification loop passes the workspace so project test
    /// commands (`make test`, `cargo test`, …) resolve the right project.
    pub async fn run_bash_command_in(
        &self,
        command: &str,
        timeout: Duration,
        dir: Option<&Path>,
    ) -> Result<ToolResult, ToolError> {
        // Guard: check the full chain (W0: timed into the turn's
        // `guard_ms` accumulator — behavior unchanged, policy first).
        // The span lives in a sync scope (dropped before any await), so
        // no `entered()` guard ever crosses an await point.
        let guard_start = Instant::now();
        let verdicts = {
            let _guard_span = tracing::debug_span!("guard_check").entered();
            Guard::check_chain(command)
        };
        self.turn_guard_ms
            .fetch_add(guard_start.elapsed().as_millis() as u64, Ordering::Relaxed);
        for verdict in &verdicts {
            match verdict {
                GuardVerdict::Block(reason) => {
                    return Err(ToolError::BashBlocked(reason.clone()));
                }
                GuardVerdict::Warn(reason) => {
                    // For now, treat warnings as blocks too (no interactive confirmation)
                    return Err(ToolError::BashBlocked(format!(
                        "{reason} (interactive confirmation not available)"
                    )));
                }
                GuardVerdict::Allow => {}
            }
        }

        // Kernel containment AFTER guard policy (ordering contract documented
        // at `Sandbox::apply`). `KITTY_SANDBOX=off` is honored ONLY when set;
        // every activation logs a warning so an unsandboxed run is never
        // silent.
        let sandbox: Option<&Sandbox> = match (&self.bash_sandbox, crate::config::sandbox_off()) {
            (Some(_), true) => {
                tracing::warn!(
                    "KITTY_SANDBOX=off: Bash command runs WITHOUT kernel \
                         containment (explicit opt-out)"
                );
                None
            }
            (sb_opt, _) => sb_opt.as_ref(),
        };

        // Execute the command with a timeout. The child is spawned in its own
        // process group via `setsid()` so we can kill the entire tree on timeout.
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .kill_on_drop(false)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = dir {
            cmd.current_dir(dir);
        }
        if let Some(sb) = sandbox {
            sb.apply(&mut cmd).map_err(|e| match e {
                SandboxError::Unsupported(_) | SandboxError::Setup(_) => {
                    ToolError::Sandbox(format!(
                        "{e}; command NOT executed (fail-closed). Fix the host \
                         (needs Linux >= 5.13 with unprivileged user namespaces) \
                         or set KITTY_SANDBOX=off to disable containment explicitly"
                    ))
                }
                other => ToolError::Sandbox(other.to_string()),
            })?;
        }
        // F3.1: Create a new process group so we can kill the entire tree on
        // timeout. The hook runs inside the forked child before exec.
        #[cfg(unix)]
        unsafe {
            cmd.pre_exec(|| {
                let ret = nix::libc::setsid();
                if ret == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        // When containment is armed, a spawn failure usually means the
        // kernel refused the namespace/seccomp chain at exec time even
        // though the cheap preflight passed (e.g. Docker's default
        // seccomp profile blocks unshare(CLONE_NEWUSER) while the sysctl
        // files still look permissive). Surface that as an actionable
        // sandbox error instead of a bare errno.
        let child = cmd.spawn().map_err(|e| {
            if sandbox.is_some() {
                ToolError::Sandbox(format!(
                    "spawn failed while arming containment: {e}. This host \
                     or container likely blocks user namespaces; set \
                     KITTY_SANDBOX=off to disable containment explicitly"
                ))
            } else {
                ToolError::Io(e)
            }
        })?;
        // Save PID before consuming child with wait_with_output.
        let child_pid = child.id();

        // W0: child-execution wall (the `SandboxOutcome::duration_ms`
        // quantity) into the turn's `sandbox_ms` accumulator. Measured
        // around spawn+wait regardless of containment, so `sandbox_ms` is
        // the sandbox-stage wall including containment overhead when armed.
        let exec_start = Instant::now();
        let output = match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => {
                self.turn_sandbox_ms
                    .fetch_add(exec_start.elapsed().as_millis() as u64, Ordering::Relaxed);
                return Err(ToolError::Io(e));
            }
            Err(_elapsed) => {
                // F3.1: Explicitly kill the entire process group on timeout.
                // SIGTERM first for graceful shutdown, then SIGKILL after 2s grace.
                #[cfg(unix)]
                if let Some(id) = child_pid {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let pgid = Pid::from_raw(-(id as i32)); // negative = process group
                    let _ = kill(pgid, Signal::SIGTERM);
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    let _ = kill(pgid, Signal::SIGKILL);
                }
                self.turn_sandbox_ms
                    .fetch_add(exec_start.elapsed().as_millis() as u64, Ordering::Relaxed);
                return Ok(ToolResult {
                    success: false,
                    output: format!("command timed out after {}s (killed)", timeout.as_secs()),
                    retry: false,
                });
            }
        };
        self.turn_sandbox_ms
            .fetch_add(exec_start.elapsed().as_millis() as u64, Ordering::Relaxed);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n--- stderr ---\n");
            }
            result.push_str(&stderr);
        }

        // Truncate very long output (F4.1: profile-driven cap).
        let max_output = self.bash_output_cap;
        if result.len() > max_output {
            result.truncate(max_output);
            result.push_str("\n... (output truncated)");
        }

        // F3.2: Redact secrets before returning output to the agent.
        let result = redact_secrets(&result);

        Ok(ToolResult {
            success: output.status.success(),
            output: result,
            retry: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ToolError, ToolRegistry};
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn bash_blocks_dangerous_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "rm -rf /"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    /// W0: `execute_bash` records guard + sandbox wall time into the
    /// turn accumulators (and `take_turn_timing` resets them).
    #[tokio::test]
    async fn w0_bash_records_guard_and_sandbox_timing() {
        let registry = ToolRegistry::new();
        registry.reset_turn_timing();
        // 50ms sleep: comfortably above ms resolution + scheduling jitter,
        // far below any timeout. `sleep` passes Guard (see the F1.3 timeout
        // tests) and `nanosleep` is seccomp-allowlisted under containment.
        let args = serde_json::json!({"command": "sleep 0.05"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(result.success, "sleep must succeed: {}", result.output);
        let (guard_ms, sandbox_ms) = registry.take_turn_timing();
        assert!(
            sandbox_ms >= 10,
            "50ms sleep must show sandbox wall time, got {sandbox_ms}ms"
        );
        // `guard_ms` is a real ms-truncated measurement of a ~µs check —
        // often 0 for a single command, so it is read but not asserted.
        let _ = guard_ms;
        // Swap-reset discipline: a second read (e.g. the next turn) sees zero.
        assert_eq!(registry.take_turn_timing(), (0, 0));
    }

    /// W0: a Guard-blocked command records guard time but no sandbox time
    /// (no child ever spawns) — and the take still resets.
    #[tokio::test]
    async fn w0_blocked_command_records_no_sandbox_time() {
        let registry = ToolRegistry::new();
        registry.reset_turn_timing();
        let args = serde_json::json!({"command": "rm -rf /"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
        let (_guard_ms, sandbox_ms) = registry.take_turn_timing();
        assert_eq!(sandbox_ms, 0, "blocked commands never spawn a child");
    }

    #[tokio::test]
    async fn bash_allows_safe_commands() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hello"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }

    #[tokio::test]
    async fn bash_command_substitution_blocked() {
        // F1.1 regression: `echo $(...)` must be blocked, not silently allowed.
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo $(rm -rf /)"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[tokio::test]
    async fn bash_timeout_returns_failed_result() {
        // F1.3 regression: a command that hangs must be killed and reported as
        // a failed tool result — never hang forever, never abort the session.
        let registry = ToolRegistry::with_timeout(1); // 1 second
        let args = serde_json::json!({"command": "sleep 10"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("timed out"),
            "got: {}",
            result.output
        );
        assert!(!result.retry);
    }

    #[tokio::test]
    async fn bash_timeout_still_fires_under_sandbox() {
        if !crate::sandbox::containment_available() {
            println!(
                "SKIP: kernel containment unavailable (Landlock/userns) — \
                 cannot verify sandbox wiring honestly"
            );
            return;
        }
        let started = std::time::Instant::now();
        let registry = ToolRegistry::with_timeout(1);
        let args = serde_json::json!({"command": "sleep 10"});
        let result = registry.execute_bash(&args).await.unwrap();
        let elapsed = started.elapsed();
        println!("sandboxed timeout test elapsed: {elapsed:?}");
        assert!(
            !result.success,
            "sleep must be killed, got: {}",
            result.output
        );
        assert!(
            result.output.contains("timed out"),
            "got: {}",
            result.output
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "timeout must fire near the 1s budget, took {elapsed:?} — \
             spawn is blocking on the sandbox exec-error channel"
        );
    }

    #[tokio::test]
    async fn bash_blocks_redirect_write() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "cat > /tmp/evil.txt"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("do not redirect") || msg.contains("Write"),
            "got: {msg}"
        );
    }

    #[tokio::test]
    async fn bash_allows_redirect_to_dev_null() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hi > /dev/null"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(
            result.success || !result.output.contains("blocked"),
            "should allow /dev/null, got: {:?}",
            result
        );
        // If it blocks, it's a failure — but /dev/null should be allowed
        // Verify no BashBlocked error was thrown
    }

    #[tokio::test]
    async fn bash_blocks_tee_write() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo x | tee file.log"});
        let err = registry.execute_bash(&args).await.unwrap_err();
        assert!(matches!(err, ToolError::BashBlocked(_)));
    }

    #[tokio::test]
    async fn bash_allows_tee_without_file() {
        let registry = ToolRegistry::new();
        let args = serde_json::json!({"command": "echo hi | tee"});
        let result = registry.execute_bash(&args).await.unwrap();
        assert!(result.success);
    }

    /// Gate-3 finding 1: execute_bash must arm kernel containment. A
    /// Guard-passing command (`cp`, no redirect/substitution) that writes
    /// OUTSIDE the sandbox fs_allow must FAIL under execute_bash, while the
    /// SAME command succeeds without the sandbox (negative control proving
    /// the denial comes from Landlock, not DAC).

    #[tokio::test]
    async fn bash_write_outside_workspace_denied_by_sandbox_with_negative_control() {
        if !crate::sandbox::containment_available() {
            println!(
                "SKIP: kernel containment unavailable (Landlock/userns) — \
                 cannot verify sandbox wiring honestly"
            );
            return;
        }
        if crate::config::sandbox_off() {
            println!("SKIP: KITTY_SANDBOX=off set — containment intentionally disabled");
            return;
        }

        // Source inside the allow-root; victim OUTSIDE any allow-root
        // (fs_allow below deliberately excludes both tempdirs' parents).
        let allow_root = tempfile::tempdir().unwrap();
        let source = allow_root.path().join("src.txt");
        std::fs::write(&source, "payload").unwrap();
        let victim_dir = tempfile::tempdir().unwrap();
        let victim = victim_dir.path().join("victim.txt");
        std::fs::write(&victim, b"original").unwrap();

        let sb = Sandbox::new(crate::sandbox::SandboxConfig {
            fs_allow: vec![allow_root.path().to_path_buf()],
            net_deny: true,
            mem_limit_mb: 512,
            cpu_timeout_secs: 30,
        });
        let registry = ToolRegistry::new().with_sandbox(sb);

        let command = format!("cp {} {}", source.display(), victim.display());
        let args = serde_json::json!({ "command": command });

        // Sandboxed: cp opens the destination O_WRONLY|O_TRUNC -> needs
        // WriteFile (+Truncate) rights outside fs_allow -> Landlock denies.
        let result = registry.execute_bash(&args).await.unwrap();
        println!(
            "sandboxed cp result: success={} output={}",
            result.success, result.output
        );
        assert!(
            !result.success,
            "CONTAINMENT BROKEN: Guard-passing write outside fs_allow \
             succeeded under execute_bash"
        );
        assert_eq!(
            std::fs::read(&victim).unwrap(),
            b"original",
            "victim file must be untouched by the denied write"
        );

        // Negative control: identical command WITHOUT sandbox succeeds.
        let plain = ToolRegistry::new().without_sandbox();
        let ctl = plain.execute_bash(&args).await.unwrap();
        println!(
            "negative-control cp result: success={} output={}",
            ctl.success, ctl.output
        );
        assert!(
            ctl.success,
            "negative control FAILED — plain cp was already denied \
             (output: {}); denial cannot be attributed to the sandbox",
            ctl.output
        );
        assert_eq!(std::fs::read(&victim).unwrap(), b"payload");
    }
}
