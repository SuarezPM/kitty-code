//! Safety guards for the agent loop.
//!
//! Provides permission gates for shell commands and chain analysis.
//! Split (audit 2026-09 Fase 1.2): pattern tables live in `patterns`,
//! parsing/write-target analysis in `shell_parse`, and the interpreter
//! table in `interpreters`.

mod interpreters;
mod patterns;
mod shell_parse;

use interpreters::INTERPRETERS;
pub(crate) use patterns::*;
pub(crate) use shell_parse::*;
pub use shell_parse::{detect_write_targets, is_write_via_redirect};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardVerdict {
    /// Command is safe to run.
    Allow,
    /// Command is dangerous, refuse to execute.
    Block(String),
    /// Command is risky, requires user confirmation.
    Warn(String),
}

/// Detect command substitution in the raw command string.
///
/// `$(` / backticks / process substitution `<(`/`>(` all make the shell
/// **execute** whatever they wrap, even when the outer command is a "safe
/// output" command like `echo` or `cat` (`echo $(rm -rf /)` really runs
/// `rm -rf /`). Variable expansion (`$VAR`, `$5`, `${x}`) is NOT execution and
/// must not trigger this path — only `$(` (which includes `${x:=$(cmd)}` and
/// `$((...))` arithmetic forms that can contain commands) and backticks do.
/// Safety guard that analyzes shell commands.
pub struct Guard;

impl Guard {
    /// Analyze a single shell command for safety.
    ///
    /// Layers in order: 0 raw pre-scan, 1 always-blocked, 1b shell-redirect
    /// write detection, 2 safe output, 3 interpreters, 4 general scan.
    #[allow(clippy::doc_markdown)]
    pub fn check_command(cmd: &str) -> GuardVerdict {
        let name = command_name(cmd).to_lowercase();

        // Layer 0: command substitution anywhere in the raw command means
        // arguments get executed, no matter who the command is.
        if has_command_substitution(cmd) {
            return Self::scan(cmd);
        }

        // Layer 1: hard-blocked commands (eval).
        if ALWAYS_BLOCKED_COMMANDS.contains(&name.as_str()) {
            return GuardVerdict::Block(format!(
                "Blocked: '{name}' is always blocked (arbitrary code execution)"
            ));
        }

        // Layer 1b: shell-redirect write detection (port of _shared/shell-write.ts)
        // Must run BEFORE safe-output allow, because `cat`/`echo` with `>` is a write.
        let targets = detect_write_targets(cmd);
        if !targets.is_empty() {
            return GuardVerdict::Block(format!(
                "use Write for new file / Edit for existing — do not redirect (writes to {})",
                targets.join(", ")
            ));
        }

        // Layer 2: safe output commands.
        if SAFE_OUTPUT_COMMANDS.contains(&name.as_str()) {
            return GuardVerdict::Allow;
        }

        // Layer 3: interpreters.
        if INTERPRETERS.contains(&name.as_str()) {
            return Self::scan(cmd);
        }

        // Layer 4: general command — strip quotes (literal data args) then scan.
        Self::scan(&strip_quotes(cmd))
    }

    /// Scan a command string for blocked/risky patterns (case-insensitive).
    fn scan(cmd: &str) -> GuardVerdict {
        let lowered = cmd.to_lowercase();

        for pattern in BLOCKED_PATTERNS {
            // Refine `dd if=` to not block benign cases:
            // - dd if=/dev/null (reading from null is safe)
            // - dd ... of=/dev/null etc. (writing to null device is safe, not a file)
            if *pattern == "dd if="
                && (lowered.contains("dd if=/dev/null")
                    || lowered.contains("of=/dev/null")
                    || lowered.contains("of=/dev/stdout")
                    || lowered.contains("of=/dev/stderr")
                    || lowered.contains("of=/dev/stdin")
                    || lowered.contains("of=/dev/tty")
                    || lowered.contains("of=/dev/zero"))
            {
                continue;
            }
            if lowered.contains(&pattern.to_lowercase()) {
                return GuardVerdict::Block(format!(
                    "Blocked: command contains dangerous pattern '{pattern}'"
                ));
            }
        }

        for pattern in RISKY_PATTERNS {
            if lowered.contains(&pattern.to_lowercase()) {
                return GuardVerdict::Warn(format!(
                    "Warning: command contains risky pattern '{pattern}'"
                ));
            }
        }

        GuardVerdict::Allow
    }

    /// Analyze a shell chain (commands separated by &&, ||, |, ;).
    /// Each command is judged independently.
    pub fn check_chain(chain: &str) -> Vec<GuardVerdict> {
        let mut results = Vec::new();
        for cmd in split_command_chain(chain) {
            let cmd = cmd.trim();
            if !cmd.is_empty() {
                results.push(Self::check_command(cmd));
            }
        }
        results
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_command_allows() {
        assert_eq!(Guard::check_command("ls -la"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("cat file.rs"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("cargo build"), GuardVerdict::Allow);
    }

    #[test]
    fn blocked_command_detected() {
        match Guard::check_command("rm -rf /") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("dd if=/dev/zero of=/dev/sda") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn risky_command_warns() {
        match Guard::check_command("rm -rf target/") {
            GuardVerdict::Warn(_) => {}
            other => panic!("Expected Warn, got {other:?}"),
        }
        match Guard::check_command("git push --force") {
            GuardVerdict::Warn(_) => {}
            other => panic!("Expected Warn, got {other:?}"),
        }
    }

    #[test]
    fn chain_split_works() {
        let results = Guard::check_chain("ls && rm -rf / && pwd");
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
        assert_eq!(results[2], GuardVerdict::Allow);
    }

    #[test]
    fn pipe_chain_split() {
        let results = Guard::check_chain("cat file.txt | grep pattern | wc -l");
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| *r == GuardVerdict::Allow));
    }

    #[test]
    fn echo_quoted_dangerous_not_blocked() {
        // Arguments to echo are just strings, not executed
        assert_eq!(Guard::check_command("echo 'rm -rf /'"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("echo \"rm -rf /\""),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn cat_quoted_dangerous_not_blocked() {
        // cat's arguments are file paths / strings, not executed
        assert_eq!(Guard::check_command("cat 'rm -rf /'"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("cat \"rm -rf /\""),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn grep_quoted_dangerous_not_blocked() {
        // grep's arguments are patterns, not executed
        assert_eq!(
            Guard::check_command("grep 'rm -rf' file.txt"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("grep \"rm -rf\" file.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn echo_without_quotes_dangerous_not_blocked() {
        // echo's arguments are strings even without quotes
        assert_eq!(Guard::check_command("echo rm -rf /"), GuardVerdict::Allow);
    }

    #[test]
    fn sed_quoted_dangerous_not_blocked() {
        assert_eq!(
            Guard::check_command("sed 's/rm -rf/foo/' file"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn sh_c_with_dangerous_payload_is_blocked() {
        // Regression: the quote-stripping bypass used to allow this.
        match Guard::check_command("sh -c 'rm -rf /'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("bash -c \"rm -rf /\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn python_c_with_system_is_blocked() {
        match Guard::check_command("python3 -c 'import os; os.system(\"rm -rf /\")'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn awk_system_is_blocked() {
        // awk was previously in SAFE_OUTPUT_COMMANDS and silently allowed this.
        match Guard::check_command("awk 'BEGIN{system(\"rm -rf /\")}'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn benign_interpreter_commands_still_allowed() {
        assert_eq!(
            Guard::check_command("python3 build.py"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("sh -c 'cargo build'"),
            GuardVerdict::Allow
        );
        assert_eq!(Guard::check_command("node index.js"), GuardVerdict::Allow);
    }

    #[test]
    fn dangerous_patterns_still_blocked() {
        // Actual dangerous commands should still be blocked
        match Guard::check_command("rm -rf /") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        match Guard::check_command("dd if=/dev/zero of=/dev/sda") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn dangerous_after_chain_operator_still_blocked() {
        // Dangerous command after && should be blocked
        let results = Guard::check_chain("echo foo && rm -rf /");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
    }

    #[test]
    fn strip_quotes_removes_single_quotes() {
        assert_eq!(strip_quotes("echo 'rm -rf /'"), "echo ");
        assert_eq!(strip_quotes("echo 'hello' world"), "echo  world");
    }

    #[test]
    fn strip_quotes_removes_double_quotes() {
        assert_eq!(strip_quotes("echo \"rm -rf /\""), "echo ");
        assert_eq!(strip_quotes("echo \"hello\" world"), "echo  world");
    }

    #[test]
    fn strip_quotes_handles_escaped_double_quotes() {
        // Escaped quotes inside double quotes don't close the quote
        // We just need to strip the quoted content, not preserve it
        assert_eq!(strip_quotes("echo \"say \\\"hi\\\"\" world"), "echo  world");
    }

    #[test]
    fn strip_quotes_preserves_single_char() {
        assert_eq!(strip_quotes("ls -la"), "ls -la");
        assert_eq!(strip_quotes("rm -rf /"), "rm -rf /");
    }

    #[test]
    fn command_name_extracts_first_word() {
        assert_eq!(command_name("rm -rf /"), "rm");
        assert_eq!(command_name("echo hello"), "echo");
        assert_eq!(command_name(""), "");
    }

    // === F1.1 regression: command substitution bypasses ===

    #[test]
    fn command_substitution_in_echo_blocked() {
        // echo is a "safe output" command, but $(...) executes its payload.
        match Guard::check_command("echo $(rm -rf /)") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn command_substitution_in_printf_blocked() {
        match Guard::check_command("printf \"$(rm -rf /)\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn command_substitution_backticks_blocked() {
        // Backticks are command substitution too.
        match Guard::check_command("echo `rm -rf /`") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn process_substitution_blocked() {
        match Guard::check_command("cat <(rm -rf /)") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn eval_always_blocked() {
        match Guard::check_command("eval \"rm -rf /\"") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
        // Even benign-looking eval is blocked: it can execute anything.
        match Guard::check_command("eval 'x=1'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn substitution_inside_sh_c_blocked() {
        // The shell executes $(...) inside its -c argument.
        match Guard::check_command("sh -c 'echo $(rm -rf /)'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block, got {other:?}"),
        }
    }

    #[test]
    fn variable_expansion_not_blocked() {
        // $VAR / $5 / ${x} are expansion, not execution.
        assert_eq!(
            Guard::check_command("echo \"cost: $5\""),
            GuardVerdict::Allow
        );
        assert_eq!(Guard::check_command("echo $HOME/path"), GuardVerdict::Allow);
    }

    #[test]
    fn literal_dangerous_text_not_blocked() {
        // Plain literal text with no substitution is data, not execution.
        assert_eq!(
            Guard::check_command("echo 'rm -rf / is dangerous'"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn backup_tarball_allowed() {
        // F1.1.3 regression (sh-backup false positive): tarball creation must
        // not be blocked by the "rm -rf" / backup-content heuristics.
        assert_eq!(
            Guard::check_command("tar -czf backup.tar.gz dir/"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn find_exec_rm_documented_decision() {
        // Decide explicitly: `find ... -exec rm {} \;` on a *named pattern*
        // (e.g. '*.tmp') is a legitimate cleanup operation, not a recursive
        // wipe. Documented Allow (matching the plan's "Allow o Warn
        // documentado"). A recursive wipe of a root path would still hit the
        // Risky `rm -r` pattern if present; here the target is a temp glob.
        assert_eq!(
            Guard::check_command("find . -name '*.tmp' -exec rm {} \\;"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn source_and_env_raw_scan() {
        // source / . / env / xargs execute their arguments: raw scan applies.
        match Guard::check_command("source backup.sh") {
            GuardVerdict::Allow => {}
            other => panic!("Expected Allow for benign source, got {other:?}"),
        }
        match Guard::check_command(". myscript.sh") {
            GuardVerdict::Allow => {}
            other => panic!("Expected Allow for benign dot-source, got {other:?}"),
        }
        match Guard::check_command("env VAR=1 sh -c 'rm -rf /'") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for env-wrapped danger, got {other:?}"),
        }
    }

    // === P0c: shell-redirect write detection (port of _shared/shell-write.ts) ===

    #[test]
    fn redirect_cat_to_file_blocked() {
        match Guard::check_command("cat > file.txt") {
            GuardVerdict::Block(msg) => {
                assert!(msg.contains("do not redirect") || msg.contains("Write"))
            }
            other => panic!("Expected Block for cat > file, got {other:?}"),
        }
        assert!(is_write_via_redirect("cat > file.txt"));
        assert_eq!(detect_write_targets("cat > file.txt"), vec!["file.txt"]);
    }

    #[test]
    fn redirect_echo_append_blocked() {
        match Guard::check_command("echo hi >> file.txt") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for echo >>, got {other:?}"),
        }
        assert!(is_write_via_redirect("echo hi >> file.txt"));
    }

    #[test]
    fn redirect_no_space_blocked() {
        assert_eq!(detect_write_targets("cat>main.py"), vec!["main.py"]);
        match Guard::check_command("cat>main.py") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for cat>main.py, got {other:?}"),
        }
    }

    #[test]
    fn tee_with_file_blocked() {
        match Guard::check_command("echo x | tee out.log") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for tee out.log, got {other:?}"),
        }
        assert_eq!(
            detect_write_targets("echo x | tee -a out.log"),
            vec!["out.log"]
        );
    }

    #[test]
    fn tee_without_args_allowed() {
        assert_eq!(Guard::check_command("echo hi | tee"), GuardVerdict::Allow);
        assert!(!is_write_via_redirect("echo hi | tee"));
        assert_eq!(detect_write_targets("echo hi | tee"), Vec::<String>::new());
    }

    #[test]
    fn tee_to_dev_null_allowed() {
        assert_eq!(
            Guard::check_command("echo x | tee /dev/null"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("echo x | tee /dev/null"));
    }

    #[test]
    fn redirect_to_dev_null_allowed() {
        assert_eq!(Guard::check_command("cat > /dev/null"), GuardVerdict::Allow);
        assert_eq!(
            Guard::check_command("cmd > /dev/null 2>&1"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("cat > /dev/null"));
        assert!(!is_write_via_redirect("cmd > /dev/null 2>&1"));
    }

    #[test]
    fn redirect_stderr_to_dev_null_allowed() {
        assert_eq!(
            Guard::check_command("grep -ri \"x\" ~/.config/ 2>/dev/null"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("make 2>/dev/null"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("make 2>/dev/null"));
    }

    #[test]
    fn redirect_to_dev_stdout_allowed() {
        assert_eq!(
            Guard::check_command("cmd > /dev/stdout"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("cmd 2> /dev/stderr"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("cmd > /dev/stdout"));
    }

    #[test]
    fn fd_duplication_not_flagged() {
        assert_eq!(detect_write_targets("make 2>&1"), Vec::<String>::new());
        assert_eq!(detect_write_targets("echo err >&2"), Vec::<String>::new());
        assert_eq!(Guard::check_command("make 2>&1"), GuardVerdict::Allow);
    }

    #[test]
    fn tar_czf_still_allowed() {
        assert_eq!(
            Guard::check_command("tar -czf backup.tar.gz dir/"),
            GuardVerdict::Allow
        );
        assert!(!is_write_via_redirect("tar -czf backup.tar.gz dir/"));
    }

    #[test]
    fn heredoc_strip_preserves_redirect() {
        let cmd = "cat > backend/main.py << 'ENDOFFILE'\nfrom fastapi import FastAPI\napp = FastAPI()\nENDOFFILE";
        assert_eq!(detect_write_targets(cmd), vec!["backend/main.py"]);
        match Guard::check_command(cmd) {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for heredoc > file, got {other:?}"),
        }
        // Heredoc body with `>` should not be mistaken for redirect
        let cmd2 = "cat > cmp.py << 'EOF'\nif a > b:\n    pass  # don't compare like this\nEOF";
        assert_eq!(detect_write_targets(cmd2), vec!["cmp.py"]);
    }

    #[test]
    fn heredoc_body_not_split_as_commands() {
        assert_eq!(
            split_command_chain("cat > f.sh << 'EOF'\nrm -rf /\nEOF"),
            vec!["cat > f.sh"]
        );
    }

    #[test]
    fn dedup_write_targets() {
        let writes = detect_write_targets("echo a > one.txt && echo b >> one.txt");
        assert_eq!(writes, vec!["one.txt"]);
        let writes2 = detect_write_targets("echo a > one.txt && echo b >> two.txt");
        assert_eq!(writes2, vec!["one.txt", "two.txt"]);
    }

    #[test]
    fn dd_of_file_blocked() {
        assert_eq!(
            detect_write_targets("dd if=/dev/zero of=disk.img bs=1M"),
            vec!["disk.img"]
        );
        match Guard::check_command("dd if=/dev/zero of=disk.img bs=1M") {
            GuardVerdict::Block(_) => {}
            other => panic!("Expected Block for dd of=, got {other:?}"),
        }
    }

    #[test]
    fn dd_of_dev_null_allowed() {
        assert_eq!(
            detect_write_targets("dd if=big.img of=/dev/null"),
            Vec::<String>::new()
        );
        assert_eq!(
            Guard::check_command("dd if=big.img of=/dev/null"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn dd_if_dev_null_not_blocked() {
        // Refined dd if= : benign input should not be blocked
        assert_eq!(
            Guard::check_command("dd if=/dev/null of=/dev/null"),
            GuardVerdict::Allow
        );
        assert_eq!(
            Guard::check_command("dd if=/dev/null bs=1 count=1"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn redirect_inside_quotes_not_flagged() {
        assert_eq!(
            detect_write_targets("grep \"a > b\" file.txt"),
            Vec::<String>::new()
        );
        assert_eq!(detect_write_targets("echo 'x >> y'"), Vec::<String>::new());
        assert_eq!(
            Guard::check_command("grep \"a > b\" file.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn input_redirect_not_flagged() {
        assert_eq!(
            detect_write_targets("wc -l < input.txt"),
            Vec::<String>::new()
        );
        assert_eq!(
            detect_write_targets("grep foo <<< \"$var\""),
            Vec::<String>::new()
        );
        assert_eq!(
            Guard::check_command("wc -l < input.txt"),
            GuardVerdict::Allow
        );
    }

    #[test]
    fn chain_each_segment_judged() {
        // ls && rm -rf / should be refused on rm, not admitted on ls
        let results = Guard::check_chain("ls && cat > evil.txt");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], GuardVerdict::Allow);
        assert!(matches!(results[1], GuardVerdict::Block(_)));
    }

    #[test]
    fn extract_redirect_targets_works() {
        assert_eq!(extract_redirect_targets("cat > file.txt"), vec!["file.txt"]);
        assert_eq!(
            extract_redirect_targets("echo hi >> file.txt"),
            vec!["file.txt"]
        );
        assert_eq!(
            extract_redirect_targets("cmd > /dev/null"),
            Vec::<String>::new()
        );
        assert_eq!(extract_redirect_targets("cmd 2>&1"), Vec::<String>::new());
    }

    #[test]
    fn is_non_destructive_target_works() {
        assert!(is_non_destructive_target("/dev/null"));
        assert!(is_non_destructive_target("/dev/stdout"));
        assert!(is_non_destructive_target("/dev/stderr"));
        assert!(is_non_destructive_target("/dev/fd/2"));
        assert!(!is_non_destructive_target("file.txt"));
        assert!(!is_non_destructive_target("/dev/null.bak"));
        assert!(is_non_destructive_target("&1"));
    }

    #[test]
    fn split_command_chain_quote_aware() {
        assert_eq!(
            split_command_chain("echo 'a && b' ; ls"),
            vec!["echo 'a && b'", "ls"]
        );
        assert_eq!(
            split_command_chain("ls && rm -rf / ; echo done | wc -l"),
            vec!["ls", "rm -rf /", "echo done", "wc -l"]
        );
    }

    #[test]
    fn strip_heredoc_bodies_works() {
        assert_eq!(
            strip_heredoc_bodies("cat > f << 'EOF'\nbody\nEOF"),
            "cat > f "
        );
        assert_eq!(
            strip_heredoc_bodies("cat > f << EOF\nbody without terminator"),
            "cat > f "
        );
        assert_eq!(
            strip_heredoc_bodies("grep foo <<< \"bar\""),
            "grep foo <<< \"bar\""
        );
    }

    #[test]
    fn strip_heredoc_bodies_multiline_kills_arithmetic_mutant() {
        // Two body lines force `consumed += line.len() + 1` to run twice;
        // a `+=` -> `*=` mutant leaves `consumed` at 0 and leaks the body.
        assert_eq!(
            strip_heredoc_bodies("cat > f << EOF\na\nb\nEOF\necho hi"),
            "cat > f echo hi"
        );
    }

    #[test]
    fn strip_heredoc_bodies_double_heredoc_rescans() {
        // Two heredocs in one command exercise the re-scan path
        // (`search_from = at` after the first strip).
        assert_eq!(
            strip_heredoc_bodies("cat > a << EOF\nx\nEOF\ncat > b << END\ny\nEND\necho done"),
            "cat > a cat > b echo done"
        );
    }
}
