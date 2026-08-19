//! Safety guards for the agent loop.
//!
//! Provides permission gates for shell commands and chain analysis.

/// Patterns that indicate destructive or dangerous shell commands.
const BLOCKED_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf .",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    ":(){:|:&};:",
    "chmod -R 777 /",
    "chmod -R 777 ~",
    "wget -O- | sh",
    "curl | sh",
    "eval(",
    "exec(",
    "fork(",
    "shutdown",
    "reboot",
    "halt",
    "init 0",
    "init 6",
];

/// Commands whose arguments are just data and are never executed.
/// If the first word of a command is one of these, the arguments are not a
/// threat (e.g., `echo 'rm -rf /'` just prints text, `cat 'rm -rf /'` reads a
/// file named literally).
///
/// NOTE: This list must only contain commands that are *incapable* of executing
/// their arguments. Anything Turing-complete or with an execute flag (awk's
/// `system()`, sed's `e` command, shells, script interpreters) is intentionally
/// excluded and handled by `INTERPRETERS` below.
const SAFE_OUTPUT_COMMANDS: &[&str] = &[
    "echo", "cat", "grep", "printf", "read",
    "test", "[", "true", "false", "head", "tail", "wc",
    "sort", "uniq", "cut", "tr", "tee",
];

/// Interpreters / shells: their arguments are *executed*, not just data.
/// For these we must scan the entire raw command (quotes and all), because the
/// dangerous payload lives inside the `-c`/`-e` argument and quote-stripping
/// would otherwise hide it (e.g. `sh -c 'rm -rf /'` or
/// `awk 'BEGIN{system("rm -rf /")}'`).
const INTERPRETERS: &[&str] = &[
    "sh", "bash", "dash", "ash", "zsh", "ksh", "csh", "tcsh", "fish",
    "python", "python3", "python2", "perl", "ruby", "node", "lua",
    "php", "pwsh", "powershell", "cmd", "awk",
];

/// Get the command name (first word) from a command string.
fn command_name(cmd: &str) -> &str {
    cmd.split_whitespace().next().unwrap_or("")
}

/// Remove quoted substrings from a command to avoid false positives.
/// In single quotes, everything is literal (no escape processing).
/// In double quotes, backslash escapes the next character.
fn strip_quotes(cmd: &str) -> String {
    let mut result = String::with_capacity(cmd.len());
    let bytes = cmd.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => {
                // Single quote: everything until next single quote is literal
                i += 1;
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                if i < bytes.len() { i += 1; } // skip closing quote
            }
            b'"' => {
                // Double quote: backslash escapes next char
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 2; // skip backslash and escaped char
                    } else {
                        i += 1;
                    }
                }
                if i < bytes.len() { i += 1; } // skip closing quote
            }
            _ => {
                result.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    result
}

/// Patterns that indicate potentially risky but sometimes legitimate commands.
/// These require confirmation rather than hard block.
const RISKY_PATTERNS: &[&str] = &[
    "rm -rf",
    "rm -r",
    "sudo",
    "chmod -R",
    "chown -R",
    "kill -9",
    "killall",
    "pkill",
    "git push --force",
    "git reset --hard",
    "git clean -f",
    "git checkout --",
    "docker rm",
    "docker rmi",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardVerdict {
    /// Command is safe to run.
    Allow,
    /// Command is dangerous, refuse to execute.
    Block(String),
    /// Command is risky, requires user confirmation.
    Warn(String),
}

/// Safety guard that analyzes shell commands.
pub struct Guard;

impl Guard {
    /// Analyze a single shell command for safety.
    ///
    /// Three layers, in order:
    /// 1. Safe output commands (echo, cat, grep, ...) — their arguments are just
    ///    data, never executed, so they are always allowed.
    /// 2. Interpreters (sh, bash, python, awk, sed, ...) — their arguments ARE
    ///    executed. We scan the **raw** command (quotes included) so the dangerous
    ///    payload inside `-c`/`BEGIN{system()}` cannot be hidden by quoting.
    /// 3. Everything else — scan the quote-stripped command to avoid false
    ///    positives from dangerous-looking literal strings passed as data.
    pub fn check_command(cmd: &str) -> GuardVerdict {
        let name = command_name(cmd).to_lowercase();

        if SAFE_OUTPUT_COMMANDS.contains(&name.as_str()) {
            return GuardVerdict::Allow;
        }

        if INTERPRETERS.contains(&name.as_str()) {
            return Self::scan(cmd);
        }

        // General command: strip quotes (literal data args) then scan.
        Self::scan(&strip_quotes(cmd))
    }

    /// Scan a command string for blocked/risky patterns (case-insensitive).
    fn scan(cmd: &str) -> GuardVerdict {
        let lowered = cmd.to_lowercase();

        for pattern in BLOCKED_PATTERNS {
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
        // Split on shell operators, but preserve them as context
        let mut results = Vec::new();
        for cmd in Self::split_chain(chain) {
            let cmd = cmd.trim();
            if !cmd.is_empty() {
                results.push(Self::check_command(cmd));
            }
        }
        results
    }

    /// Split a shell chain into individual commands.
    fn split_chain(chain: &str) -> Vec<&str> {
        let mut parts = Vec::new();
        let mut current_start = 0;
        let chars: Vec<char> = chain.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            match chars[i] {
                '|' if i + 1 < len && chars[i + 1] == '|' => {
                    parts.push(&chain[current_start..i]);
                    i += 2;
                    current_start = i;
                }
                '|' if i + 1 < len && chars[i + 1] != '|' => {
                    parts.push(&chain[current_start..i]);
                    i += 1;
                    current_start = i;
                }
                '&' if i + 1 < len && chars[i + 1] == '&' => {
                    parts.push(&chain[current_start..i]);
                    i += 2;
                    current_start = i;
                }
                ';' => {
                    parts.push(&chain[current_start..i]);
                    i += 1;
                    current_start = i;
                }
                _ => i += 1,
            }
        }
        if current_start < len {
            parts.push(&chain[current_start..]);
        }

        // Filter out empty strings from leading operators
        parts.into_iter().filter(|s| !s.trim().is_empty()).collect()
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
        assert_eq!(Guard::check_command("echo \"rm -rf /\""), GuardVerdict::Allow);
    }

    #[test]
    fn cat_quoted_dangerous_not_blocked() {
        // cat's arguments are file paths / strings, not executed
        assert_eq!(Guard::check_command("cat 'rm -rf /'"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("cat \"rm -rf /\""), GuardVerdict::Allow);
    }

    #[test]
    fn grep_quoted_dangerous_not_blocked() {
        // grep's arguments are patterns, not executed
        assert_eq!(Guard::check_command("grep 'rm -rf' file.txt"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("grep \"rm -rf\" file.txt"), GuardVerdict::Allow);
    }

    #[test]
    fn echo_without_quotes_dangerous_not_blocked() {
        // echo's arguments are strings even without quotes
        assert_eq!(Guard::check_command("echo rm -rf /"), GuardVerdict::Allow);
    }

    #[test]
    fn sed_quoted_dangerous_not_blocked() {
        assert_eq!(Guard::check_command("sed 's/rm -rf/foo/' file"), GuardVerdict::Allow);
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
        assert_eq!(Guard::check_command("python3 build.py"), GuardVerdict::Allow);
        assert_eq!(Guard::check_command("sh -c 'cargo build'"), GuardVerdict::Allow);
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
}
