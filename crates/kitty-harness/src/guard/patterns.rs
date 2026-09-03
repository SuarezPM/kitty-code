/// Patterns that indicate destructive or dangerous shell commands.
pub(crate) const BLOCKED_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "rm -rf .",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    // Fork-bomb signatures (found by property testing — spaced spellings
    // slipped past the no-space variant).
    ":(){:|:&};:",
    ":(){ :|:& };:",
    ":(){ :|:&};:",
    ":(){:|:& };:",
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
///
/// `tee` deliberately NOT included — `tee file` writes to `file` via shell
/// and must be detected as a write (see `detect_write_targets`). `tee` without
/// args is still Allow via redirect detection (no targets).
pub(crate) const SAFE_OUTPUT_COMMANDS: &[&str] = &[
    "echo", "cat", "grep", "printf", "read", "test", "[", "true", "false", "head", "tail", "wc",
    "sort", "uniq", "cut", "tr",
];

/// Interpreters / shells: their arguments are *executed*, not just data.
/// For these we must scan the entire raw command (quotes and all), because the
/// dangerous payload lives inside the `-c`/`-e` argument and quote-stripping
/// would otherwise hide it (e.g. `sh -c 'rm -rf /'` or
/// `awk 'BEGIN{system("rm -rf /")}'`).
pub(crate) const ALWAYS_BLOCKED_COMMANDS: &[&str] = &["eval"];

/// Targets that are character devices and destroy no real file.
///
/// Redirecting to these is benign (`2>/dev/null` discards output,
/// `>/dev/stdout` re-plumbs a stream) and must not be treated as a file write.
/// Matched case-sensitively. Windows `nul` intentionally NOT exempted.
pub(crate) const NON_DESTRUCTIVE_TARGETS: &[&str] = &[
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/stdin",
    "/dev/tty",
    "/dev/zero",
    "/dev/full",
    "/dev/random",
    "/dev/urandom",
];

/// Chain operators that split a shell command chain into independent commands.
pub(crate) const CHAIN_OPERATORS: &[&str] = &["&&", "||", ";", "|", "\n"];

/// Get the command name (first word) from a command string.
/// Patterns that indicate potentially risky but sometimes legitimate commands.
/// These require confirmation rather than hard block.
pub(crate) const RISKY_PATTERNS: &[&str] = &[
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
