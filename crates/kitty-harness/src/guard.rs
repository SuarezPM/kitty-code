//! Safety guards for the agent loop.
//!
//! Provides permission gates for shell commands and chain analysis.
//! Port of little-coder `_shared/shell-write.ts` for shell-redirect write detection.

use std::collections::HashSet;

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
///
/// `tee` deliberately NOT included — `tee file` writes to `file` via shell
/// and must be detected as a write (see `detect_write_targets`). `tee` without
/// args is still Allow via redirect detection (no targets).
const SAFE_OUTPUT_COMMANDS: &[&str] = &[
    "echo", "cat", "grep", "printf", "read", "test", "[", "true", "false", "head", "tail", "wc",
    "sort", "uniq", "cut", "tr",
];

/// Interpreters / shells: their arguments are *executed*, not just data.
/// For these we must scan the entire raw command (quotes and all), because the
/// dangerous payload lives inside the `-c`/`-e` argument and quote-stripping
/// would otherwise hide it (e.g. `sh -c 'rm -rf /'` or
/// `awk 'BEGIN{system("rm -rf /")}'`).
const INTERPRETERS: &[&str] = &[
    "sh",
    "bash",
    "dash",
    "ash",
    "zsh",
    "ksh",
    "csh",
    "tcsh",
    "fish",
    "python",
    "python3",
    "python2",
    "perl",
    "ruby",
    "node",
    "lua",
    "php",
    "pwsh",
    "powershell",
    "cmd",
    "awk",
    // These guys also execute their arguments, so they get the raw scan too:
    "source",
    ".",
    "env",
    "xargs",
    "timeout",
    "nohup",
];

/// Commands that are *always* blocked, no matter the arguments.
///
/// `eval` re-parses and executes its argument as shell code — there is no
/// legitimate use inside a coding agent, and any payload it wraps is
/// unanalyzable by a pattern guard. Hard-blocked by decision (F1.1).
const ALWAYS_BLOCKED_COMMANDS: &[&str] = &["eval"];

/// Targets that are character devices and destroy no real file.
///
/// Redirecting to these is benign (`2>/dev/null` discards output,
/// `>/dev/stdout` re-plumbs a stream) and must not be treated as a file write.
/// Matched case-sensitively. Windows `nul` intentionally NOT exempted.
const NON_DESTRUCTIVE_TARGETS: &[&str] = &[
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
const CHAIN_OPERATORS: &[&str] = &["&&", "||", ";", "|", "\n"];

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
                if i < bytes.len() {
                    i += 1;
                } // skip closing quote
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
                if i < bytes.len() {
                    i += 1;
                } // skip closing quote
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

/// Detect command substitution in the raw command string.
///
/// `$(` / backticks / process substitution `<(`/`>(` all make the shell
/// **execute** whatever they wrap, even when the outer command is a "safe
/// output" command like `echo` or `cat` (`echo $(rm -rf /)` really runs
/// `rm -rf /`). Variable expansion (`$VAR`, `$5`, `${x}`) is NOT execution and
/// must not trigger this path — only `$(` (which includes `${x:=$(cmd)}` and
/// `$((...))` arithmetic forms that can contain commands) and backticks do.
fn has_command_substitution(cmd: &str) -> bool {
    cmd.contains("$(") || cmd.contains('`') || cmd.contains("<(") || cmd.contains(">(")
}

// ---------------------------------------------------------------------------
// Shell-write detection (port of _shared/shell-write.ts)
// ---------------------------------------------------------------------------

/// True for a redirect target that can't destroy a real file.
///
/// Covers `/dev/null`, `/dev/stdout`, `/dev/stderr`, `/dev/stdin`, `/dev/tty`,
/// `/dev/zero`, etc. plus `/dev/fd/N`. Also treats fd duplication plumbing
/// (`&1`, `&2`, `1`, `2`) as non-destructive. Case-sensitive.
/// See little-coder `_shared/shell-write.ts` `isNonDestructiveTarget`.
pub fn is_non_destructive_target(s: &str) -> bool {
    // Trim surrounding whitespace
    let t = s.trim();
    if NON_DESTRUCTIVE_TARGETS.contains(&t) {
        return true;
    }
    // /dev/fd/N — fd plumbing by another spelling
    if let Some(rest) = t.strip_prefix("/dev/fd/") {
        if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }
    // fd duplication targets like &1, &2, and bare fd numbers 1,2
    // Also handles "2>", "&>", ">&" forms mentioned in spec — if the target
    // string itself is one of those, treat as non-destructive.
    if t == "1" || t == "2" || t.starts_with('&') {
        return true;
    }
    if t == "2>" || t == "&>" || t == ">&" || t == "2>&1" || t == ">&2" || t == "&>>" {
        return true;
    }
    false
}

/// Helper: is target an fd target (starts with &)? e.g. &1, &2
fn is_fd_target(target: &str) -> bool {
    target.starts_with('&')
}

/// Remove heredoc bodies, leaving the command line that opened them.
///
/// A heredoc body is data, not shell syntax, and analyzing it produces noise:
/// a `>` in the payload (`if a > b:`) looks like a redirect, and an apostrophe
/// (`don't`) leaves the quote scanner stuck. The redirect we care about —
/// `cat > main.py << 'EOF'` — sits on the opening line, so dropping the body
/// loses nothing.
///
/// Portable port of `stripHeredocBodies` from `_shared/shell-write.ts`.
pub fn strip_heredoc_bodies(cmd: &str) -> String {
    let mut out = cmd.to_string();
    let mut search_from: usize = 0;

    // One pass per heredoc; a command can open several (guard 32)
    for _ in 0..32 {
        if search_from >= out.len() {
            break;
        }
        let rest = &out[search_from..];
        // Find "<<"
        let Some(rel_pos) = rest.find("<<") else {
            break;
        };
        let at = search_from + rel_pos;

        // Guard: "<<<" here-string — nothing to strip, keep scanning past it
        if out.as_bytes().get(at + 2) == Some(&b'<') {
            search_from = at + 3;
            continue;
        }

        // Parse delimiter after "<<"
        let mut idx = at + 2;
        let bytes = out.as_bytes();
        // Optional '-' for <<-
        if bytes.get(idx) == Some(&b'-') {
            idx += 1;
        }
        // Skip spaces/tabs
        while idx < bytes.len() && (bytes[idx] == b' ' || bytes[idx] == b'\t') {
            idx += 1;
        }
        if idx >= bytes.len() {
            search_from = at + 2;
            continue;
        }

        let delim: String;
        let delim_end: usize;
        if bytes[idx] == b'\'' || bytes[idx] == b'"' {
            let quote = bytes[idx];
            idx += 1;
            let start = idx;
            while idx < bytes.len() && bytes[idx] != quote {
                idx += 1;
            }
            if idx >= bytes.len() {
                // Unterminated quote — not a valid heredoc
                search_from = at + 2;
                continue;
            }
            delim = out[start..idx].to_string();
            delim_end = idx + 1; // after closing quote
        } else {
            // Unquoted delimiter: [A-Za-z_][A-Za-z0-9_]*
            let start = idx;
            if start < bytes.len() && (bytes[start].is_ascii_alphabetic() || bytes[start] == b'_') {
                idx += 1;
                while idx < bytes.len()
                    && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_')
                {
                    idx += 1;
                }
                delim = out[start..idx].to_string();
                delim_end = idx;
            } else {
                search_from = at + 2;
                continue;
            }
        }

        if delim.is_empty() {
            search_from = delim_end;
            continue;
        }

        // Body starts after newline following delimiter
        let Some(nl_pos) = out[delim_end..].find('\n') else {
            // No newline — single-line command, nothing to strip
            search_from = delim_end;
            continue;
        };
        let body_start = delim_end + nl_pos + 1; // after '\n'

        // Scan lines after body_start for terminator
        let remaining = &out[body_start..];
        let lines: Vec<&str> = remaining.split('\n').collect();
        let mut consumed: usize = 0;
        let mut closed = false;
        for line in &lines {
            // consumed includes line length + 1 for '\n' except maybe last
            // For simplicity, track as we iterate
            if line.trim() == delim {
                // "<<-" allows indented terminator with tabs — trim covers it
                closed = true;
                consumed += line.len() + 1; // include newline for this line
                break;
            }
            consumed += line.len() + 1;
        }

        let body_end = if closed {
            // body_start + consumed, but clamp
            let end = body_start + consumed;
            end.min(out.len())
        } else {
            // Unterminated heredoc: rest of string is body
            out.len()
        };

        // Remove from `at` to `body_end`, but keep opening line prefix before `<<`
        // Actually JS does: out = out.slice(0, at) + out.slice(bodyEnd)
        // That removes the heredoc operator and body, leaving "cat > f "
        // For detection we want to keep the "> file" part before "<<"
        // So slicing from `at` is correct — it preserves prefix.
        out = format!("{}{}", &out[..at], &out[body_end..]);
        search_from = at;
    }

    out
}

/// Walk `cmd` tracking quote state and invoke `visit` for each char outside
/// handling. Used to keep every consumer from firing on text that merely
/// looks like shell syntax (`grep "a > b"` writes nothing).
#[allow(dead_code)]
fn scan_with_quotes<F>(cmd: &str, mut visit: F)
where
    F: FnMut(char, usize, bool),
{
    let chars: Vec<char> = cmd.chars().collect();
    let mut quote: Option<char> = None;
    let mut i = 0usize;
    // We need byte index for visit; track byte offset in parallel
    let mut byte_idx = 0usize;
    while i < chars.len() {
        let ch = chars[i];
        let ch_len = ch.len_utf8();
        if quote.is_none() && ch == '\\' {
            // Backslash escapes next char entirely (outside single quotes)
            // Skip both
            i += 1;
            if i < chars.len() {
                byte_idx += ch_len + chars[i].len_utf8();
                i += 1;
            } else {
                byte_idx += ch_len;
            }
            continue;
        }
        if quote == Some('\'') && ch == '\\' {
            // Backslash literal inside single quotes — no escaping
            visit(ch, byte_idx, true);
            byte_idx += ch_len;
            i += 1;
            continue;
        }
        if quote.is_none() && (ch == '"' || ch == '\'') {
            quote = Some(ch);
            byte_idx += ch_len;
            i += 1;
            continue;
        }
        if let Some(q) = quote {
            if ch == q {
                quote = None;
                byte_idx += ch_len;
                i += 1;
                continue;
            }
        }
        let quoted = quote.is_some();
        visit(ch, byte_idx, quoted);
        byte_idx += ch_len;
        i += 1;
    }
}

/// Split a command line into individual commands on unquoted `&&`, `||`, `;`,
/// `|`, `\n`. Heredoc bodies are stripped first so their contents are never
/// mistaken for commands.
///
/// Port of `splitCommandChain` from `_shared/shell-write.ts`.
pub fn split_command_chain(raw: &str) -> Vec<String> {
    let cmd = strip_heredoc_bodies(raw);
    let mut cuts: Vec<(usize, usize)> = Vec::new(); // (at byte idx, len)

    // Need to scan with quote tracking to find operators
    // We'll build a byte-indexed view for starts_with checks.
    // Use scan_with_quotes but need to check operators at each position.
    // Simpler: iterate byte indices with quote state.
    let bytes = cmd.as_bytes();
    let len = bytes.len();
    let mut quote: Option<u8> = None;
    let mut i = 0usize;
    while i < len {
        let b = bytes[i];
        // Handle escapes
        if quote.is_none() && b == b'\\' {
            // Skip next byte
            if i + 1 < len {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if quote == Some(b'\'') && b == b'\\' {
            i += 1;
            continue;
        }
        if quote.is_none() && (b == b'"' || b == b'\'') {
            quote = Some(b);
            i += 1;
            continue;
        }
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        // Not quoted — check for chain operators
        let quoted = false;
        if !quoted {
            let mut matched: Option<usize> = None;
            for op in CHAIN_OPERATORS {
                let op_bytes = op.as_bytes();
                if i + op_bytes.len() <= len && &bytes[i..i + op_bytes.len()] == op_bytes {
                    // For "&&" and "||" ensure we don't double-count;
                    // also check we aren't inside already-recorded operator
                    if let Some(last) = cuts.last() {
                        if i < last.0 + last.1 {
                            matched = None;
                            break;
                        }
                    }
                    // Prefer longest match: check && and || before & and |
                    // Since we iterate in order ["&&","||",";","|","\n"] this is fine,
                    // but need to ensure "&&" matched before "&" — but we only have "&&", not single "&"
                    // So longest is covered.
                    matched = Some(op_bytes.len());
                    break;
                }
            }
            if let Some(op_len) = matched {
                cuts.push((i, op_len));
                i += op_len;
                continue;
            }
        }
        i += 1;
    }

    let mut segments: Vec<String> = Vec::new();
    let mut start = 0usize;
    for (at, op_len) in &cuts {
        let seg = cmd[start..*at].trim().to_string();
        if !seg.is_empty() {
            segments.push(seg);
        }
        start = at + op_len;
    }
    let last_seg = cmd[start..].trim().to_string();
    if !last_seg.is_empty() {
        segments.push(last_seg);
    }
    segments
}

/// Tokenize helper — split on unquoted whitespace, keeping quoted runs together.
/// Also treats unquoted `>` and `<` as word boundaries (so `cat>f` is two words).
pub fn tokenize(s: &str) -> Vec<String> {
    split_words(s)
}

/// Split on unquoted whitespace, keeping quoted runs together.
/// Port of `splitWords` from `_shared/shell-write.ts`.
pub fn split_words(s: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut quote: Option<u8> = None;
    let mut i = 0usize;

    while i < len {
        let b = bytes[i];
        if quote.is_none() && b == b'\\' {
            // Escape next char outside single quotes
            cur.push(b as char);
            if i + 1 < len {
                cur.push(bytes[i + 1] as char);
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if quote.is_none() && (b == b'"' || b == b'\'') {
            quote = Some(b);
            cur.push(b as char);
            i += 1;
            continue;
        }
        if let Some(q) = quote {
            if b == q {
                quote = None;
                cur.push(b as char);
                i += 1;
                continue;
            }
        }
        if quote.is_none() && (b == b' ' || b == b'\t' || b == b'\n' || b == b'\r') {
            if !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
            i += 1;
            continue;
        }
        // Unquoted redirect ends current word (`cat>f` is two words)
        if quote.is_none() && (b == b'>' || b == b'<') {
            if !cur.is_empty() {
                words.push(cur.clone());
                cur.clear();
            }
            // Do not push redirect itself as word; just split
            i += 1;
            continue;
        }
        cur.push(b as char);
        i += 1;
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

/// The first whitespace-delimited word of `s`, honoring quotes.
fn first_word(s: &str) -> String {
    split_words(s).into_iter().next().unwrap_or_default()
}

/// Unquote a word: strip matching outer quotes or unescape backslashes.
fn unquote(word: &str) -> String {
    let bytes = word.as_bytes();
    if word.len() >= 2
        && (bytes[0] == b'"' || bytes[0] == b'\'')
        && bytes[word.len() - 1] == bytes[0]
    {
        return word[1..word.len() - 1].to_string();
    }
    // Replace \X with X
    let mut out = String::with_capacity(word.len());
    let mut chars = word.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract redirect targets (`>`/`>>`) from a single segment, outside quotes.
/// Returns paths after `>`/`>>`, filtered for non-destructive and deduped internally.
pub fn extract_redirect_targets(segment: &str) -> Vec<String> {
    let mut targets: Vec<String> = Vec::new();
    let bytes = segment.as_bytes();
    let len = bytes.len();
    let mut quote: Option<u8> = None;
    let mut i = 0usize;

    // Track positions of '>' outside quotes
    let mut redirects: Vec<(usize, bool)> = Vec::new(); // (at after >, is_append)

    while i < len {
        let b = bytes[i];
        if quote.is_none() && b == b'\\' {
            if i + 1 < len {
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if quote == Some(b'\'') && b == b'\\' {
            i += 1;
            continue;
        }
        if quote.is_none() && (b == b'"' || b == b'\'') {
            quote = Some(b);
            i += 1;
            continue;
        }
        if let Some(q) = quote {
            if b == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        // Not quoted
        if b == b'>' {
            // Don't record second '>' of ">>" twice
            if i > 0 && bytes[i - 1] == b'>' {
                i += 1;
                continue;
            }
            let is_append = i + 1 < len && bytes[i + 1] == b'>';
            let at = i + if is_append { 2 } else { 1 };
            redirects.push((at, is_append));
        }
        i += 1;
    }

    for (at, _is_append) in redirects {
        if at >= segment.len() {
            continue;
        }
        let rest = &segment[at..];
        // Process substitution `>(cmd)` is not a file target
        if rest.trim_start().starts_with('(') {
            continue;
        }
        let target = first_word(rest);
        if target.is_empty() || is_fd_target(&target) {
            continue;
        }
        let unquoted = unquote(&target);
        if unquoted.is_empty() || is_non_destructive_target(&unquoted) {
            continue;
        }
        targets.push(unquoted);
    }

    // Dedup within this segment (preserve first)
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for t in targets {
        if seen.contains(&t) {
            continue;
        }
        seen.insert(t.clone());
        deduped.push(t);
    }
    deduped
}

/// Extract `tee` file targets from a segment.
/// `tee [-a] FILE...` writes every non-flag argument.
pub fn extract_tee_targets(segment: &str) -> Vec<String> {
    let mut out = Vec::new();
    for seg in split_command_chain(segment) {
        let words = split_words(&seg);
        if words.is_empty() {
            continue;
        }
        // Check if first word is exactly "tee" (case-sensitive)
        if words[0] != "tee" {
            continue;
        }
        for w in words.iter().skip(1) {
            if w.starts_with('-') {
                continue;
            }
            let u = unquote(w);
            if u.is_empty() || is_non_destructive_target(&u) {
                continue;
            }
            out.push(u);
        }
    }
    // Also handle segment without chain splitting (if caller already segmented)
    // But split_command_chain already handles single segment, so this covers it.
    // For direct segment that's a single command, the loop above already works.
    // Dedup
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for t in out {
        if seen.contains(&t) {
            continue;
        }
        seen.insert(t.clone());
        deduped.push(t);
    }
    deduped
}

/// Extract `dd of=PATH` targets.
pub fn extract_dd_targets(segment: &str) -> Vec<String> {
    let mut out = Vec::new();
    for seg in split_command_chain(segment) {
        let words = split_words(&seg);
        if words.is_empty() || words[0] != "dd" {
            continue;
        }
        for w in words.iter().skip(1) {
            if let Some(path) = w.strip_prefix("of=") {
                let u = unquote(path);
                if u.is_empty() || is_non_destructive_target(&u) {
                    continue;
                }
                out.push(u);
            } else if *w == "of" {
                // Handle `dd of = file` with spaces (unlikely but cover)
                // Look ahead — but split_words already splits on whitespace,
                // so "of" alone would be separate; next word might be "="?
                // Skip for now.
                continue;
            }
        }
    }
    // Dedup
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for t in out {
        if seen.contains(&t) {
            continue;
        }
        seen.insert(t.clone());
        deduped.push(t);
    }
    deduped
}

/// Every path `cmd` writes to via shell redirection or write-by-design tools.
///
/// Covers `>`, `>>`, `tee`, and `dd of=`. Ignores fd duplication
/// (`2>&1`, `>&2`), process substitution (`>(…)`), here-strings/heredocs
/// (`<<`, `<<<`), and anything inside quotes.
///
/// Port of `detectWriteTargets` from `_shared/shell-write.ts`.
pub fn detect_write_targets(cmd: &str) -> Vec<String> {
    let stripped = strip_heredoc_bodies(cmd);
    let mut writes: Vec<String> = Vec::new();

    // Redirects: scan whole stripped command for > outside quotes
    // We need to handle redirects that may appear in any segment, but scanning
    // whole stripped command once is sufficient (like JS).
    // However to mirror JS, we scan redirects on whole cmd, not per segment.
    writes.extend(extract_redirect_targets(&stripped));

    // For tee and dd, use chain splitting
    for segment in split_command_chain(&stripped) {
        // For tee/dd we already use split inside, but to avoid double splitting,
        // handle per segment directly without re-splitting:
        // Use direct word parsing for this segment only
        let words = split_words(&segment);
        if words.is_empty() {
            continue;
        }
        if words[0] == "tee" {
            for w in words.iter().skip(1) {
                if w.starts_with('-') {
                    continue;
                }
                let u = unquote(w);
                if u.is_empty() || is_non_destructive_target(&u) {
                    continue;
                }
                writes.push(u);
            }
        }
        if words[0] == "dd" {
            for w in words.iter().skip(1) {
                if let Some(path) = w.strip_prefix("of=") {
                    let u = unquote(path);
                    if u.is_empty() || is_non_destructive_target(&u) {
                        continue;
                    }
                    writes.push(u);
                }
            }
        }
    }

    // De-duplicate by path, keeping first kind, and filter non-destructive
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for w in writes {
        if w.is_empty() || seen.contains(&w) || is_non_destructive_target(&w) {
            continue;
        }
        seen.insert(w.clone());
        deduped.push(w);
    }
    deduped
}

/// True when the command writes to the filesystem through the shell.
pub fn is_write_via_redirect(cmd: &str) -> bool {
    !detect_write_targets(cmd).is_empty()
}

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

    /// Split a shell chain into individual commands (legacy, quote-unaware).
    /// Kept for compatibility; new code should use `split_command_chain`.
    #[allow(dead_code)]
    fn split_chain(chain: &str) -> Vec<&str> {
        // Delegate to new quote-aware splitter and leak memory for &str compat
        // For tests we just use old logic to avoid lifetime issues, but we
        // keep it for external callers that rely on &str.
        // Instead, implement as before but ensure it filters correctly.
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
}
