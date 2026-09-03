//! Shell command parsing and write-target analysis: tokenizer, chain
//! splitter, heredoc stripping, and redirect/tee/dd target extraction.

use super::patterns::{CHAIN_OPERATORS, NON_DESTRUCTIVE_TARGETS};
use std::collections::HashSet;

pub(crate) fn command_name(cmd: &str) -> &str {
    cmd.split_whitespace().next().unwrap_or("")
}

/// Remove quoted substrings from a command to avoid false positives.
/// In single quotes, everything is literal (no escape processing).
/// In double quotes, backslash escapes the next character.
pub(crate) fn strip_quotes(cmd: &str) -> String {
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
pub(crate) fn has_command_substitution(cmd: &str) -> bool {
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
pub(crate) fn is_fd_target(target: &str) -> bool {
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
pub(crate) fn first_word(s: &str) -> String {
    split_words(s).into_iter().next().unwrap_or_default()
}

/// Unquote a word: strip matching outer quotes or unescape backslashes.
pub(crate) fn unquote(word: &str) -> String {
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
