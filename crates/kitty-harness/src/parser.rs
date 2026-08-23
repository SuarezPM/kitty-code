//! Forgiving parser for tool calls from model output.
//!
//! Parses tool calls from multiple formats (first-match wins):
//! - Native Ollama `tool_calls` in the message object
//! - `<|tool_call|>` special tags (Qwen-style chat templates)
//! - `[TOOL_CALLS]` markers (Mistral-style arrays)
//! - `<tool_call>` XML blocks in text content
//! - JSON objects embedded in ```json / ```tool fences
//! - `<json>` blocks
//! - Bare JSON objects embedded in prose
//! - Reasoning models that embed tool calls in `<think>` blocks
//!
//! Recovery layers for weak local models (ordered, applied only when earlier
//! layers produce nothing usable):
//! - Lenient JSON repair ([`repair_json`]): smart quotes → straight quotes,
//!   single → double quotes, trailing commas removed, unquoted keys quoted.
//! - Fence stripping before JSON parse attempts.
//! - Empty-args recovery: when a tool name is detected but its arguments are
//!   empty/invalid, a bounded text window around the mention is scanned for a
//!   nearby JSON object to attach.
//! - Fuzzy tool-name mapping: case-insensitive + Levenshtein ≤ 2 against the
//!   known tool set (see [`ToolCallParser::with_known_tools`]).

use serde_json::Value;

/// A parsed tool call with name and arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Canonical tool names used for fuzzy-name matching when no explicit set is provided.
pub const DEFAULT_KNOWN_TOOLS: &[&str] = &["Read", "Edit", "Write", "Bash", "Glob", "Grep"];

/// Maximum edit distance for fuzzy tool-name matching.
const FUZZY_MAX_DISTANCE: usize = 2;

/// Half-size (in bytes) of the text window scanned around a tool-name mention
/// during empty-args recovery (~8K total window).
const RECOVERY_WINDOW_HALF_BYTES: usize = 4 * 1024;

/// Parser for tool calls from model output.
pub struct ToolCallParser {
    known_tools: Vec<String>,
}

/// Canonical error emitted when a tool call intent is detected but parsing fails.
const MALFORMED_ERROR: &str = "Malformed tool call detected. Expected format: {\"name\":\"ToolName\",\"arguments\":{...}} or <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>. Available tools: Read, Edit, Write, Bash, Glob, Grep.";

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            known_tools: DEFAULT_KNOWN_TOOLS
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        }
    }

    /// Restrict fuzzy tool-name matching to a specific tool subset.
    ///
    /// Well-formed calls naming tools outside this subset are left untouched.
    pub fn with_known_tools(mut self, tools: &[&str]) -> Self {
        if !tools.is_empty() {
            self.known_tools = tools.iter().map(|s| s.to_string()).collect();
        }
        self
    }

    /// Parse all tool calls from the model's output text.
    /// Returns empty vec if none found (caller decides what to do).
    pub fn parse(&self, content: &str) -> Vec<ParsedToolCall> {
        let results = self.parse_internal(content);

        if results.is_empty() && self.has_tool_call_intent(content) {
            tracing::warn!(content = %content, "malformed tool call detected (parse returned empty but intent heuristic matched)");
        }

        results
    }

    /// Parse with diagnostics: returns calls plus a list of diagnostic error strings.
    ///
    /// Heuristic: if content contains a tool call intent marker (`<tool_call>`,
    /// `<|tool_call|>`, `[TOOL_CALLS]`, ` ```json `, `<json>`, `{"name"`,
    /// `{"function"`, `{"tool"`) but `parse()` yields no calls, a single
    /// diagnostic string is returned explaining the expected format.
    pub fn parse_with_diagnostics(&self, content: &str) -> (Vec<ParsedToolCall>, Vec<String>) {
        let calls = self.parse_internal(content);
        if calls.is_empty() && self.has_tool_call_intent(content) {
            (calls, vec![MALFORMED_ERROR.to_string()])
        } else {
            (calls, Vec::new())
        }
    }

    /// Try to parse, returning `Err` with a diagnostic if a malformed intent is detected.
    ///
    /// Returns `Ok(calls)` when parsing succeeds — including the legitimate empty case
    /// (no intent). Returns `Err(message)` only when intent heuristics fire but no
    /// call could be parsed.
    pub fn try_parse_with_error(&self, content: &str) -> Result<Vec<ParsedToolCall>, String> {
        let (calls, diagnostics) = self.parse_with_diagnostics(content);
        if let Some(err) = diagnostics.into_iter().next() {
            Err(err)
        } else {
            Ok(calls)
        }
    }

    /// Heuristic detector for "model tried to call a tool but syntax was broken".
    fn has_tool_call_intent(&self, content: &str) -> bool {
        content.contains("<tool_call>")
            || content.contains("<|tool_call|>")
            || content.contains("[TOOL_CALLS]")
            || content.contains("```json")
            || content.contains("<json>")
            || content.contains(r#"{"name""#)
            || content.contains(r#"{"function""#)
            || content.contains(r#"{"tool""#)
    }

    /// Internal parsing logic without diagnostics side-effects.
    ///
    /// Layer order (first-match wins for the tagged/marker forms; the original
    /// XML + fenced-block union semantics are preserved):
    /// 1. `<|tool_call|>` special tags
    /// 2. `[TOOL_CALLS]` marker
    /// 3. `<tool_call>` XML blocks
    /// 4. ```json / ```tool fenced blocks
    /// 5. `<json>` blocks
    /// 6. standalone JSON (fence-stripped, repaired)
    /// 7. `<think>` embedded content
    /// 8. empty-args recovery (nearby JSON attached to a mentioned tool)
    /// 9. fuzzy canonicalization of tool names
    fn parse_internal(&self, content: &str) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();

        // 1. `<|tool_call|>` special-tag form (Qwen-style templates).
        if results.is_empty() {
            results.extend(self.parse_delimited_calls(
                content,
                "<|tool_call|>",
                &[
                    "<|end|>",
                    "<|end_tool_call|>",
                    "<|tool_call_end|>",
                    "</tool_call>",
                ],
            ));
        }

        // 2. `[TOOL_CALLS]` marker (Mistral-style array or single object).
        if results.is_empty() {
            results.extend(self.parse_tool_calls_marker(content));
        }

        // 3. Try XML <tool_call> blocks
        results.extend(self.parse_xml_tool_calls(content));

        // 4. Try JSON blocks (```json / ```tool ... ```)
        results.extend(self.parse_json_code_blocks(content));

        // 5. Try <json>...</json> blocks
        if results.is_empty() {
            results.extend(self.parse_delimited_calls(content, "<json>", &["</json>"]));
        }

        // 6. Try standalone JSON objects that look like tool calls
        if results.is_empty() {
            results.extend(self.parse_standalone_json(content));
        }

        // 7. Try extracting from <think> blocks (reasoning models)
        if results.is_empty() {
            if let Some(thinking_content) = self.extract_thinking(content) {
                results.extend(self.parse_xml_tool_calls(thinking_content));
                results.extend(self.parse_json_code_blocks(thinking_content));
            }
        }

        // 8. Empty-args recovery: every extracted call lacks usable arguments
        //    (or nothing was extracted at all) — look for a JSON payload near
        //    a tool-name mention.
        if results.iter().all(|c| is_empty_arguments(&c.arguments)) {
            if let Some(call) = self.recover_missing_arguments(content) {
                results.clear();
                results.push(call);
            }
        }

        // 9. Fuzzy canonicalization of tool names against the known set.
        for call in &mut results {
            call.name = self.canonical_tool_name(&call.name);
        }

        results
    }

    /// Parse `opener <payload> closer` regions, trying each closer and taking
    /// the earliest match. Unclosed openers consume the rest of the content.
    fn parse_delimited_calls(
        &self,
        content: &str,
        opener: &str,
        closers: &[&str],
    ) -> Vec<ParsedToolCall> {
        let mut results = Vec::new();
        let mut search_start = 0;

        while let Some(rel) = content[search_start..].find(opener) {
            let inner_start = search_start + rel + opener.len();

            let mut best: Option<(usize, usize)> = None; // (inner_end_rel, resume_rel)
            for closer in closers {
                if let Some(erel) = content[inner_start..].find(closer) {
                    if best.is_none_or(|(bend, _)| erel < bend) {
                        best = Some((erel, erel + closer.len()));
                    }
                }
            }

            let (inner, resume, closed) = match best {
                Some((end_rel, resume_rel)) => (
                    &content[inner_start..inner_start + end_rel],
                    inner_start + resume_rel,
                    true,
                ),
                None => (&content[inner_start..], content.len(), false),
            };

            if let Some(tc) = self.parse_tool_call_json(inner.trim()) {
                results.push(tc);
            }
            if !closed {
                break;
            }
            search_start = resume;
        }

        results
    }

    /// Parse `<tool_call>{"name":"...","arguments":{...}}</tool_call>` XML blocks.
    fn parse_xml_tool_calls(&self, content: &str) -> Vec<ParsedToolCall> {
        self.parse_delimited_calls(content, "<tool_call>", &["</tool_call>"])
    }

    /// Parse `[TOOL_CALLS] [{...}, {...}]` markers (Mistral-style).
    fn parse_tool_calls_marker(&self, content: &str) -> Vec<ParsedToolCall> {
        const MARKER: &str = "[TOOL_CALLS]";
        let idx = match content.find(MARKER) {
            Some(i) => i + MARKER.len(),
            None => return Vec::new(),
        };
        let rest = content[idx..].trim_start();

        // Preferred form: a JSON array of call objects.
        if let Some(&(s, e)) = balanced_spans(rest, '[', ']').first() {
            if let Some(Value::Array(items)) = parse_lenient_value(&rest[s..=e]) {
                let calls: Vec<_> = items
                    .iter()
                    .filter_map(|v| self.tool_call_from_value(v))
                    .collect();
                if !calls.is_empty() {
                    return calls;
                }
            }
        }

        // Fallback: a single object right after the marker.
        if let Some(&(s, e)) = balanced_spans(rest, '{', '}').first() {
            if let Some(tc) = self.parse_tool_call_json(&rest[s..=e]) {
                return vec![tc];
            }
        }

        Vec::new()
    }

    /// Parse JSON code blocks: ```json ... ``` and ```tool ... ```
    fn parse_json_code_blocks(&self, content: &str) -> Vec<ParsedToolCall> {
        const OPENERS: [&str; 2] = ["```json", "```tool"];
        let mut results = Vec::new();
        let mut search_start = 0;

        while let Some((abs_start, opener_len)) = next_fence_opener(content, search_start, &OPENERS)
        {
            let json_start = abs_start + opener_len;

            if let Some(end) = content[json_start..].find("```") {
                let json_str = content[json_start..json_start + end].trim();
                if let Some(tc) = self.parse_tool_call_json(json_str) {
                    results.push(tc);
                }
                search_start = json_start + end + 3;
            } else {
                let json_str = content[json_start..].trim();
                if let Some(tc) = self.parse_tool_call_json(json_str) {
                    results.push(tc);
                }
                break;
            }
        }

        results
    }

    /// Try to parse standalone JSON that looks like a tool call.
    /// Strips wrapping code fences and repairs near-miss JSON before giving up.
    fn parse_standalone_json(&self, content: &str) -> Vec<ParsedToolCall> {
        let stripped = strip_code_fences(content);

        // Try parsing the entire content as a single tool call
        if let Some(tc) = self.parse_tool_call_json(stripped) {
            return vec![tc];
        }

        // Try finding JSON objects in the text (string-aware brace scan)
        let mut results = Vec::new();
        let mut spans = balanced_spans(stripped, '{', '}');
        spans.sort_by_key(|&(s, _)| s);
        for (s, e) in spans {
            if let Some(tc) = self.parse_tool_call_json(&stripped[s..=e]) {
                results.push(tc);
            }
        }

        results
    }

    /// Extract content from <think>...</think> blocks.
    fn extract_thinking<'a>(&self, content: &'a str) -> Option<&'a str> {
        let start = content.find("<think>")?;
        let inner_start = start + 8;
        let end = content[inner_start..].find("</think>")?;
        Some(content[inner_start..inner_start + end].trim())
    }

    /// Try to parse a JSON string as a tool call.
    /// Handles: {"name":"...","arguments":{...}}, {"function":{"name":...}},
    /// and {"tool":"...","parameters":{...}}.
    ///
    /// Falls back to [`repair_json`] when strict parsing fails. A missing
    /// `arguments` field yields an empty object so the empty-args recovery
    /// layer can fill it in later.
    fn parse_tool_call_json(&self, json_str: &str) -> Option<ParsedToolCall> {
        if let Some(val) = parse_lenient_value(json_str) {
            return self.tool_call_from_value(&val);
        }
        None
    }

    /// Detect tool-call shapes in an already-parsed JSON value.
    fn tool_call_from_value(&self, val: &Value) -> Option<ParsedToolCall> {
        // Format 1: {"name": "tool_name", "arguments": {...}}
        if let Some(name) = val.get("name").and_then(Value::as_str) {
            let args = val.get("arguments").cloned().unwrap_or_else(empty_object);
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments: args,
            });
        }

        // Format 2: {"function": {"name": "tool_name", "arguments": {...}}}
        if let Some(func) = val.get("function") {
            if let Some(name) = func.get("name").and_then(Value::as_str) {
                let args = func.get("arguments").cloned().unwrap_or_else(empty_object);
                return Some(ParsedToolCall {
                    name: name.to_string(),
                    arguments: args,
                });
            }
        }

        // Format 3: {"tool": "name", "parameters": {...}}
        if let (Some(name), Some(params)) = (
            val.get("tool").and_then(Value::as_str),
            val.get("parameters"),
        ) {
            return Some(ParsedToolCall {
                name: name.to_string(),
                arguments: params.clone(),
            });
        }

        None
    }

    /// Empty-args recovery: find a known tool name mentioned in the text and
    /// scan a bounded window around the mention for a JSON object to use as
    /// its arguments.
    fn recover_missing_arguments(&self, content: &str) -> Option<ParsedToolCall> {
        let (pos, tool) = self.find_tool_mention(content)?;

        let w_start =
            floor_char_boundary(content, pos.saturating_sub(RECOVERY_WINDOW_HALF_BYTES));
        let w_end = ceil_char_boundary(
            content,
            (pos + tool.len() + RECOVERY_WINDOW_HALF_BYTES).min(content.len()),
        );
        let window = &content[w_start..w_end];
        let mention_off = pos - w_start;

        let mut spans = balanced_spans(window, '{', '}');
        spans.sort_by_key(|&(s, _)| s);
        // Prefer candidates at/after the mention (typical "prose: {args}"
        // ordering), then fall back to preceding objects.
        let (after, before): (Vec<_>, Vec<_>) =
            spans.into_iter().partition(|&(s, _)| s >= mention_off);

        for (s, e) in after.into_iter().chain(before) {
            let candidate = &window[s..=e];
            let Some(val) = parse_lenient_value(candidate) else {
                continue;
            };

            // A complete embedded tool call with real arguments wins outright.
            if let Some(tc) = self.tool_call_from_value(&val) {
                if !is_empty_arguments(&tc.arguments) {
                    return Some(ParsedToolCall {
                        name: self.canonical_tool_name(&tc.name),
                        arguments: tc.arguments,
                    });
                }
                continue; // name-shell with empty args — keep scanning
            }

            // A bare arguments object sitting next to the mention.
            if let Value::Object(map) = &val {
                if !map.is_empty()
                    && !map.contains_key("name")
                    && !map.contains_key("function")
                    && !map.contains_key("tool")
                {
                    return Some(ParsedToolCall {
                        name: tool.clone(),
                        arguments: val,
                    });
                }
            }
        }

        None
    }

    /// First (leftmost) word-bounded, case-insensitive mention of a known tool.
    fn find_tool_mention(&self, content: &str) -> Option<(usize, String)> {
        let mut best: Option<(usize, String)> = None;
        for tool in &self.known_tools {
            if let Some(pos) = find_word_bounded(content, tool) {
                if best.as_ref().is_none_or(|(bp, _)| pos < *bp) {
                    best = Some((pos, tool.clone()));
                }
            }
        }
        best
    }

    /// Map a raw tool name onto the canonical known set:
    /// exact match → case-insensitive match → Levenshtein ≤ 2 (closest wins,
    /// ties broken by list order) → unchanged.
    fn canonical_tool_name(&self, name: &str) -> String {
        if self.known_tools.iter().any(|t| t == name) {
            return name.to_string();
        }
        if let Some(exact) = self.known_tools.iter().find(|t| t.eq_ignore_ascii_case(name)) {
            return exact.clone();
        }
        let lowered = name.to_lowercase();
        let mut best: Option<(&String, usize)> = None;
        for tool in &self.known_tools {
            let d = levenshtein(&lowered, &tool.to_lowercase());
            if d <= FUZZY_MAX_DISTANCE && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((tool, d));
            }
        }
        best.map(|(t, _)| t.clone())
            .unwrap_or_else(|| name.to_string())
    }
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a JSON value strictly, falling back to [`repair_json`] once.
fn parse_lenient_value(s: &str) -> Option<Value> {
    if let Ok(val) = serde_json::from_str::<Value>(s) {
        return Some(val);
    }
    serde_json::from_str::<Value>(&repair_json(s)).ok()
}

/// Best-effort repair of near-miss JSON emitted by weak models.
///
/// Transformations (all context-aware — never applied inside double-quoted
/// strings, so string contents survive untouched):
/// - smart quotes (“ ” „ ‟ ‘ ’ ‚ ‛) → straight quotes
/// - single-quoted strings/values → double-quoted; embedded `"` gets escaped,
///   `\'` becomes a literal apostrophe
/// - trailing commas before `}` / `]` removed
/// - unquoted object keys (`name:`) wrapped in quotes
pub fn repair_json(input: &str) -> String {
    let normalized = normalize_smart_quotes(input);
    let chars: Vec<char> = normalized.chars().collect();
    let mut out = String::with_capacity(normalized.len() + 8);
    let mut in_string = false;
    let mut single_origin = false;
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if in_string {
            match c {
                '\\' if i + 1 < chars.len() => {
                    if chars[i + 1] == '\'' {
                        // \' is invalid in JSON strings; emit a bare apostrophe.
                        out.push('\'');
                    } else {
                        out.push('\\');
                        out.push(chars[i + 1]);
                    }
                    i += 2;
                }
                '"' if single_origin => {
                    // Embedded double quote inside a converted single-quoted
                    // string must be escaped.
                    out.push_str("\\\"");
                    i += 1;
                }
                '"' => {
                    in_string = false;
                    out.push('"');
                    i += 1;
                }
                '\'' if single_origin => {
                    in_string = false;
                    out.push('"');
                    i += 1;
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                single_origin = false;
                out.push('"');
                i += 1;
            }
            '\'' => {
                in_string = true;
                single_origin = true;
                out.push('"');
                i += 1;
            }
            ',' if next_nonspace_is_closer(&chars, i) => {
                // Trailing comma — drop it.
                i += 1;
            }
            c if is_identifier_start(c) => {
                let mut j = i + 1;
                while j < chars.len() && is_identifier_continue(chars[j]) {
                    j += 1;
                }
                let mut k = j;
                while k < chars.len() && chars[k].is_whitespace() {
                    k += 1;
                }
                if j > i + 1 && k < chars.len() && chars[k] == ':' {
                    // Unquoted object key — quote it conservatively.
                    out.push('"');
                    out.extend(chars[i..j].iter());
                    out.push('"');
                    i = j;
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }

    out
}

fn normalize_smart_quotes(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\u{2018}' | '\u{2019}' | '\u{201A}' | '\u{201B}' => '\'',
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' => '"',
            other => other,
        })
        .collect()
}

fn next_nonspace_is_closer(chars: &[char], comma_idx: usize) -> bool {
    let mut j = comma_idx + 1;
    while j < chars.len() && chars[j].is_whitespace() {
        j += 1;
    }
    j < chars.len() && (chars[j] == '}' || chars[j] == ']')
}

fn is_identifier_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

fn is_identifier_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Byte ranges `(start, end)` inclusive of every balanced `open...close` span,
/// ignoring delimiters inside double-quoted strings.
fn balanced_spans(text: &str, open: char, close: char) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            c if c == open => stack.push(i),
            c if c == close => {
                if let Some(s) = stack.pop() {
                    spans.push((s, i));
                }
            }
            _ => {}
        }
    }

    spans
}

/// Strip a wrapping ```lang ... ``` fence (json/tool/plain) from `s`.
/// Returns `s.trim()` unchanged when no leading fence is present.
fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    let Some(rest) = t.strip_prefix("```") else {
        return t;
    };
    // Skip the optional language token up to the newline.
    let body = match rest.find('\n') {
        Some(nl) => &rest[nl + 1..],
        None => rest,
    };
    let body = body.strip_suffix("```").unwrap_or(body);
    body.trim()
}

/// True when `args` is null, non-object, or an empty object — i.e. unusable.
fn is_empty_arguments(args: &Value) -> bool {
    match args {
        Value::Null => true,
        Value::Object(map) => map.is_empty(),
        _ => false,
    }
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

/// Case-insensitive, word-bounded search for `needle` (ASCII) in `hay`.
fn find_word_bounded(hay: &str, needle: &str) -> Option<usize> {
    let nb = needle.as_bytes();
    let hb = hay.as_bytes();
    if nb.is_empty() || hb.len() < nb.len() {
        return None;
    }
    for i in 0..=(hb.len() - nb.len()) {
        if !hb[i..i + nb.len()].eq_ignore_ascii_case(nb) {
            continue;
        }
        let prev_ok =
            i == 0 || !(hb[i - 1].is_ascii_alphanumeric() || hb[i - 1] == b'_');
        let next_i = i + nb.len();
        let next_ok = next_i == hb.len()
            || !(hb[next_i].is_ascii_alphanumeric() || hb[next_i] == b'_');
        if prev_ok && next_ok {
            return Some(i);
        }
    }
    None
}

fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn ceil_char_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// Classic two-row Levenshtein edit distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Earliest occurrence of any opener at/after `from`, as `(abs_pos, opener_len)`.
fn next_fence_opener(content: &str, from: usize, openers: &[&str]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for op in openers {
        if let Some(rel) = content[from..].find(op) {
            let abs = from + rel;
            if best.is_none_or(|(bpos, _)| abs < bpos) {
                best = Some((abs, op.len()));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_xml_tool_call() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "foo.rs");
    }

    #[test]
    fn parse_json_code_block() {
        let parser = ToolCallParser::new();
        let input = r#"```json
{"name":"Bash","arguments":{"command":"ls -la"}}
```"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
    }

    #[test]
    fn parse_function_format() {
        let parser = ToolCallParser::new();
        let input =
            r#"{"function":{"name":"Edit","arguments":{"path":"a.rs","old":"x","new":"y"}}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Edit");
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call>
some text here
<tool_call>{"name":"Edit","arguments":{"path":"a.rs","old":"x","new":"y"}}</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Edit");
    }

    #[test]
    fn parse_from_thinking_block() {
        let parser = ToolCallParser::new();
        let input = r#"<think>I need to read the file first.</think><tool_call>{"name":"Read","arguments":{"path":"main.rs"}}</tool_call>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn parse_empty_content() {
        let parser = ToolCallParser::new();
        let calls = parser.parse("");
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_malformed_json_gracefully() {
        let parser = ToolCallParser::new();
        let input = "<tool_call>{not json}</tool_call>";
        let calls = parser.parse(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_unclosed_xml_tag() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn parse_tool_format() {
        let parser = ToolCallParser::new();
        let input = r#"{"tool":"Grep","parameters":{"pattern":"fn main"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    // --- diagnostics hardening tests ---

    #[test]
    fn diagnostics_malformed_xml_returns_error() {
        let parser = ToolCallParser::new();
        let input = "<tool_call>{not json}</tool_call>";
        let calls = parser.parse(input);
        assert!(
            calls.is_empty(),
            "parse should still return empty for malformed"
        );

        let (calls2, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls2.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(
            diagnostics[0].contains("Malformed"),
            "diagnostic should contain Malformed, got: {}",
            diagnostics[0]
        );
        assert!(
            diagnostics[0].contains("Available tools"),
            "diagnostic should list available tools"
        );

        let err = parser.try_parse_with_error(input).unwrap_err();
        assert!(err.contains("Malformed"));
    }

    #[test]
    fn diagnostics_valid_xml_no_error() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"foo.rs"}}</tool_call>"#;
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert_eq!(calls.len(), 1);
        assert!(
            diagnostics.is_empty(),
            "valid call should produce no diagnostics"
        );
        assert!(parser.try_parse_with_error(input).is_ok());
    }

    #[test]
    fn diagnostics_plain_text_no_error() {
        let parser = ToolCallParser::new();
        let input = "Hello world, this is just plain text without any tool intent.";
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert!(
            diagnostics.is_empty(),
            "plain text should not trigger diagnostic"
        );
        assert!(parser.try_parse_with_error(input).is_ok());
        assert!(parser.try_parse_with_error(input).unwrap().is_empty());
    }

    #[test]
    fn diagnostics_malformed_json_block() {
        let parser = ToolCallParser::new();
        let input = "```json\n{not valid json}\n```";
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("Malformed"));
    }

    #[test]
    fn diagnostics_malformed_standalone_name() {
        let parser = ToolCallParser::new();
        // Contains {"name" heuristic but invalid JSON / missing arguments field
        let input = r#"{"name": "Read", "arguments": not_json}"#;
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(calls.is_empty());
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].contains("Malformed"));
    }

    #[test]
    fn diagnostics_empty_no_error() {
        let parser = ToolCallParser::new();
        let (calls, diagnostics) = parser.parse_with_diagnostics("");
        assert!(calls.is_empty());
        assert!(diagnostics.is_empty());
        assert!(parser.try_parse_with_error("").is_ok());
    }

    // --- repair_json unit tests (each transformation isolated) ---

    #[test]
    fn repair_single_to_double_quotes() {
        assert_eq!(repair_json("{'name': 'Read'}"), r#"{"name": "Read"}"#);
    }

    #[test]
    fn repair_trailing_comma_object() {
        assert_eq!(repair_json(r#"{"name": "Read",}"#), r#"{"name": "Read"}"#);
    }

    #[test]
    fn repair_trailing_comma_array() {
        assert_eq!(repair_json("[1, 2, 3, ]"), "[1, 2, 3 ]");
    }

    #[test]
    fn repair_unquoted_keys() {
        assert_eq!(
            repair_json(r#"{name: "Read", arguments: {"path": "x.rs"}}"#),
            r#"{"name": "Read", "arguments": {"path": "x.rs"}}"#
        );
    }

    #[test]
    fn repair_smart_quotes() {
        let input = "{\u{201c}name\u{201d}: \u{2018}Read\u{2019}}";
        assert_eq!(repair_json(input), r#"{"name": "Read"}"#);
    }

    #[test]
    fn repair_escapes_double_quotes_inside_single_quoted_strings() {
        assert_eq!(
            repair_json(r#"{'command': 'echo "hi"'}"#),
            r#"{"command": "echo \"hi\""}"#
        );
    }

    #[test]
    fn repair_apostrophe_escape_becomes_literal() {
        assert_eq!(
            repair_json(r#"{'path': 'don\'t.rs'}"#),
            r#"{"path": "don't.rs"}"#
        );
    }

    #[test]
    fn repair_valid_json_unchanged() {
        let valid = r#"{"name":"Read","arguments":{"path":"a b, c}d"}}"#;
        assert_eq!(repair_json(valid), valid);
    }

    #[test]
    fn repair_leaves_value_identifiers_alone() {
        // Identifier VALUES without a following colon must not be quoted.
        let input = r#"{"name": "Read", "arguments": not_json}"#;
        assert_eq!(repair_json(input), input);
    }

    #[test]
    fn repair_keeps_commas_and_braces_inside_strings() {
        let input = r#"{"cmd": "echo a, } ] b"}"#;
        assert_eq!(repair_json(input), input);
    }

    // --- recovery layer integration tests (real malformed samples) ---

    #[test]
    fn parse_special_tag_form() {
        let parser = ToolCallParser::new();
        let input = r#"<|tool_call|>{"name":"Read","arguments":{"path":"foo.rs"}}<|end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "foo.rs");
    }

    #[test]
    fn parse_special_tag_unclosed() {
        let parser = ToolCallParser::new();
        let input = r#"<|tool_call|>{"name":"Bash","arguments":{"command":"ls"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn parse_tool_calls_marker_array() {
        let parser = ToolCallParser::new();
        let input = r#"[TOOL_CALLS] [{"name":"Read","arguments":{"path":"a.rs"}},{"name":"Bash","arguments":{"command":"ls"}}]"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Bash");
    }

    #[test]
    fn parse_tool_calls_marker_single_object() {
        let parser = ToolCallParser::new();
        let input = r#"[TOOL_CALLS] {"name":"Grep","arguments":{"pattern":"fn main"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    #[test]
    fn parse_tool_fence_block() {
        let parser = ToolCallParser::new();
        let input = "```tool\n{\"name\":\"Write\",\"arguments\":{\"path\":\"a.txt\",\"content\":\"hi\"}}\n```";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["content"], "hi");
    }

    #[test]
    fn parse_json_tag_block() {
        let parser = ToolCallParser::new();
        let input = r#"<json>{"name":"Glob","arguments":{"pattern":"*.rs"}}</json>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Glob");
        assert_eq!(calls[0].arguments["pattern"], "*.rs");
    }

    #[test]
    fn parse_fenced_single_quotes_recovered() {
        let parser = ToolCallParser::new();
        let input = "```json\n{'name': 'Bash', 'arguments': {'command': 'cargo test'}}\n```";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "cargo test");
    }

    #[test]
    fn parse_fenced_trailing_comma_recovered() {
        let parser = ToolCallParser::new();
        let input = "```json\n{\"name\":\"Edit\",\"arguments\":{\"path\":\"a.rs\",}}\n```";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Edit");
        assert_eq!(calls[0].arguments["path"], "a.rs");
    }

    #[test]
    fn parse_fenced_unquoted_keys_recovered() {
        let parser = ToolCallParser::new();
        let input = "```json\n{name: \"Grep\", arguments: {pattern: \"TODO\"}}\n```";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "TODO");
    }

    #[test]
    fn parse_prose_with_embedded_json() {
        let parser = ToolCallParser::new();
        let input = r#"I'll inspect the module now using Read: {"path": "src/lib.rs"}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "src/lib.rs");
    }

    #[test]
    fn parse_name_shell_with_args_nearby() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Bash","arguments":{}}</tool_call>
{"command":"ls -la","cwd":"/tmp"}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls -la");
        assert_eq!(calls[0].arguments["cwd"], "/tmp");
    }

    #[test]
    fn parse_name_only_yields_empty_args_call() {
        let parser = ToolCallParser::new();
        let input = r#"{"name":"Read"}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn recovery_respects_window_cap() {
        let parser = ToolCallParser::new();
        // Args JSON sits ~9K chars past the mention — beyond the 8K window.
        let filler = "x".repeat(9000);
        let input = format!("I will use Read now. {filler} {{\"path\":\"a.rs\"}}");
        let calls = parser.parse(&input);
        assert!(calls.is_empty(), "expected no recovery beyond window cap");
    }

    #[test]
    fn recovery_requires_json_payload() {
        let parser = ToolCallParser::new();
        let input = "I will use Bash to run the tests now.";
        let calls = parser.parse(input);
        assert!(calls.is_empty(), "prose mention without JSON must not synthesize a call");
    }

    // --- fuzzy tool-name tests ---

    #[test]
    fn fuzzy_canonicalizes_lowercase() {
        let parser = ToolCallParser::new();
        let input = r#"{"name":"read","arguments":{"path":"x.rs"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn fuzzy_canonicalizes_typo_within_two_edits() {
        let parser = ToolCallParser::new();
        let input = r#"{"name":"Bashe","arguments":{"command":"ls"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
    }

    #[test]
    fn fuzzy_unknown_name_preserved() {
        let parser = ToolCallParser::new();
        let input = r#"{"name":"deploy_worker","arguments":{"env":"prod"}}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "deploy_worker");
    }

    #[test]
    fn with_known_tools_restricts_fuzzy_matching() {
        let parser = ToolCallParser::new().with_known_tools(&["Fetch"]);
        let typo = parser.parse(r#"{"name":"fetxh","arguments":{"url":"https://x.dev"}}"#);
        assert_eq!(typo.len(), 1);
        assert_eq!(typo[0].name, "Fetch");

        // "read" is not within distance 2 of "Fetch" — left untouched.
        let unmapped = parser.parse(r#"{"name":"read","arguments":{"path":"x"}}"#);
        assert_eq!(unmapped.len(), 1);
        assert_eq!(unmapped[0].name, "read");
    }

    // --- regression guards for well-formed multi-format inputs ---

    #[test]
    fn xml_and_fence_union_semantics_preserved() {
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call>
```json
{"name":"Bash","arguments":{"command":"ls"}}
```"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Bash");
    }

    #[test]
    fn wellformed_names_never_mutated() {
        let parser = ToolCallParser::new();
        for tool in DEFAULT_KNOWN_TOOLS {
            let input = format!(r#"{{"name":"{tool}","arguments":{{"k":"v"}}}}"#);
            let calls = parser.parse(&input);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].name, *tool);
        }
    }

    #[test]
    fn diagnostics_new_intent_markers_fire() {
        let parser = ToolCallParser::new();
        for marker_input in [
            "<|tool_call|>{garbage}<|end|>".to_string(),
            "[TOOL_CALLS] {garbage}".to_string(),
            "<json>{garbage}</json>".to_string(),
        ] {
            let (calls, diagnostics) = parser.parse_with_diagnostics(&marker_input);
            assert!(calls.is_empty(), "input: {marker_input}");
            assert_eq!(diagnostics.len(), 1, "input: {marker_input}");
            assert!(diagnostics[0].contains("Malformed"));
        }
    }

    // --- helper unit tests ---

    #[test]
    fn levenshtein_basics() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("read", "Read"), 1); // case-sensitive; callers lowercase first
        assert_eq!(levenshtein("bashe", "bash"), 1);
        assert_eq!(levenshtein("reda", "read"), 2);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn balanced_spans_ignores_braces_in_strings() {
        let text = r#"pre {"a": "}{"} post {"b": 1}"#;
        let spans = balanced_spans(text, '{', '}');
        assert_eq!(spans.len(), 2);
        assert_eq!(&text[spans[0].0..=spans[0].1], r#"{"a": "}{"}"#);
        assert_eq!(&text[spans[1].0..=spans[1].1], r#"{"b": 1}"#);
    }

    #[test]
    fn strip_code_fences_variants() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), r#"{"a":1}"#);
        assert_eq!(strip_code_fences("```\n{\"a\":1}\n```"), r#"{"a":1}"#);
        assert_eq!(strip_code_fences("{\"a\":1}"), "{\"a\":1}");
        assert_eq!(strip_code_fences("  text only  "), "text only");
    }

    #[test]
    fn find_word_bounded_boundaries() {
        assert_eq!(find_word_bounded("please Read this", "Read"), Some(7));
        assert_eq!(find_word_bounded("already done", "read"), None);
        assert_eq!(find_word_bounded("my_Read var", "Read"), None);
        assert_eq!(find_word_bounded("READS many", "Read"), None);
        assert_eq!(find_word_bounded("READ:", "read"), Some(0));
    }
}
