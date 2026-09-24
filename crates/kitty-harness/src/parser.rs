//! Forgiving parser for tool calls from model output.
//!
//! Parses tool calls from multiple formats (first-match wins):
//! - Native Ollama `tool_calls` in the message object
//! - `<|tool_call|>` special tags (Qwen-style chat templates)
//! - `[TOOL_CALLS]` markers (Mistral-style arrays)
//! - `<tool_call>` XML blocks in text content
//! - Python-kwarg calls `[name(k='v', ...)]` in `<|tool_call_start|>`
//!   (MiniCPM5 / Liquid-style templates)
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

use std::borrow::Cow;

use serde_json::Value;

/// A parsed tool call with name and arguments.
///
/// Salvage convention: when a call's arguments payload cannot be parsed as a
/// JSON object, the call is retained with `arguments` set to
/// `{"_raw": "<best-effort body>"}` instead of being dropped outright.
/// `_raw`-only arguments count as empty for [`is_empty_arguments`], so the
/// downstream empty-args recovery layer can still attach real arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Value,
    /// Server-issued call id when parsed from a typed `ToolCall` (None for
    /// calls parsed out of free text, which carry no id).
    pub id: Option<String>,
}

/// Canonical tool names used for fuzzy-name matching when no explicit set is provided.
pub const DEFAULT_KNOWN_TOOLS: &[&str] = &["Read", "Edit", "Write", "Bash", "Glob", "Grep"];

/// Maximum edit distance for fuzzy tool-name matching.
const FUZZY_MAX_DISTANCE: usize = 2;

/// Half-size (in bytes) of the text window scanned around a tool-name mention
/// during empty-args recovery (~8K total window).
const RECOVERY_WINDOW_HALF_BYTES: usize = 4 * 1024;

/// Cap (in bytes) for late/recovery text scans: tool-call markers appear near
/// the end of streamed buffers, so scanning unbounded content is an O(n) hot
/// spot on long conversations.
const RECOVER_SCAN_MAX_BYTES: usize = 32_000;

/// Maximum accepted tool-name length for inline-name payloads (`NAME{json}`).
const MAX_INLINE_NAME_LEN: usize = 64;

/// Key prefixes introducing the target tool in Gemma channel-tagged forms.
const CHANNEL_TARGET_KEYS: [&str; 3] = ["to=", "tool=", "name="];

/// How far (in bytes) past a channel target (`to=NAME`) to look for the
/// following `<|message|>` / `<message>` tag.
const MESSAGE_TAG_LOOKAHEAD_BYTES: usize = 200;

/// Marker introducing the loose Gemma `call:NAME{json}` fragment form.
const LOOSE_CALL_MARKER: &str = "call:";

/// Opener for MiniCPM5/Liquid-style Python-kwarg tool calls:
/// `<|tool_call_start|>[write_file(path='a.py')]<|tool_call_end|>`.
const KWARG_CALL_OPENER: &str = "<|tool_call_start|>";
/// Closers accepted for the kwarg form (the `_start` opener pairs with any).
const KWARG_CALL_CLOSERS: &[&str] = &["<|tool_call_end|>", "<|end_tool_call|>", "</tool_call>"];
/// Maximum nesting depth for Python-literal kwarg values (lists/dicts).
const KWARG_MAX_DEPTH: u32 = 8;

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
            || content.contains("<|tool_call_start|>")
            || content.contains("[TOOL_CALLS]")
            || content.contains("```json")
            || content.contains("<json>")
            || content.contains(r#"{"name""#)
            || content.contains(r#"{"function""#)
            || content.contains(r#"{"tool""#)
    }

    /// Internal parsing logic without diagnostics side-effects.
    ///
    /// Content is first normalized for Gemma quote-token artifacts
    /// ([`normalize_gemma_quote_tokens`]).
    ///
    /// Layer order (first-match wins for the tagged/marker forms; the original
    /// XML + fenced-block union semantics are preserved):
    /// 1. `<|tool_call|>` special tags
    ///    1b. asymmetric `<|tool_call>` opener (Gemma 4 wild variant)
    ///    1c. `<|tool_call_start|>` + Python-kwarg calls (MiniCPM5/Liquid)
    /// 2. `[TOOL_CALLS]` marker
    /// 3. `<tool_call>` XML blocks
    /// 4. ```json / ```tool fenced blocks
    /// 5. `<json>` blocks
    /// 6. standalone JSON (fence-stripped, repaired)
    /// 7. `<think>` embedded content
    ///    7b. Gemma channel-tagged / loose `call:NAME{json}` fragments (tail 32K)
    /// 8. empty-args recovery (nearby JSON attached to a mentioned tool)
    /// 9. fuzzy canonicalization of tool names
    fn parse_internal(&self, content: &str) -> Vec<ParsedToolCall> {
        let content = normalize_gemma_quote_tokens(content);
        let content: &str = content.as_ref();
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

        // 1b. Asymmetric `<|tool_call>` opener (Gemma 4 wild variant, no
        //     trailing pipe). Must run after the symmetric form because
        //     `<|tool_call>` is a prefix of `<|tool_call|>`.
        if results.is_empty() {
            results.extend(self.parse_asymmetric_tool_calls(content));
        }

        // 1c. `<|tool_call_start|>[name(k='v', ...)]` (MiniCPM5/Liquid-style
        //     Python kwargs). Runs after the JSON-tag layers so JSON
        //     envelopes keep priority; disjoint openers (no JSON layer
        //     matches `<|tool_call_start|>`) keep every call owned by one
        //     layer. Names are canonicalized by the fuzzy layer afterwards.
        if results.is_empty() {
            results.extend(self.parse_python_kwarg_calls(content));
        }

        // 2. `[TOOL_CALLS]` marker (Mistral-style array or single object).
        if results.is_empty() {
            results.extend(self.parse_tool_calls_marker(content));
        }

        // 3. XML <tool_call> blocks — fully subsumed by the asymmetric layer
        //    above (its opener/closer/payload sets are supersets), so it only
        //    runs when that layer produced nothing. Keeping every bare block
        //    owned by exactly ONE layer prevents the same call from being
        //    parsed twice and dispatched twice. The standalone helper remains
        //    for the <think>-block path below.
        if results.is_empty() {
            results.extend(self.parse_xml_tool_calls(content));
        }

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

        // 7b. Gemma channel-tagged and loose `call:NAME{json}` fragments
        //     (late recovery, scanned within the tail of the content only).
        if results.is_empty() {
            results.extend(self.parse_channel_tagged_forms(content));
        }
        if results.is_empty() {
            results.extend(self.parse_loose_call_forms(content));
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

        // Collapse repeated identical calls for SIDE-EFFECT-FREE tools only:
        // a repeated read carries no information (and overlapping opener
        // prefixes can yield the same read twice), while identical stateful
        // calls (Bash/Edit/Write) are legitimate sequences or retries and are
        // always kept.
        const READ_ONLY_TOOLS: [&str; 3] = ["Read", "Glob", "Grep"];
        let mut unique: Vec<ParsedToolCall> = Vec::with_capacity(results.len());
        for call in results {
            let duplicate_read_only = READ_ONLY_TOOLS.contains(&call.name.as_str())
                && unique
                    .iter()
                    .any(|s| s.name == call.name && s.arguments == call.arguments);
            if !duplicate_read_only {
                unique.push(call);
            }
        }
        results = unique;

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
        self.parse_delimited_calls_with(content, opener, closers, |payload| {
            self.parse_tool_call_json(payload)
        })
    }

    /// Like [`Self::parse_delimited_calls`], with a pluggable payload parser
    /// (the asymmetric Gemma layer accepts inline-name payloads in addition
    /// to JSON envelopes).
    fn parse_delimited_calls_with(
        &self,
        content: &str,
        opener: &str,
        closers: &[&str],
        parse_payload: impl Fn(&str) -> Option<ParsedToolCall>,
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

            if let Some(tc) = parse_payload(inner.trim()) {
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

    /// Parse `<|tool_call_start|>...` regions with a Python-kwarg payload
    /// parser (see [`Self::parse_python_kwarg_call`]).
    fn parse_python_kwarg_calls(&self, content: &str) -> Vec<ParsedToolCall> {
        self.parse_delimited_calls_with(content, KWARG_CALL_OPENER, KWARG_CALL_CLOSERS, |payload| {
            self.parse_python_kwarg_call(payload)
        })
    }

    /// Parse one `[name(k='v', n=3)]` (or bare `name(k='v')`) payload into a
    /// call. Returns `None` when the shape is not a kwarg call — the layer
    /// then yields nothing, later layers still run, and the intent heuristic
    /// reports the attempt as malformed rather than dropping it silently.
    fn parse_python_kwarg_call(&self, payload: &str) -> Option<ParsedToolCall> {
        let mut body = payload.trim();
        if let Some(stripped) = body.strip_prefix('[') {
            body = stripped.strip_suffix(']')?.trim();
        }
        let paren = body.find('(')?;
        let name = body[..paren].trim();
        if !is_word_like(name) || name.len() > MAX_INLINE_NAME_LEN {
            return None;
        }
        let after = body[paren + 1..].trim_start();
        let close = find_matching_paren(after)?;
        let args_str = after[..close].trim();
        if !after[close + 1..].trim().is_empty() {
            return None; // trailing garbage after `)` — not a clean call
        }
        let mut args = serde_json::Map::new();
        if !args_str.is_empty() {
            for item in split_top_level(args_str, ',') {
                let (key, value) = parse_kwarg_item(item.trim(), 0)?;
                if !is_word_like(key) {
                    return None;
                }
                args.insert(key.to_string(), value);
            }
        }
        Some(ParsedToolCall {
            name: name.to_string(),
            arguments: Value::Object(args),
            id: None,
        })
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

    /// Parse asymmetric Gemma tool-call blocks across the opener variants
    /// seen in the wild — `<|tool_call>` (no trailing pipe), the symmetric
    /// `<|tool_call|>`, and bare `<tool_call>` — each combinable with any of
    /// the closers `<tool_call|>`, `<|end_tool_call|>`, `</tool_call>`. The
    /// payload may be a JSON envelope or the inline-name form `NAME{json}`.
    fn parse_asymmetric_tool_calls(&self, content: &str) -> Vec<ParsedToolCall> {
        const OPENERS: [&str; 3] = ["<|tool_call>", "<|tool_call|>", "<tool_call>"];
        const CLOSERS: [&str; 3] = ["<tool_call|>", "<|end_tool_call|>", "</tool_call>"];
        let mut results = Vec::new();
        // Opener variants overlap by design: the pipeless form is a strict
        // prefix of the piped one, so symmetric-tag content is probed by both
        // passes. A misaligned pass yields a shifted payload that fails to
        // parse and self-suppresses; when two passes do parse the same block,
        // the results are identical calls, and repeated identical READS are
        // collapsed at the end of parse_internal. The XML layer below is
        // gated off once this layer produces anything, so a bare block always
        // has exactly one owning layer and is never dispatched twice.
        for opener in OPENERS {
            results.extend(
                self.parse_delimited_calls_with(content, opener, &CLOSERS, |payload| {
                    self.parse_tool_call_json(payload)
                        .or_else(|| self.parse_inline_name_payload(payload))
                }),
            );
        }
        results
    }

    /// Parse the inline-name payload form `NAME{json}` (Gemma 4): the trimmed
    /// prefix before the first balanced `{...}` object is the tool name. The
    /// prefix must be non-empty, ≤ [`MAX_INLINE_NAME_LEN`] chars, and word-like.
    fn parse_inline_name_payload(&self, payload: &str) -> Option<ParsedToolCall> {
        let brace = payload.find('{')?;
        let (s, e) = *balanced_spans(payload, '{', '}')
            .iter()
            .find(|&&(s, _)| s == brace)?;
        let name = payload[..brace].trim();
        if !is_word_like(name) || name.len() > MAX_INLINE_NAME_LEN {
            return None;
        }
        let arguments = parse_lenient_value(&payload[s..=e])?;
        Some(ParsedToolCall {
            name: name.to_string(),
            arguments,
            id: None,
        })
    }

    /// Parse Gemma channel-tagged intent fragments such as
    /// `<|channel|>commentary to=WebSearch <|message|>{"query":"x"}<|im_end|>`
    /// (also `tool=` / `name=`, tags possibly without pipes). Scanned within
    /// the last [`RECOVER_SCAN_MAX_BYTES`] bytes only; the leftmost candidate
    /// whose message payload is a non-empty JSON object wins.
    fn parse_channel_tagged_forms(&self, content: &str) -> Vec<ParsedToolCall> {
        let tail = recover_scan_tail(content);
        let mut candidates: Vec<(usize, &str)> = Vec::new();
        for tool in &self.known_tools {
            for key in CHANNEL_TARGET_KEYS {
                let needle = format!("{key}{tool}");
                if let Some(pos) = find_word_bounded(tail, &needle) {
                    candidates.push((pos, tool));
                }
            }
        }
        candidates.sort_by_key(|&(pos, _)| pos);
        for (pos, tool) in candidates {
            if let Some(call) = channel_payload_after_target(tail, pos, tool) {
                return vec![call];
            }
        }
        Vec::new()
    }

    /// Parse loose Gemma `call:NAME{json}` fragments. Scanned within the last
    /// [`RECOVER_SCAN_MAX_BYTES`] bytes only; every well-formed occurrence
    /// yields a call (names are canonicalized by the fuzzy layer afterwards).
    fn parse_loose_call_forms(&self, content: &str) -> Vec<ParsedToolCall> {
        let tail = recover_scan_tail(content);
        let bytes = tail.as_bytes();
        let mut results = Vec::new();
        let mut search = 0;
        while let Some(rel) = tail[search..].find(LOOSE_CALL_MARKER) {
            let abs = search + rel;
            search = abs + LOOSE_CALL_MARKER.len();
            // Require a word boundary before `call` (skip e.g. `recall:`).
            if abs > 0 && (bytes[abs - 1].is_ascii_alphanumeric() || bytes[abs - 1] == b'_') {
                continue;
            }
            if let Some((name, arguments)) = parse_loose_call_payload(&tail[search..]) {
                results.push(ParsedToolCall {
                    name: name.to_string(),
                    arguments,
                    id: None,
                });
            }
        }
        results
    }

    /// Try to parse a JSON string as a tool call.
    /// Handles: {"name":"...","arguments":{...}}, {"function":{"name":...}},
    /// and {"tool":"...","parameters":{...}}.
    ///
    /// Falls back to [`repair_json`] when strict parsing fails. When even
    /// repair cannot parse the body, a recoverable tool name is salvaged and
    /// the call is retained with its best-effort body as `_raw` (see
    /// [`ParsedToolCall`]) instead of being dropped. A missing arguments
    /// field yields an empty object so the empty-args recovery layer can
    /// fill it in later.
    fn parse_tool_call_json(&self, json_str: &str) -> Option<ParsedToolCall> {
        if let Some(val) = parse_lenient_value(json_str) {
            return self.tool_call_from_value(&val);
        }
        // Embedded-object tolerance: a balanced JSON object inside otherwise
        // noisy payload text (e.g. an unrecognized closer tag after it)
        // parses from its own span before anything else is tried. The
        // earliest-starting span is the OUTERMOST object.
        if let Some(&(s, e)) = balanced_spans(json_str, '{', '}')
            .iter()
            .min_by_key(|&&(s, _)| s)
        {
            if let Some(val) = parse_lenient_value(&json_str[s..=e]) {
                if let Some(tc) = self.tool_call_from_value(&val) {
                    return Some(tc);
                }
            }
        }
        // Salvage: the JSON could not be parsed even after repair. If a tool
        // name is still recoverable, retain the call with its best-effort
        // body as `_raw` instead of dropping it outright (empty arguments
        // let downstream recovery layers fill them in).
        let name = salvage_name_from_body(json_str)?;
        Some(ParsedToolCall {
            name,
            arguments: serde_json::json!({ "_raw": json_str.trim() }),
            id: None,
        })
    }

    /// Detect tool-call shapes in an already-parsed JSON value.
    ///
    /// Arguments are read through the alias chain `arguments` → `parameters`
    /// → `args`. A non-object payload is retained as an `_raw` string on the
    /// call rather than poisoning dispatch with unusable arguments.
    fn tool_call_from_value(&self, val: &Value) -> Option<ParsedToolCall> {
        // Format 1: {"name": "tool_name", "arguments": {...}}
        if let Some(name) = val.get("name").and_then(Value::as_str) {
            return Some(finish_call(name, args_field(val)));
        }

        // Format 2: {"function": {"name": "tool_name", "arguments": {...}}}
        if let Some(func) = val.get("function") {
            if let Some(name) = func.get("name").and_then(Value::as_str) {
                return Some(finish_call(name, args_field(func)));
            }
        }

        // Format 3: {"tool": "name", "parameters": {...}}
        if let Some(name) = val.get("tool").and_then(Value::as_str) {
            if let Some(args) = args_field(val) {
                return Some(finish_call(name, Some(args)));
            }
        }

        None
    }

    /// Empty-args recovery: find a known tool name mentioned in the text and
    /// scan a bounded window around the mention for a JSON object to use as
    /// its arguments. The scan is capped to the last
    /// [`RECOVER_SCAN_MAX_BYTES`] bytes: tool-call markers appear near the end
    /// of streamed buffers, so scanning unbounded content is an O(n) hot spot.
    fn recover_missing_arguments(&self, content: &str) -> Option<ParsedToolCall> {
        // All indices below index into the tail slice.
        let content = recover_scan_tail(content);

        let (pos, tool) = self.find_tool_mention(content)?;

        let w_start = floor_char_boundary(content, pos.saturating_sub(RECOVERY_WINDOW_HALF_BYTES));
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
                    let id = val.get("id").and_then(Value::as_str).map(str::to_string);
                    return Some(ParsedToolCall {
                        name: self.canonical_tool_name(&tc.name),
                        arguments: tc.arguments,
                        id,
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
                        id: None,
                    });
                }
            }
        }

        // Salvage chain: structured fragment forms scoped to the same
        // window — channel-tagged targets, loose `call:NAME{json}`, then an
        // inline `NAME{json}` starting right at a mention of the tool.
        if let Some(call) = self
            .parse_channel_tagged_forms(window)
            .into_iter()
            .find(|c| c.name.eq_ignore_ascii_case(&tool))
        {
            return Some(call);
        }
        if let Some(call) = self
            .parse_loose_call_forms(window)
            .into_iter()
            .find(|c| c.name.eq_ignore_ascii_case(&tool))
        {
            return Some(call);
        }
        let mut scan = mention_off;
        while let Some(rel) = find_word_bounded(&window[scan..], &tool) {
            let start = scan + rel;
            if let Some(call) = self.parse_inline_name_payload(&window[start..]) {
                if !is_empty_arguments(&call.arguments) {
                    return Some(call);
                }
            }
            // `tool` is ASCII, so this advance stays on a char boundary and
            // always makes progress.
            scan = start + tool.len();
            if scan >= window.len() {
                break;
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
        if let Some(exact) = self
            .known_tools
            .iter()
            .find(|t| t.eq_ignore_ascii_case(name))
        {
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

/// True when `s` is a non-empty identifier-style word: ASCII alphanumeric /
/// underscore, starting with a letter or underscore.
fn is_word_like(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_identifier_start(c) => chars.all(is_identifier_continue),
        _ => false,
    }
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

/// Normalize Gemma tokenizer artifacts where `<|"|>` / `<|'|>` appear in place
/// of plain quote characters inside tool-call payloads. Returns a borrowed
/// slice unchanged (zero allocation) when no token is present.
fn normalize_gemma_quote_tokens(content: &str) -> Cow<'_, str> {
    const DOUBLE_TOKEN: &str = "<|\"|>";
    const SINGLE_TOKEN: &str = "<|'|>";
    if !content.contains(DOUBLE_TOKEN) && !content.contains(SINGLE_TOKEN) {
        return Cow::Borrowed(content);
    }
    Cow::Owned(
        content
            .replace(DOUBLE_TOKEN, "\"")
            .replace(SINGLE_TOKEN, "'"),
    )
}

/// True when `args` is null, non-object, empty, or carries only the `_raw`
/// salvage payload — i.e. no usable arguments yet.
fn is_empty_arguments(args: &Value) -> bool {
    match args {
        Value::Null => true,
        Value::Object(map) => map.is_empty() || (map.len() == 1 && map.contains_key("_raw")),
        _ => false,
    }
}

/// First present arguments field among the accepted aliases
/// (`arguments` → `parameters` → `args`).
fn args_field(val: &Value) -> Option<&Value> {
    ["arguments", "parameters", "args"]
        .iter()
        .find_map(|key| val.get(key))
}

/// Assemble the final call for a recovered name: object payloads clone
/// through verbatim, absent payloads become an empty object, and non-object
/// payloads are retained as the `_raw` salvage string (see
/// [`ParsedToolCall`]).
fn finish_call(name: &str, args: Option<&Value>) -> ParsedToolCall {
    match args {
        None => ParsedToolCall {
            name: name.to_string(),
            arguments: empty_object(),
            id: None,
        },
        Some(v) if v.is_object() => ParsedToolCall {
            name: name.to_string(),
            arguments: v.clone(),
            id: None,
        },
        Some(v) => ParsedToolCall {
            name: name.to_string(),
            arguments: serde_json::json!({ "_raw": v.to_string() }),
            id: None,
        },
    }
}

/// Best-effort extraction of a `"name": "X"` value from JSON that could not
/// be parsed even after repair. The value must be word-like and within
/// [`MAX_INLINE_NAME_LEN`].
fn salvage_name_from_body(body: &str) -> Option<String> {
    const KEY: &str = "\"name\"";
    let key = body.find(KEY)?;
    let rest = body[key + KEY.len()..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let name = &rest[..end];
    (is_word_like(name) && name.len() <= MAX_INLINE_NAME_LEN).then(|| name.to_string())
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
        let prev_ok = i == 0 || !(hb[i - 1].is_ascii_alphanumeric() || hb[i - 1] == b'_');
        let next_i = i + nb.len();
        let next_ok =
            next_i == hb.len() || !(hb[next_i].is_ascii_alphanumeric() || hb[next_i] == b'_');
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

/// Last [`RECOVER_SCAN_MAX_BYTES`] bytes of `content` (char-boundary safe).
fn recover_scan_tail(content: &str) -> &str {
    let start = floor_char_boundary(
        content,
        content.len().saturating_sub(RECOVER_SCAN_MAX_BYTES),
    );
    &content[start..]
}

/// From a `to=NAME`-style match at `pos`, find the next `<|message|>` /
/// `<message>` tag within [`MESSAGE_TAG_LOOKAHEAD_BYTES`] and parse the first
/// balanced JSON object after it. The TAG lookup is bounded, but the payload
/// scan runs on the unbounded remainder so large arguments are not clipped.
/// Requires a non-empty JSON object payload.
fn channel_payload_after_target(tail: &str, pos: usize, tool: &str) -> Option<ParsedToolCall> {
    let window_end = ceil_char_boundary(tail, (pos + MESSAGE_TAG_LOOKAHEAD_BYTES).min(tail.len()));
    let window = &tail[pos..window_end];
    let (tag_rel, tag_len) = window
        .find("<|message|>")
        .map(|rel| (rel, "<|message|>".len()))
        .into_iter()
        .chain(window.find("<message>").map(|rel| (rel, "<message>".len())))
        .min_by_key(|&(rel, _)| rel)?;
    // Convert to an absolute offset in `tail`, then scan the UNBOUNDED
    // remainder for the payload — only the tag lookup is window-bounded.
    let after = &tail[pos + tag_rel + tag_len..];
    let (s, e) = *balanced_spans(after, '{', '}')
        .iter()
        .min_by_key(|&&(s, _)| s)?;
    let Value::Object(map) = parse_lenient_value(&after[s..=e])? else {
        return None;
    };
    if map.is_empty() {
        return None;
    }
    Some(ParsedToolCall {
        name: tool.to_string(),
        arguments: Value::Object(map),
        id: None,
    })
}

/// Parse the payload following a `call:` marker: optional whitespace, a word
/// (the raw tool name), optional whitespace, then one balanced `{...}` JSON
/// object. Returns `(name, arguments)`; `None` when the shape does not match
/// or the JSON cannot be parsed.
fn parse_loose_call_payload(rest: &str) -> Option<(&str, Value)> {
    let rest = rest.trim_start();
    let name_end = rest
        .find(|c: char| !is_identifier_continue(c))
        .unwrap_or(rest.len());
    if name_end == 0 {
        return None;
    }
    let name = &rest[..name_end];
    let rest = rest[name_end..].trim_start();
    let (s, e) = *balanced_spans(rest, '{', '}')
        .iter()
        .find(|&&(s, _)| s == 0)?;
    Some((name, parse_lenient_value(&rest[s..=e])?))
}

/// Byte index of the `)` closing the paren opened just before `s` (depth
/// starts at 1). Skips quoted regions (both quote styles, backslash
/// escapes); nested `(`/`)` adjust depth, other brackets are inert. All
/// returned indices land on ASCII bytes, so slicing at them is safe.
fn find_matching_paren(s: &str) -> Option<usize> {
    let mut depth = 1u32;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    for (i, b) in s.bytes().enumerate() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == q {
                quote = None;
            }
            continue;
        }
        match b {
            b'\'' | b'"' => quote = Some(b),
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split `s` on `delim` (must be ASCII: `,`, `=`, `:`) at bracket depth 0,
/// quote-aware (both styles, backslash escapes). Depth counts `(`, `[`,
/// `{` uniformly, so commas inside nested literals never split.
fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    debug_assert!(delim.is_ascii());
    let mut parts = Vec::new();
    let mut depth = 0u32;
    let mut quote: Option<u8> = None;
    let mut escaped = false;
    let mut start = 0;
    let delim_b = delim as u8;
    for (i, b) in s.bytes().enumerate() {
        if let Some(q) = quote {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == q {
                quote = None;
            }
            continue;
        }
        match b {
            b'\'' | b'"' => quote = Some(b),
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth = depth.saturating_sub(1),
            b if b == delim_b && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

/// Parse one `key=literal` item into its key slice and JSON value.
/// Returns `None` unless the item is exactly one top-level `=` pair.
fn parse_kwarg_item(item: &str, depth: u32) -> Option<(&str, Value)> {
    let kv = split_top_level(item, '=');
    if kv.len() != 2 {
        return None;
    }
    let key = kv[0].trim();
    let raw = kv[1].trim();
    if key.is_empty() || raw.is_empty() {
        return None;
    }
    Some((key, parse_python_literal(raw, depth)?))
}

/// Parse one Python literal into JSON: quoted strings (both styles, with
/// escapes), `True`/`False`/`None`, ints, finite floats, `[...]` lists and
/// `{...}` dicts (string or identifier keys). Returns `None` for anything
/// else so the caller degrades to `_raw`/malformed instead of guessing.
fn parse_python_literal(s: &str, depth: u32) -> Option<Value> {
    if depth > KWARG_MAX_DEPTH {
        return None;
    }
    let s = s.trim();
    if s.len() >= 2 && (s.starts_with('\'') || s.starts_with('"')) {
        return parse_python_string(s).map(Value::String);
    }
    match s {
        "True" => return Some(Value::Bool(true)),
        "False" => return Some(Value::Bool(false)),
        "None" => return Some(Value::Null),
        _ => {}
    }
    if s.starts_with('[') {
        let inner = s.strip_prefix('[')?.strip_suffix(']')?;
        if inner.trim().is_empty() {
            return Some(Value::Array(Vec::new()));
        }
        let items: Option<Vec<Value>> = split_top_level(inner, ',')
            .into_iter()
            .map(|it| parse_python_literal(it.trim(), depth + 1))
            .collect();
        return items.map(Value::Array);
    }
    if s.starts_with('{') {
        let inner = s.strip_prefix('{')?.strip_suffix('}')?;
        let mut map = serde_json::Map::new();
        if !inner.trim().is_empty() {
            for entry in split_top_level(inner, ',') {
                let kv = split_top_level(entry, ':');
                if kv.len() != 2 {
                    return None;
                }
                let raw_key = kv[0].trim();
                let key = if raw_key.len() >= 2
                    && (raw_key.starts_with('\'') || raw_key.starts_with('"'))
                {
                    parse_python_string(raw_key)?
                } else if is_word_like(raw_key) {
                    raw_key.to_string()
                } else {
                    return None;
                };
                let value = parse_python_literal(kv[1].trim(), depth + 1)?;
                map.insert(key, value);
            }
        }
        return Some(Value::Object(map));
    }
    if let Ok(n) = s.parse::<i64>() {
        return Some(Value::from(n));
    }
    if let Ok(f) = s.parse::<f64>() {
        return if f.is_finite() {
            Some(Value::from(f))
        } else {
            None // "nan"/"inf" parse but are not JSON
        };
    }
    None
}

/// Parse one Python string literal (single or double quoted) with backslash
/// escapes (`\n \t \r \\ \' \" \0`; unknown escapes kept verbatim). An
/// unescaped interior quote terminates nothing — the literal is malformed
/// and yields `None`.
fn parse_python_string(s: &str) -> Option<String> {
    let quote = *s.as_bytes().first()?;
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    if s.len() < 2 || !s.ends_with(quote as char) {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '0' => out.push('\0'),
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        } else if c == quote as char {
            return None; // unescaped interior quote: not one literal
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Classic two-row Levenshtein edit distance.
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
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
    fn salvage_invalid_args_retains_call_with_raw_body() {
        let parser = ToolCallParser::new();
        // Contains {"name" but the arguments payload is not JSON: P14 keeps
        // the call alive with a best-effort `_raw` body instead of dropping.
        let input = r#"{"name": "Read", "arguments": not_json}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["_raw"], input);
        // `_raw`-only arguments count as empty so recovery can still fire.
        assert!(is_empty_arguments(&calls[0].arguments));
    }

    #[test]
    fn salvage_non_object_args_field_retains_raw_value() {
        let parser = ToolCallParser::new();
        // arguments present but not an object → retained as `_raw`.
        let input = r#"{"name": "Bash", "arguments": "ls -la"}"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        // The raw payload is the JSON-serialized form of the field.
        assert_eq!(
            calls[0].arguments["_raw"],
            serde_json::Value::String("\"ls -la\"".to_string())
        );
        assert!(is_empty_arguments(&calls[0].arguments));
    }

    #[test]
    fn diagnostics_malformed_standalone_name() {
        let parser = ToolCallParser::new();
        // Contains {"name" heuristic but invalid JSON / missing arguments field
        let input = r#"{"name": "Read", "arguments": not_json}"#;
        let (calls, diagnostics) = parser.parse_with_diagnostics(input);
        // P14 salvage retains the call, so no diagnostic is emitted.
        assert_eq!(calls.len(), 1);
        assert!(diagnostics.is_empty());
        assert_eq!(calls[0].arguments["_raw"], input);
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
        assert!(
            calls.is_empty(),
            "prose mention without JSON must not synthesize a call"
        );
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

    // --- Tier 7 recovery gaps: Gemma variants ---

    #[test]
    fn parse_gemma_quote_tokens_normalized() {
        let parser = ToolCallParser::new();
        let input =
            r#"<|tool_call|>{"name":<|'|>Read<|'|>,"arguments":{"path":<|"|>foo.rs<|"|>}}<|end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "foo.rs");
    }

    #[test]
    fn parse_asymmetric_opener_json_envelope() {
        let parser = ToolCallParser::new();
        let input = r#"<|tool_call>{"name":"Edit","arguments":{"path":"a.rs"}}<|end_tool_call|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Edit");
        assert_eq!(calls[0].arguments["path"], "a.rs");
    }

    #[test]
    fn parse_asymmetric_opener_inline_name() {
        let parser = ToolCallParser::new();
        let input = r#"<|tool_call>read{"path":"a.rs"}<tool_call|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].name, "Read",
            "inline raw name must be canonicalized"
        );
        assert_eq!(calls[0].arguments["path"], "a.rs");
    }

    #[test]
    fn parse_asymmetric_layer_accepts_bare_opener_with_wild_closer() {
        // P8: bare `<tool_call>` opener combined with a wild closer that the
        // XML layer alone cannot handle.
        let parser = ToolCallParser::new();
        let input = r#"<tool_call>{"name":"Edit","arguments":{"path":"a.rs"}}<tool_call|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1, "exactly one call, no cross-layer duplicate");
        assert_eq!(calls[0].name, "Edit");
        assert_eq!(calls[0].arguments["path"], "a.rs");
    }

    #[test]
    fn parse_asymmetric_bare_opener_without_duplicate_dispatch() {
        // P8: a well-formed bare block must still yield exactly ONE call even
        // though both the asymmetric layer and the XML layer can parse it.
        let parser = ToolCallParser::new();
        let input =
            r#"prefix <tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call> tail"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");

        // Two bare blocks → exactly two distinct calls.
        let two = r#"<tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call>
mid
<tool_call>{"name":"Edit","arguments":{"path":"b.rs","old":"x","new":"y"}}</tool_call>"#;
        let calls = parser.parse(two);
        assert_eq!(calls.len(), 2, "got: {calls:?}");
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Edit");
    }

    #[test]
    fn duplicate_identical_stateful_calls_survive() {
        // G3-4: two IDENTICAL Bash calls in one response are an intentional
        // sequence — dedup must not collapse stateful tools.
        let parser = ToolCallParser::new();
        let block = r#"<tool_call>{"name":"Bash","arguments":{"command":"make test"}}</tool_call>"#;
        let input = format!("{block}\n{block}");
        let calls = parser.parse(&input);
        assert_eq!(
            calls.len(),
            2,
            "stateful duplicates must survive: {calls:?}"
        );
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[1].name, "Bash");
    }

    #[test]
    fn duplicate_identical_read_only_call_collapses() {
        // G3-4: a repeated identical read carries no information — collapse
        // it (also covers any residual cross-layer double-parse of reads).
        let parser = ToolCallParser::new();
        let block = r#"<tool_call>{"name":"Read","arguments":{"path":"a.rs"}}</tool_call>"#;
        let input = format!("{block}\n{block}");
        let calls = parser.parse(&input);
        assert_eq!(calls.len(), 1, "read-only duplicates collapse: {calls:?}");
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn recover_fallback_channel_form_near_mention() {
        // P13: prose mentions Bash; the only structured payload is a
        // channel-tagged fragment — the salvage chain must find it.
        let parser = ToolCallParser::new();
        let input = r#"I will use Bash now <|channel|>commentary to=Bash <|message|>{"command":"ls -la"}<|im_end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls -la");
    }

    #[test]
    fn recover_fallback_inline_name_after_mention() {
        // P13: bare-object window scan fails (no standalone `{...}` object),
        // inline `NAME{json}` right after the mention wins.
        let parser = ToolCallParser::new();
        let input = r#"let me Grep{"pattern":"fn main"} for it"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    #[test]
    fn recover_fallback_negative_no_payload_at_all() {
        // P13 bad fixture: mention without any structured form nearby.
        let parser = ToolCallParser::new();
        let input = r#"I would like to use the Grep tool someday, please."#;
        let calls = parser.parse(input);
        assert!(calls.is_empty(), "got: {calls:?}");
    }

    #[test]
    fn parse_args_alias_accepted() {
        // P15: `args` is accepted as an alias for arguments/parameters.
        let parser = ToolCallParser::new();

        let direct = r#"{"name":"Bash","args":{"command":"ls"}}"#;
        let calls = parser.parse(direct);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls");

        let nested = r#"{"function":{"name":"Bash","args":{"command":"ls"}}}"#;
        let calls = parser.parse(nested);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], "ls");

        let tool_form = r#"{"tool":"Bash","args":{"command":"ls"}}"#;
        let calls = parser.parse(tool_form);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn parse_args_alias_bad_fixture_missing_everything() {
        // P15 bad fixture: name present but no arguments field at all →
        // empty arguments, not an error.
        let parser = ToolCallParser::new();
        let calls = parser.parse(r#"{"name":"Bash"}"#);
        assert_eq!(calls.len(), 1);
        assert!(is_empty_arguments(&calls[0].arguments));
    }

    #[test]
    fn parse_loose_call_form() {
        let parser = ToolCallParser::new();
        let input = r#"thought dump call:Bash{"command":"ls"} end"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn parse_channel_tagged_to_variant() {
        let parser = ToolCallParser::new();
        let input = r#"<|channel|>commentary to=Read <|message|>{"path":"a.rs"}<|im_end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "a.rs");
    }

    #[test]
    fn parse_channel_tagged_tool_variant_without_pipes() {
        let parser = ToolCallParser::new();
        let input = r#"<channel>commentary tool=Grep <message>{"pattern":"fn main"}</message>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    #[test]
    fn parse_channel_tagged_large_payload_beyond_lookahead_window() {
        // The JSON payload closes well beyond MESSAGE_TAG_LOOKAHEAD_BYTES
        // after the target match — only the TAG lookup may be window-bounded,
        // the payload must still parse in full.
        let body = "x".repeat(250);
        let input = format!(
            r#"<|channel|>commentary tool=Write <|message|>{{"path":"src/main.rs","content":"{body}"}}<|im_end|>"#
        );
        let parser = ToolCallParser::new();
        let calls = parser.parse(&input);
        assert_eq!(calls.len(), 1, "payload >200B must not be clipped");
        assert_eq!(calls[0].name, "Write");
        assert_eq!(calls[0].arguments["path"], "src/main.rs");
        assert_eq!(calls[0].arguments["content"], body);
    }

    #[test]
    fn loose_call_without_braces_yields_nothing() {
        let parser = ToolCallParser::new();
        let input = "I should call:Bash soon, after the tests finish.";
        let calls = parser.parse(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn channel_tag_without_message_payload_yields_nothing() {
        let parser = ToolCallParser::new();
        let input = "<|channel|>commentary to=Bash <|im_end|>";
        let calls = parser.parse(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn asymmetric_closer_without_opener_yields_nothing() {
        let parser = ToolCallParser::new();
        let input = "junk <tool_call|> not json </tool_call|> trailing";
        let calls = parser.parse(input);
        assert!(calls.is_empty());
    }

    #[test]
    fn symmetric_layer_takes_precedence_over_asymmetric() {
        let parser = ToolCallParser::new();
        // Two complete symmetric calls: only layer 1's closer set (`<|end|>`)
        // can split these; the asymmetric layer would see one unclosed blob.
        let input = r#"<|tool_call|>{"name":"Read","arguments":{"path":"a.rs"}}<|end|>
<|tool_call|>{"name":"Bash","arguments":{"command":"ls"}}<|end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Bash");
    }

    #[test]
    fn recover_scan_tail_allows_marker_in_last_32k() {
        let parser = ToolCallParser::new();
        let filler = "x".repeat(33_000);
        let marker = r#"call:Bash{"command":"ls"}"#;
        let input = format!("{filler}\n{marker}");
        let calls = parser.parse(&input);
        assert_eq!(calls.len(), 1, "marker inside the 32K tail must parse");
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn recover_scan_tail_drops_marker_beyond_tail() {
        let parser = ToolCallParser::new();
        let filler = "x".repeat(40_000);
        let marker = r#"call:Bash{"command":"ls"}"#;
        let input = format!("{marker}{filler}");
        let calls = parser.parse(&input);
        assert!(
            calls.is_empty(),
            "marker beyond the 32K tail must not parse"
        );
    }

    #[test]
    fn empty_args_recovery_bounded_to_last_32k() {
        let parser = ToolCallParser::new();
        let filler = "x".repeat(33_000);
        let args = r#"{"path":"a.rs"}"#;
        let input = format!("I will use Read now. {args}{filler}");
        let calls = parser.parse(&input);
        assert!(
            calls.is_empty(),
            "mention beyond the 32K tail must not drive recovery"
        );
    }

    // ── Layer 1c: MiniCPM5/Liquid-style Python-kwarg calls ──────────

    #[test]
    fn parse_kwarg_basic_single_string() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Read(path='foo.rs')]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "foo.rs");
    }

    #[test]
    fn parse_kwarg_multiple_mixed_types() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Write(path=\"a.py\", count=3, ratio=1.5, \
            verbose=True, other=False, missing=None, neg=-7)]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        let a = &calls[0].arguments;
        assert_eq!(a["path"], "a.py");
        assert_eq!(a["count"], 3);
        assert_eq!(a["ratio"], 1.5);
        assert_eq!(a["verbose"], true);
        assert_eq!(a["other"], false);
        assert!(a["missing"].is_null());
        assert_eq!(a["neg"], -7);
    }

    #[test]
    fn parse_kwarg_escapes() {
        let parser = ToolCallParser::new();
        let input = r#"<|tool_call_start|>[Write(path='a.py', content='line1\nline2\'q\\')]<|tool_call_end|>"#;
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["content"], "line1\nline2'q\\");
    }

    #[test]
    fn parse_kwarg_unordered_keys() {
        // Kwarg order must not matter: content-first still parses.
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Edit(content='new', path='a.rs')]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Edit");
        assert_eq!(calls[0].arguments["path"], "a.rs");
        assert_eq!(calls[0].arguments["content"], "new");
    }

    #[test]
    fn parse_kwarg_list_and_dict() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Bash(command='ls', tags=['a', 'b'], \
            opts={'k': 1, v: True})]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["tags"], serde_json::json!(["a", "b"]));
        assert_eq!(
            calls[0].arguments["opts"],
            serde_json::json!({"k": 1, "v": true})
        );
    }

    #[test]
    fn parse_kwarg_bare_form_without_brackets() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>Bash(command='ls')<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
        assert_eq!(calls[0].arguments["command"], "ls");
    }

    #[test]
    fn parse_kwarg_empty_args() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Read()]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert!(calls[0].arguments.as_object().unwrap().is_empty());
    }

    #[test]
    fn parse_kwarg_name_case_canonicalized() {
        // Layer 9 fuzzy layer folds `read` onto `Read` afterwards.
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[read(path='a.rs')]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
    }

    #[test]
    fn parse_kwarg_unclosed_opener_consumes_rest() {
        let parser = ToolCallParser::new();
        let input = "thinking out loud <|tool_call_start|>[Bash(command='ls')]";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Bash");
    }

    #[test]
    fn parse_kwarg_multiple_regions() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Read(path='a.rs')]<|tool_call_end|> \
            then <|tool_call_start|>[Bash(command='ls')]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[1].name, "Bash");
    }

    #[test]
    fn parse_kwarg_json_layers_keep_priority() {
        // First-match-wins: a JSON envelope in layer 1 beats a kwarg call.
        let parser = ToolCallParser::new();
        let input = "<|tool_call|>{\"name\":\"Read\",\"arguments\":{\"path\":\"j.rs\"}}<|end|> \
            <|tool_call_start|>[Bash(command='ls')]<|tool_call_end|>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Read");
        assert_eq!(calls[0].arguments["path"], "j.rs");
    }

    #[test]
    fn parse_kwarg_trailing_garbage_rejected() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Read(path='a.rs')] junk<|tool_call_end|>";
        assert!(parser.parse(input).is_empty());
    }

    #[test]
    fn parse_kwarg_malformed_reports_intent() {
        // Garbage inside the markers: nothing parses, but the intent
        // heuristic fires so the caller gets a diagnostic, not silence.
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>garbage((<|tool_call_end|>";
        assert!(parser.parse(input).is_empty());
        let (_, diagnostics) = parser.parse_with_diagnostics(input);
        assert!(!diagnostics.is_empty());
    }

    #[test]
    fn parse_kwarg_nan_inf_rejected() {
        // "nan"/"inf" parse as f64 but are not JSON — must not leak in.
        let parser = ToolCallParser::new();
        assert!(parser
            .parse("<|tool_call_start|>[Bash(x=nan)]<|tool_call_end|>")
            .is_empty());
        assert!(parser
            .parse("<|tool_call_start|>[Bash(x=inf)]<|tool_call_end|>")
            .is_empty());
    }

    #[test]
    fn parse_kwarg_alternate_closer() {
        let parser = ToolCallParser::new();
        let input = "<|tool_call_start|>[Grep(pattern='foo')]</tool_call>";
        let calls = parser.parse(input);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "Grep");
    }
}

#[cfg(test)]
mod proptest_unit {
    // W5-code: proptest unitarios (migra 2 casos fijos a strategies).
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        #[test]
        fn xml_envelope_roundtrip(
            tool in prop::sample::select(vec!["Read", "Bash", "Grep", "Glob", "Edit", "Write"]),
            path in "[a-z0-9_/]{1,24}\\.[a-z]{1,3}",
        ) {
            let parser = ToolCallParser::new();
            let input = format!(r#"<tool_call>{{"name":"{tool}","arguments":{{"path":"{path}"}}}}</tool_call>"#);
            let calls = parser.parse(&input);
            prop_assert_eq!(calls.len(), 1);
            prop_assert_eq!(&calls[0].name, tool);
            prop_assert_eq!(&calls[0].arguments["path"], &serde_json::json!(path));
        }
        #[test]
        fn arbitrary_text_never_panics(s in ".*{0,128}") {
            let parser = ToolCallParser::new();
            let _ = parser.parse(&s);
            let _ = parser.try_parse_with_error(&s);
        }
    }
}
