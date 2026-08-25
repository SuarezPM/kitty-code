//! Tolerance ladder for str-replace edits: graduated fuzzy matchers tried in
//! order after an exact `old_string` match fails, so indentation drift,
//! whitespace noise, escaped-literal confusion, and anchor-only recall still
//! land instead of spiraling into retry exhaustion.
//!
//! Every strategy yields concrete candidate spans (verbatim substrings of the
//! file content); the orchestrator ([`run_ladder`]) applies the
//! disproportionate-match guard to each candidate before accepting it, then
//! resolves occurrence counts against `replace_all`. A fuzzy match yielding
//! several occurrences without `replace_all` is reported ambiguous — the
//! caller surfaces the usual "provide more context" guidance.

use crate::parser::levenshtein;

/// Block-anchor similarity floor (single and multiple candidates).
const BLOCK_ANCHOR_SIMILARITY_THRESHOLD: f64 = 0.65;

/// Context-aware middle-line trimmed-equality floor.
const CONTEXT_AWARE_MIDDLE_RATIO: f64 = 0.5;

/// Files larger than this skip fuzzy matching entirely (same scale as the
/// agent's `TRUNCATION_MAX_BYTES` output cap): scanning a file far bigger
/// than anything fed back to the model is pure waste, and exact matching has
/// already failed by the time the ladder runs.
const FUZZY_MAX_CONTENT_BYTES: usize = 512 * 1024;

/// Block-anchor candidates larger than this skip the Levenshtein middle
/// comparison — a span this big would be rejected by the disproportionate
/// guard for any multi-line `old_string` anyway, so comparing is waste.
const BLOCK_ANCHOR_MAX_SPAN_BYTES: usize = 64 * 1024;

/// How a tolerance-ladder attempt resolved.
#[derive(Debug)]
pub(crate) enum LadderOutcome {
    /// A guarded candidate matched; [`LadderHit::all_occurrences`] is set
    /// when the span occurs multiple times and `replace_all` was requested.
    Hit(LadderHit),
    /// Candidates matched at several positions without `replace_all`.
    Ambiguous,
    /// No strategy produced a viable candidate.
    NotFound,
}

/// One accepted tolerance-ladder match.
#[derive(Debug)]
pub(crate) struct LadderHit {
    /// Verbatim span in the (LF-normalized) file content to replace.
    pub span: String,
    /// Terse strategy tag surfaced to the model, e.g. `line-trim`.
    pub label: &'static str,
    /// Multiple occurrences exist and `replace_all=true` — replace them all.
    pub all_occurrences: bool,
}

/// Run the tolerance ladder over `content` looking for `find`.
///
/// Strategy order: line-trim → whitespace-normalized → trimmed-boundary →
/// escape-normalized → indent-flexible → context-aware → block-anchor; the
/// first strategy producing a vetted match short-circuits the rest. Every
/// candidate passes through [`is_disproportionate_match`] — the guard is part
/// of the ladder itself and is never bypassed. Oversized files bail before
/// any strategy runs (see [`FUZZY_MAX_CONTENT_BYTES`]).
pub(crate) fn run_ladder(content: &str, find: &str, replace_all: bool) -> LadderOutcome {
    if find.is_empty() || content.len() > FUZZY_MAX_CONTENT_BYTES {
        return LadderOutcome::NotFound;
    }

    type Strategy = fn(&str, &str) -> Vec<String>;
    const STRATEGIES: [(&str, Strategy); 7] = [
        ("line-trim", line_trimmed_candidates),
        ("whitespace-normalized", whitespace_normalized_candidates),
        ("trimmed-boundary", trimmed_boundary_candidates),
        ("escape-normalized", escape_normalized_candidates),
        ("indent-flexible", indentation_flexible_candidates),
        ("context-aware", context_aware_candidates),
        ("block-anchor", block_anchor_candidates),
    ];

    let mut ambiguous = false;
    for (label, strategy) in STRATEGIES {
        // Candidate spans repeat whenever the drifted block occurs several
        // times; vetting each DISTINCT span once keeps the pass O(spans)
        // instead of O(candidates × file).
        let mut tried: std::collections::HashSet<String> = std::collections::HashSet::new();
        for span in strategy(content, find) {
            if !tried.insert(span.clone()) {
                continue;
            }
            if !content.contains(&span) {
                // Defensive: strategies promise verbatim spans.
                continue;
            }
            if is_disproportionate_match(&span, find) {
                continue;
            }
            let occurrences = content.matches(&span).count();
            if occurrences > 1 && !replace_all {
                ambiguous = true;
                continue;
            }
            return LadderOutcome::Hit(LadderHit {
                span,
                label,
                all_occurrences: occurrences > 1,
            });
        }
    }

    if ambiguous {
        LadderOutcome::Ambiguous
    } else {
        LadderOutcome::NotFound
    }
}

/// Atomicity guard: reject candidates whose span is wildly larger than the
/// requested `old_string` (many more lines, or far more trimmed content), so
/// a loose anchor match can never swallow half a file in one edit.
pub(crate) fn is_disproportionate_match(search: &str, old_string: &str) -> bool {
    let old_lines = old_string.split('\n').count();
    let search_lines = search.split('\n').count();
    if search_lines >= (old_lines + 3).max(old_lines * 2) {
        return true;
    }
    if old_lines == 1 {
        return false;
    }
    let old_trimmed = old_string.trim().len();
    let search_trimmed = search.trim().len();
    search_trimmed > old_trimmed.saturating_add(500).max(old_trimmed * 4)
}

// ---------------------------------------------------------------------------
// Strategy 1: per-line trim equivalence
// ---------------------------------------------------------------------------

/// Match blocks whose lines are trim-equal to `find`'s lines, returning the
/// ORIGINAL (untrimmed) block from the content.
fn line_trimmed_candidates(content: &str, find: &str) -> Vec<String> {
    let original_lines: Vec<&str> = content.split('\n').collect();
    let mut search_lines: Vec<&str> = find.split('\n').collect();
    if search_lines.last() == Some(&"") {
        search_lines.pop();
    }
    let mut out = Vec::new();

    for i in 0..=original_lines.len().saturating_sub(search_lines.len()) {
        let matches = search_lines
            .iter()
            .zip(original_lines[i..].iter())
            .all(|(s, o)| s.trim() == o.trim());
        if !matches {
            continue;
        }
        out.push(original_lines[i..i + search_lines.len()].join("\n"));
    }
    out
}

// ---------------------------------------------------------------------------
// Strategy 2: whitespace-run collapse
// ---------------------------------------------------------------------------

/// Collapse every whitespace run to a single space (`trim` + `\s+`→`"`).
fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Match lines/blocks whose whitespace-normalized form equals `find`; also
/// matches substrings of a line where the words of `find` appear separated by
/// arbitrary whitespace runs.
fn whitespace_normalized_candidates(content: &str, find: &str) -> Vec<String> {
    let normalized_find = normalize_whitespace(find);
    let mut out = Vec::new();

    for line in content.split('\n') {
        let normalized_line = normalize_whitespace(line);
        if normalized_line == normalized_find {
            out.push(line.to_string());
            continue;
        }
        if !normalized_line.contains(&normalized_find) {
            continue;
        }
        // Locate the actual span whose words are separated by flexible
        // whitespace inside this one line.
        let words: Vec<&str> = find.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let pattern = words
            .iter()
            .copied()
            .map(regex::escape)
            .collect::<Vec<_>>()
            .join("\\s+");
        if let Ok(re) = regex::Regex::new(&pattern) {
            if let Some(m) = re.find(line) {
                out.push(m.as_str().to_string());
            }
        }
    }

    let find_line_count = find.split('\n').count();
    if find_line_count > 1 {
        let lines: Vec<&str> = content.split('\n').collect();
        for i in 0..=lines.len().saturating_sub(find_line_count) {
            let block = lines[i..i + find_line_count].join("\n");
            if normalize_whitespace(&block) == normalized_find {
                out.push(block);
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Strategy 3: whole-string trim equivalence
// ---------------------------------------------------------------------------

/// Match when the fully-trimmed `find` appears verbatim, or when a block's
/// trimmed form equals the trimmed `find`.
fn trimmed_boundary_candidates(content: &str, find: &str) -> Vec<String> {
    let trimmed_find = find.trim();
    if trimmed_find == find {
        // Already trimmed; nothing new to try.
        return Vec::new();
    }
    let mut out = Vec::new();
    if content.contains(trimmed_find) {
        out.push(trimmed_find.to_string());
    }

    let find_line_count = find.split('\n').count();
    let lines: Vec<&str> = content.split('\n').collect();
    for i in 0..=lines.len().saturating_sub(find_line_count) {
        let block = lines[i..i + find_line_count].join("\n");
        if block.trim() == trimmed_find {
            out.push(block);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Strategy 4: escape-sequence normalization
// ---------------------------------------------------------------------------

/// Best-effort unescape of literal `\n` `\t` `\r` `\'` `\"` `` \` `` `\\`
/// `\$` and backslash-before-newline sequences (left-to-right scan; unknown
/// escapes pass through untouched).
fn unescape_literal_escapes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() {
            match chars[i + 1] {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                '`' => out.push('`'),
                '\\' => out.push('\\'),
                '$' => out.push('$'),
                '\n' => out.push('\n'),
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
            i += 2;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// Match when the unescaped `find` appears verbatim, or when a block's
/// unescaped form equals the unescaped `find`.
fn escape_normalized_candidates(content: &str, find: &str) -> Vec<String> {
    let unescaped_find = unescape_literal_escapes(find);
    let mut out = Vec::new();

    if unescaped_find != find && content.contains(&unescaped_find) {
        out.push(unescaped_find.clone());
    }

    let find_line_count = unescaped_find.split('\n').count();
    let lines: Vec<&str> = content.split('\n').collect();
    for i in 0..=lines.len().saturating_sub(find_line_count) {
        let block = lines[i..i + find_line_count].join("\n");
        if unescape_literal_escapes(&block) == unescaped_find {
            out.push(block);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Strategy 5: minimum-indentation dedent equivalence
// ---------------------------------------------------------------------------

/// Strip the common leading whitespace of all non-empty lines (empty lines
/// are left untouched).
fn dedent(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let min_indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.chars().take_while(|c| c.is_whitespace()).count())
        .min();
    match min_indent {
        Some(min) if min > 0 => lines
            .iter()
            .map(|l| {
                if l.trim().is_empty() {
                    (*l).to_string()
                } else {
                    l.chars().skip(min).collect::<String>()
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => text.to_string(),
    }
}

/// Match blocks whose dedented form equals the dedented `find`.
fn indentation_flexible_candidates(content: &str, find: &str) -> Vec<String> {
    let normalized_find = dedent(find);
    let find_line_count = find.split('\n').count();
    let lines: Vec<&str> = content.split('\n').collect();
    let mut out = Vec::new();
    for i in 0..=lines.len().saturating_sub(find_line_count) {
        let block = lines[i..i + find_line_count].join("\n");
        if dedent(&block) == normalized_find {
            out.push(block);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Strategy 6: anchor lines + ≥50% trimmed-equal middle (equal height)
// ---------------------------------------------------------------------------

/// Match blocks whose first/last trimmed lines equal `find`'s anchors and
/// whose middle lines are at least 50% trim-equal (same block height).
fn context_aware_candidates(content: &str, find: &str) -> Vec<String> {
    let mut find_lines: Vec<&str> = find.split('\n').collect();
    if find_lines.len() < 3 {
        return Vec::new();
    }
    if find_lines.last() == Some(&"") {
        find_lines.pop();
    }
    let content_lines: Vec<&str> = content.split('\n').collect();
    let first = find_lines[0].trim();
    let last = find_lines[find_lines.len() - 1].trim();
    let mut out = Vec::new();

    for i in 0..content_lines.len() {
        if content_lines[i].trim() != first {
            continue;
        }
        for (j, line) in content_lines.iter().enumerate().skip(i + 2) {
            if line.trim() != last {
                continue;
            }
            let block_lines = &content_lines[i..=j];
            if block_lines.len() == find_lines.len() {
                let mut matching = 0usize;
                let mut total = 0usize;
                for k in 1..block_lines.len() - 1 {
                    let b = block_lines[k].trim();
                    let f = find_lines[k].trim();
                    if !b.is_empty() || !f.is_empty() {
                        total += 1;
                        if b == f {
                            matching += 1;
                        }
                    }
                }
                if total == 0 || (matching as f64 / total as f64) >= CONTEXT_AWARE_MIDDLE_RATIO {
                    out.push(block_lines.join("\n"));
                }
            }
            break;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Strategy 7: anchor lines + Levenshtein-fuzzy middle (size-bounded)
// ---------------------------------------------------------------------------

/// Match blocks by trimmed first/last anchors with a line-count delta ≤25%
/// and a Levenshtein-similarity floor averaged over the middle lines.
fn block_anchor_candidates(content: &str, find: &str) -> Vec<String> {
    let mut search_lines: Vec<&str> = find.split('\n').collect();
    if search_lines.len() < 3 {
        return Vec::new();
    }
    if search_lines.last() == Some(&"") {
        search_lines.pop();
    }
    let content_lines: Vec<&str> = content.split('\n').collect();
    let first = search_lines[0].trim();
    let last = search_lines[search_lines.len() - 1].trim();
    let search_block_size = search_lines.len();
    let max_line_delta = (search_block_size / 4).max(1);

    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for i in 0..content_lines.len() {
        if content_lines[i].trim() != first {
            continue;
        }
        for (j, line) in content_lines.iter().enumerate().skip(i + 2) {
            if line.trim() == last {
                if (j - i + 1).abs_diff(search_block_size) <= max_line_delta {
                    candidates.push((i, j));
                }
                break;
            }
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    // Average Levenshtein similarity of the middle lines of one candidate.
    let middle_similarity = |start: usize, end: usize| -> f64 {
        let actual_block_size = end - start + 1;
        let lines_to_check = (search_block_size - 2).min(actual_block_size - 2);
        if lines_to_check == 0 {
            return 1.0; // anchors only — accept on anchor strength
        }
        let mut sum = 0.0f64;
        let mut compared = 0usize;
        let mut j = 1usize;
        while j < search_block_size - 1 && j < actual_block_size - 1 {
            let orig = content_lines[start + j].trim();
            let srch = search_lines[j].trim();
            let max_len = orig.chars().count().max(srch.chars().count());
            if max_len > 0 {
                let d = levenshtein(orig, srch);
                sum += 1.0 - (d as f64 / max_len as f64);
                compared += 1;
            }
            j += 1;
        }
        if compared == 0 {
            1.0
        } else {
            sum / compared as f64
        }
    };

    let mut out = Vec::new();
    for &(start, end) in &candidates {
        let span = content_lines[start..=end].join("\n");
        if span.len() > BLOCK_ANCHOR_MAX_SPAN_BYTES {
            // Levenshtein over a span this size is waste: the guard would
            // reject it for any multi-line old_string regardless of outcome.
            continue;
        }
        if middle_similarity(start, end) >= BLOCK_ANCHOR_SIMILARITY_THRESHOLD {
            out.push(span);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRIFTED: &str = "fn main() {\n    let x = 1;\n    let y = 2;\n}\n";

    // -- guard ------------------------------------------------------------

    #[test]
    fn guard_rejects_oversized_line_counts() {
        let old = "a\nb";
        let huge = "a\nb\nc\nd\ne\nf\ng\nh"; // 8 lines vs 2 → ≥ max(5, 4)
        assert!(is_disproportionate_match(huge, old));
        let close = "a\nb\nc"; // 3 lines vs 2 → below max(5, 4)
        assert!(!is_disproportionate_match(close, old));
    }

    #[test]
    fn guard_rejects_oversized_length_but_spares_single_line_olds() {
        // Length rule only applies to multi-line olds.
        let old_one = "compact";
        let big_one = "x".repeat(old_one.len() * 8 + 600);
        assert!(!is_disproportionate_match(&big_one, old_one));
        assert!(is_disproportionate_match("a\nb\nc\nd\ne\nf", old_one));

        let old_multi = "a\ncompact";
        let big_multi = format!("a\n{}", "x".repeat(old_multi.trim().len() * 8 + 600));
        assert!(is_disproportionate_match(&big_multi, old_multi));
    }

    #[test]
    fn guard_accepts_reasonable_span() {
        assert!(!is_disproportionate_match("    indented();", "indented();"));
    }

    // -- line-trim ----------------------------------------------------------

    #[test]
    fn line_trimmed_positive_indentation_drift() {
        let find = "let x = 1;\n    let y = 2;";
        let hits = line_trimmed_candidates(DRIFTED, find);
        assert_eq!(hits, vec!["    let x = 1;\n    let y = 2;"]);
    }

    #[test]
    fn line_trimmed_negative_on_content_drift() {
        assert!(line_trimmed_candidates(DRIFTED, "let x = 9;").is_empty());
    }

    #[test]
    fn line_trimmed_guard_rejects_giant_indentation_span() {
        // Two-line find whose trim-equal block carries ~1200 columns of
        // indentation — the span dwarfs the trimmed-length budget.
        let pad = " ".repeat(1200);
        let content = format!("{pad}a();\n{pad}b();\n");
        assert!(matches!(
            run_ladder(&content, "a();\nb();", false),
            LadderOutcome::NotFound
        ));
    }

    // -- whitespace-normalized ---------------------------------------------

    #[test]
    fn whitespace_normalized_positive_full_line_and_substring() {
        // Full-line equality after collapse.
        let hits = whitespace_normalized_candidates("foo\tbar   baz\n", "foo bar baz");
        assert_eq!(hits, vec!["foo\tbar   baz"]);
        // Substring of one line with flexible gaps.
        let hits = whitespace_normalized_candidates("x = foo\tbar baz; y\n", "foo bar baz");
        assert_eq!(hits, vec!["foo\tbar baz"]);
    }

    #[test]
    fn whitespace_normalized_negative_on_word_change() {
        assert!(whitespace_normalized_candidates(DRIFTED, "let z = 1;").is_empty());
    }

    #[test]
    fn whitespace_normalized_guard_rejects_giant_gap_span() {
        // Two-line find matched across a ~900-column whitespace run — the
        // verbatim span exceeds the trimmed-length budget.
        let content = format!("a{}b\nc\n", " ".repeat(900));
        assert!(matches!(
            run_ladder(&content, "a b\nc", false),
            LadderOutcome::NotFound
        ));
    }

    // -- trimmed-boundary ---------------------------------------------------

    #[test]
    fn trimmed_boundary_positive_untrimmed_find() {
        let content = "start\n\n  target body  \n\nend";
        let hits = trimmed_boundary_candidates(content, "target body\n");
        assert_eq!(hits.first().map(String::as_str), Some("target body"));
    }

    #[test]
    fn trimmed_boundary_negative_when_absent_and_when_already_trimmed() {
        assert!(trimmed_boundary_candidates(DRIFTED, "missing body\n").is_empty());
        // Already-trimmed find short-circuits even when it WOULD match.
        assert!(trimmed_boundary_candidates(DRIFTED, "let x = 1;").is_empty());
    }

    // -- escape-normalized --------------------------------------------------

    #[test]
    fn escape_normalized_positive_unescaped_containment() {
        let content = "msg := \"hi\"\nbye\n";
        let find = "msg := \\\"hi\\\"\\nbye";
        let hits = escape_normalized_candidates(content, find);
        assert!(hits.contains(&"msg := \"hi\"\nbye".to_string()));
    }

    #[test]
    fn escape_normalized_negative_without_escape_relation() {
        assert!(escape_normalized_candidates(DRIFTED, "let x \\\\= 3;").is_empty());
    }

    #[test]
    fn escape_normalized_guard_rejects_multiline_balloon() {
        // One escaped old line unescapes into five real lines; the verbatim
        // unescaped span trips the line-count arm of the guard.
        let old = "q\\nq\\nq\\nq\\nq";
        let content = "q\nq\nq\nq\nq\n";
        assert!(matches!(
            run_ladder(content, old, false),
            LadderOutcome::NotFound
        ));
    }

    // -- indent-flexible ------------------------------------------------------

    #[test]
    fn indentation_flexible_positive_dedent_shift() {
        let find = "if ok {\n    go();\n}";
        let content = "        if ok {\n            go();\n        }\n";
        let hits = indentation_flexible_candidates(content, find);
        assert_eq!(hits, vec!["        if ok {\n            go();\n        }"]);
    }

    #[test]
    fn indentation_flexible_negative_on_inner_change() {
        assert!(indentation_flexible_candidates(DRIFTED, "let q = 1;\nlet r = 2;").is_empty());
        // Empty input dedents to itself.
        assert_eq!(dedent(""), "");
    }

    // -- context-aware --------------------------------------------------------

    #[test]
    fn context_aware_positive_half_middle_tolerance() {
        let content = "head\nalpha\nMIDDLE-CHANGED\ngamma\ntail\n";
        let find = "head\nalpha\nbeta\ngamma\ntail\n";
        let hits = context_aware_candidates(content, find);
        assert_eq!(hits, vec!["head\nalpha\nMIDDLE-CHANGED\ngamma\ntail"]);
    }

    #[test]
    fn context_aware_negative_below_ratio_or_height_mismatch() {
        // Three changed middles out of three → 0% < 50% → no match.
        let content = "head\nC1\nC2\nC3\ntail\n";
        assert!(context_aware_candidates(content, "head\na\nb\nc\ntail\n").is_empty());
        // Anchor distance changes block height → height equality fails.
        let tall = "top\n1\n2\n3\n4\n5\nbottom\n";
        assert!(context_aware_candidates(tall, "top\nmid\nbottom\n").is_empty());
    }

    // -- block-anchor ----------------------------------------------------------

    #[test]
    fn block_anchor_positive_fuzzy_middle_within_delta() {
        let content = "def f():\n    alpha beta\n    gamma delta\n# end f\n";
        let find = "def f():\n    alfa betta\n    gamma delte\n# end f\n";
        let hits = block_anchor_candidates(content, find);
        assert_eq!(
            hits,
            vec!["def f():\n    alpha beta\n    gamma delta\n# end f"]
        );
    }

    #[test]
    fn block_anchor_negative_when_similarity_too_low() {
        let content = "def f():\n    zzz yyy\n    xxx www\n# end f\n";
        let find = "def f():\n    alfa betta\n    gamma delte\n# end f\n";
        assert!(block_anchor_candidates(content, find).is_empty());
    }

    #[test]
    fn block_anchor_negative_when_height_delta_exceeded() {
        // Anchor distance gives a 9-line block vs a 3-line find (delta 1).
        let content = "A\n1\n2\n3\n4\n5\n6\n7\nZ\n";
        assert!(matches!(
            run_ladder(content, "A\nmid\nZ\n", false),
            LadderOutcome::NotFound
        ));
    }

    // -- ladder orchestration ----------------------------------------------------

    #[test]
    fn ladder_order_prefers_earlier_strategy() {
        // Indentation drift fixable by BOTH line-trim (1st) and
        // whitespace-normalized (2nd) resolves via the earlier strategy.
        match run_ladder("do_it();\n", "\tdo_it();", false) {
            LadderOutcome::Hit(hit) => assert_eq!(hit.label, "line-trim"),
            other => panic!("expected hit, got {other:?}"),
        }
    }

    #[test]
    fn ladder_reports_ambiguous_without_replace_all() {
        match run_ladder("  x = 1;\nmid\n  x = 1;\n", "x = 1;", false) {
            LadderOutcome::Ambiguous => {}
            other => panic!("expected ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn ladder_replace_all_covers_every_occurrence() {
        let content = "  x = 1;\nmid\n  x = 1;\n";
        match run_ladder(content, "x = 1;", true) {
            LadderOutcome::Hit(hit) => {
                assert!(hit.all_occurrences);
                assert_eq!(hit.label, "line-trim");
                assert_eq!(hit.span, "  x = 1;");
                assert_eq!(content.matches(&hit.span).count(), 2);
            }
            other => panic!("expected hit, got {other:?}"),
        }
    }

    #[test]
    fn ladder_never_returns_unguarded_hit() {
        // Any Hit returned must have passed the disproportionate guard —
        // the ladder cannot emit a candidate the guard would reject.
        let content = format!("ok({})\n", "arg ".repeat(200));
        match run_ladder(&content, "ok(arg)", true) {
            LadderOutcome::Hit(hit) => assert!(!is_disproportionate_match(&hit.span, "ok(arg)")),
            LadderOutcome::NotFound | LadderOutcome::Ambiguous => {}
        }
    }

    #[test]
    fn ladder_notfound_when_nothing_matches() {
        assert!(matches!(
            run_ladder(DRIFTED, "totally absent text\nwith lines", false),
            LadderOutcome::NotFound
        ));
    }

    #[test]
    fn ladder_bails_on_oversized_files_but_not_normal_ones() {
        // G3-2: files beyond the fuzzy size cap bail before any strategy runs.
        let mut big = String::with_capacity(FUZZY_MAX_CONTENT_BYTES + 8192);
        while big.len() <= FUZZY_MAX_CONTENT_BYTES {
            big.push_str("filler line\n");
        }
        big.push_str("    target_value();\n");
        assert!(matches!(
            run_ladder(&big, "target_value();", false),
            LadderOutcome::NotFound
        ));

        // Control: the same drifted line in a normal-size file still matches.
        let small = "filler\n    target_value();\n";
        assert!(matches!(
            run_ladder(small, "target_value();", false),
            LadderOutcome::Hit(_)
        ));
    }

    #[test]
    fn block_anchor_skips_oversized_candidates() {
        // G3-2: candidates beyond the span cap skip Levenshtein comparison —
        // an identical middle would match without the cap.
        let huge = "x".repeat(BLOCK_ANCHOR_MAX_SPAN_BYTES + 1);
        let content = format!("A\n{huge}\nZ");
        let find = format!("A\n{huge}\nZ");
        assert!(block_anchor_candidates(&content, &find).is_empty());
    }
}
