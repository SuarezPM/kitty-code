//! Harness micro-benchmarks (Fase 3.3): parser hot path. These measure the
//! HARNESS, not the model — they close the "no profiling" gap flagged by the
//! 2026-09 audit before any further optimization is attempted.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use kitty_harness::parser::ToolCallParser;

const CLEAN_NATIVE: &str = r#"I'll read the file.
{"name": "Read", "arguments": {"path": "src/main.rs"}}"#;

const MIXED_REPAIR: &str = "Let me try:\n<tool_call>\n{\"name\": \"Edit\", \"arguments\": {\"path\": \"a.rs\", \"old_hash\": \"abc123\", \"new_content\": \"fn main() {}\"}}\n</tool_call>\nDone.";

fn bench_parser(c: &mut Criterion) {
    c.bench_function("parser_clean_native", |b| {
        let parser = ToolCallParser::new();
        b.iter(|| parser.parse(black_box(CLEAN_NATIVE)))
    });
    c.bench_function("parser_mixed_xml_stream", |b| {
        let parser = ToolCallParser::new();
        b.iter(|| parser.parse(black_box(MIXED_REPAIR)))
    });
}

criterion_group!(benches, bench_parser);
criterion_main!(benches);
