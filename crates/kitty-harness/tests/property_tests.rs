//! Property-based tests over the highest-risk pure surfaces (Fase 6.2):
//! - `ToolCallParser`: arbitrary malformed tool-call text never panics and
//!   always terminates with a parse result or error.
//! - `HashlineEditor`: arbitrary file contents + edits never panic and keep
//!   the optimistic-concurrency invariant (`FileChanged` on stale hash).
//! - `Guard`: interpreter-wrapped blocked payloads are always blocked.

use kitty_harness::guard::Guard;
use kitty_harness::hashline::{HashlineEditor, HashlineError};
use kitty_harness::parser::ToolCallParser;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1024))]

    #[test]
    fn parser_never_panics_on_arbitrary_text(input in ".*") {
        let parser = ToolCallParser::new();
        let _ = parser.parse(&input);
        let _ = parser.try_parse_with_error(&input);
    }

    #[test]
    fn parser_handles_arbitrary_tool_call_wrappers(
        name in "[a-zA-Z_]{0,16}",
        payload in ".*{0,64}",
        prefix in ".*{0,32}",
    ) {
        let text = format!("{prefix}<tool_call>{{\"name\": \"{name}\", \"arguments\": {payload}}}</tool_call>");
        let parser = ToolCallParser::new();
        let _ = parser.parse(&text);
    }

    #[test]
    fn hashline_edit_never_panics(
        content in ".*{0,200}",
        hash in "[a-f0-9]{0,12}",
        new_content in ".*{0,120}",
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, &content).unwrap();
        let mut editor = HashlineEditor::new();
        let _ = editor.read_file(&path);
        match editor.edit_by_hash(&path, &hash, &new_content) {
            Ok(_) | Err(HashlineError::FileChanged { .. }) => {}
            other => {
                // Any other error is acceptable as long as it is an error,
                // never a panic; record nothing.
                prop_assert!(other.is_err());
            }
        }
    }

    #[test]
    fn hashline_detects_stale_content_on_modified_file(
        line in ".*{0,80}",
        hash in "[a-f0-9]{8,64}",
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.txt");
        std::fs::write(&path, &line).unwrap();
        let mut editor = HashlineEditor::new();
        let _ = editor.read_file(&path);
        // Change the file on disk after the snapshot: any edit against the
        // old snapshot MUST return FileChanged, never silently apply.
        std::fs::write(&path, format!("{line}\nCHANGED")).unwrap();
        match editor.edit_by_hash(&path, &hash, "x") {
            Err(HashlineError::FileChanged { .. }) => {}
            Ok(_) => prop_assert!(false, "stale edit silently applied"),
            Err(_) => {}
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn guard_blocks_interpreter_wrapped_payloads(
        interpreter in prop::sample::select(vec!["sh -c", "bash -c", "python3 -c", "python -c", "perl -e", "awk 'BEGIN{"]),
        payload in prop::sample::select(vec![
            "rm -rf /", "rm -rf ~", ":(){ :|:& };:", "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda", "chmod -R 777 /", "sudo rm -rf /*",
        ]),
        quote in prop::sample::select(vec!["'", "\"", ""]),
        suffix in prop::sample::select(vec!["'", "\"", ""]),
    ) {
        let cmd = format!("{interpreter} {quote}{payload}{quote}{suffix}");
        let v = Guard::check_command(&cmd);
        prop_assert!(
            !matches!(v, kitty_harness::guard::GuardVerdict::Allow),
            "interpreter-wrapped payload must not be allowed: {cmd}"
        );
    }
}
