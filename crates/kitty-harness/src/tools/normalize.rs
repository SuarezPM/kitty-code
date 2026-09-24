//! Tool-call normalization — canonical names + argument-key aliases.
//!
//! W2 split of `tools.rs` by symbol: weaker models emit near-miss tool names
//! and argument keys; both fold onto the canonical forms here. Pure free
//! functions, no I/O. Re-exported through `tools.rs` so `crate::tools::*`
//! paths (e.g. `agent/response.rs`) keep working unchanged.

use crate::parser::ParsedToolCall;

/// Canonical tool names served by [`crate::tools::ToolRegistry`].
const REAL_TOOLS: [&str; 6] = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"];

/// Common hallucinated synonyms mapped onto the real tools. Never consulted
/// before the exact/case-insensitive real-name check, so a synonym can never
/// shadow a canonical tool name.
const TOOL_SYNONYMS: &[(&str, &str)] = &[
    // str_replace-style editors → Edit
    ("str_replace", "Edit"),
    ("str_replace_editor", "Edit"),
    ("str_replace_based_edit_tool", "Edit"),
    ("edit_file", "Edit"),
    // file viewers → Read
    ("view", "Read"),
    ("read_file", "Read"),
    ("cat", "Read"),
    // file creators → Write
    ("write_file", "Write"),
    ("create_file", "Write"),
    // shell aliases → Bash
    ("shell", "Bash"),
    ("terminal", "Bash"),
    ("run_command", "Bash"),
    // directory listings / glob-style names → Glob
    ("find", "Glob"),
    ("ls", "Glob"),
    ("list_files", "Glob"),
    ("list_dir", "Glob"),
    // search synonyms → Grep
    ("search", "Grep"),
    ("search_files", "Grep"),
];

/// Argument-key aliases folded onto the canonical keys, applied ONLY when the
/// canonical key is absent from the arguments object.
const ARG_KEY_ALIASES: &[(&str, &str)] = &[
    ("file_path", "path"),
    ("filepath", "path"),
    ("file", "path"),
    ("cmd", "command"),
    ("query", "pattern"),
    ("old_str", "old_string"),
    ("old_text", "old_string"),
    ("new_str", "new_string"),
    ("new_text", "new_string"),
];

/// Extract the salvage body when arguments carry ONLY an `_raw` string (the
/// parser's unparsable-JSON retention shape). Mixed objects pass through.
pub(crate) fn raw_only_arguments(args: &serde_json::Value) -> Option<&str> {
    let map = args.as_object()?;
    if map.len() == 1 {
        map.get("_raw")?.as_str()
    } else {
        None
    }
}

/// Attempt to repair a `_raw` salvage body into usable JSON arguments.
///
/// Uses [`crate::json_repair::repair_json`] to fix control chars and
/// invalid escapes. Returns `Some(repaired_value)` on success,
/// `None` if the body cannot be salvaged.
pub(crate) fn attempt_raw_repair(raw: &str) -> Option<serde_json::Value> {
    // First try the parser's structural repair (smart quotes, trailing commas)
    let structurally_repaired = crate::parser::repair_json(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&structurally_repaired) {
        if v.is_object() {
            return Some(v);
        }
    }
    // Then try the low-level repair (control chars, invalid escapes)
    let repaired = crate::json_repair::repair_json(raw);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&repaired) {
        if v.is_object() {
            return Some(v);
        }
    }
    None
}

/// Fold one incoming call onto canonical tool/argument naming.
///
/// Idempotent: canonical names pass through unchanged and aliased keys are
/// only moved when the canonical key is absent, so re-normalizing an already
/// normalized call is a no-op.
pub(crate) fn normalize_tool_call(tc: &ParsedToolCall) -> ParsedToolCall {
    ParsedToolCall {
        name: normalize_tool_name(&tc.name),
        arguments: rekey_arg_aliases(tc.arguments.clone()),
        id: tc.id.clone(),
    }
}

/// Resolve `name` onto the canonical casing of a real tool (case-insensitive)
/// or a known hallucinated synonym; unknown names pass through unchanged.
pub(crate) fn normalize_tool_name(name: &str) -> String {
    if let Some(real) = REAL_TOOLS.iter().find(|t| t.eq_ignore_ascii_case(name)) {
        return (*real).to_string();
    }
    if let Some((_, real)) = TOOL_SYNONYMS
        .iter()
        .find(|(s, _)| s.eq_ignore_ascii_case(name))
    {
        return (*real).to_string();
    }
    name.to_string()
}

/// Re-key aliased argument names onto canonical ones; an alias is only moved
/// when the canonical key is absent, and consumed aliases are removed.
pub(crate) fn rekey_arg_aliases(args: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut map) = args else {
        return args;
    };
    for (alias, canonical) in ARG_KEY_ALIASES {
        if let Some(value) = map.remove(*alias) {
            map.entry((*canonical).to_string()).or_insert(value);
        }
    }
    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::super::{ToolError, ToolRegistry};
    use super::*;
    use crate::parser::ParsedToolCall;
    use std::io::Write;

    fn temp_file(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[tokio::test]
    async fn alias_synonym_name_dispatches_to_real_tool() {
        // "str_replace" is not a real tool; it must dispatch to Edit.
        let f = temp_file("alpha\nbeta");
        let mut registry = ToolRegistry::new();
        registry
            .execute_read(&serde_json::json!({"path": f.path().to_str().unwrap()}))
            .await
            .unwrap();
        let tc = ParsedToolCall {
            name: "str_replace".to_string(),
            arguments: serde_json::json!({
                "path": f.path().to_str().unwrap(),
                "old_string": "beta",
                "new_string": "BETA"
            }),
            id: None,
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(result.success, "got: {}", result.output);
        assert_eq!(std::fs::read_to_string(f.path()).unwrap(), "alpha\nBETA");
    }

    #[tokio::test]
    async fn alias_arg_rekey_maps_onto_canonical_keys() {
        // file_path → path, cmd → command (only when canonical absent).
        let f = temp_file("content");
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"file_path": f.path().to_str().unwrap()}),
            id: None,
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(result.success, "file_path alias must satisfy Read's `path`");

        let normalized = normalize_tool_call(&ParsedToolCall {
            name: "Bash".to_string(),
            arguments: serde_json::json!({"cmd": "echo hi", "extra": 1}),
            id: None,
        });
        assert_eq!(normalized.arguments["command"], "echo hi");
        assert!(normalized.arguments.get("cmd").is_none());
        assert_eq!(normalized.arguments["extra"], 1);
    }

    #[tokio::test]
    async fn alias_canonical_key_wins_over_alias() {
        // When the canonical key is already present, the alias is dropped
        // and the canonical value is kept.
        let normalized = normalize_tool_call(&ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({"path": "real.txt", "file_path": "fake.txt"}),
            id: None,
        });
        assert_eq!(normalized.arguments["path"], "real.txt");
        assert!(normalized.arguments.get("file_path").is_none());
    }

    #[tokio::test]
    async fn alias_real_names_pass_through_with_canonical_casing() {
        // Case-insensitive resolution onto canonical casing...
        assert_eq!(normalize_tool_name("read"), "Read");
        assert_eq!(normalize_tool_name("BASH"), "Bash");
        assert_eq!(normalize_tool_name("Glob"), "Glob");
        // ...and a synonym table entry can never shadow a real name.
        assert_eq!(normalize_tool_name("Edit"), "Edit");
    }

    #[tokio::test]
    async fn alias_unknown_names_pass_through_unchanged() {
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "TotallyMadeUp".to_string(),
            arguments: serde_json::json!({}),
            id: None,
        };
        let err = registry.execute(&tc).await.unwrap_err();
        match err {
            ToolError::NotFound(name) => {
                assert_eq!(name, "TotallyMadeUp", "unknown names must pass through");
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn alias_normalization_is_idempotent() {
        // G3-1: the agent loop normalizes upstream of the quality gate and
        // execute() normalizes again — the second pass must be a no-op.
        let once = normalize_tool_call(&ParsedToolCall {
            name: "str_replace".to_string(),
            arguments: serde_json::json!({
                "file_path": "a.rs",
                "old_str": "x",
                "new_str": "y"
            }),
            id: None,
        });
        let twice = normalize_tool_call(&once);
        assert_eq!(once, twice);
        assert_eq!(once.name, "Edit");
        assert_eq!(once.arguments["path"], "a.rs");
        assert_eq!(once.arguments["old_string"], "x");
        assert_eq!(once.arguments["new_string"], "y");
    }

    #[tokio::test]
    async fn raw_only_arguments_get_actionable_diagnostic() {
        // G3-5: salvaged `_raw`-only args must explain WHAT was wrong and
        // show the recovered body instead of a generic missing-key error.
        let mut registry = ToolRegistry::new();
        let tc = ParsedToolCall {
            name: "Read".to_string(),
            arguments: serde_json::json!({
                "_raw": "{\"name\": \"Read\", \"path\": no_close"
            }),
            id: None,
        };
        let result = registry.execute(&tc).await.unwrap();
        assert!(!result.success);
        assert!(
            result.output.contains("not valid JSON"),
            "got: {}",
            result.output
        );
        assert!(
            result.output.contains("no_close"),
            "recovered body must be shown, got: {}",
            result.output
        );
    }
}
