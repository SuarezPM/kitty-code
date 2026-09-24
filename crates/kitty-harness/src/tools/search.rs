//! Glob/Grep tool implementations (W2 split of `tools.rs`, por símbolo).
//!
//! Grep file reads are todo-file 1-op via `tokio::fs` (W1).
use super::{ToolError, ToolRegistry, ToolResult};

impl ToolRegistry {
    /// Execute a Glob tool call.
    pub(crate) async fn execute_glob(
        &self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Glob".to_string(),
                reason: "missing 'pattern'".to_string(),
            })?;

        let entries: Vec<String> = glob::glob(pattern)
            .map_err(|e| ToolError::Glob(e.to_string()))?
            .filter_map(|entry| entry.ok())
            .map(|path| path.display().to_string())
            .collect();

        if entries.is_empty() {
            Ok(ToolResult {
                success: true,
                output: "No files matched".to_string(),
                retry: false,
            })
        } else {
            Ok(ToolResult {
                success: true,
                output: entries.join("\n"),
                retry: false,
            })
        }
    }

    /// Execute a Grep tool call.
    pub(crate) async fn execute_grep(
        &self,
        args: &serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs {
                tool: "Grep".to_string(),
                reason: "missing 'pattern'".to_string(),
            })?;

        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let include = args.get("include").and_then(|v| v.as_str());

        let re = regex::Regex::new(pattern)
            .map_err(|e| ToolError::Grep(format!("Invalid regex: {e}")))?;

        let mut results = Vec::new();
        let walker = ignore::WalkBuilder::new(path)
            .hidden(false)
            .git_ignore(true)
            .build();

        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }

            let entry_path = entry.path();
            if let Some(incl) = include {
                if let Some(ext) = entry_path.extension() {
                    if ext != incl.trim_start_matches('.') {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            if let Ok(content) = tokio::fs::read_to_string(entry_path).await {
                for (line_num, line) in content.lines().enumerate() {
                    if re.is_match(line) {
                        results.push(format!(
                            "{}:{}:{}",
                            entry_path.display(),
                            line_num + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }

        // Truncate if too many results
        let max_results = 200;
        if results.len() > max_results {
            let total = results.len();
            results.truncate(max_results);
            results.push(format!("... and {total} total matches (truncated)"));
        }

        Ok(ToolResult {
            success: true,
            output: if results.is_empty() {
                "No matches found".to_string()
            } else {
                results.join("\n")
            },
            retry: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::ToolRegistry;

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("b.txt"), "hello").unwrap();

        let registry = ToolRegistry::new();
        let pattern = format!("{}/*.rs", dir.path().display());
        let args = serde_json::json!({"pattern": pattern});
        let result = registry.execute_glob(&args).await.unwrap();
        assert!(result.success);
        assert!(result.output.contains("a.rs"));
        assert!(!result.output.contains("b.txt"));
    }

    #[tokio::test]
    async fn grep_respects_gitignore() {
        // F3.2: Grep uses ignore crate — files in .gitignore'd dirs are excluded.
        let dir = tempfile::tempdir().unwrap();
        // Init a git repo so ignore crate picks up .gitignore
        std::process::Command::new("git")
            .args(["init", dir.path().to_str().unwrap()])
            .output()
            .unwrap();
        // Create .gitignore excluding target/
        std::fs::write(dir.path().join(".gitignore"), "target/\n").unwrap();
        // Create a file that SHOULD match (not gitignored)
        std::fs::write(dir.path().join("src.rs"), "needle here").unwrap();
        // Create a file in target/ that should NOT match
        let target_dir = dir.path().join("target");
        std::fs::create_dir_all(&target_dir).unwrap();
        std::fs::write(target_dir.join("build.rs"), "needle here").unwrap();

        let registry = ToolRegistry::new();
        let args = serde_json::json!({
            "pattern": "needle",
            "path": dir.path().to_str().unwrap()
        });
        let result = registry.execute_grep(&args).await.unwrap();
        assert!(result.success);
        assert!(
            result.output.contains("src.rs"),
            "should find src.rs, got: {}",
            result.output
        );
        assert!(
            !result.output.contains("target/build.rs"),
            "should NOT find target/build.rs, got: {}",
            result.output
        );
    }
}
