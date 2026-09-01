//! Aider Polyglot benchmark loader — official pattern with config.json parsing,
//! test file hiding, retry loop, and subprocess timeouts.
//!
//! Loads exercises from bench/official/polyglot-benchmark/<lang>/exercises/practice/
//! Reads .meta/config.json for file lists, copies only solution files to tempdir,
//! runs the agent, materializes tests AFTER agent finishes, then runs unit tests.
//! Retries once on failure with test output fed back.

use std::path::{Path, PathBuf};
use serde::Deserialize;

/// Parsed .meta/config.json for an Exercism exercise.
#[derive(Debug, Clone, Deserialize)]
pub struct ExerciseConfig {
    pub files: ExerciseFiles,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExerciseFiles {
    pub solution: Vec<String>,
    pub test: Vec<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub example: Vec<String>,
}

/// A Polyglot exercise loaded from official Exercism format.
#[derive(Debug, Clone)]
pub struct PolyglotExercise {
    pub id: String,
    pub lang: String,
    pub exercise_dir: PathBuf,
    pub config: ExerciseConfig,
    pub prompt: String,
}

/// Result of running one exercise.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PolyglotResult {
    pub id: String,
    pub lang: String,
    pub passed: bool,
    pub latency_ms: u64,
    pub attempts: usize,
    pub verdict: String,
    pub error: Option<String>,
}

/// Load exercise config from .meta/config.json.
pub fn load_exercise_config(exercise_dir: &Path) -> Option<ExerciseConfig> {
    let config_path = exercise_dir.join(".meta").join("config.json");
    let content = std::fs::read_to_string(&config_path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Build prompt from .docs/ directory files.
pub fn build_prompt(exercise_dir: &Path, config: &ExerciseConfig) -> String {
    let mut prompt = String::new();

    // Read introduction.md (optional)
    let intro = exercise_dir.join(".docs").join("introduction.md");
    if intro.exists() {
        if let Ok(content) = std::fs::read_to_string(&intro) {
            prompt.push_str(&content);
            prompt.push_str("\n\n");
        }
    }

    // Read instructions.md (required)
    let instructions = exercise_dir.join(".docs").join("instructions.md");
    if let Ok(content) = std::fs::read_to_string(&instructions) {
        prompt.push_str(&content);
        prompt.push_str("\n\n");
    }

    // Read instructions.append.md (optional)
    let append = exercise_dir.join(".docs").join("instructions.append.md");
    if append.exists() {
        if let Ok(content) = std::fs::read_to_string(&append) {
            prompt.push_str(&content);
            prompt.push_str("\n\n");
        }
    }

    // Add suffix with file list
    let file_list = config.files.solution.join(", ");
    prompt.push_str(&format!(
        "Use the above instructions to modify the supplied files: {file_list}\n\
         Don't change the names of existing functions or classes.\n\
         Only use standard libraries."
    ));

    prompt
}

/// Copy only solution files + .docs/ to tempdir (NO test files, NO .meta/).
pub fn copy_solution_files_only(
    src: &Path,
    dst: &Path,
    config: &ExerciseConfig,
) -> std::io::Result<()> {
    // Copy solution files
    for file in &config.files.solution {
        let src_path = src.join(file);
        let dst_path = dst.join(file);
        if let Some(parent) = dst_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if src_path.exists() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }

    // Copy .docs/ directory (for reference)
    let docs_src = src.join(".docs");
    let docs_dst = dst.join(".docs");
    if docs_src.exists() {
        copy_dir_all(&docs_src, &docs_dst)?;
    }

    Ok(())
}

/// Copy test files from original to tempdir (AFTER agent finishes).
pub fn materialize_tests(
    src: &Path,
    dst: &Path,
    config: &ExerciseConfig,
) -> std::io::Result<()> {
    for file in &config.files.test {
        let src_path = src.join(file);
        let dst_path = dst.join(file);
        if let Some(parent) = dst_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if src_path.exists() {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Run tests with timeout.
pub async fn run_test_with_timeout(
    exercise_dir: &Path,
    lang: &str,
    timeout_secs: u64,
) -> (bool, String) {
    let cmd = test_command(lang);
    let args: Vec<&str> = cmd.split_whitespace().collect();
    let program = args[0];
    let extra_args: Vec<&str> = args[1..].to_vec();

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokio::process::Command::new(program)
            .args(&extra_args)
            .current_dir(exercise_dir)
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let combined = if stderr.is_empty() {
                stdout
            } else {
                format!("{stdout}\n--- stderr ---\n{stderr}")
            };
            (output.status.success(), combined)
        }
        Ok(Err(e)) => (false, format!("Failed to run test: {e}")),
        Err(_) => (false, format!("Test timed out after {timeout_secs}s")),
    }
}

/// Get test command for a language.
pub fn test_command(lang: &str) -> &'static str {
    match lang {
        "python" => "python3 -m pytest -x -q",
        "rust" => "cargo test -- --include-ignored",
        "go" => "go test ./...",
        "javascript" => "npm test --silent",
        "cpp" => "make test",
        "java" => "./gradlew test",
        _ => "echo 'unknown language'",
    }
}

/// Build retry prompt with test output.
pub fn retry_prompt(test_output: &str, config: &ExerciseConfig) -> String {
    let file_list = config.files.solution.join(", ");
    let truncated = if test_output.len() > 4000 {
        // Take last 4000 chars, UTF-8 safe
        test_output
            .chars()
            .rev()
            .take(4000)
            .collect::<String>()
    } else {
        test_output.to_string()
    };
    format!(
        "The tests failed. Output:\n```\n{truncated}\n```\n\
         The tests are correct, don't try to change them.\n\
         Fix the implementation in {file_list}."
    )
}

/// Copy directory recursively.
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Load all Polyglot exercises from the benchmark directory.
pub fn load_polyglot_exercises(base_dir: &Path) -> Vec<PolyglotExercise> {
    let mut exercises = Vec::new();
    let languages = ["python", "javascript", "rust", "go", "cpp", "java"];

    for lang in &languages {
        let practice_dir = base_dir.join(lang).join("exercises").join("practice");
        if !practice_dir.exists() {
            continue;
        }

        for entry in std::fs::read_dir(&practice_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
        {
            let exercise_dir = entry.path();
            if !exercise_dir.is_dir() {
                continue;
            }

            let exercise_name = exercise_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Some(config) = load_exercise_config(&exercise_dir) {
                let prompt = build_prompt(&exercise_dir, &config);
                exercises.push(PolyglotExercise {
                    id: format!(
                        "{}-{}",
                        lang.chars().next().unwrap_or('x'),
                        exercise_name
                    ),
                    lang: lang.to_string(),
                    exercise_dir,
                    config,
                    prompt,
                });
            }
        }
    }

    exercises
}

/// Generate Aider-compatible report.
pub fn generate_polyglot_report(results: &[PolyglotResult]) -> serde_json::Value {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let rate = if total > 0 {
        (passed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let by_language: std::collections::HashMap<String, serde_json::Value> =
        results.iter().fold(
            std::collections::HashMap::new(),
            |mut acc, r| {
                let entry = acc.entry(r.lang.clone()).or_insert_with(|| {
                    serde_json::json!({"attempted": 0, "passed": 0, "failed": 0})
                });
                let obj = entry.as_object_mut().unwrap();
                *obj.get_mut("attempted").unwrap() =
                    serde_json::json!(obj["attempted"].as_i64().unwrap_or(0) + 1);
                if r.passed {
                    *obj.get_mut("passed").unwrap() =
                        serde_json::json!(obj["passed"].as_i64().unwrap_or(0) + 1);
                } else {
                    *obj.get_mut("failed").unwrap() =
                        serde_json::json!(obj["failed"].as_i64().unwrap_or(0) + 1);
                }
                acc
            },
        );

    serde_json::json!({
        "suite": "polyglot",
        "total": total,
        "passed": passed,
        "rate_pct": rate,
        "by_language": by_language,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let meta = dir.path().join(".meta");
        std::fs::create_dir_all(&meta).unwrap();
        std::fs::write(
            meta.join("config.json"),
            r#"{"files":{"solution":["src/lib.rs"],"test":["tests/test.rs"],"example":[]}}"#,
        )
        .unwrap();

        let config = load_exercise_config(dir.path()).unwrap();
        assert_eq!(config.files.solution, vec!["src/lib.rs"]);
        assert_eq!(config.files.test, vec!["tests/test.rs"]);
    }

    #[test]
    fn build_prompt_includes_instructions() {
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join(".docs");
        std::fs::create_dir_all(&docs).unwrap();
        std::fs::write(docs.join("instructions.md"), "Implement the exercise.").unwrap();

        let config = ExerciseConfig {
            files: ExerciseFiles {
                solution: vec!["lib.rs".into()],
                test: vec!["tests/test.rs".into()],
                example: vec![],
            },
        };

        let prompt = build_prompt(dir.path(), &config);
        assert!(prompt.contains("Implement the exercise."));
        assert!(prompt.contains("lib.rs"));
        assert!(prompt.contains("Don't change the names"));
    }

    #[test]
    fn test_command_per_language() {
        assert_eq!(test_command("python"), "python3 -m pytest -x -q");
        assert_eq!(test_command("rust"), "cargo test -- --include-ignored");
        assert_eq!(test_command("go"), "go test ./...");
    }

    #[test]
    fn retry_prompt_includes_output() {
        let config = ExerciseConfig {
            files: ExerciseFiles {
                solution: vec!["lib.rs".into()],
                test: vec!["tests/test.rs".into()],
                example: vec![],
            },
        };
        let prompt = retry_prompt("test failed: assertion error", &config);
        assert!(prompt.contains("test failed"));
        assert!(prompt.contains("lib.rs"));
        assert!(prompt.contains("don't try to change them"));
    }

    #[test]
    fn copy_solution_files_excludes_tests() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        // Create source files
        std::fs::create_dir_all(src.path().join("src")).unwrap();
        std::fs::write(src.path().join("src/lib.rs"), "fn main() {}").unwrap();
        std::fs::create_dir_all(src.path().join("tests")).unwrap();
        std::fs::write(src.path().join("tests/test.rs"), "#[test] fn t() {}").unwrap();
        std::fs::create_dir_all(src.path().join(".docs")).unwrap();
        std::fs::write(src.path().join(".docs/instructions.md"), "docs").unwrap();
        std::fs::create_dir_all(src.path().join(".meta")).unwrap();
        std::fs::write(src.path().join(".meta/config.json"), "{}").unwrap();

        let config = ExerciseConfig {
            files: ExerciseFiles {
                solution: vec!["src/lib.rs".into()],
                test: vec!["tests/test.rs".into()],
                example: vec![],
            },
        };

        copy_solution_files_only(src.path(), dst.path(), &config).unwrap();

        // Solution file present
        assert!(dst.path().join("src/lib.rs").exists());
        // .docs/ present
        assert!(dst.path().join(".docs/instructions.md").exists());
        // Test file NOT present
        assert!(!dst.path().join("tests/test.rs").exists());
        // .meta/ NOT present
        assert!(!dst.path().join(".meta").exists());
    }

    #[test]
    fn materialize_tests_copies_test_files() {
        let src = tempfile::tempdir().unwrap();
        let dst = tempfile::tempdir().unwrap();

        std::fs::create_dir_all(src.path().join("tests")).unwrap();
        std::fs::write(src.path().join("tests/test.rs"), "#[test] fn t() {}").unwrap();

        let config = ExerciseConfig {
            files: ExerciseFiles {
                solution: vec!["src/lib.rs".into()],
                test: vec!["tests/test.rs".into()],
                example: vec![],
            },
        };

        materialize_tests(src.path(), dst.path(), &config).unwrap();
        assert!(dst.path().join("tests/test.rs").exists());
        let content = std::fs::read_to_string(dst.path().join("tests/test.rs")).unwrap();
        assert_eq!(content, "#[test] fn t() {}");
    }

    #[test]
    fn retry_prompt_truncates_long_output() {
        let config = ExerciseConfig {
            files: ExerciseFiles {
                solution: vec!["lib.rs".into()],
                test: vec![],
                example: vec![],
            },
        };
        let long_output = "x".repeat(8000);
        let prompt = retry_prompt(&long_output, &config);
        // Should be truncated to ~4000 chars + framing
        assert!(prompt.len() < 8000);
        assert!(prompt.contains("The tests failed"));
    }
}
