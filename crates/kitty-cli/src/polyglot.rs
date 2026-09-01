//! Aider Polyglot benchmark loader — 225 Exercism exercises with unit test grading.
//!
//! Loads exercises from bench/official/polyglot-benchmark/<lang>/exercises/practice/
//! Extracts prompts from .docs/ directories (instructions.md + instructions.append.md)
//! Runs real unit tests (pytest, go test, cargo test, etc.)
//! Generates results comparable to the public leaderboard.

use std::path::{Path, PathBuf};

/// A Polyglot exercise loaded from Exercism format.
#[derive(Debug, Clone)]
pub struct PolyglotExercise {
    pub id: String,
    pub lang: String,
    pub prompt: String,
    pub exercise_dir: PathBuf,
}

/// Result of running one exercise.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PolyglotResult {
    pub id: String,
    pub lang: String,
    pub passed: bool,
    pub latency_ms: u64,
    pub verdict: String,
    pub error: Option<String>,
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

        let entries: Vec<_> = std::fs::read_dir(&practice_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        for entry in entries {
            let exercise_dir = entry.path();
            let exercise_name = exercise_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Some(prompt) = extract_prompt(&exercise_dir) {
                exercises.push(PolyglotExercise {
                    id: format!("polyglot_{}-{}", lang, exercise_name),
                    lang: lang.to_string(),
                    prompt,
                    exercise_dir,
                });
            }
        }
    }

    exercises
}

/// Extract the prompt from an exercise's .docs/ directory.
fn extract_prompt(exercise_dir: &Path) -> Option<String> {
    let docs_dir = exercise_dir.join(".docs");
    if !docs_dir.exists() {
        return None;
    }

    let instructions = docs_dir.join("instructions.md");
    let append = docs_dir.join("instructions.append.md");

    let mut parts = Vec::new();

    if let Ok(content) = std::fs::read_to_string(&instructions) {
        parts.push(content);
    }
    if let Ok(content) = std::fs::read_to_string(&append) {
        parts.push(content);
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Run the test suite for an exercise and return the result.
pub async fn run_polyglot_exercise(exercise: &PolyglotExercise) -> PolyglotResult {
    let start = std::time::Instant::now();

    let (success, error) = match exercise.lang.as_str() {
        "python" => run_pytest(&exercise.exercise_dir).await,
        "go" => run_go_test(&exercise.exercise_dir).await,
        "rust" => run_cargo_test(&exercise.exercise_dir).await,
        "javascript" => run_node_test(&exercise.exercise_dir).await,
        "java" => run_java_test(&exercise.exercise_dir).await,
        "cpp" => run_cpp_test(&exercise.exercise_dir).await,
        _ => (false, Some(format!("unsupported language: {}", exercise.lang))),
    };

    let latency = start.elapsed().as_millis() as u64;

    PolyglotResult {
        id: exercise.id.clone(),
        lang: exercise.lang.clone(),
        passed: success,
        latency_ms: latency,
        verdict: if success { "pass" } else { "fail" }.to_string(),
        error,
    }
}

async fn run_pytest(exercise_dir: &Path) -> (bool, Option<String>) {
    let output = tokio::process::Command::new("python3")
        .args(["-m", "pytest", "-x", "--tb=short", "-q"])
        .current_dir(exercise_dir)
        .output()
        .await;

    match output {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn run_go_test(exercise_dir: &Path) -> (bool, Option<String>) {
    let output = tokio::process::Command::new("go")
        .args(["test", "-v", "./..."])
        .current_dir(exercise_dir)
        .output()
        .await;

    match output {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn run_cargo_test(exercise_dir: &Path) -> (bool, Option<String>) {
    let output = tokio::process::Command::new("cargo")
        .args(["test", "--quiet"])
        .current_dir(exercise_dir)
        .output()
        .await;

    match output {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn run_node_test(exercise_dir: &Path) -> (bool, Option<String>) {
    let output = tokio::process::Command::new("node")
        .args(["--test"])
        .current_dir(exercise_dir)
        .output()
        .await;

    match output {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn run_java_test(exercise_dir: &Path) -> (bool, Option<String>) {
    // Find all .java files for compilation
    let mut java_files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(exercise_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".java") {
                    java_files.push(name.to_string());
                }
            }
        }
    }
    if java_files.is_empty() {
        return (false, Some("no .java files found".to_string()));
    }

    // Compile
    let mut cmd = tokio::process::Command::new("javac");
    cmd.arg("-d").arg("bin");
    for f in &java_files {
        cmd.arg(f);
    }
    cmd.current_dir(exercise_dir);
    let compile = cmd.output().await;

    match compile {
        Ok(o) if !o.status.success() => {
            return (
                false,
                Some(String::from_utf8_lossy(&o.stderr).to_string()),
            );
        }
        Err(e) => return (false, Some(e.to_string())),
        _ => {}
    }

    // Find the class with main (first .java file without "Test" in name)
    let main_class = java_files
        .iter()
        .find(|f| !f.contains("Test"))
        .and_then(|f| f.strip_suffix(".java"))
        .unwrap_or("Main");

    let run = tokio::process::Command::new("java")
        .args(["-cp", "bin", main_class])
        .current_dir(exercise_dir)
        .output()
        .await;

    match run {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

async fn run_cpp_test(exercise_dir: &Path) -> (bool, Option<String>) {
    // Find .cpp files
    let mut cpp_files: Vec<String> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(exercise_dir) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".cpp") && !name.contains("test") {
                    cpp_files.push(name.to_string());
                }
            }
        }
    }
    if cpp_files.is_empty() {
        // Try src/ subdirectory
        let src_dir = exercise_dir.join("src");
        if let Ok(entries) = std::fs::read_dir(&src_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".cpp") {
                        cpp_files.push(format!("src/{}", name));
                    }
                }
            }
        }
    }
    if cpp_files.is_empty() {
        return (false, Some("no .cpp files found".to_string()));
    }

    let sources = cpp_files.join(" ");

    // Also include header file base names without extension
    let compile = tokio::process::Command::new("g++")
        .args(["-std=c++17", "-o", "test", &sources])
        .current_dir(exercise_dir)
        .output()
        .await;

    match compile {
        Ok(o) if !o.status.success() => {
            return (
                false,
                Some(String::from_utf8_lossy(&o.stderr).to_string()),
            );
        }
        Err(e) => return (false, Some(e.to_string())),
        _ => {}
    }

    let run = tokio::process::Command::new("./test")
        .current_dir(exercise_dir)
        .output()
        .await;

    match run {
        Ok(o) => (
            o.status.success(),
            Some(String::from_utf8_lossy(&o.stderr).to_string()),
        ),
        Err(e) => (false, Some(e.to_string())),
    }
}

/// Generate a Polyglot benchmark report in Aider-compatible format.
pub fn generate_polyglot_report(results: &[PolyglotResult]) -> serde_json::Value {
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let rate = if total > 0 {
        (passed as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let mut by_language: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    for r in results {
        let entry = by_language
            .entry(r.lang.clone())
            .or_insert_with(|| {
                serde_json::json!({
                    "attempted": 0,
                    "passed": 0,
                    "failed": 0,
                })
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
    }

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
    fn extract_prompt_from_instructions_md() {
        let dir = PathBuf::from("/tmp/test-exercise-polyglot");
        std::fs::create_dir_all(dir.join(".docs")).unwrap();
        std::fs::write(dir.join(".docs").join("instructions.md"), "Test prompt").unwrap();

        let prompt = extract_prompt(&dir);
        assert_eq!(prompt, Some("Test prompt".to_string()));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extract_prompt_combines_instructions_and_append() {
        let dir = PathBuf::from("/tmp/test-exercise-polyglot-append");
        std::fs::create_dir_all(dir.join(".docs")).unwrap();
        std::fs::write(
            dir.join(".docs").join("instructions.md"),
            "Base instructions",
        )
        .unwrap();
        std::fs::write(
            dir.join(".docs").join("instructions.append.md"),
            "Extra details",
        )
        .unwrap();

        let prompt = extract_prompt(&dir);
        assert_eq!(
            prompt,
            Some("Base instructions\n\nExtra details".to_string())
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extract_prompt_none_when_no_docs() {
        let dir = PathBuf::from("/tmp/test-exercise-polyglot-nodocs");
        std::fs::create_dir_all(&dir).unwrap();

        let prompt = extract_prompt(&dir);
        assert_eq!(prompt, None);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn generate_report() {
        let results = vec![
            PolyglotResult {
                id: "polyglot_python-test1".into(),
                lang: "python".into(),
                passed: true,
                latency_ms: 100,
                verdict: "pass".into(),
                error: None,
            },
            PolyglotResult {
                id: "polyglot_python-test2".into(),
                lang: "python".into(),
                passed: false,
                latency_ms: 200,
                verdict: "fail".into(),
                error: None,
            },
        ];

        let report = generate_polyglot_report(&results);
        assert_eq!(report["total"], 2);
        assert_eq!(report["passed"], 1);
        assert!((report["rate_pct"].as_f64().unwrap() - 50.0).abs() < 1e-9);
    }
}
