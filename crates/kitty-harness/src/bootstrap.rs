//! Bootstrap detection — scans workspace for marker files on first turn.
//!
//! Detects project type (Rust, Node.js, Python, Go) by checking for
//! indicator files and caches the result. Exposes a `Test` tool that
//! auto-detects the right test command.

use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BootstrapError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("No test command available (no project type detected)")]
    NoTestCommand,
}

/// Detected project type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectType {
    Rust,
    Node,
    Python,
    Go,
}

impl ProjectType {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            ProjectType::Rust => "Rust/Cargo",
            ProjectType::Node => "Node/npm",
            ProjectType::Python => "Python",
            ProjectType::Go => "Go",
        }
    }

    /// The test command for this project type.
    pub fn test_command(&self) -> &str {
        match self {
            ProjectType::Rust => "cargo test",
            ProjectType::Node => "npm test",
            ProjectType::Python => "pytest",
            ProjectType::Go => "go test ./...",
        }
    }

    /// Build commands for this project type.
    pub fn build_commands(&self) -> Vec<String> {
        match self {
            ProjectType::Rust => vec![
                "cargo build".to_string(),
                "cargo test".to_string(),
                "cargo clippy".to_string(),
            ],
            ProjectType::Node => vec!["npm run build".to_string(), "npm test".to_string()],
            ProjectType::Python => vec!["pytest".to_string()],
            ProjectType::Go => vec!["go build ./...".to_string(), "go test ./...".to_string()],
        }
    }
}

/// Bootstrap detection result.
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// All detected project types.
    pub project_types: Vec<ProjectType>,
    /// Injected system hint message.
    pub hint: String,
    /// The test command (from the first detected project).
    pub test_command: Option<String>,
}

/// Detects workspace type and caches the result.
pub struct Bootstrap {
    /// Cached detection result.
    cached: Option<BootstrapResult>,
    /// Working directory to scan.
    workspace: std::path::PathBuf,
}

impl Bootstrap {
    /// Create a new bootstrap detector for the given workspace.
    pub fn new(workspace: impl AsRef<Path>) -> Self {
        Self {
            cached: None,
            workspace: workspace.as_ref().to_path_buf(),
        }
    }

    /// Run detection (idempotent — caches result).
    pub fn detect(&mut self) -> &BootstrapResult {
        if let Some(ref cached) = self.cached {
            return cached;
        }

        let mut project_types = Vec::new();

        if self.workspace.join("Cargo.toml").exists() {
            project_types.push(ProjectType::Rust);
        }
        if self.workspace.join("package.json").exists() {
            project_types.push(ProjectType::Node);
        }
        if self.workspace.join("pyproject.toml").exists()
            || self.workspace.join("setup.py").exists()
        {
            project_types.push(ProjectType::Python);
        }
        if self.workspace.join("go.mod").exists() {
            project_types.push(ProjectType::Go);
        }

        let hint = if project_types.is_empty() {
            String::new()
        } else {
            let labels: Vec<&str> = project_types.iter().map(|p| p.label()).collect();
            let test_cmd = project_types[0].test_command();
            format!(
                "Project detected: {}. Use `{}` to run tests.",
                labels.join(", "),
                test_cmd
            )
        };

        let test_command = project_types.first().map(|p| p.test_command().to_string());

        self.cached = Some(BootstrapResult {
            project_types,
            hint,
            test_command,
        });

        self.cached.as_ref().unwrap()
    }

    /// Execute the test command for the detected project type.
    pub fn execute_test(&self) -> Result<String, BootstrapError> {
        let result = self.cached.as_ref().ok_or(BootstrapError::NoTestCommand)?;

        let cmd = result
            .test_command
            .as_ref()
            .ok_or(BootstrapError::NoTestCommand)?;

        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.workspace)
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push_str("\n--- stderr ---\n");
            }
            result.push_str(&stderr);
        }

        Ok(result)
    }

    /// Get the cached result without running detection.
    pub fn result(&self) -> Option<&BootstrapResult> {
        self.cached.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_rust_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert_eq!(result.project_types, vec![ProjectType::Rust]);
        assert!(result.hint.contains("Rust/Cargo"));
        assert_eq!(result.test_command.as_deref(), Some("cargo test"));
    }

    #[test]
    fn detect_node_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert_eq!(result.project_types, vec![ProjectType::Node]);
        assert!(result.hint.contains("Node/npm"));
        assert_eq!(result.test_command.as_deref(), Some("npm test"));
    }

    #[test]
    fn detect_python_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("pyproject.toml"), "").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert_eq!(result.project_types, vec![ProjectType::Python]);
        assert!(result.hint.contains("Python"));
        assert_eq!(result.test_command.as_deref(), Some("pytest"));
    }

    #[test]
    fn detect_go_project() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("go.mod"), "module test").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert_eq!(result.project_types, vec![ProjectType::Go]);
        assert!(result.hint.contains("Go"));
        assert_eq!(result.test_command.as_deref(), Some("go test ./..."));
    }

    #[test]
    fn detect_multiple_projects() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        std::fs::write(dir.path().join("package.json"), "{}").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert_eq!(result.project_types.len(), 2);
        assert!(result.project_types.contains(&ProjectType::Rust));
        assert!(result.project_types.contains(&ProjectType::Node));
    }

    #[test]
    fn detect_no_project() {
        let dir = tempfile::tempdir().unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let result = bootstrap.detect();
        assert!(result.project_types.is_empty());
        assert!(result.hint.is_empty());
        assert!(result.test_command.is_none());
    }

    #[test]
    fn detect_is_cached() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]").unwrap();
        let mut bootstrap = Bootstrap::new(dir.path());
        let r1 = bootstrap.detect();
        let r1_ptr = r1 as *const BootstrapResult;
        let r2 = bootstrap.detect();
        let r2_ptr = r2 as *const BootstrapResult;
        assert!(std::ptr::eq(r1_ptr, r2_ptr));
    }
}
