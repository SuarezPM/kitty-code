//! Plugin system — static registry, no dlopen.
//!
//! `Plugin` trait + `PluginRegistry` (`Vec<Box<dyn Plugin>>`) + `load_from_manifest`
//! for `plugins.toml`. Third-party tools register without touching core `tools.rs`.
//! No `libloading`/`dlopen` — all plugins are statically linked via `register`.

use std::path::Path;

use kitty_runtime::ToolDefinition;
use thiserror::Error;

use crate::tools::ToolResult;

/// Errors for plugin registration and manifest loading.
#[derive(Error, Debug)]
pub enum PluginError {
    #[error("tool already exists: {0}")]
    DuplicateTool(String),

    #[error("tool not found: {0}")]
    NotFound(String),

    #[error("manifest error: {0}")]
    Manifest(String),

    #[error("execution error: {0}")]
    Execution(String),
}

/// Third-party plugin interface.
///
/// Each plugin exposes a name, a list of [`ToolDefinition`]s, and an `execute`
/// handler. Implementors are `Send + Sync` for `PluginRegistry` storage.
pub trait Plugin: Send + Sync {
    /// Stable plugin identifier (e.g., `"hello"`).
    fn name(&self) -> &str;

    /// Tool definitions contributed by this plugin.
    fn tools(&self) -> Vec<ToolDefinition>;

    /// Execute a tool owned by this plugin.
    ///
    /// `tool_name` is one of `self.tools()[*].function.name`. Return
    /// `Err(PluginError::NotFound)` if the tool is unknown to this plugin.
    fn execute(&self, tool_name: &str, args: &serde_json::Value)
        -> Result<ToolResult, PluginError>;
}

/// Static registry of plugins.
///
/// Holds `Vec<Box<dyn Plugin>>` and enforces unique tool names across all
/// registered plugins (collision → `DuplicateTool`, no overwrite).
pub struct PluginRegistry {
    plugins: Vec<Box<dyn Plugin>>,
}

impl std::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginRegistry")
            .field(
                "plugins",
                &self.plugins.iter().map(|p| p.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Register a plugin. Fails if any of its tool names collide with an
    /// already-registered tool.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) -> Result<(), PluginError> {
        let existing: std::collections::HashSet<String> = self.tool_names().into_iter().collect();
        for t in plugin.tools() {
            if existing.contains(&t.function.name) {
                return Err(PluginError::DuplicateTool(t.function.name.clone()));
            }
        }
        // Also check duplicate within the plugin itself
        let mut seen = std::collections::HashSet::new();
        for t in plugin.tools() {
            if !seen.insert(t.function.name.clone()) {
                return Err(PluginError::DuplicateTool(t.function.name));
            }
        }
        self.plugins.push(plugin);
        Ok(())
    }

    /// Alias for `register` to match spec wording `register_plugin`.
    pub fn register_plugin(&mut self, plugin: Box<dyn Plugin>) -> Result<(), PluginError> {
        self.register(plugin)
    }

    /// All tool definitions contributed by registered plugins.
    pub fn tools(&self) -> Vec<ToolDefinition> {
        self.plugins.iter().flat_map(|p| p.tools()).collect()
    }

    /// Names of all registered plugin tools.
    pub fn tool_names(&self) -> Vec<String> {
        self.plugins
            .iter()
            .flat_map(|p| p.tools())
            .map(|t| t.function.name.clone())
            .collect()
    }

    /// Whether a tool name is provided by any plugin.
    pub fn contains_tool(&self, name: &str) -> bool {
        self.plugins
            .iter()
            .any(|p| p.tools().iter().any(|t| t.function.name == name))
    }

    /// Execute a plugin tool by name.
    ///
    /// Returns `None` if no plugin owns `name`, otherwise `Some(result)`.
    pub fn execute(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<Result<ToolResult, PluginError>> {
        for p in &self.plugins {
            if p.tools().iter().any(|t| t.function.name == tool_name) {
                return Some(p.execute(tool_name, args));
            }
        }
        None
    }

    /// Load a registry from a `plugins.toml` manifest.
    ///
    /// Manifest format:
    /// ```toml
    /// [[plugin]]
    /// name = "hello"
    /// enabled = true
    /// ```
    /// Known builtin `hello` is instantiated as [`HelloPlugin`] when enabled.
    /// Unknown names are ignored with a warning (no `dlopen`).
    pub fn load_from_manifest(path: &Path) -> Result<Self, PluginError> {
        let manifest = load_manifest(path)?;
        let mut registry = Self::new();
        for entry in manifest.plugins() {
            if !entry.enabled.unwrap_or(true) {
                continue;
            }
            match entry.name.as_str() {
                "hello" => {
                    registry.register(Box::new(HelloPlugin))?;
                }
                other => {
                    tracing::warn!(plugin = %other, "unknown plugin in manifest, skipping (no dlopen)");
                }
            }
        }
        Ok(registry)
    }

    /// Load from `plugins.toml` if present, else return an empty registry.
    pub fn load_from_manifest_or_empty(path: &Path) -> Self {
        Self::load_from_manifest(path).unwrap_or_else(|_| Self::new())
    }
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct PluginManifest {
    #[serde(default)]
    plugin: Vec<PluginEntry>,
}

impl PluginManifest {
    fn plugins(self) -> Vec<PluginEntry> {
        self.plugin
    }
}

#[derive(Debug, serde::Deserialize)]
struct PluginEntry {
    name: String,
    #[serde(default)]
    enabled: Option<bool>,
}

fn load_manifest(path: &Path) -> Result<PluginManifest, PluginError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PluginError::Manifest(format!("read {}: {e}", path.display())))?;
    parse_manifest(&content)
}

/// Async manifest loader: todo-file 1-op via tokio::fs (W1). Same parse as
/// [`load_manifest`]; startup callers keep the sync path for compat.
#[allow(dead_code)] // W1 async variant for runtimes; exercised in tests
async fn load_manifest_async(path: &Path) -> Result<PluginManifest, PluginError> {
    let content = tokio::fs::read_to_string(path)
        .await
        .map_err(|e| PluginError::Manifest(format!("read {}: {e}", path.display())))?;
    parse_manifest(&content)
}

fn parse_manifest(content: &str) -> Result<PluginManifest, PluginError> {
    // Prefer `toml` crate if content looks like TOML; fallback to JSON for tests.
    // We try TOML first, then JSON, so tests can use either.
    if let Ok(m) = toml::from_str::<PluginManifest>(content) {
        return Ok(m);
    }
    if let Ok(m) = serde_json::from_str::<PluginManifest>(content) {
        return Ok(m);
    }
    Err(PluginError::Manifest(
        "invalid manifest: expected TOML [[plugin]] entries".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Builtin dummy plugin for tests and examples
// ---------------------------------------------------------------------------

/// Plugin exposing two tools — used to test isolation and core-untouched invariant.
pub struct TwoToolPlugin;

impl Plugin for TwoToolPlugin {
    fn name(&self) -> &str {
        "two_tool"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new(
                "ToolA",
                "First tool from two-tool plugin",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "input": { "type": "string", "description": "input value" }
                    },
                    "required": ["input"]
                }),
            ),
            ToolDefinition::new(
                "ToolB",
                "Second tool from two-tool plugin",
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string", "description": "value" }
                    },
                    "required": ["value"]
                }),
            ),
        ]
    }

    fn execute(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolResult, PluginError> {
        match tool_name {
            "ToolA" => {
                let v = args.get("input").and_then(|x| x.as_str()).unwrap_or("");
                Ok(ToolResult {
                    success: true,
                    output: format!("ToolA:{v}"),
                    retry: false,
                })
            }
            "ToolB" => {
                let v = args.get("value").and_then(|x| x.as_str()).unwrap_or("");
                Ok(ToolResult {
                    success: true,
                    output: format!("ToolB:{v}"),
                    retry: false,
                })
            }
            other => Err(PluginError::NotFound(other.to_string())),
        }
    }
}

/// Dummy plugin used in tests — provides a single `HelloTool`.
pub struct HelloPlugin;

impl Plugin for HelloPlugin {
    fn name(&self) -> &str {
        "hello"
    }

    fn tools(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition::new(
            "HelloTool",
            "Dummy hello tool from plugin system",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "name to greet" }
                },
                "required": ["name"]
            }),
        )]
    }

    fn execute(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolResult, PluginError> {
        if tool_name != "HelloTool" {
            return Err(PluginError::NotFound(tool_name.to_string()));
        }
        let who = args.get("name").and_then(|v| v.as_str()).unwrap_or("world");
        Ok(ToolResult {
            success: true,
            output: format!("hello {who} from plugin"),
            retry: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn hello_registry() -> PluginRegistry {
        let mut r = PluginRegistry::new();
        r.register(Box::new(HelloPlugin)).unwrap();
        r
    }

    #[test]
    fn plugin_registry_registers_dummy_and_lists_tools() {
        let r = hello_registry();
        assert_eq!(r.len(), 1);
        assert!(r.contains_tool("HelloTool"));
        let names = r.tool_names();
        assert!(names.contains(&"HelloTool".to_string()));
        let tools = r.tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "HelloTool");
    }

    #[test]
    fn plugin_execute_hello_tool_success() {
        let r = hello_registry();
        let args = serde_json::json!({"name": "kitty"});
        let res = r.execute("HelloTool", &args).unwrap().unwrap();
        assert!(res.success);
        assert!(res.output.contains("hello kitty"));
    }

    #[test]
    fn plugin_duplicate_tool_name_blocked() {
        let mut r = hello_registry();
        // Second HelloPlugin collides on HelloTool
        let err = r.register(Box::new(HelloPlugin)).unwrap_err();
        assert!(matches!(err, PluginError::DuplicateTool(_)));
        assert!(err.to_string().contains("tool already exists"));
    }

    #[test]
    fn plugin_execute_unknown_returns_none() {
        let r = hello_registry();
        assert!(r.execute("UnknownTool", &serde_json::json!({})).is_none());
    }

    #[test]
    fn plugin_load_from_manifest_parses_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plugins.toml");
        std::fs::write(
            &path,
            r#"
[[plugin]]
name = "hello"
enabled = true
"#,
        )
        .unwrap();
        let r = PluginRegistry::load_from_manifest(&path).unwrap();
        assert_eq!(r.len(), 1);
        assert!(r.contains_tool("HelloTool"));
    }

    #[test]
    fn plugin_load_manifest_disabled_skips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plugins.toml");
        std::fs::write(
            &path,
            r#"
[[plugin]]
name = "hello"
enabled = false
"#,
        )
        .unwrap();
        let r = PluginRegistry::load_from_manifest(&path).unwrap();
        assert!(r.is_empty());
    }

    #[test]
    fn plugin_load_missing_manifest_error() {
        let path = Path::new("/tmp/kitty-nonexistent-plugins-xyz.toml");
        let err = PluginRegistry::load_from_manifest(path).unwrap_err();
        assert!(matches!(err, PluginError::Manifest(_)));
    }

    #[test]
    fn plugin_register_plugin_alias_works() {
        let mut r = PluginRegistry::new();
        r.register_plugin(Box::new(HelloPlugin)).unwrap();
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn plugin_manifest_unknown_name_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plugins.toml");
        std::fs::write(
            &path,
            r#"
[[plugin]]
name = "unknown_plugin_xyz"
enabled = true
"#,
        )
        .unwrap();
        let r = PluginRegistry::load_from_manifest(&path).unwrap();
        assert!(r.is_empty());
    }

    #[allow(non_snake_case)]
    #[test]
    fn PluginRegistry_trait_object_safety() {
        fn assert_plugin<T: Plugin>() {}
        assert_plugin::<HelloPlugin>();
        let _: Box<dyn Plugin> = Box::new(HelloPlugin);
    }

    #[test]
    fn plugin_two_tool_plugin_registers_two_tools_core_intact() {
        let mut r = PluginRegistry::new();
        r.register(Box::new(TwoToolPlugin)).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.tools().len(), 2);
        assert!(r.contains_tool("ToolA"));
        assert!(r.contains_tool("ToolB"));
        let args_a = serde_json::json!({"input": "hello"});
        let res_a = r.execute("ToolA", &args_a).unwrap().unwrap();
        assert!(res_a.success);
        assert!(res_a.output.contains("ToolA:hello"));
        let args_b = serde_json::json!({"value": "world"});
        let res_b = r.execute("ToolB", &args_b).unwrap().unwrap();
        assert!(res_b.success);
        assert!(res_b.output.contains("ToolB:world"));
        let core = crate::tools::standard_tool_definitions();
        assert_eq!(core.len(), 6, "core must remain 6 tools, no modification");
        assert!(!core.iter().any(|d| d.function.name == "ToolA"));
        assert!(!core.iter().any(|d| d.function.name == "ToolB"));
    }

    #[test]
    fn plugin_isolation_between_plugins_and_error_isolated() {
        let mut r = PluginRegistry::new();
        r.register(Box::new(HelloPlugin)).unwrap();
        r.register(Box::new(TwoToolPlugin)).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r.tools().len(), 3);
        // Duplicate across plugins blocked
        struct DupToolA;
        impl Plugin for DupToolA {
            fn name(&self) -> &str {
                "dup"
            }
            fn tools(&self) -> Vec<ToolDefinition> {
                vec![ToolDefinition::new(
                    "ToolA",
                    "dup",
                    serde_json::json!({"type":"object","properties":{}}),
                )]
            }
            fn execute(&self, _: &str, _: &serde_json::Value) -> Result<ToolResult, PluginError> {
                Ok(ToolResult {
                    success: true,
                    output: String::new(),
                    retry: false,
                })
            }
        }
        let err = r.register(Box::new(DupToolA)).unwrap_err();
        assert!(matches!(err, PluginError::DuplicateTool(_)));
        // Original tools still executable — isolation
        let res = r
            .execute("HelloTool", &serde_json::json!({"name":"x"}))
            .unwrap()
            .unwrap();
        assert!(res.success);
        let res2 = r
            .execute("ToolB", &serde_json::json!({"value":"y"}))
            .unwrap()
            .unwrap();
        assert!(res2.success);
    }

    #[test]
    fn plugin_core_not_modified_after_registry_ops() {
        let core_before = crate::tools::standard_tool_definitions();
        let mut r = PluginRegistry::new();
        r.register(Box::new(HelloPlugin)).unwrap();
        r.register(Box::new(TwoToolPlugin)).unwrap();
        let core_after = crate::tools::standard_tool_definitions();
        assert_eq!(core_before.len(), core_after.len());
        assert_eq!(core_after.len(), 6);
        for n in ["Read", "Edit", "Write", "Bash", "Glob", "Grep"] {
            assert!(
                core_after.iter().any(|d| d.function.name == n),
                "missing core {n}"
            );
        }
        // Plugin tools not in core
        for n in ["HelloTool", "ToolA", "ToolB"] {
            assert!(!core_after.iter().any(|d| d.function.name == n));
        }
    }

    /// W1 red-green: async manifest loader parses the same registry as sync.
    #[tokio::test]
    async fn plugin_load_manifest_async_matches_sync() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plugins.toml");
        std::fs::write(&path, "[[plugin]]\nname = \"hello\"\nenabled = true\n").unwrap();
        let sync_r = PluginRegistry::load_from_manifest(&path).unwrap();
        let manifest = load_manifest_async(&path).await.unwrap();
        assert_eq!(manifest.plugins().len(), 1);
        assert_eq!(sync_r.len(), 1);
        assert!(sync_r.contains_tool("HelloTool"));
    }
}
