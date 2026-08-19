use serde::{Deserialize, Serialize};

use crate::error::{Result, RuntimeError};

// ---------------------------------------------------------------------------
// Tool wire types (used by Ollama API and kitty-harness)
// ---------------------------------------------------------------------------

/// Tool definition sent to Ollama API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn required_string(args: &serde_json::Value, key: &str) -> Result<String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| RuntimeError::InvalidArgument(format!("missing required argument: {key}")))
}

pub fn optional_string(args: &serde_json::Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(String::from)
}

pub fn optional_u64(args: &serde_json::Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64())
}

pub fn json_params(props: &[(&str, &str, &str)]) -> serde_json::Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, ty, desc) in props {
        properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": ty,
                "description": desc,
            }),
        );
        required.push(name.to_string());
    }
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_serializes() {
        let def = ToolDefinition::new("Read", "Read a file", json_params(&[("path", "string", "File path")]));
        let json = serde_json::to_string(&def).unwrap();
        assert!(json.contains("\"name\":\"Read\""));
        assert!(json.contains("\"type\":\"function\""));
    }

    #[test]
    fn required_string_missing() {
        let args = serde_json::json!({"other": "value"});
        let result = required_string(&args, "path");
        assert!(result.is_err());
    }

    #[test]
    fn required_string_present() {
        let args = serde_json::json!({"path": "/tmp/test.rs"});
        let result = required_string(&args, "path");
        assert_eq!(result.unwrap(), "/tmp/test.rs");
    }
}
