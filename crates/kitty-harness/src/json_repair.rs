//! JSON repair pipeline for malformed tool call arguments from small models.
//!
//! Fixes: raw control chars, invalid escapes, type coercion (string→int/bool),
//! union types, nested objects, optional null normalization.
//!
//! Complements the structural repair in [`crate::parser::repair_json`] (smart
//! quotes, trailing commas, unquoted keys) with low-level byte-level fixes
//! and schema-aware type coercion.

use serde_json::Value;

/// Repair malformed JSON string — fix control chars and invalid escapes.
///
/// Handles raw control characters (escaping as `\uXXXX`) and invalid escape
/// sequences (doubling the backslash). This is a LOWER-LEVEL repair than
/// [`crate::parser::repair_json`] which handles structural issues like smart
/// quotes and trailing commas.
pub fn repair_json(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(&next) = chars.peek() {
                    match next {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' => {
                            result.push('\\');
                            result.push(next);
                            chars.next();
                        }
                        _ => {
                            // Invalid escape — double the backslash
                            result.push('\\');
                            result.push('\\');
                        }
                    }
                } else {
                    result.push('\\');
                }
            }
            '\x00'..='\x1f' => {
                // Raw control char — escape it
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => result.push(c),
        }
    }

    result
}

/// Coerce JSON values to match a target schema type.
///
/// Converts "1" → 1 for integers, "true" → true for booleans,
/// null → "" for strings, recursively handles nested objects/arrays.
pub fn coerce_with_schema(value: &mut Value, schema: &Value) {
    // Handle anyOf/oneOf/allOf at schema-object level before matching value type
    if let Value::Object(schema_obj) = schema {
        if let Some(Value::Array(variants)) = schema_obj
            .get("anyOf")
            .or_else(|| schema_obj.get("oneOf"))
            .or_else(|| schema_obj.get("allOf"))
        {
            for variant in variants {
                if try_coerce(value, variant) {
                    remove_optional_nulls(value, schema);
                    return;
                }
            }
        }

        // Direct leaf coercion: schema has "type" and value is a leaf
        if schema_obj.get("properties").is_none() {
            try_coerce(value, schema);
        }
    }

    match schema {
        Value::Object(schema_obj) => {
            if let Value::Object(obj) = value {
                // Recurse into properties
                if let Some(Value::Object(props_obj)) = schema_obj.get("properties") {
                    for (key, prop_schema) in props_obj {
                        if let Some(val) = obj.get_mut(key.as_str()) {
                            coerce_with_schema(val, prop_schema);
                        }
                    }
                }
            }
        }
        Value::Array(items) => {
            if let Value::Array(arr) = value {
                for item in arr {
                    if let Some(item_schema) = items.first() {
                        coerce_with_schema(item, item_schema);
                    }
                }
            }
        }
        _ => {}
    }

    // Remove null for optional properties
    remove_optional_nulls(value, schema);
}

fn try_coerce(value: &mut Value, schema: &Value) -> bool {
    if let Value::Object(obj) = schema {
        match obj.get("type").and_then(|v| v.as_str()) {
            Some("string") => {
                if let Value::Number(_) = value {
                    let s = value.to_string();
                    *value = Value::String(s);
                    return true;
                }
                if let Value::Bool(b) = value {
                    *value = Value::String(b.to_string());
                    return true;
                }
            }
            Some("integer") | Some("number") => {
                if let Value::String(s) = value {
                    if let Ok(n) = s.parse::<i64>() {
                        *value = Value::Number(n.into());
                        return true;
                    }
                }
            }
            Some("boolean") => {
                if let Value::String(s) = value {
                    match s.as_str() {
                        "true" => {
                            *value = Value::Bool(true);
                            return true;
                        }
                        "false" => {
                            *value = Value::Bool(false);
                            return true;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn remove_optional_nulls(value: &mut Value, schema: &Value) {
    if let (Value::Object(obj), Value::Object(schema_obj)) = (value, schema) {
        if let Some(Value::Object(props_obj)) = schema_obj.get("properties") {
            let keys_to_remove: Vec<String> = obj
                .iter()
                .filter(|(_, v)| v.is_null())
                .filter(|(k, _)| {
                    props_obj
                        .get(k.as_str())
                        .and_then(|p| p.get("type"))
                        .map(|t| t != "null")
                        .unwrap_or(false)
                })
                .map(|(k, _)| k.clone())
                .collect();

            for key in keys_to_remove {
                obj.remove(&key);
            }
        }
    }
}

/// Full repair pipeline: repair → parse → coerce → return.
///
/// Returns the repaired Value or the original parse error.
/// First tries parsing as-is, then applies control-char/escape repair,
/// and finally runs schema-aware coercion if a schema is provided.
pub fn repair_and_coerce(input: &str, schema: Option<&Value>) -> Result<Value, String> {
    // First try parsing as-is
    if let Ok(v) = serde_json::from_str::<Value>(input) {
        let mut result = v;
        if let Some(s) = schema {
            coerce_with_schema(&mut result, s);
        }
        return Ok(result);
    }

    // Try repairing
    let repaired = repair_json(input);
    match serde_json::from_str::<Value>(&repaired) {
        Ok(mut v) => {
            if let Some(s) = schema {
                coerce_with_schema(&mut v, s);
            }
            Ok(v)
        }
        Err(e) => Err(format!("JSON repair failed: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repair_invalid_escape() {
        let input = r#"{"key": "value\n"}"#; // This is actually valid
        let result = repair_json(input);
        assert!(serde_json::from_str::<Value>(&result).is_ok());
    }

    #[test]
    fn repair_bare_control_char() {
        // Raw tab character inside a string value
        let input = "{\"key\": \"value\tbroken\"}";
        let result = repair_json(input);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        // JSON interprets \u0009 back to tab — key is the repaired-parse that matters
        assert_eq!(parsed["key"], "value\tbroken");
    }

    #[test]
    fn repair_invalid_escape_sequence() {
        // \q is not a valid JSON escape
        let input = r#"{"key": "value\qtest"}"#;
        let result = repair_json(input);
        let parsed: Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["key"], r#"value\qtest"#);
    }

    #[test]
    fn coerce_string_to_int() {
        let mut val = serde_json::json!({"count": "5"});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}}
        });
        coerce_with_schema(&mut val, &schema);
        assert_eq!(val["count"], 5);
    }

    #[test]
    fn coerce_string_to_bool() {
        let mut val = serde_json::json!({"flag": "true"});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"flag": {"type": "boolean"}}
        });
        coerce_with_schema(&mut val, &schema);
        assert_eq!(val["flag"], true);
    }

    #[test]
    fn remove_optional_null() {
        let mut val = serde_json::json!({"name": null, "age": 25});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });
        coerce_with_schema(&mut val, &schema);
        assert!(val.get("name").is_none());
        assert_eq!(val["age"], 25);
    }

    #[test]
    fn coerce_nested_objects() {
        let mut val = serde_json::json!({"nested": {"count": "10"}});
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}}
                }
            }
        });
        coerce_with_schema(&mut val, &schema);
        assert_eq!(val["nested"]["count"], 10);
    }

    #[test]
    fn repair_and_coerce_full() {
        let input = r#"{"count": "5", "flag": "true"}"#;
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "flag": {"type": "boolean"}
            }
        });
        let result = repair_and_coerce(input, Some(&schema)).unwrap();
        assert_eq!(result["count"], 5);
        assert_eq!(result["flag"], true);
    }

    #[test]
    fn repair_and_coerce_without_schema() {
        let input = r#"{"key": "value"}"#;
        let result = repair_and_coerce(input, None).unwrap();
        assert_eq!(result["key"], "value");
    }
}
