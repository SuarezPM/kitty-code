//! K2-Horizon tool call parser (`<ifm|...>` blocks).
//!
//! Formats (from the K2 chat template; `tool_call_format` selects the
//! emission, default `xml`):
//! - json: `<ifm|tool_calls>\n<ifm|tool_call>{"name":"...","arguments":{...}}</ifm|tool_call>\n...</ifm|tool_calls>`
//! - xml: `<ifm|tool_call>NAME\n<ifm|arg_key>KEY</ifm|arg_key>\n<ifm|arg_value>VALUE</ifm|arg_value>\n...</ifm|tool_call>`
//! - xml_typed: xml plus `<ifm|arg_type>TYPE</ifm|arg_type>` after each key
//!   (types are informational; values stay verbatim strings, schema types
//!   apply later like the Qwen parser).

use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static K2_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();
static K2_ARG_KEY_REGEX: OnceLock<Regex> = OnceLock::new();
static K2_ARG_VALUE_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct K2HorizonParser;

fn block_re() -> &'static Regex {
    K2_BLOCK_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<ifm\|tool_call>(?P<inner>.*?)</ifm\|tool_call>").unwrap()
    })
}

impl ToolFormatParser for K2HorizonParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        // Also matches the `<ifm|tool_calls>` block opener (prefix), which is
        // what streaming sees first.
        text.contains("<ifm|tool_call")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::K2Horizon
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            k2_tool_call_lark(tools, false),
            tools,
            "arguments",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            k2_tool_call_lark(tools, true),
            tools,
            "arguments",
            false,
        )
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        if !block_re().is_match(message) {
            return Ok(None);
        }
        parse_k2_tool_calls(message)
    }
}

fn k2_tool_call_lark(tools: &[Tool], include_wrapper: bool) -> String {
    let open = r#""<ifm|tool_call>""#;
    let close = r#""</ifm|tool_call>""#;
    let start = if include_wrapper {
        format!(r#"start: {open} (json_call | xml_call) ("\n"? {open} (json_call | xml_call))*"#)
    } else {
        format!(r#"start: (json_call | xml_call) ("\n"? {open} (json_call | xml_call))*"#)
    };
    let json_call = if include_wrapper {
        r#"json_call: @json_body "</ifm|tool_call>""#
    } else {
        "json_call: @json_body </ifm|tool_call>"
    };
    let _ = tools;
    format!(
        "{start}\n{json_call}\nxml_call: \"\\n\"? k2_xml_function (\"\\n\"? k2_xml_function)* \"\\n\"? {close}\n{}",
        k2_xml_rules(tools),
    )
}

fn lark_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

fn k2_xml_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();
    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("k2_xml_tool_{tool_idx}");
        let opener = lark_string(&format!("{}\n", tool.function.name));
        rules.push(format!("{branch}: {opener} k2_xml_params_{tool_idx}"));
        rules.push(format!("k2_xml_params_{tool_idx}: (k2_xml_param \"\\n\"?)*"));
        branches.push(branch);
    }
    if branches.is_empty() {
        rules.push("k2_xml_function: /.+/".to_string());
    } else {
        rules.push(format!("k2_xml_function: {}", branches.join(" | ")));
    }
    rules.push(
        "k2_xml_param: \"<ifm|arg_key>\" /[^<]+/ \"</ifm|arg_key>\" \"\\n\"? (\"<ifm|arg_type>\" /[^<]+/ \"</ifm|arg_type>\" \"\\n\"?)? \"<ifm|arg_value>\" k2_xml_value \"</ifm|arg_value>\"".to_string(),
    );
    rules.push("k2_xml_value: (/[^<]+/ | \"<\" /[^i]/ | \"<i\" /[^f]/ | \"<if\" /[^m]/)*".to_string());
    rules.join("\n")
}

#[derive(serde::Serialize)]
struct K2ToolCall {
    name: String,
    arguments: Value,
}

fn parse_k2_tool_calls(message: &str) -> Result<Option<String>> {
    let mut calls = Vec::new();
    for caps in block_re().captures_iter(message) {
        let inner = caps.name("inner").unwrap().as_str().trim();
        if inner.is_empty() {
            continue;
        }
        if let Some(xml) = parse_k2_xml_call(inner) {
            calls.push(xml);
            continue;
        }
        match serde_json::from_str::<Value>(inner) {
            Ok(value) => calls.push(value),
            Err(_) => return Ok(None),
        }
    }
    match calls.len() {
        0 => Ok(None),
        1 => Ok(Some(
            serde_json::to_string(&calls[0]).map_err(candle_core::Error::msg)?,
        )),
        _ => Ok(Some(
            serde_json::to_string(&calls).map_err(candle_core::Error::msg)?,
        )),
    }
}

/// XML-ish form: function name on the first line, then arg triples.
/// Returns `None` when the block is not XML-shaped (caller tries JSON).
fn parse_k2_xml_call(inner: &str) -> Option<Value> {
    if !inner.contains("<ifm|arg_key>") {
        return None;
    }
    let key_re = K2_ARG_KEY_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<ifm\|arg_key>(?P<key>.*?)</ifm\|arg_key>").unwrap()
    });
    let value_re = K2_ARG_VALUE_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<ifm\|arg_value>(?P<value>.*?)</ifm\|arg_value>").unwrap()
    });
    let name_end = inner.find("<ifm|arg_key>")?;
    let name = inner[..name_end].trim().to_string();
    if name.is_empty() || name.contains('<') {
        return None;
    }
    let keys: Vec<String> = key_re
        .captures_iter(inner)
        .map(|c| c.name("key").unwrap().as_str().trim().to_string())
        .collect();
    let values: Vec<String> = value_re
        .captures_iter(inner)
        .map(|c| {
            let v = c.name("value").unwrap().as_str();
            let v = v.strip_prefix('\n').unwrap_or(v);
            v.strip_suffix('\n').unwrap_or(v).to_string()
        })
        .collect();
    if keys.is_empty() || keys.len() != values.len() {
        return None;
    }
    let mut arguments = Map::new();
    for (key, value) in keys.into_iter().zip(values.into_iter()) {
        arguments.insert(key, Value::String(value));
    }
    serde_json::to_value(K2ToolCall {
        name,
        arguments: Value::Object(arguments),
    })
    .ok()
}

#[cfg(test)]
mod tests {
    use super::K2HorizonParser;
    use crate::tools::parsers::ToolFormatParser;
    use serde_json::Value;

    #[test]
    fn parses_k2_json_tool_call() {
        let parsed = K2HorizonParser
            .parse(
                "<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"Read\", \"arguments\": {\"path\": \"a.rs\"}}</ifm|tool_call>\n</ifm|tool_calls>",
            )
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "Read");
        assert_eq!(value["arguments"]["path"], "a.rs");
    }

    #[test]
    fn parses_k2_xml_tool_call() {
        let parsed = K2HorizonParser
            .parse(
                "<ifm|tool_calls>\n<ifm|tool_call>Read\n<ifm|arg_key>path</ifm|arg_key>\n<ifm|arg_value>a.rs</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>",
            )
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "Read");
        assert_eq!(value["arguments"]["path"], "a.rs");
    }

    #[test]
    fn parses_k2_xml_typed_tool_call() {
        let parsed = K2HorizonParser
            .parse(
                "<ifm|tool_call>Bash\n<ifm|arg_key>command</ifm|arg_key>\n<ifm|arg_type>string</ifm|arg_type>\n<ifm|arg_value>echo hi</ifm|arg_value>\n</ifm|tool_call>",
            )
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "Bash");
        // Types are informational; values stay verbatim strings.
        assert_eq!(value["arguments"]["command"], "echo hi");
    }

    #[test]
    fn detects_k2_prefix_and_rejects_plain_text() {
        assert!(K2HorizonParser.could_be_tool_call("<ifm|tool_calls>\n"));
        assert!(K2HorizonParser.could_be_tool_call("<ifm|tool_call>Read"));
        assert!(!K2HorizonParser.could_be_tool_call("just some prose"));
        assert!(K2HorizonParser
            .parse("just some prose")
            .unwrap()
            .is_none());
    }

    #[test]
    fn multiple_k2_calls_come_back_as_array() {
        let parsed = K2HorizonParser
            .parse(
                "<ifm|tool_call>{\"name\": \"A\", \"arguments\": {}}</ifm|tool_call><ifm|tool_call>{\"name\": \"B\", \"arguments\": {}}</ifm|tool_call>",
            )
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);
    }
}
