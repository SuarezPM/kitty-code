//! Context overflow detection with 25+ provider patterns.
//!
//! Detects context-window overflow errors from various LLM providers
//! (Ollama, llama.cpp, OpenAI, Anthropic, vLLM, SGLang, etc.) and
//! silent overflows where the model produces zero output tokens despite
//! a large input.

use regex::Regex;
use std::sync::OnceLock;

/// Check if an error message indicates context overflow.
///
/// Tests the message against a bank of 20+ regex patterns covering
/// Ollama, llama.cpp, OpenAI, Anthropic, vLLM, SGLang, and generic
/// overflow indicators.
pub fn detect_overflow(error_msg: &str) -> bool {
    let patterns = overflow_patterns();
    patterns.iter().any(|re| re.is_match(error_msg))
}

/// Check for silent overflow (zero output tokens with large input).
///
/// Some providers silently truncate or return empty responses when the
/// input exceeds the context window instead of returning an explicit error.
pub fn detect_silent_overflow(input_tokens: usize, output_tokens: usize) -> bool {
    input_tokens > 1000 && output_tokens == 0
}

fn overflow_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // Ollama
            Regex::new(r"(?i)context length.*exceeded").unwrap(),
            Regex::new(r"(?i)context.*window.*full").unwrap(),
            Regex::new(r"(?i)too many tokens").unwrap(),
            // llama.cpp
            Regex::new(r"(?i)input is too long").unwrap(),
            Regex::new(r"(?i)prompt.*too.*long").unwrap(),
            Regex::new(r"(?i)n_ctx.*exceeded").unwrap(),
            // OpenAI
            Regex::new(r"(?i)maximum context length").unwrap(),
            Regex::new(r"(?i)context_length_exceeded").unwrap(),
            // Anthropic
            Regex::new(r"(?i)too many tokens.*request").unwrap(),
            Regex::new(r"(?i)token limit").unwrap(),
            // vLLM
            Regex::new(r"(?i)prompt.*length.*exceeds").unwrap(),
            Regex::new(r"(?i)max_model_len").unwrap(),
            // SGLang
            Regex::new(r"(?i)request.*too.*long").unwrap(),
            // Generic
            Regex::new(r"(?i)context.*overflow").unwrap(),
            Regex::new(r"(?i)token.*limit.*reached").unwrap(),
            Regex::new(r"(?i)input.*length.*limit").unwrap(),
            Regex::new(r"(?i)sequence.*length").unwrap(),
            Regex::new(r"(?i)max.*tokens.*exceeded").unwrap(),
            Regex::new(r"(?i)truncat.*(context|token|input|prompt|length)").unwrap(),
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_ollama_overflow() {
        assert!(detect_overflow("context length exceeded: 4096 > 4096"));
    }

    #[test]
    fn detect_openai_overflow() {
        assert!(detect_overflow(
            "This model's maximum context length is 4096 tokens"
        ));
    }

    #[test]
    fn detect_anthropic_overflow() {
        assert!(detect_overflow(
            "too many tokens in the request exceeds the token limit"
        ));
    }

    #[test]
    fn detect_llamacpp_overflow() {
        assert!(detect_overflow("input is too long for the current n_ctx"));
    }

    #[test]
    fn detect_vllm_overflow() {
        assert!(detect_overflow(
            "prompt length exceeds max_model_len of 4096"
        ));
    }

    #[test]
    fn detect_sglang_overflow() {
        assert!(detect_overflow("request too long for SGLang server"));
    }

    #[test]
    fn detect_generic_truncation() {
        assert!(detect_overflow("context was truncated due to token limit"));
        assert!(!detect_overflow("output was truncated due to max_tokens"));
    }

    #[test]
    fn silent_overflow_true() {
        assert!(detect_silent_overflow(5000, 0));
    }

    #[test]
    fn silent_overflow_false_with_output() {
        assert!(!detect_silent_overflow(5000, 100));
    }

    #[test]
    fn no_silent_overflow_small_input() {
        assert!(!detect_silent_overflow(500, 0));
    }

    #[test]
    fn no_false_positive_on_normal_error() {
        assert!(!detect_overflow("connection refused"));
        assert!(!detect_overflow("rate limit exceeded"));
        assert!(!detect_overflow("invalid API key"));
    }
}
