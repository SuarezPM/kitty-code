//! SSE framing for OpenAI-compatible chat streams.
//!
//! Wire format: each event is `data: <json>\n\n`, the stream ends with
//! `data: [DONE]\n\n`. `tool_calls` travel as deltas inside `delta`
//! (same shape as non-stream `message.tool_calls`); the final frame carries
//! `usage`. No simulated progress events.

use crate::response::{ChatCompletionChunkResponse, Usage};

/// Wrap a JSON payload as one SSE event.
pub fn data_event(json: &str) -> String {
    format!("data: {json}\n\n")
}

/// Terminal event every stream must end with.
pub fn done_frame() -> &'static str {
    "data: [DONE]\n\n"
}

/// Serialize one chat chunk (content or tool_calls delta) as an SSE event.
pub fn chunk_to_frame(chunk: &ChatCompletionChunkResponse) -> String {
    let json = serde_json::to_string(chunk).unwrap_or_else(|_| "{\"object\":\"chat.completion.chunk\"}".to_string());
    data_event(&json)
}

/// Final event carrying usage. Emitted before `[DONE]`.
/// `model`, `id`, `created` mirror the in-flight chunk so clients can join.
pub fn final_frame(id: &str, model: &str, created: u128, usage: &Usage) -> String {
    let payload = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": usage,
    });
    data_event(&payload.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_shape() {
        let f = data_event("{}");
        assert_eq!(f, "data: {}\n\n");
        assert_eq!(done_frame(), "data: [DONE]\n\n");
    }

    #[test]
    fn final_carries_usage() {
        let u = Usage {
            completion_tokens: 5,
            prompt_tokens: 7,
            total_tokens: 12,
            prompt_tokens_details: None,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        };
        let f = final_frame("x", "m", 1, &u);
        assert!(f.starts_with("data: "));
        assert!(f.contains("\"usage\""));
        assert!(f.contains("12"));
    }
}
