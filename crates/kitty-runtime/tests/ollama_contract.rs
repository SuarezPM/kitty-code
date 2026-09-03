//! HTTP contract tests for `OllamaClient` against a wiremock server
//! (Fase 6.3): payload shape (`truncate:false` + session `num_ctx`), the
//! 400 context-overflow mapping to the typed overflow error, and the
//! transport-timeout mapping.

use kitty_runtime::client::{OllamaClient, ENV_NUM_CTX};
use kitty_runtime::Message;
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn test_messages() -> Vec<Message> {
    vec![Message::user("ping")]
}

#[tokio::test]
async fn chat_sends_truncate_false_and_session_num_ctx() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .and(body_partial_json(serde_json::json!({
            "model": "test-model",
            "truncate": false,
            "options": { "num_ctx": 8192 }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "model": "test-model",
            "message": { "role": "assistant", "content": "pong" },
            "done": true
        })))
        .mount(&server)
        .await;
    let client = OllamaClient::new(&server.uri()).with_num_ctx(8192);
    let resp = client
        .chat("test-model", test_messages(), None, None)
        .await
        .expect("chat succeeds");
    assert_eq!(resp.message.content.as_deref(), Some("pong"));
}

#[tokio::test]
async fn chat_400_overflow_maps_to_typed_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_string("input length exceeds context window (8192 tokens, given 9000)"),
        )
        .mount(&server)
        .await;
    let client = OllamaClient::new(&server.uri()).with_num_ctx(8192);
    let err = client
        .chat("test-model", test_messages(), None, None)
        .await
        .expect_err("overflow must error");
    assert!(
        err.is_overflow(),
        "400 overflow body must map to the typed overflow error, got: {err}"
    );
}

#[tokio::test]
async fn chat_timeout_maps_to_transport_error() {
    use std::time::Duration;
    let server = MockServer::start().await;
    // Delay far beyond a 1-second client timeout: the client-side deadline
    // fires and must surface as a transient/transport error.
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_delay(Duration::from_secs(30)))
        .mount(&server)
        .await;
    // SAFETY of env mutation: this binary's other tests never read this var.
    std::env::set_var("KITTY_OLLAMA_TIMEOUT_SECS", "1");
    let client = OllamaClient::new(&server.uri()).with_num_ctx(4096);
    let err = client
        .chat("test-model", test_messages(), None, None)
        .await
        .expect_err("unresponsive server must error");
    assert!(
        err.is_transient(),
        "transport timeout must map to the transient/transport error, got: {err}"
    );
}

#[test]
fn env_num_ctx_documented_in_runtime_const() {
    assert_eq!(ENV_NUM_CTX, "KITTY_NUM_CTX");
}
