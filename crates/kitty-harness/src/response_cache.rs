//! Idempotent response cache for identical retries (W5-code, SDD).
//!
//! Keys a `ChatResponse` by BLAKE3 over `(model, messages, tools, options)`
//! debug shapes, so a byte-identical retry (overflow-compaction retry,
//! thinking-budget retry, transport retry) can be served without burning
//! another model call. Same-process only, bounded size (FIFO eviction),
//! disabled by default (`Agent::with_response_cache` opts in).
//!
//! Code-only in this lane: hit-rate effects are measured in the serial
//! window, never asserted here beyond bookkeeping correctness.

use std::collections::{HashMap, VecDeque};

use kitty_runtime::{ChatResponse, Message, ToolDefinition};

/// Cache key: BLAKE3 over the request's debug shapes (all deterministic
/// field-ordered types; `serde_json::Value` maps are `BTreeMap`-ordered).
#[must_use]
pub fn cache_key(
    model: &str,
    messages: &[Message],
    tools: &Option<Vec<ToolDefinition>>,
    options: &Option<kitty_runtime::client::RequestOptions>,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\n");
    hasher.update(format!("{messages:?}").as_bytes());
    hasher.update(b"\n");
    hasher.update(format!("{tools:?}").as_bytes());
    hasher.update(b"\n");
    hasher.update(format!("{options:?}").as_bytes());
    hex::encode(hasher.finalize().as_bytes())
}

/// Bounded FIFO cache of whole responses by request key.
#[derive(Debug, Default)]
pub struct ResponseCache {
    cap: usize,
    map: HashMap<String, ChatResponse>,
    order: VecDeque<String>,
    hits: u64,
    misses: u64,
}

impl ResponseCache {
    /// Create with a max entry count (0 = degenerate: never stores).
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            map: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a key (counts hit/miss).
    pub fn get(&mut self, key: &str) -> Option<ChatResponse> {
        match self.map.get(key) {
            Some(resp) => {
                self.hits += 1;
                Some(resp.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Store a response (FIFO-evicts past `cap`; `cap == 0` stores nothing).
    pub fn insert(&mut self, key: String, resp: ChatResponse) {
        if self.cap == 0 {
            return;
        }
        if !self.map.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.map.insert(key, resp);
        while self.map.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
    }

    /// `(hits, misses)` bookkeeping.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Entries currently held.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether no entries are held.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kitty_runtime::Message;

    fn test_response(text: &str) -> ChatResponse {
        ChatResponse {
            model: "test".into(),
            message: Message::assistant(text),
            done: true,
            eval_count: None,
            eval_duration: None,
            prompt_eval_count: None,
        }
    }

    #[test]
    fn key_is_stable_and_request_sensitive() {
        let msgs = vec![Message::user("hi")];
        let k1 = cache_key("m", &msgs, &None, &None);
        let k2 = cache_key("m", &msgs, &None, &None);
        assert_eq!(k1, k2);
        assert_ne!(k1, cache_key("other", &msgs, &None, &None));
        assert_ne!(k1, cache_key("m", &[Message::user("bye")], &None, &None));
    }

    #[test]
    fn insert_get_hit_miss_bookkeeping() {
        let mut c = ResponseCache::new(8);
        let key = cache_key("m", &[Message::user("hi")], &None, &None);
        assert!(c.get(&key).is_none());
        c.insert(key.clone(), test_response("yo"));
        let hit = c.get(&key).expect("stored response must hit");
        assert_eq!(hit.message.content.as_deref(), Some("yo"));
        assert_eq!(c.stats(), (1, 1));
    }

    #[test]
    fn fifo_evicts_past_cap() {
        let mut c = ResponseCache::new(2);
        for i in 0..3 {
            let k = cache_key("m", &[Message::user(format!("q{i}"))], &None, &None);
            c.insert(k, test_response("r"));
        }
        assert_eq!(c.len(), 2);
        let first = cache_key("m", &[Message::user("q0")], &None, &None);
        assert!(c.get(&first).is_none(), "oldest entry must be evicted");
    }

    #[test]
    fn zero_cap_stores_nothing() {
        let mut c = ResponseCache::new(0);
        let key = cache_key("m", &[Message::user("hi")], &None, &None);
        c.insert(key.clone(), test_response("r"));
        assert!(c.is_empty());
    }
}
