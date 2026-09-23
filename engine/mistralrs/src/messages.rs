use std::{
    collections::HashMap,
    fmt::Display,
    future::Future,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::*;
use either::Either;
use indexmap::IndexMap;
use serde_json::{json, Value};

/// A type which can be used as a chat request.
///
/// Implemented by [`TextMessages`], [`MultimodalMessages`], and [`RequestBuilder`].
pub trait RequestLike {
    /// Borrow the current list of chat messages.
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>];
    /// Take ownership of the messages, converting them into a [`RequestMessage`].
    fn take_messages(&mut self) -> RequestMessage;
    /// Take any custom logits processors, if configured.
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>>;
    /// Take the adapter selection, if configured.
    fn take_adapter(&mut self) -> Option<AdapterSelection>;
    /// Whether log-probabilities should be returned.
    fn return_logprobs(&self) -> bool;
    /// Take the generation constraint (regex, JSON schema, grammar, or none).
    fn take_constraint(&mut self) -> Constraint;
    /// Take the tools and tool-choice policy, if any.
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)>;
    /// Take the sampling parameters.
    fn take_sampling_params(&mut self) -> SamplingParams;
    /// Whether to silently truncate prompts that exceed the model's context length.
    fn truncate_sequence(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Plain text (chat) messages.
///
/// No constraints, logits processors, logprobs, tools, or adapters.
///
/// Sampling is deterministic.
pub struct TextMessages {
    messages: Vec<IndexMap<String, MessageContent>>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
}

impl From<TextMessages> for Vec<IndexMap<String, MessageContent>> {
    fn from(value: TextMessages) -> Self {
        value.messages
    }
}

#[derive(Debug, Clone, PartialEq)]
/// A chat message role.
pub enum TextMessageRole {
    /// The human user.
    User,
    /// The model / assistant.
    Assistant,
    /// System prompt providing instructions to the model.
    System,
    /// Output from a tool call.
    Tool,
    /// A custom role string.
    Custom(String),
}

impl Display for TextMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::System => write!(f, "system"),
            Self::Tool => write!(f, "tool"),
            Self::Custom(c) => write!(f, "{c}"),
        }
    }
}

impl Default for TextMessages {
    fn default() -> Self {
        Self::new()
    }
}

impl TextMessages {
    /// Create an empty text message list.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            enable_thinking: None,
            reasoning_effort: None,
        }
    }

    /// Append a message with the given role and text content.
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Remove all messages.
    pub fn clear(mut self) -> Self {
        self.messages.clear();
        self
    }

    /// Enable extended thinking (chain-of-thought) for models that support it.
    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Set the reasoning effort for models that support it.
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }
}

impl RequestLike for TextMessages {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }
    fn take_messages(&mut self) -> RequestMessage {
        let mut other = Vec::new();
        std::mem::swap(&mut other, &mut self.messages);
        RequestMessage::Chat {
            messages: other,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort,
            tool_call_format: None,
        }
    }
    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        None
    }
    fn take_adapter(&mut self) -> Option<AdapterSelection> {
        None
    }
    fn return_logprobs(&self) -> bool {
        false
    }
    fn take_constraint(&mut self) -> Constraint {
        Constraint::None
    }
    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)> {
        None
    }
    fn take_sampling_params(&mut self) -> SamplingParams {
        SamplingParams::deterministic()
    }
}


/// A way to add messages with finer control given.
///
/// This includes control over:
/// - Logits processors
/// - Constraints
/// - Logprobs
/// - Tools
/// - Sampling
/// - Enable thinking for models that support the configuration
pub struct RequestBuilder {
    messages: Vec<IndexMap<String, MessageContent>>,
    logits_processors: Vec<Arc<dyn CustomLogitsProcessor>>,
    adapter: Option<AdapterSelection>,
    return_logprobs: bool,
    constraint: Constraint,
    tools: Vec<Tool>,
    tool_choice: ToolChoice,
    sampling_params: SamplingParams,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    truncate_sequence: bool,
    tool_call_format: Option<String>,
}

impl Default for RequestBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl From<TextMessages> for RequestBuilder {
    fn from(value: TextMessages) -> Self {
        Self {
            messages: value.messages,
            logits_processors: Vec::new(),
            adapter: None,
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: SamplingParams::deterministic(),
            enable_thinking: value.enable_thinking,
            reasoning_effort: value.reasoning_effort,
            truncate_sequence: false,
            tool_call_format: None,
        }
    }
}


impl RequestBuilder {
    /// Create an empty request builder with deterministic sampling defaults.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            logits_processors: Vec::new(),
            adapter: None,
            return_logprobs: false,
            constraint: Constraint::None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            sampling_params: SamplingParams::deterministic(),
            enable_thinking: None,
            reasoning_effort: None,
            truncate_sequence: false,
            tool_call_format: None,
        }
    }











    /// Add a message to the request.
    ///
    /// For messages with tool calls, use [`Self::add_message_with_tool_call`].
    /// For messages with tool outputs, use [`Self::add_tool_message`].
    pub fn add_message(mut self, role: TextMessageRole, text: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ]));
        self
    }

    /// Add a message with the output of a tool call.
    pub fn add_tool_message(mut self, tool_content: impl ToString, tool_id: impl ToString) -> Self {
        self.messages.push(IndexMap::from([
            (
                "role".to_string(),
                Either::Left(TextMessageRole::Tool.to_string()),
            ),
            (
                "content".to_string(),
                Either::Left(tool_content.to_string()),
            ),
            (
                "tool_call_id".to_string(),
                Either::Left(tool_id.to_string()),
            ),
        ]));
        self
    }

    /// Append an assistant message that includes tool call results.
    pub fn add_message_with_tool_call(
        mut self,
        role: TextMessageRole,
        text: impl ToString,
        tool_calls: Vec<ToolCallResponse>,
    ) -> Self {
        let tool_messages = tool_calls
            .iter()
            .map(|t| {
                IndexMap::from([
                    ("id".to_string(), Value::String(t.id.clone())),
                    ("type".to_string(), Value::String(t.tp.to_string())),
                    (
                        "function".to_string(),
                        json!({
                            "name": t.function.name,
                            "arguments": t.function.arguments,
                        }),
                    ),
                ])
            })
            .collect();
        self.messages.push(IndexMap::from([
            ("role".to_string(), Either::Left(role.to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
            ("tool_calls".to_string(), Either::Right(tool_messages)),
        ]));
        self
    }

    /// The default tool choice is auto.
    pub fn set_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = tools;
        self
    }

    /// Control how the model selects tools (auto, required, none, or a specific tool).
    pub fn set_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = tool_choice;
        self
    }



    /// Request log-probabilities for each generated token.
    pub fn return_logprobs(mut self, return_logprobs: bool) -> Self {
        self.return_logprobs = return_logprobs;
        self
    }

    /// Apply a generation constraint (regex, JSON schema, or grammar).
    pub fn set_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set the sampling parameters as given.
    pub fn set_sampling(mut self, params: SamplingParams) -> Self {
        self.sampling_params = params;
        self
    }

    /// Set the sampling parameters for deterministic generation.
    /// This sets up the parameters so that there is:
    /// - No temperature, topk, topp, minp
    /// - No penalties, stop tokens, or logit bias
    /// - No maximum length
    pub fn set_deterministic_sampler(mut self) -> Self {
        self.sampling_params = SamplingParams::deterministic();
        self
    }

    /// Set the sampling temperature. Higher values increase randomness.
    pub fn set_sampler_temperature(mut self, temperature: f64) -> Self {
        self.sampling_params.temperature = Some(temperature);
        self
    }

    /// Limit sampling to the top-k most probable tokens.
    pub fn set_sampler_topk(mut self, topk: usize) -> Self {
        self.sampling_params.top_k = Some(topk);
        self
    }

    /// Nucleus sampling: only consider tokens whose cumulative probability exceeds this threshold.
    pub fn set_sampler_topp(mut self, topp: f64) -> Self {
        self.sampling_params.top_p = Some(topp);
        self
    }

    /// Min-p sampling: filter tokens below this fraction of the top token's probability.
    pub fn set_sampler_minp(mut self, minp: f64) -> Self {
        self.sampling_params.min_p = Some(minp);
        self
    }

    /// Return the top-n log-probabilities per token position.
    pub fn set_sampler_topn_logprobs(mut self, top_n_logprobs: usize) -> Self {
        self.sampling_params.top_n_logprobs = top_n_logprobs;
        self
    }

    /// Penalize tokens proportionally to how often they have appeared so far.
    pub fn set_sampler_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.sampling_params.frequency_penalty = Some(frequency_penalty);
        self
    }

    /// Penalize tokens that have appeared at all, regardless of frequency.
    pub fn set_sampler_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.sampling_params.presence_penalty = Some(presence_penalty);
        self
    }

    /// Set stop tokens that terminate generation when produced.
    pub fn set_sampler_stop_toks(mut self, stop_toks: StopTokens) -> Self {
        self.sampling_params.stop_toks = Some(stop_toks);
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn set_sampler_max_len(mut self, max_len: usize) -> Self {
        self.sampling_params.max_len = Some(max_len);
        self
    }

    /// Keep generating past the model's EOS tokens; explicit stops and the token limit still apply.
    pub fn set_sampler_ignore_eos(mut self, ignore_eos: bool) -> Self {
        self.sampling_params.ignore_eos = ignore_eos;
        self
    }

    /// Apply a bias to specific token IDs during sampling.
    pub fn set_sampler_logits_bias(mut self, logits_bias: HashMap<u32, f32>) -> Self {
        self.sampling_params.logits_bias = Some(logits_bias);
        self
    }

    /// Generate multiple independent completions for the same prompt.
    pub fn set_sampler_n_choices(mut self, n_choices: usize) -> Self {
        self.sampling_params.n_choices = n_choices;
        self
    }

    /// Configure DRY (Don't Repeat Yourself) sampling parameters.
    pub fn set_sampler_dry_params(mut self, dry_params: DrySamplingParams) -> Self {
        self.sampling_params.dry_params = Some(dry_params);
        self
    }

    /// Enable extended thinking (chain-of-thought) for models that support it.
    pub fn enable_thinking(mut self, enable_thinking: bool) -> Self {
        self.enable_thinking = Some(enable_thinking);
        self
    }

    /// Set the reasoning effort for models that support it.
    pub fn with_reasoning_effort(mut self, reasoning_effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    /// Truncate prompts that exceed the model's maximum context length.
    pub fn with_truncate_sequence(mut self, truncate_sequence: bool) -> Self {
        self.truncate_sequence = truncate_sequence;
        self
    }

    /// Set the tool-call presentation format for templates that support
    /// several (`json` | `xml` | `xml_typed`, e.g. K2-Horizon). `None`
    /// (default) leaves the template's own default in place (`xml` for K2).
    /// Unknown values are ignored downstream (template default wins), never
    /// erroring mid-turn.
    pub fn with_tool_call_format(mut self, tool_call_format: impl ToString) -> Self {
        self.tool_call_format = Some(tool_call_format.to_string());
        self
    }
}

impl RequestLike for RequestBuilder {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }

    fn take_messages(&mut self) -> RequestMessage {
        let mut other = Vec::new();
        std::mem::swap(&mut other, &mut self.messages);
        RequestMessage::Chat {
            messages: other,
            enable_thinking: self.enable_thinking,
            reasoning_effort: self.reasoning_effort,
            tool_call_format: self.tool_call_format.take(),
        }
    }

    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        if self.logits_processors.is_empty() {
            None
        } else {
            let mut other = Vec::new();
            std::mem::swap(&mut other, &mut self.logits_processors);
            Some(other)
        }
    }

    fn take_adapter(&mut self) -> Option<AdapterSelection> {
        self.adapter.take()
    }

    fn return_logprobs(&self) -> bool {
        self.return_logprobs
    }

    fn take_constraint(&mut self) -> Constraint {
        let mut other = Constraint::None;
        std::mem::swap(&mut other, &mut self.constraint);
        other
    }

    fn take_tools(&mut self) -> Option<(Vec<Tool>, ToolChoice)> {
        if self.tools.is_empty() {
            None
        } else {
            let mut other_ts = Vec::new();
            std::mem::swap(&mut other_ts, &mut self.tools);
            let mut other_tc = ToolChoice::Auto;
            std::mem::swap(&mut other_tc, &mut self.tool_choice);
            Some((other_ts, other_tc))
        }
    }

    fn take_sampling_params(&mut self) -> SamplingParams {
        let mut other = SamplingParams::deterministic();
        std::mem::swap(&mut other, &mut self.sampling_params);
        other
    }

    fn truncate_sequence(&self) -> bool {
        self.truncate_sequence
    }

}


