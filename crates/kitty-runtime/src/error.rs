use thiserror::Error;

/// Errors that can occur in the kitty-runtime crate.
#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Ollama API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// F2: the server rejected the prompt because it does not fit the model's
    /// context window (`truncate:false` turns silent truncation into this
    /// audible failure). `requested` is the `num_ctx` we asked for; the raw
    /// server message usually states the actual available size.
    #[error(
        "Context window exceeded: requested num_ctx={requested} but the prompt does not fit \
         ({message}). Lower KITTY_NUM_CTX (e.g. KITTY_NUM_CTX=16384) and restart the session."
    )]
    ContextOverflow { requested: u32, message: String },

    #[error("Model not available: {0}")]
    ModelNotFound(String),

    #[error("Connection refused: {0}")]
    ConnectionRefused(String),

    #[error("Tool execution failed: {tool}: {reason}")]
    ToolExecution { tool: String, reason: String },

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Ollama server returned unexpected response: {0}")]
    UnexpectedResponse(String),
}

pub type Result<T> = std::result::Result<T, RuntimeError>;
