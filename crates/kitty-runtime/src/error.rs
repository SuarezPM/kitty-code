use thiserror::Error;

/// Classification of errors for retry decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Retryable: timeout, connection reset, 5xx
    Transient,
    /// Rate limited: 429
    RateLimit,
    /// Permanent: 400, 401, 403, unknown
    Permanent,
    /// Context overflow: needs compaction
    Overflow,
    /// Quota/billing: don't retry
    Quota,
}

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

impl RuntimeError {
    /// Whether this error is transient and worth retrying (F3.3).
    pub fn is_transient(&self) -> bool {
        match self {
            RuntimeError::Api { status, .. } => {
                matches!(status, 429 | 500 | 502 | 503 | 504)
            }
            RuntimeError::ConnectionRefused(_) => true,
            RuntimeError::Io(e) => {
                // Connection reset, broken pipe, etc.
                e.kind() == std::io::ErrorKind::ConnectionReset
                    || e.kind() == std::io::ErrorKind::ConnectionAborted
                    || e.kind() == std::io::ErrorKind::BrokenPipe
            }
            // Http errors from reqwest include timeouts and network failures.
            // Treat reqwest::Error as transient for connection-level issues.
            RuntimeError::Http(e) => e.is_timeout() || e.is_connect(),
            _ => false,
        }
    }

    /// Classify the error type for retry decisions.
    pub fn error_class(&self) -> ErrorClass {
        match self {
            RuntimeError::Api { status, .. } => match status {
                429 => ErrorClass::RateLimit,
                500 | 502 | 503 | 504 => ErrorClass::Transient,
                _ => ErrorClass::Permanent,
            },
            RuntimeError::ConnectionRefused(_) => ErrorClass::Transient,
            RuntimeError::ContextOverflow { .. } => ErrorClass::Overflow,
            RuntimeError::Io(e) => match e.kind() {
                std::io::ErrorKind::ConnectionReset
                | std::io::ErrorKind::ConnectionAborted
                | std::io::ErrorKind::BrokenPipe => ErrorClass::Transient,
                _ => ErrorClass::Permanent,
            },
            RuntimeError::Http(e) => {
                if e.is_timeout() || e.is_connect() {
                    ErrorClass::Transient
                } else {
                    ErrorClass::Permanent
                }
            }
            _ => ErrorClass::Permanent,
        }
    }

    /// Check if error is retryable (transient or rate-limit).
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.error_class(),
            ErrorClass::Transient | ErrorClass::RateLimit
        )
    }

    /// Check if error is an overflow that needs compaction.
    pub fn is_overflow(&self) -> bool {
        matches!(self.error_class(), ErrorClass::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api_err(status: u16) -> RuntimeError {
        RuntimeError::Api {
            status,
            message: "test".into(),
        }
    }

    #[test]
    fn transient_429() {
        assert!(api_err(429).is_transient());
    }
    #[test]
    fn transient_500() {
        assert!(api_err(500).is_transient());
    }
    #[test]
    fn transient_502() {
        assert!(api_err(502).is_transient());
    }
    #[test]
    fn transient_503() {
        assert!(api_err(503).is_transient());
    }
    #[test]
    fn transient_504() {
        assert!(api_err(504).is_transient());
    }
    #[test]
    fn non_transient_400() {
        assert!(!api_err(400).is_transient());
    }
    #[test]
    fn non_transient_401() {
        assert!(!api_err(401).is_transient());
    }
    #[test]
    fn non_transient_404() {
        assert!(!api_err(404).is_transient());
    }
    #[test]
    fn transient_connection_refused() {
        assert!(RuntimeError::ConnectionRefused("test".into()).is_transient());
    }
    #[test]
    fn transient_io_connection_reset() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
        assert!(RuntimeError::Io(io).is_transient());
    }
    #[test]
    fn non_transient_io_other() {
        let io = std::io::Error::other("other");
        assert!(!RuntimeError::Io(io).is_transient());
    }
    #[test]
    fn non_transient_context_overflow() {
        assert!(!RuntimeError::ContextOverflow {
            requested: 0,
            message: "test".into()
        }
        .is_transient());
    }
}
