//! Serving security policy: bind scope, token auth, rate limit, CORS.
//!
//! Pure policy layer — no listener lives here (the thin binary wires it to
//! HTTP; the axum `serve` call sites are out of scope for this lane). Rules:
//! - `ServingConfig::default()` binds `127.0.0.1` (loopback-only, no token
//!   needed) — the old default-open posture stays valid ONLY on loopback.
//! - `validate()` fails closed: a non-loopback bind WITHOUT `KITTY_API_TOKEN`
//!   is a startup error, closing the `check_auth(None, _) → Ok` hole
//!   (`api.rs`) for every external bind.
//! - Admin/Logs/Metrics routes require the bearer whenever a token is
//!   configured or the bind is external (`check_route`); `/health` stays open.
//! - `RateLimiter` is a std+tokio token bucket (no new deps).
//! - `CorsConfig` is an allow-list + pure header builder. Wiring
//!   `tower_http::CorsLayer` needs a NEW direct dep in
//!   `mistralrs-core/Cargo.toml` (HELD-dep, lane forbids it); the exact hook
//!   is documented on [`CorsConfig::allow_origin`] for the listener owner.

use tokio::sync::Mutex;
use tokio::time::Instant;

use super::api::check_auth;
use super::errors::KittyError;

/// Env knobs (all optional; documented defaults apply when unset).
pub const ENV_BIND: &str = "KITTY_BIND";
pub const ENV_PORT: &str = "KITTY_PORT";
pub const ENV_TOKEN: &str = "KITTY_API_TOKEN";
pub const ENV_REQUIRE_TOKEN: &str = "KITTY_REQUIRE_TOKEN";
pub const ENV_RPS: &str = "KITTY_RATE_LIMIT_RPS";
pub const ENV_BURST: &str = "KITTY_RATE_LIMIT_BURST";
pub const ENV_CORS: &str = "KITTY_CORS_ORIGINS";

/// Loopback default: never serves off-host unless the operator opts in.
pub const DEFAULT_BIND: &str = "127.0.0.1";
/// Default port for the thin serving binary.
pub const DEFAULT_PORT: u16 = 8080;

#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Sustained requests/second per limiter.
    pub rps: u32,
    /// Max burst before shedding.
    pub burst: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self { rps: 30, burst: 60 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CorsConfig {
    /// Exact allowed origins (`"*"` allows any). Empty = deny all.
    pub allow_origins: Vec<String>,
}

impl CorsConfig {
    /// HELD-dep hook: the future axum listener applies this as
    /// `tower_http::cors::CorsLayer::new().allow_origin(...)` per allowed
    /// origin (or `.allow_origin(Any)` for `"*"`); until that dep lands,
    /// apply [`Self::response_headers`] on responses instead.
    pub fn allow_origin(&self, origin: &str) -> bool {
        self.allow_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// Pure `Access-Control-*` headers for an allowed origin; empty when the
    /// origin is not allowed (fail-closed: no header, browser blocks).
    pub fn response_headers(&self, origin: Option<&str>) -> Vec<(String, String)> {
        let Some(origin) = origin else { return vec![] };
        if !self.allow_origin(origin) {
            return vec![];
        }
        let echo = if self.allow_origins.iter().any(|o| o == "*") {
            "*"
        } else {
            origin
        };
        vec![
            ("Access-Control-Allow-Origin".to_string(), echo.to_string()),
            (
                "Access-Control-Allow-Methods".to_string(),
                "GET, POST, OPTIONS".to_string(),
            ),
            (
                "Access-Control-Allow-Headers".to_string(),
                "Content-Type, Authorization".to_string(),
            ),
            ("Access-Control-Max-Age".to_string(), "600".to_string()),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct ServingConfig {
    /// Bind address. Default loopback; anything else forces a token.
    pub bind: String,
    pub port: u16,
    /// Deploy token (`KITTY_API_TOKEN`). `None` = loopback-only mode.
    pub api_token: Option<String>,
    /// Explicit opt-in even on loopback (`KITTY_REQUIRE_TOKEN=1`).
    pub require_token: bool,
    pub rate_limit: RateLimitConfig,
    pub cors: CorsConfig,
}

impl Default for ServingConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND.to_string(),
            port: DEFAULT_PORT,
            api_token: None,
            require_token: false,
            rate_limit: RateLimitConfig::default(),
            cors: CorsConfig::default(),
        }
    }
}

impl ServingConfig {
    /// Read all `KITTY_*` knobs. Total (never fails); [`Self::validate`]
    /// is the fail-closed gate before serving.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(b) = std::env::var(ENV_BIND) {
            if !b.trim().is_empty() {
                cfg.bind = b.trim().to_string();
            }
        }
        if let Ok(p) = std::env::var(ENV_PORT) {
            if let Ok(port) = p.trim().parse::<u16>() {
                cfg.port = port;
            }
        }
        cfg.api_token = std::env::var(ENV_TOKEN)
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());
        cfg.require_token = matches!(
            std::env::var(ENV_REQUIRE_TOKEN)
                .ok()
                .as_deref()
                .map(str::trim),
            Some("1") | Some("true") | Some("yes")
        );
        if let Ok(r) = std::env::var(ENV_RPS) {
            if let Ok(rps) = r.trim().parse::<u32>() {
                if rps > 0 {
                    cfg.rate_limit.rps = rps;
                }
            }
        }
        if let Ok(b) = std::env::var(ENV_BURST) {
            if let Ok(burst) = b.trim().parse::<u32>() {
                if burst > 0 {
                    cfg.rate_limit.burst = burst;
                }
            }
        }
        if let Ok(origins) = std::env::var(ENV_CORS) {
            cfg.cors.allow_origins = origins
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect();
        }
        cfg
    }

    /// True for loopback binds (`127.*`, `::1`, `localhost`).
    pub fn is_loopback(&self) -> bool {
        let b = self.bind.trim();
        b == "localhost" || b == "::1" || b.starts_with("127.")
    }

    /// Effective token requirement: explicit opt-in OR any off-host bind.
    pub fn effective_require_token(&self) -> bool {
        self.require_token || !self.is_loopback()
    }

    /// Fail-closed startup gate. External bind without a token is an error;
    /// loopback stays token-optional (documented single-user default).
    pub fn validate(&self) -> Result<(), KittyError> {
        if self.port == 0 {
            return Err(KittyError::Validation {
                detail: "serving port 0 is not allowed — set KITTY_PORT".to_string(),
            });
        }
        if self.effective_require_token() && self.api_token.is_none() {
            return Err(KittyError::Validation {
                detail: format!(
                    "refusing to serve on non-loopback bind `{}` without KITTY_API_TOKEN \
                     (set a token or bind {DEFAULT_BIND})",
                    self.bind
                ),
            });
        }
        Ok(())
    }

    /// Route scope: Admin/Logs/Metrics always gated; everything else
    /// (chat, models, health) stays open. Query strings are ignored.
    pub fn auth_required(path: &str) -> bool {
        let route = path.split('?').next().unwrap_or(path);
        const GATED: [&str; 3] = ["/admin", "/logs", "/metrics"];
        GATED
            .iter()
            .any(|p| route == *p || route.starts_with(&format!("{p}/")))
    }

    /// Per-request gate combining scope + token state. External binds can
    /// never pass without a configured token (even if `validate` was
    /// skipped); loopback without a token stays open by design.
    pub fn check_route(&self, path: &str, bearer: Option<&str>) -> Result<(), KittyError> {
        if !Self::auth_required(path) {
            return Ok(());
        }
        if self.effective_require_token() && self.api_token.is_none() {
            return Err(KittyError::Validation {
                detail: "unauthorized: external bind requires KITTY_API_TOKEN".to_string(),
            });
        }
        check_auth(self.api_token.as_deref(), bearer)
    }

    /// `bind:port` for the listener call site.
    pub fn addr(&self) -> String {
        format!("{}:{}", self.bind, self.port)
    }
}

/// Dependency-free token bucket (tokio clock). `false` = shed load.
pub struct RateLimiter {
    rps: f64,
    burst: f64,
    state: Mutex<(f64, Instant)>,
}

impl RateLimiter {
    pub fn new(cfg: &RateLimitConfig) -> Self {
        Self {
            rps: cfg.rps.max(1) as f64,
            burst: cfg.burst.max(1) as f64,
            state: Mutex::new((cfg.burst.max(1) as f64, Instant::now())),
        }
    }

    pub async fn try_acquire(&self) -> bool {
        let mut guard = self.state.lock().await;
        let now = Instant::now();
        let elapsed = now.duration_since(guard.1).as_secs_f64();
        guard.0 = (guard.0 + elapsed * self.rps).min(self.burst);
        guard.1 = now;
        if guard.0 >= 1.0 {
            guard.0 -= 1.0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn external() -> ServingConfig {
        ServingConfig {
            bind: "0.0.0.0".to_string(),
            api_token: Some("secret".to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn default_is_loopback_and_validates_open() {
        let cfg = ServingConfig::default();
        assert_eq!(cfg.bind, "127.0.0.1");
        assert!(cfg.is_loopback());
        assert!(cfg.validate().is_ok());
        // Loopback without token: documented open (previous behavior kept).
        assert!(cfg.check_route("/admin/load", None).is_ok());
        assert!(cfg.check_route("/health", None).is_ok());
    }

    #[test]
    fn external_bind_without_token_fails_closed() {
        let cfg = ServingConfig {
            bind: "0.0.0.0".to_string(),
            ..Default::default()
        };
        assert!(!cfg.is_loopback());
        assert!(cfg.validate().is_err());
        // Even if validate() were skipped, requests still deny.
        assert!(cfg.check_route("/admin/load", Some("secret")).is_err());
        assert!(cfg.check_route("/metrics", None).is_err());
        // Health stays open regardless.
        assert!(cfg.check_route("/health", None).is_ok());
        // Other loopback spellings.
        for b in ["127.0.0.2", "::1", "localhost"] {
            let c = ServingConfig {
                bind: b.to_string(),
                ..Default::default()
            };
            assert!(c.is_loopback(), "{b}");
            assert!(c.validate().is_ok());
        }
    }

    #[test]
    fn token_branches_per_route() {
        let cfg = external();
        assert!(cfg.validate().is_ok());
        for gated in [
            "/admin/load",
            "/admin/status",
            "/logs",
            "/logs?tail=10",
            "/metrics",
        ] {
            assert!(cfg.check_route(gated, None).is_err(), "{gated} missing");
            assert!(
                cfg.check_route(gated, Some("wrong")).is_err(),
                "{gated} wrong"
            );
            assert!(
                cfg.check_route(gated, Some("secret")).is_ok(),
                "{gated} valid"
            );
        }
        for open in ["/health", "/v1/models", "/v1/chat/completions"] {
            assert!(cfg.check_route(open, None).is_ok(), "{open} must stay open");
        }
        // Explicit opt-in enforces the token even on loopback.
        let strict = ServingConfig {
            require_token: true,
            api_token: Some("secret".to_string()),
            ..Default::default()
        };
        assert!(strict.check_route("/metrics", None).is_err());
        assert!(strict.check_route("/metrics", Some("secret")).is_ok());
        assert!(strict.check_route("/health", None).is_ok());
    }

    #[test]
    fn port_zero_rejected() {
        let cfg = ServingConfig {
            port: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[tokio::test]
    async fn rate_limiter_bursts_then_sheds_then_refills() {
        let lim = RateLimiter::new(&RateLimitConfig { rps: 20, burst: 2 });
        assert!(lim.try_acquire().await);
        assert!(lim.try_acquire().await);
        assert!(!lim.try_acquire().await);
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert!(lim.try_acquire().await);
    }

    #[test]
    fn cors_is_allow_list_fail_closed() {
        let deny = CorsConfig::default();
        assert!(deny.response_headers(Some("https://x.test")).is_empty());
        let cfg = CorsConfig {
            allow_origins: vec!["https://app.test".to_string()],
        };
        let headers = cfg.response_headers(Some("https://app.test"));
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Access-Control-Allow-Origin" && v == "https://app.test"));
        assert!(cfg.response_headers(Some("https://evil.test")).is_empty());
        assert!(cfg.response_headers(None).is_empty());
    }
}
