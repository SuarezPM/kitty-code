//! Espejo de integración de `serving/config.rs` (unit tests :265-379).
//!
//! Replica los 6 tests unitarios como test de integración contra la API
//! pública (`mistralrs_core::serving`). Si este target no compila por deps
//! ajenas preexistentes del workspace (no por este fichero), HELD
//! documentado aquí y se reporta sin forzar.

use mistralrs_core::serving::{CorsConfig, RateLimitConfig, RateLimiter, ServingConfig};

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
