//! `kitty update` — version check (10-min, plan SDD-ESTADO-ARTE §4).
//!
//! Compares the local version ([`local_version`], via
//! `kitty_harness::version::current`) against the latest GitHub release tag.
//! Pure decision logic ([`decide`], [`normalize_tag`], [`extract_tag`],
//! rendering) is unit-tested without network; only [`fetch_latest_tag`]
//! touches the network, and every failure there is fail-closed: a loud
//! message plus `Err` (exit ≠ 0), never a silent success.
//!
//! Release source: `GET {releases_url}` (GitHub `releases/latest` JSON,
//! `tag_name` field). `KITTY_UPDATE_URL` overrides the URL (tests, mirrors);
//! `KITTY_UPDATE_LATEST` bypasses the network with a fixed tag (tests,
//! offline pinning). Tag comparison strips one leading `v`/`V` because
//! GitHub tags are `vX.Y.Z` while the harness semver core is `X.Y.Z`.

use anyhow::{Context, Result};

/// `owner/repo` used for release URLs (matches workspace `repository`).
pub const GITHUB_OWNER_REPO: &str = "SuarezPM/kitty-code";
/// Default `releases/latest` API URL.
pub const DEFAULT_RELEASES_URL: &str =
    "https://api.github.com/repos/SuarezPM/kitty-code/releases/latest";
/// Env override for the release JSON URL (tests, mirrors).
pub const ENV_UPDATE_URL: &str = "KITTY_UPDATE_URL";
/// Env override with a fixed latest tag (bypasses the network).
pub const ENV_UPDATE_LATEST: &str = "KITTY_UPDATE_LATEST";
/// Network timeout: the check must fail fast and loud, never hang silently.
pub const FETCH_TIMEOUT_SECS: u64 = 10;

/// Running version (harness primitive — single source of truth).
#[must_use]
pub fn local_version() -> &'static str {
    kitty_harness::version::current()
}

/// Strip one leading `v`/`V` and surrounding whitespace.
///
/// GitHub tags ship as `v0.2.0`; the harness [`parse_version`](kitty_harness::version::parse_version)
/// only accepts the numeric `X.Y.Z` core, so `v`-prefixed tags must be
/// normalized before comparison (never by changing the harness).
#[must_use]
pub fn normalize_tag(s: &str) -> &str {
    let t = s.trim();
    t.strip_prefix('v')
        .or_else(|| t.strip_prefix('V'))
        .unwrap_or(t)
}

/// `true` iff `latest_raw` is strictly newer than `have` (fail-closed:
/// `false` on any unparseable input — never prompts on garbage).
#[must_use]
pub fn is_update_available(have: &str, latest_raw: &str) -> bool {
    kitty_harness::version::is_newer(normalize_tag(have), normalize_tag(latest_raw))
}

/// Extract `tag_name` from a GitHub `releases/latest` JSON body.
///
/// Returns `None` on invalid JSON, missing/non-string `tag_name`, or an
/// empty tag — every one of those is a fail-closed error upstream, never a
/// silent "up to date".
#[must_use]
pub fn extract_tag(body: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let tag = v.get("tag_name")?.as_str()?.trim();
    if tag.is_empty() {
        return None;
    }
    Some(tag.to_string())
}

/// Outcome of a successful check (both variants exit 0; only fetch/parse
/// failures exit ≠ 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateDecision {
    /// Local is current (equal or newer than the release tag).
    UpToDate { current: String, latest: String },
    /// A strictly newer release tag exists.
    UpdateAvailable { current: String, latest: String },
}

impl UpdateDecision {
    /// `true` iff a newer release exists.
    #[must_use]
    pub fn update_available(&self) -> bool {
        matches!(self, Self::UpdateAvailable { .. })
    }
}

/// Pure decision: `None` when either side is unparseable (fail-closed —
/// the caller turns this into a loud error, never into "up to date").
///
/// The stored `latest` keeps the raw trimmed tag (e.g. `v0.2.0`) so URLs
/// and messages show exactly what the release calls itself.
#[must_use]
pub fn decide(current: &str, latest_raw: &str) -> Option<UpdateDecision> {
    let latest = latest_raw.trim().to_string();
    if kitty_harness::version::parse_version(normalize_tag(current)).is_none()
        || kitty_harness::version::parse_version(normalize_tag(&latest)).is_none()
    {
        return None;
    }
    if is_update_available(current, &latest) {
        Some(UpdateDecision::UpdateAvailable {
            current: current.trim().to_string(),
            latest,
        })
    } else {
        Some(UpdateDecision::UpToDate {
            current: current.trim().to_string(),
            latest,
        })
    }
}

/// Human-readable rendering (default `kitty update` output, Spanish).
#[must_use]
pub fn format_human(decision: &UpdateDecision) -> String {
    match decision {
        UpdateDecision::UpToDate { current, latest } => {
            format!("kitty {current}: al día (última release {latest}).")
        }
        UpdateDecision::UpdateAvailable { current, latest } => {
            let url = release_url_for(latest);
            format!(
                "kitty {current}: hay actualización {latest} disponible.\n\
                 Release: {url}\n\
                 Actualiza con el instalador o `cargo install` y re-ejecuta `kitty update`."
            )
        }
    }
}

/// Machine-readable rendering (`kitty update --json`).
#[must_use]
pub fn format_json(decision: &UpdateDecision) -> String {
    let (current, latest) = match decision {
        UpdateDecision::UpToDate { current, latest }
        | UpdateDecision::UpdateAvailable { current, latest } => (current, latest),
    };
    serde_json::json!({
        "current": current,
        "latest": latest,
        "update_available": decision.update_available(),
    })
    .to_string()
}

/// Human release page for a tag (tag shown verbatim, e.g. `v0.2.0`).
#[must_use]
pub fn release_url_for(tag_raw: &str) -> String {
    format!(
        "https://github.com/{GITHUB_OWNER_REPO}/releases/tag/{}",
        tag_raw.trim()
    )
}

/// Release JSON URL: `KITTY_UPDATE_URL` wins, else the GitHub default.
#[must_use]
pub fn releases_url() -> String {
    std::env::var(ENV_UPDATE_URL)
        .ok()
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| DEFAULT_RELEASES_URL.to_string())
}

/// Fixed latest tag from `KITTY_UPDATE_LATEST` (network bypass); `None`
/// when unset or blank.
#[must_use]
pub fn latest_override() -> Option<String> {
    std::env::var(ENV_UPDATE_LATEST)
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// Fetch the latest release tag over HTTPS.
///
/// Fail-closed: any transport, status, or parse problem is an `Err` with a
/// clear Spanish message (the caller prints it loudly and exits ≠ 0).
pub async fn fetch_latest_tag(url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .user_agent("kitty-cli/update")
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .build()
        .context("construir cliente HTTPS para el check de versión")?;
    let body = client
        .get(url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .await
        .with_context(|| {
            format!(
                "sin red: no se pudo contactar {url} (revisa la conexión o fija {} / {})",
                ENV_UPDATE_URL, ENV_UPDATE_LATEST
            )
        })?
        .error_for_status()
        .with_context(|| format!("el servidor de releases devolvió error para {url}"))?
        .text()
        .await
        .context("leer el cuerpo de la respuesta de releases")?;
    extract_tag(&body).with_context(|| {
        format!("respuesta de releases sin `tag_name` útil desde {url} (no se asume versión)")
    })
}

/// Resolve the latest tag: env override first (no network), else fetch.
pub async fn resolve_latest() -> Result<String> {
    if let Some(tag) = latest_override() {
        return Ok(tag);
    }
    fetch_latest_tag(&releases_url()).await
}

/// `kitty update` entry point: check + loud report.
///
/// - Up to date / update available → print (human or JSON) + `Ok` (exit 0).
/// - No network, bad status, or unparseable tag → `eprintln!` + `Err`
///   (exit ≠ 0, nunca silencioso).
pub async fn run_update(json: bool) -> Result<()> {
    let current = local_version().to_string();
    let latest_raw = match resolve_latest().await {
        Ok(tag) => tag,
        Err(e) => {
            eprintln!("kitty update: sin red o release ilegible (local {current}): {e:#}");
            return Err(e);
        }
    };
    match decide(&current, &latest_raw) {
        Some(decision) => {
            if json {
                println!("{}", format_json(&decision));
            } else {
                println!("{}", format_human(&decision));
            }
            Ok(())
        }
        None => {
            eprintln!(
                "kitty update: tag remoto ininteligible {latest_raw:?} (local {current}): \
                 no se asume versión — revisa {} o la red y reintenta",
                ENV_UPDATE_URL
            );
            anyhow::bail!("tag remoto ininteligible {latest_raw:?} frente a local {current}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn normalize_strips_single_v_prefix() {
        assert_eq!(normalize_tag("v0.2.0"), "0.2.0");
        assert_eq!(normalize_tag("V1.0.0"), "1.0.0");
        assert_eq!(normalize_tag("  v0.2.0  "), "0.2.0");
        assert_eq!(normalize_tag("0.2.0"), "0.2.0");
        // Only one prefix is stripped; the rest stays and fails parse upstream.
        assert_eq!(normalize_tag("vv1.0.0"), "v1.0.0");
    }

    #[test]
    fn local_version_is_parseable_after_normalize() {
        assert!(
            kitty_harness::version::parse_version(normalize_tag(local_version())).is_some(),
            "CARGO_PKG_VERSION must parse: {}",
            local_version()
        );
    }

    #[test]
    fn update_available_compares_numeric_cores() {
        assert!(is_update_available("0.1.0", "0.2.0"));
        assert!(is_update_available("0.1.0", "v0.2.0"));
        assert!(is_update_available("v0.1.0", "0.1.1"));
        assert!(is_update_available("0.9.9", "0.10.0"));
        assert!(!is_update_available("0.2.0", "0.2.0"));
        assert!(!is_update_available("v0.2.0", "0.2.0"));
        assert!(!is_update_available("0.2.0", "0.1.9"));
    }

    #[test]
    fn update_check_is_fail_closed_on_garbage() {
        assert!(!is_update_available("0.1.0", "latest"));
        assert!(!is_update_available("0.1.0", ""));
        assert!(!is_update_available("", "0.2.0"));
        assert!(!is_update_available("0.1.0", "vv1.0.0"));
        assert!(decide("0.1.0", "latest").is_none());
        assert!(decide("0.1.0", "").is_none());
        assert!(decide("0.1", "0.2.0").is_none());
    }

    #[test]
    fn decide_keeps_raw_tag_for_display() {
        assert_eq!(
            decide("0.1.0", "v0.2.0"),
            Some(UpdateDecision::UpdateAvailable {
                current: "0.1.0".to_string(),
                latest: "v0.2.0".to_string(),
            })
        );
        assert_eq!(
            decide("0.2.0", "v0.2.0"),
            Some(UpdateDecision::UpToDate {
                current: "0.2.0".to_string(),
                latest: "v0.2.0".to_string(),
            })
        );
        // Downgrade is "up to date" (never prompt), not an error.
        assert!(matches!(
            decide("0.2.0", "0.1.9"),
            Some(UpdateDecision::UpToDate { .. })
        ));
    }

    #[test]
    fn extract_tag_reads_github_latest_shape() {
        let body = r#"{"tag_name":"v0.2.0","name":"v0.2.0"}"#;
        assert_eq!(extract_tag(body).as_deref(), Some("v0.2.0"));
        assert_eq!(extract_tag("not json"), None);
        assert_eq!(extract_tag(r#"{"name":"no tag"}"#), None);
        assert_eq!(extract_tag(r#"{"tag_name":42}"#), None);
        assert_eq!(extract_tag(r#"{"tag_name":"  "}"#), None);
        assert_eq!(
            extract_tag(r#"{"tag_name":"  v0.3.0  "}"#).as_deref(),
            Some("v0.3.0")
        );
    }

    #[test]
    fn human_and_json_render_both_versions() {
        let up = UpdateDecision::UpToDate {
            current: "0.1.0".to_string(),
            latest: "v0.1.0".to_string(),
        };
        let msg = format_human(&up);
        assert!(msg.contains("0.1.0"), "got {msg}");
        assert!(msg.contains("al día"), "got {msg}");

        let avail = UpdateDecision::UpdateAvailable {
            current: "0.1.0".to_string(),
            latest: "v0.2.0".to_string(),
        };
        let msg = format_human(&avail);
        assert!(msg.contains("0.1.0") && msg.contains("v0.2.0"), "got {msg}");
        assert!(
            msg.contains(&release_url_for("v0.2.0")),
            "must link the release: {msg}"
        );

        let parsed: serde_json::Value =
            serde_json::from_str(&format_json(&avail)).expect("valid json");
        assert_eq!(parsed["current"], "0.1.0");
        assert_eq!(parsed["latest"], "v0.2.0");
        assert_eq!(parsed["update_available"], true);
    }

    #[test]
    fn release_url_points_at_owner_repo_tag() {
        let url = release_url_for("v0.2.0");
        assert!(
            url.contains(GITHUB_OWNER_REPO) && url.ends_with("/v0.2.0"),
            "got {url}"
        );
    }

    #[test]
    fn update_subcommand_parses_with_and_without_json() {
        let cli = crate::Cli::try_parse_from(["kitty", "update"]).unwrap();
        match cli.command {
            crate::Commands::Update { json } => assert!(!json),
            _ => panic!("expected update"),
        }
        let cli = crate::Cli::try_parse_from(["kitty", "update", "--json"]).unwrap();
        match cli.command {
            crate::Commands::Update { json } => assert!(json),
            _ => panic!("expected update --json"),
        }
    }

    /// Fail-closed sin red: puerto cerrado → `Err` ruidoso, nunca `Ok`
    /// silencioso. Sin red externa (localhost con nada escuchando).
    #[tokio::test]
    async fn fetch_against_closed_port_fails_loudly() {
        let err = fetch_latest_tag("http://127.0.0.1:9/kitty-update-probe")
            .await
            .expect_err("closed port must fail");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("sin red"),
            "error must say sin red clearly, got: {msg}"
        );
    }
}
