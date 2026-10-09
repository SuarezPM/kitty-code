//! Version primitives for `kitty update` (10-min, SDD estado-arte).
//!
//! Pure semver-core comparison without new dependencies. The network fetch
//! (latest release lookup) and the `update` UX live in the cli lane — this
//! module gives it tested primitives: [`current`], [`parse_version`], and
//! [`is_newer`]. Fail-closed: unparseable input never reports an update.

/// Running crate version (`CARGO_PKG_VERSION` of kitty-harness).
#[must_use]
pub fn current() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Parse the numeric `major.minor.patch` core, ignoring any `-pre`/`+build`
/// suffix. Returns `None` on non-numeric cores.
#[must_use]
pub fn parse_version(s: &str) -> Option<(u64, u64, u64)> {
    let core = s.split(['-', '+']).next()?.trim();
    let mut parts = core.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// `true` iff `latest` is strictly newer than `have` (numeric core
/// comparison). `false` on equality, downgrade, or any unparseable input
/// (never prompt an update on garbage).
#[must_use]
pub fn is_newer(have: &str, latest: &str) -> bool {
    match (parse_version(have), parse_version(latest)) {
        (Some(h), Some(l)) => l > h,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_is_parseable() {
        assert!(
            parse_version(current()).is_some(),
            "CARGO_PKG_VERSION must parse: {}",
            current()
        );
    }

    #[test]
    fn orders_versions_numerically() {
        assert!(is_newer("0.1.0", "0.2.0"));
        assert!(is_newer("0.1.0", "0.1.1"));
        assert!(is_newer("0.9.9", "0.10.0"));
        assert!(!is_newer("0.2.0", "0.2.0"));
        assert!(!is_newer("0.2.0", "0.1.9"));
        assert!(!is_newer("1.0.0", "0.99.99"));
    }

    #[test]
    fn ignores_suffixes_and_rejects_garbage() {
        assert!(is_newer("0.1.0", "0.1.1-rc.1"));
        assert!(!is_newer("0.1.0", "latest"));
        assert!(!is_newer("", "0.1.0"));
        assert!(!is_newer("0.1.0", ""));
        assert!(!is_newer("0.1", "0.2.0"));
        assert!(!is_newer("0.1.0", "0.1.0.1"));
    }
}
