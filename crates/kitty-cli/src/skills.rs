//! Skill discovery (W3): walker `SKILL.md` + cableado a `format_skills_xml`.
//!
//! [`discover_skills`] walks each configured root (non-recursive depth cap to
//! avoid accidental home-wide scans) looking for `SKILL.md` files and derives
//! [`Skill`](kitty_harness::skills::Skill) records from them:
//! - `name` = parent directory name (`skills/bun-build/SKILL.md` → `bun-build`).
//! - `description` = first non-empty, non-heading line of the file.
//! - `location` = parent directory (absolute if the root was absolute).
//!
//! Rendering stays in `kitty-harness`
//! ([`format_skills_xml`](kitty_harness::skills::format_skills_xml)) — this
//! module only discovers. `kitty skills` prints the XML block the agent
//! injects plus a one-line-per-skill summary.

use std::path::{Path, PathBuf};

/// Re-export for CLI rendering.
pub use kitty_harness::skills::{format_skills_xml, Skill};

/// Default skill roots, in priority order (first hit wins on name clash):
/// explicit `KITTY_SKILLS_DIR` (colon-separated), then `./skills`, then
/// `~/.config/opencode/skills` (opencode-compatible lane on this machine).
pub fn default_skill_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(raw) = std::env::var("KITTY_SKILLS_DIR") {
        dirs.extend(
            raw.split(':')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
        );
    }
    dirs.push(PathBuf::from("skills"));
    if let Ok(home) = std::env::var("HOME") {
        dirs.push(
            PathBuf::from(home)
                .join(".config")
                .join("opencode")
                .join("skills"),
        );
    }
    dirs
}

/// Depth cap for the manual walker (skill repos are shallow: `<root>/<name>/SKILL.md`).
const MAX_DEPTH: usize = 4;

/// Discover skills under `roots`. Missing roots are skipped silently; files
/// that cannot be read are skipped (discovery never fails the CLI).
pub fn discover_skills(roots: &[PathBuf]) -> Vec<Skill> {
    let mut out: Vec<Skill> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for root in roots {
        walk_dir(root, 0, &mut out, &mut seen);
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn walk_dir(
    dir: &Path,
    depth: usize,
    out: &mut Vec<Skill>,
    seen: &mut std::collections::HashSet<String>,
) {
    if depth > MAX_DEPTH {
        return;
    }
    // Direct hit: `<dir>/SKILL.md`.
    let candidate = dir.join("SKILL.md");
    if candidate.is_file() {
        if let Some(skill) = load_skill(dir, &candidate) {
            if seen.insert(skill.name.clone()) {
                out.push(skill);
            }
        }
        // A skill dir is a leaf — do not descend into it.
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    // Deterministic order for stable CLI output.
    let mut subdirs: Vec<PathBuf> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Skip hidden and dependency dirs.
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with('.') || name == "node_modules" || name == "target" {
                    continue;
                }
            }
            subdirs.push(path);
        }
    }
    subdirs.sort();
    for sub in subdirs {
        walk_dir(&sub, depth + 1, out, seen);
    }
}

fn load_skill(dir: &Path, skill_md: &Path) -> Option<Skill> {
    let name = dir.file_name()?.to_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let text = std::fs::read_to_string(skill_md).ok()?;
    let description = first_description_line(&text).unwrap_or_else(|| name.clone());
    Some(Skill {
        name,
        description,
        location: dir.to_path_buf(),
    })
}

/// First non-empty line that is not a markdown heading or fence.
///
/// YAML front-matter `description:` wins when present (opencode-style skills
/// carry it); otherwise the first content line. Bare `key: value` lines are
/// skipped so the summary never shows `name: foo` instead of the description.
fn first_description_line(text: &str) -> Option<String> {
    if let Some(desc) = front_matter_description(text) {
        return Some(desc);
    }
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with("```") || t.starts_with("---") {
            continue;
        }
        // Skip `key: value` / `key:` front-matter-ish lines (no spaces
        // before the colon); legit prose with colons keeps spaces.
        if let Some(colon) = t.find(':') {
            if !t[..colon].chars().any(char::is_whitespace) {
                continue;
            }
        }
        return Some(t.chars().take(160).collect());
    }
    None
}

/// `description:` inside a leading `---` YAML front-matter block.
fn front_matter_description(text: &str) -> Option<String> {
    let mut lines = text.lines();
    if lines.next()?.trim() != "---" {
        return None;
    }
    for line in lines {
        let t = line.trim();
        if t == "---" || t == "..." {
            break;
        }
        if let Some(rest) = t.strip_prefix("description:") {
            let d = rest.trim().trim_matches('"').trim();
            if !d.is_empty() {
                return Some(d.chars().take(160).collect());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn write_skill(root: &Path, name: &str, body: &str) -> PathBuf {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut f = std::fs::File::create(dir.join("SKILL.md")).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        dir
    }

    #[test]
    fn discovers_skill_by_dir_name() {
        let root = tempfile::tempdir().unwrap();
        write_skill(
            root.path(),
            "bun-build",
            "# bun-build\n\nCreate optimized production bundles.\n",
        );
        let skills = discover_skills(&[root.path().to_path_buf()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "bun-build");
        assert!(
            skills[0].description.contains("optimized production"),
            "got {:?}",
            skills[0].description
        );
    }

    #[test]
    fn front_matter_description_wins() {
        let root = tempfile::tempdir().unwrap();
        write_skill(
            root.path(),
            "fm",
            "---\nname: fm\ndescription: Real front-matter summary.\n---\n\n# Fm\n\nBody prose.\n",
        );
        let skills = discover_skills(&[root.path().to_path_buf()]);
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "Real front-matter summary.");
    }

    #[test]
    fn missing_root_is_empty_not_error() {
        let skills = discover_skills(&[PathBuf::from("/tmp/kitty-no-skills-dir-xyz-123")]);
        assert!(skills.is_empty());
    }

    #[test]
    fn first_hit_wins_on_name_clash() {
        let r1 = tempfile::tempdir().unwrap();
        let r2 = tempfile::tempdir().unwrap();
        write_skill(r1.path(), "dup", "# dup\n\nFrom root one.\n");
        write_skill(r2.path(), "dup", "# dup\n\nFrom root two.\n");
        let skills = discover_skills(&[r1.path().to_path_buf(), r2.path().to_path_buf()]);
        assert_eq!(skills.len(), 1);
        assert!(skills[0].description.contains("root one"));
    }

    #[test]
    fn wires_into_format_skills_xml() {
        let root = tempfile::tempdir().unwrap();
        write_skill(root.path(), "s1", "# s1\n\nDoes things.\n");
        let skills = discover_skills(&[root.path().to_path_buf()]);
        let xml = format_skills_xml(&skills);
        assert!(xml.contains("<available_skills>"));
        assert!(xml.contains("s1"));
    }

    #[test]
    fn skills_sorted_by_name() {
        let root = tempfile::tempdir().unwrap();
        write_skill(root.path(), "zzz", "# zzz\n\nZ.\n");
        write_skill(root.path(), "aaa", "# aaa\n\nA.\n");
        let skills = discover_skills(&[root.path().to_path_buf()]);
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "aaa");
        assert_eq!(skills[1].name, "zzz");
    }
}
