//! Progressive skill disclosure — skills listed by name+description in prompt,
//! full content loaded on-demand via Read tool.
//!
//! Skills are discovered from SKILL.md files in configured directories.
//! Only name + description appear in the system prompt (saving tokens);
//! the full skill content is read on-demand by the agent when needed.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A discovered skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Short identifier (e.g. "bun-build").
    pub name: String,
    /// One-line description of what the skill covers.
    pub description: String,
    /// Absolute path to the skill directory (parent of SKILL.md).
    pub location: std::path::PathBuf,
}

/// Discover skills from one or more skill directories.
///
/// Each directory is scanned for `SKILL.md` files. Duplicate names are
/// skipped with a warning (first wins).
pub fn discover_skills(skill_dirs: &[&Path]) -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen_names = std::collections::HashSet::new();

    for dir in skill_dirs {
        if !dir.exists() {
            continue;
        }

        for entry in std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some("SKILL.md") {
                if let Some(skill) = parse_skill_file(&path) {
                    if seen_names.insert(skill.name.clone()) {
                        skills.push(skill);
                    }
                }
            }
        }
    }

    skills
}

/// Parse a SKILL.md file for name and description.
///
/// Expects simple frontmatter-style lines: `name: ...` and `description: ...`.
fn parse_skill_file(path: &std::path::Path) -> Option<Skill> {
    let content = std::fs::read_to_string(path).ok()?;

    let mut name = None;
    let mut description = None;

    for line in content.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("name:") {
            name = Some(val.trim().to_string());
        } else if let Some(val) = line.strip_prefix("description:") {
            description = Some(val.trim().to_string());
        }
    }

    let name = name?;
    let description = description.unwrap_or_default();
    let location = path.parent()?.to_path_buf();

    Some(Skill {
        name,
        description,
        location,
    })
}

/// Format skills as XML for the system prompt.
///
/// Returns an empty string when no skills are available (avoids injecting
/// an empty XML block into the prompt).
pub fn format_skills_xml(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return String::new();
    }

    let mut xml = String::from("<available_skills>\n");
    for skill in skills {
        xml.push_str(&format!(
            "  <skill>\n    <name>{}</name>\n    <description>{}</description>\n    <location>{}</location>\n  </skill>\n",
            skill.name,
            skill.description,
            skill.location.display()
        ));
    }
    xml.push_str("</available_skills>\n");
    xml
}

/// Get the full content path for a skill by name.
pub fn skill_content_path(skills: &[Skill], name: &str) -> Option<std::path::PathBuf> {
    skills
        .iter()
        .find(|s| s.name == name)
        .map(|s| s.location.join("SKILL.md"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_skills_xml_empty() {
        assert_eq!(format_skills_xml(&[]), "");
    }

    #[test]
    fn format_skills_xml_with_skills() {
        let skills = vec![Skill {
            name: "test-skill".to_string(),
            description: "A test skill".to_string(),
            location: std::path::PathBuf::from("/tmp/skills"),
        }];
        let xml = format_skills_xml(&skills);
        assert!(xml.contains("<available_skills>"));
        assert!(xml.contains("test-skill"));
        assert!(xml.contains("A test skill"));
        assert!(xml.contains("</available_skills>"));
    }

    #[test]
    fn skill_content_path_found() {
        let skills = vec![Skill {
            name: "my-skill".to_string(),
            description: "desc".to_string(),
            location: std::path::PathBuf::from("/opt/skills/my-skill"),
        }];
        let path = skill_content_path(&skills, "my-skill");
        assert_eq!(
            path,
            Some(std::path::PathBuf::from("/opt/skills/my-skill/SKILL.md"))
        );
    }

    #[test]
    fn skill_content_path_not_found() {
        let skills: Vec<Skill> = vec![];
        assert_eq!(skill_content_path(&skills, "nope"), None);
    }

    #[test]
    fn discover_skills_empty_dirs() {
        let skills = discover_skills(&[]);
        assert!(skills.is_empty());
    }

    #[test]
    fn discover_skills_nonexistent_dir() {
        let skills = discover_skills(&[std::path::Path::new("/nonexistent/path")]);
        assert!(skills.is_empty());
    }
}
