//! Progressive skill disclosure — skills listed by name+description in prompt,
//! full content loaded on-demand via Read tool.
//!
//! Skills are discovered from SKILL.md files in configured directories.
//! Only name + description appear in the system prompt (saving tokens);
//! the full skill content is read on-demand by the agent when needed.

use serde::{Deserialize, Serialize};

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
}
