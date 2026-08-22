//! Skills — reusable capability definitions injected into the system prompt.
//!
//! A skill bundles instructions, and optionally examples, for one capability.
//! Loading them at run time rather than baking them into the prompt means the
//! set can change without touching agent code.
//!
//! Injection is placed *before* any cache point so the skill text sits in the
//! cacheable prefix. Appending after it would invalidate the cache on every
//! turn, which for a long skill set costs more than the skills are worth.
//!
//! Ported from upstream `vended_plugins/skills/`.

use serde::{Deserialize, Serialize};

use crate::types::content::{SystemContentBlock, SystemPrompt};

/// One reusable capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    /// One line, used when listing skills.
    pub description: String,
    /// The instructions injected into the system prompt.
    pub instructions: String,
}

impl Skill {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: instructions.into(),
        }
    }

    /// How this skill appears in the injected prompt.
    pub fn render(&self) -> String {
        format!(
            "## {}\n{}\n\n{}",
            self.name, self.description, self.instructions
        )
    }
}

/// A set of skills that can be injected into a system prompt.
#[derive(Debug, Clone, Default)]
pub struct SkillSet {
    skills: Vec<Skill>,
}

impl SkillSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    pub fn extend(mut self, skills: impl IntoIterator<Item = Skill>) -> Self {
        self.skills.extend(skills);
        self
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn skills(&self) -> &[Skill] {
        &self.skills
    }

    /// The full injected text.
    pub fn render(&self) -> String {
        let bodies: Vec<String> = self.skills.iter().map(Skill::render).collect();
        format!("# Available skills\n\n{}", bodies.join("\n\n"))
    }

    /// Inject these skills into `prompt`.
    ///
    /// The skill text is placed before the first cache point, keeping it inside
    /// the cacheable prefix — appending after it would invalidate the cache
    /// every turn.
    ///
    /// This is upstream's `fix(skills): preserve cache points in system prompt
    /// during skills injection`.
    pub fn inject(&self, prompt: Option<&SystemPrompt>) -> SystemPrompt {
        if self.is_empty() {
            return prompt
                .cloned()
                .unwrap_or_else(|| SystemPrompt::Text(String::new()));
        }

        let injected = SystemContentBlock::Text {
            text: self.render(),
        };

        let Some(prompt) = prompt else {
            return SystemPrompt::Blocks(vec![injected]);
        };

        let (_, blocks) = prompt.split();

        // Insert ahead of the first cache point; with none, append.
        let insert_at = blocks
            .iter()
            .position(|b| matches!(b, SystemContentBlock::CachePoint(_)))
            .unwrap_or(blocks.len());

        let mut result = blocks;
        result.insert(insert_at, injected);
        SystemPrompt::Blocks(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::CachePoint;

    fn skills() -> SkillSet {
        SkillSet::new()
            .with_skill(Skill::new("search", "Find things", "Use the search tool."))
            .with_skill(Skill::new("write", "Produce text", "Be concise."))
    }

    #[test]
    fn rendering_includes_every_skill() {
        let text = skills().render();
        assert!(text.contains("search"));
        assert!(text.contains("Use the search tool."));
        assert!(text.contains("write"));
        assert!(text.contains("Be concise."));
    }

    #[test]
    fn injecting_into_a_plain_prompt_keeps_both() {
        let prompt = SystemPrompt::from("You are helpful.");
        let injected = skills().inject(Some(&prompt));

        let text = injected.as_text().expect("text rendering");
        assert!(text.contains("You are helpful."));
        assert!(text.contains("Available skills"));
    }

    #[test]
    fn injection_lands_before_the_cache_point() {
        // Appending after it would invalidate the cached prefix every turn,
        // which for a long skill set costs more than the skills are worth.
        let prompt = SystemPrompt::Blocks(vec![
            SystemContentBlock::Text {
                text: "base instructions".into(),
            },
            SystemContentBlock::CachePoint(CachePoint::new()),
        ]);

        let injected = skills().inject(Some(&prompt));
        let SystemPrompt::Blocks(blocks) = injected else {
            panic!("expected structured blocks");
        };

        let cache_index = blocks
            .iter()
            .position(|b| matches!(b, SystemContentBlock::CachePoint(_)))
            .expect("cache point survives");
        let skills_index = blocks
            .iter()
            .position(|b| matches!(b, SystemContentBlock::Text { text } if text.contains("Available skills")))
            .expect("skills injected");

        assert!(
            skills_index < cache_index,
            "skills must sit inside the cacheable prefix"
        );
    }

    #[test]
    fn the_cache_point_is_never_dropped() {
        let prompt = SystemPrompt::Blocks(vec![SystemContentBlock::CachePoint(
            CachePoint::new().with_ttl("1h"),
        )]);

        let injected = skills().inject(Some(&prompt));
        let SystemPrompt::Blocks(blocks) = injected else {
            panic!("expected blocks");
        };

        assert!(blocks
            .iter()
            .any(|b| matches!(b, SystemContentBlock::CachePoint(cp) if cp.ttl.as_deref() == Some("1h"))));
    }

    #[test]
    fn injecting_with_no_prompt_yields_just_the_skills() {
        let injected = skills().inject(None);
        assert!(injected.as_text().unwrap().contains("Available skills"));
    }

    #[test]
    fn an_empty_skill_set_leaves_the_prompt_untouched() {
        let prompt = SystemPrompt::from("original");
        let injected = SkillSet::new().inject(Some(&prompt));
        assert_eq!(injected.as_text().as_deref(), Some("original"));
    }

    #[test]
    fn skills_round_trip_through_serde() {
        let skill = Skill::new("a", "b", "c");
        let json = serde_json::to_string(&skill).unwrap();
        assert_eq!(serde_json::from_str::<Skill>(&json).unwrap(), skill);
    }
}
