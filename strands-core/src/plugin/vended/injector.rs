//! Context injection — adding content to the conversation from outside it.
//!
//! Retrieved documents, memory recall, environment facts: things the agent
//! should see but that no tool call produced. Injecting them as structured
//! blocks (rather than concatenating into a prompt) keeps them attributable and
//! lets cache points sit where they belong.
//!
//! Ported from upstream `vended_plugins/context_injector/` and `injection/`.

use crate::types::content::{SystemContentBlock, SystemPrompt};

/// Where injected content is placed relative to a cache point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InjectionPlacement {
    /// Before the first cache point, inside the cacheable prefix.
    ///
    /// Right for content that is stable across turns — instructions, a fixed
    /// document set. Wrong for anything that changes each turn, which would
    /// invalidate the cache every time.
    BeforeCachePoint,
    /// After the last cache point.
    ///
    /// Right for per-turn content — a fresh memory recall, current time — where
    /// keeping the cached prefix intact matters more than caching this.
    ///
    /// Upstream v1.53 `feat: add injected content behind cache points`.
    AfterCachePoint,
}

/// One piece of injected content.
#[derive(Debug, Clone, PartialEq)]
pub struct InjectedContent {
    /// Names the source, so the model can weigh it.
    pub source: String,
    pub content: String,
}

impl InjectedContent {
    pub fn new(source: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            content: content.into(),
        }
    }

    /// Render with an XML-ish wrapper naming the source.
    ///
    /// Tagging matters: unlabelled injected text reads to the model as its own
    /// instructions, which is how retrieved content ends up being followed as
    /// though it were a directive.
    pub fn render(&self) -> String {
        format!(
            "<injected source=\"{}\">\n{}\n</injected>",
            escape(&self.source),
            self.content
        )
    }
}

/// Escape the characters that would break out of an attribute.
fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Injects content into a system prompt.
#[derive(Debug, Clone)]
pub struct ContextInjector {
    items: Vec<InjectedContent>,
    placement: InjectionPlacement,
}

impl Default for ContextInjector {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            placement: InjectionPlacement::AfterCachePoint,
        }
    }
}

impl ContextInjector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_placement(mut self, placement: InjectionPlacement) -> Self {
        self.placement = placement;
        self
    }

    pub fn add(mut self, item: InjectedContent) -> Self {
        self.items.push(item);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// The rendered block, or `None` when there is nothing to inject.
    pub fn render(&self) -> Option<String> {
        if self.items.is_empty() {
            return None;
        }
        Some(
            self.items
                .iter()
                .map(InjectedContent::render)
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    }

    /// Inject into `prompt` at the configured placement.
    pub fn inject(&self, prompt: Option<&SystemPrompt>) -> SystemPrompt {
        let Some(text) = self.render() else {
            return prompt
                .cloned()
                .unwrap_or_else(|| SystemPrompt::Text(String::new()));
        };

        let block = SystemContentBlock::Text { text };

        let Some(prompt) = prompt else {
            return SystemPrompt::Blocks(vec![block]);
        };

        let (_, mut blocks) = prompt.split();

        let index = match self.placement {
            InjectionPlacement::BeforeCachePoint => blocks
                .iter()
                .position(|b| matches!(b, SystemContentBlock::CachePoint(_)))
                .unwrap_or(blocks.len()),
            InjectionPlacement::AfterCachePoint => blocks
                .iter()
                .rposition(|b| matches!(b, SystemContentBlock::CachePoint(_)))
                .map(|i| i + 1)
                .unwrap_or(blocks.len()),
        };

        blocks.insert(index, block);
        SystemPrompt::Blocks(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::content::CachePoint;

    fn injector() -> ContextInjector {
        ContextInjector::new().add(InjectedContent::new("memory", "user prefers dark mode"))
    }

    fn prompt_with_cache_point() -> SystemPrompt {
        SystemPrompt::Blocks(vec![
            SystemContentBlock::Text {
                text: "base".into(),
            },
            SystemContentBlock::CachePoint(CachePoint::new()),
        ])
    }

    fn position_of(blocks: &[SystemContentBlock], needle: &str) -> usize {
        blocks
            .iter()
            .position(|b| matches!(b, SystemContentBlock::Text { text } if text.contains(needle)))
            .unwrap_or_else(|| panic!("{needle} not found"))
    }

    fn cache_position(blocks: &[SystemContentBlock]) -> usize {
        blocks
            .iter()
            .position(|b| matches!(b, SystemContentBlock::CachePoint(_)))
            .expect("cache point present")
    }

    #[test]
    fn injected_content_is_tagged_with_its_source() {
        // Unlabelled injected text reads to the model as its own instructions,
        // which is how retrieved content ends up followed as a directive.
        let text = InjectedContent::new("memory", "a fact").render();
        assert!(text.contains(r#"source="memory""#));
        assert!(text.contains("a fact"));
        assert!(text.contains("</injected>"));
    }

    #[test]
    fn a_source_cannot_break_out_of_the_attribute() {
        let text = InjectedContent::new(r#"evil" onload="x"#, "content").render();
        assert!(!text.contains(r#"onload="x""#), "attribute escaped: {text}");
        assert!(text.contains("&quot;"));
    }

    #[test]
    fn per_turn_content_defaults_to_after_the_cache_point() {
        // Content that changes each turn must not invalidate the cached prefix.
        let injected = injector().inject(Some(&prompt_with_cache_point()));
        let SystemPrompt::Blocks(blocks) = injected else {
            panic!("expected blocks");
        };
        assert!(position_of(&blocks, "dark mode") > cache_position(&blocks));
    }

    #[test]
    fn stable_content_can_be_placed_inside_the_cacheable_prefix() {
        let injected = injector()
            .with_placement(InjectionPlacement::BeforeCachePoint)
            .inject(Some(&prompt_with_cache_point()));

        let SystemPrompt::Blocks(blocks) = injected else {
            panic!("expected blocks");
        };
        assert!(position_of(&blocks, "dark mode") < cache_position(&blocks));
    }

    #[test]
    fn injection_appends_when_there_is_no_cache_point() {
        let prompt = SystemPrompt::from("base instructions");
        let text = injector().inject(Some(&prompt)).as_text().unwrap();
        assert!(text.contains("base instructions"));
        assert!(text.contains("dark mode"));
    }

    #[test]
    fn the_cache_point_always_survives() {
        for placement in [
            InjectionPlacement::BeforeCachePoint,
            InjectionPlacement::AfterCachePoint,
        ] {
            let injected = injector()
                .with_placement(placement)
                .inject(Some(&prompt_with_cache_point()));
            let SystemPrompt::Blocks(blocks) = injected else {
                panic!("expected blocks");
            };
            assert!(
                blocks
                    .iter()
                    .any(|b| matches!(b, SystemContentBlock::CachePoint(_))),
                "{placement:?} dropped the cache point"
            );
        }
    }

    #[test]
    fn an_empty_injector_leaves_the_prompt_untouched() {
        let prompt = SystemPrompt::from("original");
        let injected = ContextInjector::new().inject(Some(&prompt));
        assert_eq!(injected.as_text().as_deref(), Some("original"));
        assert!(ContextInjector::new().render().is_none());
    }

    #[test]
    fn multiple_items_are_all_injected() {
        let injector = ContextInjector::new()
            .add(InjectedContent::new("a", "first"))
            .add(InjectedContent::new("b", "second"));

        let text = injector.render().unwrap();
        assert!(text.contains("first") && text.contains("second"));
        assert_eq!(injector.len(), 2);
    }
}
