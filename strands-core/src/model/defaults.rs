//! Default model metadata — context window limits for known model ids.
//!
//! Best-effort lookup table sourced from provider documentation. Unknown
//! models return `None` and callers degrade gracefully (proactive compression
//! simply stays off) rather than guessing a limit and truncating wrongly.
//!
//! Applies to providers with well-known, fixed model ids. Providers using
//! local or custom ids (Ollama, llama.cpp) are deliberately absent — their
//! context window depends on deployment config, not on a static table.

/// Context window limit assumed when a model reports none.
pub const DEFAULT_CONTEXT_WINDOW_LIMIT: u64 = 200_000;

/// Known model id → context window limit, in tokens.
///
/// Sorted by provider. For ids carrying a cross-region prefix (e.g.
/// `us.anthropic.claude-sonnet-4-6`), [`get_context_window_limit`] strips the
/// prefix before lookup, so only the base id needs an entry here.
static CONTEXT_WINDOW_LIMITS: &[(&str, u64)] = &[
    // Anthropic (direct API)
    ("claude-sonnet-4-6", 1_000_000),
    ("claude-sonnet-4-20250514", 1_000_000),
    ("claude-sonnet-4-5", 200_000),
    ("claude-sonnet-4-5-20250929", 200_000),
    ("claude-opus-4-6", 1_000_000),
    ("claude-opus-4-6-20260205", 1_000_000),
    ("claude-opus-4-7", 1_000_000),
    ("claude-opus-4-7-20260416", 1_000_000),
    ("claude-opus-4-8", 1_000_000),
    ("claude-opus-5", 1_000_000),
    ("claude-fable-5", 1_000_000),
    ("claude-sonnet-5", 1_000_000),
    ("claude-opus-4-5", 200_000),
    ("claude-opus-4-5-20251101", 200_000),
    ("claude-opus-4-20250514", 200_000),
    ("claude-opus-4-1", 200_000),
    ("claude-opus-4-1-20250805", 200_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-haiku-4-5-20251001", 200_000),
    ("claude-3-7-sonnet-20250219", 200_000),
    ("claude-3-5-sonnet-20241022", 200_000),
    ("claude-3-5-sonnet-20240620", 200_000),
    ("claude-3-5-haiku-20241022", 200_000),
    ("claude-3-opus-20240229", 200_000),
    ("claude-3-haiku-20240307", 200_000),
    // Bedrock Anthropic (base model IDs — cross-region prefixes stripped by get_context_window_limit)
    ("anthropic.claude-sonnet-4-6", 1_000_000),
    ("anthropic.claude-sonnet-4-20250514-v1:0", 1_000_000),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", 200_000),
    ("anthropic.claude-opus-4-6-v1", 1_000_000),
    ("anthropic.claude-opus-4-7", 1_000_000),
    ("anthropic.claude-opus-4-8", 1_000_000),
    ("anthropic.claude-opus-5", 1_000_000),
    ("anthropic.claude-fable-5", 1_000_000),
    ("anthropic.claude-sonnet-5", 1_000_000),
    ("anthropic.claude-opus-4-5-20251101-v1:0", 200_000),
    ("anthropic.claude-opus-4-20250514-v1:0", 200_000),
    ("anthropic.claude-opus-4-1-20250805-v1:0", 200_000),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", 200_000),
    ("anthropic.claude-haiku-4-5@20251001", 200_000),
    ("anthropic.claude-3-7-sonnet-20250219-v1:0", 200_000),
    ("anthropic.claude-3-7-sonnet-20240620-v1:0", 200_000),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", 200_000),
    ("anthropic.claude-3-5-sonnet-20240620-v1:0", 200_000),
    ("anthropic.claude-3-5-haiku-20241022-v1:0", 200_000),
    ("anthropic.claude-3-opus-20240229-v1:0", 200_000),
    ("anthropic.claude-3-haiku-20240307-v1:0", 200_000),
    ("anthropic.claude-3-sonnet-20240229-v1:0", 200_000),
    ("anthropic.claude-mythos-preview", 1_000_000),
    // Bedrock Amazon Nova
    ("amazon.nova-pro-v1:0", 300_000),
    ("amazon.nova-lite-v1:0", 300_000),
    ("amazon.nova-micro-v1:0", 128_000),
    ("amazon.nova-premier-v1:0", 1_000_000),
    ("amazon.nova-2-lite-v1:0", 1_000_000),
    ("amazon.nova-2-pro-preview-20251202-v1:0", 1_000_000),
    // OpenAI
    ("gpt-5.6", 1_050_000),
    ("gpt-5.6-sol", 1_050_000),
    ("gpt-5.6-terra", 1_050_000),
    ("gpt-5.6-luna", 1_050_000),
    ("gpt-5.5", 1_050_000),
    ("gpt-5.5-pro", 1_050_000),
    ("gpt-5.4", 1_050_000),
    ("gpt-5.4-pro", 1_050_000),
    ("gpt-5.4-mini", 272_000),
    ("gpt-5.4-nano", 272_000),
    ("gpt-5.2", 272_000),
    ("gpt-5.2-pro", 272_000),
    ("gpt-5.1", 272_000),
    ("gpt-5", 272_000),
    ("gpt-5-mini", 272_000),
    ("gpt-5-nano", 272_000),
    ("gpt-5-pro", 128_000),
    ("gpt-4.1", 1_047_576),
    ("gpt-4.1-mini", 1_047_576),
    ("gpt-4.1-nano", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4o-mini", 128_000),
    ("gpt-4-turbo", 128_000),
    ("o3", 200_000),
    ("o3-mini", 200_000),
    ("o3-pro", 200_000),
    ("o4-mini", 200_000),
    ("o1", 200_000),
    // Google Gemini
    ("gemini-2.5-flash", 1_048_576),
    ("gemini-2.5-flash-lite", 1_048_576),
    ("gemini-2.5-pro", 1_048_576),
    ("gemini-2.0-flash", 1_048_576),
    ("gemini-2.0-flash-lite", 1_048_576),
    ("gemini-3-pro-preview", 1_048_576),
    ("gemini-3-flash-preview", 1_048_576),
    ("gemini-3.6-flash", 1_048_576),
    ("gemini-3.5-flash", 1_048_576),
    ("gemini-3.5-flash-lite", 1_048_576),
    ("gemini-3.1-flash-lite", 1_048_576),
    ("gemini-3.1-pro-preview", 1_048_576),
    ("gemini-3.1-flash-lite-preview", 1_048_576),
    // Mistral
    ("mistral-large-latest", 262_144),
    ("mistral-large-2512", 262_144),
    ("mistral-large-3", 262_144),
    ("mistral-medium-latest", 131_072),
    ("mistral-medium-2505", 131_072),
    ("mistral-small-latest", 131_072),
    ("mistral-small-3-2-2506", 131_072),];

/// Look up the context window limit for a model id.
///
/// Two prefix strips are attempted when the direct lookup misses:
///
/// - `<prefix>.` — Bedrock cross-region ids (`us.`, `eu.`, `global.`) resolve
///   without needing an entry per region.
/// - `<vendor>/` — OpenRouter-style ids (`anthropic/claude-opus-5`) resolve to
///   their base model. Not in upstream, which has no OpenRouter provider.
pub fn get_context_window_limit(model_id: &str) -> Option<u64> {
    if let Some(limit) = lookup(model_id) {
        return Some(limit);
    }

    for sep in ['.', '/'] {
        if let Some((_, rest)) = model_id.split_once(sep) {
            if let Some(limit) = lookup(rest) {
                tracing::debug!(
                    model_id,
                    stripped_id = rest,
                    "Resolved context window limit via prefix strip"
                );
                return Some(limit);
            }
        }
    }

    None
}

fn lookup(model_id: &str) -> Option<u64> {
    CONTEXT_WINDOW_LIMITS
        .iter()
        .find(|(id, _)| *id == model_id)
        .map(|(_, limit)| *limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ids_resolve_directly() {
        assert_eq!(get_context_window_limit("claude-opus-5"), Some(1_000_000));
        assert_eq!(get_context_window_limit("gpt-4o"), Some(128_000));
        assert_eq!(
            get_context_window_limit("gemini-2.5-pro"),
            Some(1_048_576)
        );
    }

    #[test]
    fn cross_region_prefixes_are_stripped() {
        // The table holds only the base id; the prefix strip is what makes
        // every region resolve without duplicating 100 entries per region.
        assert_eq!(
            get_context_window_limit("us.anthropic.claude-opus-5"),
            Some(1_000_000)
        );
        assert_eq!(
            get_context_window_limit("eu.anthropic.claude-3-opus-20240229-v1:0"),
            Some(200_000)
        );
    }

    #[test]
    fn openrouter_vendor_prefixes_are_stripped() {
        assert_eq!(
            get_context_window_limit("anthropic/claude-opus-5"),
            Some(1_000_000)
        );
        assert_eq!(get_context_window_limit("openai/gpt-4o"), Some(128_000));
    }

    #[test]
    fn unknown_ids_return_none_rather_than_a_default() {
        // Returning None matters: callers disable proactive compression rather
        // than compressing against an invented limit.
        assert_eq!(get_context_window_limit("llama3.2"), None);
        assert_eq!(get_context_window_limit("some-local-model"), None);
    }

    #[test]
    fn table_has_no_duplicate_ids() {
        let mut ids: Vec<&str> = CONTEXT_WINDOW_LIMITS.iter().map(|(id, _)| *id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate model id in the limits table");
    }
}
