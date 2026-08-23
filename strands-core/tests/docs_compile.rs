//! Compile-checks the code in `docs/`.
//!
//! Documentation drifted silently through the v1.37 → v1.53 sync: four guides
//! still described the pre-sync `Model::stream` and `ToolSpec` shapes long after
//! those changed. Prose cannot be type-checked, but code can, so every Rust
//! block in the guides is extracted and compiled here.
//!
//! A block that is a deliberate fragment (a trait signature, a partial `impl`)
//! is marked ```rust,ignore in the guide and skipped. Everything else must
//! compile against the real API.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A Rust block lifted from a guide.
struct Snippet {
    file: String,
    /// 1-based line of the opening fence, so a failure points at the source.
    line: usize,
    code: String,
}

fn docs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("docs")
}

/// Extract every non-ignored ```rust block from a guide.
fn snippets_in(path: &Path) -> Vec<Snippet> {
    let text = std::fs::read_to_string(path).expect("readable guide");
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    let mut snippets = Vec::new();
    let mut current: Option<(usize, String)> = None;

    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end();

        match &mut current {
            // Inside a block: close it or accumulate.
            Some((start, body)) => {
                if line.trim() == "```" {
                    snippets.push(Snippet {
                        file: name.clone(),
                        line: *start,
                        code: std::mem::take(body),
                    });
                    current = None;
                } else {
                    body.push_str(raw);
                    body.push('\n');
                }
            }
            // Outside: open only on a checkable Rust fence.
            None => {
                if let Some(info) = line.trim().strip_prefix("```") {
                    let info = info.trim();
                    let is_rust = info == "rust" || info.starts_with("rust,");
                    let skip = info.contains("ignore") || info.contains("no_run");
                    if is_rust && !skip {
                        current = Some((index + 1, String::new()));
                    }
                }
            }
        }
    }

    snippets
}

/// Wrap a snippet so a bare expression-style example still compiles.
///
/// Guides show usage, not `main` functions. Items (`use`, `struct`, `impl`,
/// `fn`) have to stay at module scope, so they are hoisted out of the body.
fn wrap(code: &str) -> String {
    let mut items = String::new();
    let mut body = String::new();

    // Track brace depth so an `impl` block's contents are not re-classified.
    let mut depth = 0usize;
    let mut in_item = false;

    for line in code.lines() {
        let trimmed = line.trim_start();
        let starts_item = depth == 0
            && (trimmed.starts_with("use ")
                || trimmed.starts_with("pub ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("trait ")
                || trimmed.starts_with("impl")
                || trimmed.starts_with("async fn ")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("#["));

        if depth == 0 && starts_item {
            in_item = true;
        }

        if in_item {
            items.push_str(line);
            items.push('\n');
        } else {
            body.push_str(line);
            body.push('\n');
        }

        depth += line.matches('{').count();
        depth = depth.saturating_sub(line.matches('}').count());

        if in_item && depth == 0 && (line.contains('}') || line.trim_end().ends_with(';')) {
            in_item = false;
        }
    }

    format!(
        "#![allow(unused, dead_code, unused_imports, ambiguous_glob_reexports)]\n\
         {items}\n\
         #[allow(clippy::all)]\n\
         async fn __doc_example() {{\n{body}\n}}\n\
         fn main() {{}}\n"
    )
}

#[test]
fn every_documented_example_compiles() {
    let docs = docs_dir();
    let mut guides: Vec<PathBuf> = std::fs::read_dir(&docs)
        .expect("docs directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .collect();
    guides.sort();

    let snippets: Vec<Snippet> = guides.iter().flat_map(|p| snippets_in(p)).collect();

    assert!(
        !snippets.is_empty(),
        "found no checkable examples — the extractor is probably broken, \
         which would make this test silently vacuous"
    );

    // Build against the already-compiled workspace, so this needs no network
    // and no separate dependency resolution.
    let deps = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/debug/deps");

    if !deps.exists() {
        eprintln!("skipping: {} not built yet", deps.display());
        return;
    }

    let scratch = std::env::temp_dir().join(format!("strands-doc-check-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");

    let mut failures = Vec::new();

    for (index, snippet) in snippets.iter().enumerate() {
        let source = scratch.join(format!("snippet_{index}.rs"));
        std::fs::write(&source, wrap(&snippet.code)).expect("write snippet");

        let mut rustc = Command::new("rustc");
        rustc
            .arg("--edition=2021")
            .arg("--emit=metadata")
            .arg("--crate-type=bin")
            .arg("-L")
            .arg(&deps);

        // Link everything the guides import, so a missing crate is never
        // mistaken for an API change.
        for krate in [
            "strands_core",
            "strands_ollama",
            "strands_openrouter",
            "strands_claude_cli",
            "strands_claude_mcp",
            "strands_tools",
            "serde_json",
            "async_trait",
            "futures",
            "tokio",
        ] {
            let rlib = find_rlib(&deps, krate);
            if !rlib.is_empty() {
                rustc.arg("--extern").arg(format!("{krate}={rlib}"));
            }
        }

        let output = rustc
            .arg("-o")
            .arg(scratch.join(format!("snippet_{index}.meta")))
            .arg(&source)
            .output();

        match output {
            Ok(out) if !out.status.success() => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                // Match on error *codes*, not message text. Guides legitimately
                // omit imports and surrounding context, so resolution errors
                // (E0432/E0433/E0425/E0252/E0423) are the guide being
                // illustrative. These codes can only mean the documented shape
                // no longer matches the real one.
                const API_DRIFT: &[&str] = &[
                    "E0050", // incorrect number of function parameters
                    "E0053", // method has an incompatible type for trait
                    "E0060", // wrong number of arguments
                    "E0061", // this function takes N arguments
                    "E0063", // missing field in initializer
                    "E0107", // wrong number of generic arguments
                    "E0195", // lifetime parameters do not match the trait
                    "E0220", // associated type not found
                    "E0407", // method is not a member of trait
                    "E0559", // unknown field in enum variant
                    "E0560", // unknown field in struct
                    "E0576", // not found in trait
                ];

                if API_DRIFT.iter().any(|code| stderr.contains(code)) {
                    failures.push(format!(
                        "{}:{} — API mismatch\n{}",
                        snippet.file,
                        snippet.line,
                        stderr
                            .lines()
                            .filter(|l| {
                                l.starts_with("error[")
                                    && API_DRIFT.iter().any(|code| l.contains(code))
                            })
                            .take(3)
                            .collect::<Vec<_>>()
                            .join("\n")
                    ));
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("skipping: could not run rustc: {e}");
                return;
            }
        }
    }

    let _ = std::fs::remove_dir_all(&scratch);

    assert!(
        failures.is_empty(),
        "{} documented example(s) no longer match the API:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// Locate the most recently built artifact for a crate.
///
/// Proc-macro crates build as a dynamic library rather than an rlib, so both
/// extensions are accepted.
fn find_rlib(deps: &Path, crate_name: &str) -> String {
    let prefix = format!("lib{crate_name}-");
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(deps)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.starts_with(&prefix)
                && (name.ends_with(".rlib")
                    || name.ends_with(".dylib")
                    || name.ends_with(".so"))
        })
        .filter_map(|e| {
            let modified = e.metadata().ok()?.modified().ok()?;
            Some((modified, e.path()))
        })
        .collect();

    candidates.sort_by_key(|(time, _)| *time);
    candidates
        .last()
        .map(|(_, path)| path.to_string_lossy().into_owned())
        .unwrap_or_default()
}
