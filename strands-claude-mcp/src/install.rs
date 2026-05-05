use std::path::PathBuf;
use std::process::Command;

use crate::cargo_target_candidates;

/// Search for the `strands-claude-mcp-shim` binary in the usual places.
///
/// Order:
/// 1. Same directory as the current executable (Tauri sidecar).
/// 2. Tauri bundle resources (macOS app bundles put non-binary assets here,
///    but if the shim is shipped as a sidecar, Tauri colocates it).
/// 3. The Cargo workspace's `target/{debug,release}/` for development.
/// 4. PATH.
pub fn find_shim_binary() -> Result<PathBuf, String> {
    let bin_name = if cfg!(target_os = "windows") {
        "strands-claude-mcp-shim.exe"
    } else {
        "strands-claude-mcp-shim"
    };

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(bin_name);
            if candidate.exists() {
                return Ok(candidate);
            }
            // Tauri Resources fallback.
            for rel in ["../Resources", "../Resources/resources"] {
                let candidate = dir.join(rel).join(bin_name);
                if candidate.exists() {
                    return Ok(candidate.canonicalize().unwrap_or(candidate));
                }
            }
        }
    }

    for candidate in cargo_target_candidates() {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    if let Ok(out) = Command::new("sh")
        .args(["-lc", &format!("command -v {bin_name}")])
        .output()
    {
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(PathBuf::from(path));
            }
        }
    }

    Err(format!(
        "{bin_name} not found. Build it with: cargo build -p strands-claude-mcp"
    ))
}

/// Register an MCP server named `name` with Claude Code at user scope.
/// Idempotent: removes any prior registration with the same name first.
///
/// The shim is invoked as `<shim> --name <name> --port <port>`. Claude spawns
/// it as needed; the shim then connects to your in-process bridge.
pub fn install(name: &str, port: u16) -> Result<String, String> {
    let shim = find_shim_binary()?;
    let shim_str = shim.to_string_lossy().to_string();

    // `claude mcp add` is idempotent only after `remove`; suppress remove errors
    // on the first install via 2>/dev/null.
    let cmd = format!(
        "claude mcp remove {name} -s user 2>/dev/null; \
         claude mcp add {name} -s user -- {shim_str} --name {name} --port {port}"
    );
    let out = Command::new("sh")
        .args(["-lc", &cmd])
        .output()
        .map_err(|e| format!("spawn `claude mcp add`: {e} (is the Claude CLI installed?)"))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!(
            "claude mcp add failed (exit {}): {stderr}",
            out.status.code().unwrap_or(-1)
        ));
    }
    Ok(shim_str)
}

/// Unregister a previously installed server. Errors are swallowed because
/// `claude mcp remove` exits non-zero if the server isn't registered, and
/// callers usually don't care to distinguish that from a real failure.
pub fn uninstall(name: &str) {
    let _ = Command::new("sh")
        .args([
            "-lc",
            &format!(
                "claude mcp remove {name} -s user 2>/dev/null; \
                 claude mcp remove {name} -s local 2>/dev/null"
            ),
        ])
        .output();
}
