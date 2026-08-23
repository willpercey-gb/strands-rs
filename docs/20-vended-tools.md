# Ready-Made Tools

The `strands-tools` crate ships tools you would otherwise write yourself. It is
separate from `strands-core` so an agent that only needs the loop does not pull
in an HTTP stack.

```toml
strands-tools = { version = "0.1", features = ["shell", "file-editor", "http"] }
```

`shell`, `file-editor`, `sleep`, `stop` and `sandbox` are on by default; `http`
is opt-in.

## sleep / stop

```rust,ignore
use strands_tools::{SleepTool, StopTool};

let stop = StopTool::new();
let signal = stop.signal();

let agent = Agent::builder()
    .model(model)
    .tool(SleepTool::new())
    .tool(stop)
    .build()?;

if signal.is_stopped() {
    println!("model stopped: {:?}", signal.reason());
}
```

`sleep` is capped per call, so a model that misreads a unit gets an error rather
than parking the agent for an hour.

## shell

```rust,ignore
use strands_tools::ShellTool;

let shell = ShellTool::new()
    .with_timeout(Duration::from_secs(30))
    .with_working_dir("/srv/project")
    .with_allowlist(["git", "cargo", "ls"]);
```

Annotated as **destructive** and **open-world**, because a permission layer
reads those hints and understating them would be worse than omitting them.

> The allowlist is a coarse guard, not a boundary. The command still runs
> through a shell, so anything on the list that can spawn a subprocess
> (`find -exec`, `xargs`, an interpreter) escapes it. For untrusted input use a
> [sandbox](#sandboxes).

## file_editor

```rust,ignore
use strands_tools::FileEditorTool;

let editor = FileEditorTool::new().with_root("/srv/project");
```

Edits are exact-string replacements requiring a **unique** match. Line numbers
drift as soon as the model makes one edit; an ambiguous replacement is refused
rather than silently editing the wrong occurrence.

Always set `with_root` — without it the model can read and rewrite anything the
process can.

## http_request

```rust,ignore
use strands_tools::HttpRequestTool;

let http = HttpRequestTool::new()
    .with_allowed_hosts(["api.example.com"]);
```

Handing a model outbound HTTP is an SSRF surface. By default it refuses
non-HTTP schemes and any loopback, link-local, private or shared-address-space
destination — cloud metadata endpoints included — and disables redirects so a
hop cannot route around the check.

`allow_private_networks(true)` only when the destination is genuinely trusted.

## Sandboxes

```rust,ignore
use strands_tools::{Command, DockerSandbox, Sandbox};

let sandbox = DockerSandbox::new("alpine:3");
sandbox.start().await?;
let out = sandbox.run(&Command::new("ls -la")).await?;
```

Networking is **disabled by default** — an agent that can reach the network from
inside the sandbox has escaped most of the point of one. Enabling it contains
filesystem and process damage but not exfiltration.

`Sandbox::is_isolated()` exists so a caller can refuse untrusted input on
`LocalEnvironment`, which is a sandbox in name only and warns on every run.
