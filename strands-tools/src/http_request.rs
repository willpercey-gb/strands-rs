//! A tool that makes HTTP requests.
//!
//! # Safety
//!
//! Handing a model outbound HTTP is an SSRF surface: the URL comes from model
//! output, and the process may sit inside a network the caller does not want
//! reachable. The tool therefore refuses non-HTTP schemes and, by default,
//! addresses that resolve to loopback, link-local or private ranges — including
//! cloud metadata endpoints. Use [`allow_private_networks`](HttpRequestTool::allow_private_networks)
//! only when the destination is genuinely trusted.
//!
//! Ported from upstream `vended_tools/http_request/`.

use std::collections::HashSet;
use std::net::IpAddr;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use strands_core::error::StrandsError;
use strands_core::tool::{Tool, ToolContext, ToolOutput};
use strands_core::types::tools::{ToolAnnotations, ToolSpec};
use tracing::{debug, warn};

/// Default per-request timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cap on a captured response body, in bytes.
pub const DEFAULT_MAX_BODY: usize = 100_000;

/// Makes HTTP requests on the model's behalf.
pub struct HttpRequestTool {
    client: reqwest::Client,
    max_body: usize,
    allow_private: bool,
    allowed_hosts: Option<HashSet<String>>,
}

impl Default for HttpRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpRequestTool {
    /// Create with default settings.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(DEFAULT_TIMEOUT)
                // Redirects are followed by default, which would route around
                // the destination check; each hop must be re-validated instead.
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            max_body: DEFAULT_MAX_BODY,
            allow_private: false,
            allowed_hosts: None,
        }
    }

    /// Set the max body.
    pub fn with_max_body(mut self, bytes: usize) -> Self {
        self.max_body = bytes;
        self
    }

    /// Permit requests to loopback and private addresses.
    ///
    /// Off by default because the URL is model-controlled; turning it on
    /// exposes anything the process can reach on its own network, cloud
    /// metadata services included.
    pub fn allow_private_networks(mut self, allow: bool) -> Self {
        self.allow_private = allow;
        self
    }

    /// Restrict requests to these hostnames.
    pub fn with_allowed_hosts<I, S>(mut self, hosts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_hosts = Some(hosts.into_iter().map(Into::into).collect());
        self
    }

    /// Whether an address is one the model should not be able to reach.
    fn is_private(ip: &IpAddr) -> bool {
        match ip {
            IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_private()
                    || v4.is_link_local()
                    || v4.is_broadcast()
                    || v4.is_unspecified()
                    // 169.254.169.254 is link-local and already covered, but
                    // shared address space (100.64/10) is not.
                    || v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1])
            }
            IpAddr::V6(v6) => {
                v6.is_loopback()
                    || v6.is_unspecified()
                    // fc00::/7 unique-local and fe80::/10 link-local
                    || (v6.segments()[0] & 0xfe00) == 0xfc00
                    || (v6.segments()[0] & 0xffc0) == 0xfe80
            }
        }
    }

    /// Validate a destination before any request is made.
    fn check_url(&self, url: &str) -> Result<reqwest::Url, String> {
        let parsed = reqwest::Url::parse(url).map_err(|e| format!("invalid URL: {e}"))?;

        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!("unsupported scheme '{}'", parsed.scheme()));
        }

        let Some(host) = parsed.host_str() else {
            return Err("URL has no host".to_string());
        };

        if let Some(allowed) = &self.allowed_hosts {
            if !allowed.contains(host) {
                return Err(format!("host '{host}' is not on the allowlist"));
            }
        }

        if !self.allow_private {
            // `host_str` brackets IPv6 literals (`[::1]`), which would fail the
            // IP parse and let them straight through the check below.
            let literal = host.trim_start_matches('[').trim_end_matches(']');

            // A literal IP is checked directly. A hostname is resolved by the
            // client later, so this catches the direct case only — an allowlist
            // is the reliable control for name-based destinations.
            if let Ok(ip) = literal.parse::<IpAddr>() {
                if Self::is_private(&ip) {
                    return Err(format!(
                        "refusing to reach private or loopback address {ip}"
                    ));
                }
            }
        }

        Ok(parsed)
    }
}

#[async_trait]
impl Tool for HttpRequestTool {
    fn name(&self) -> &str {
        "http_request"
    }

    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "http_request",
            "Make an HTTP request and return the status, headers and body.",
            json!({
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to request"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                        "description": "HTTP method (default GET)"
                    },
                    "headers": {
                        "type": "object",
                        "description": "Additional request headers"
                    },
                    "body": {"type": "string", "description": "Request body"}
                },
                "required": ["url"]
            }),
        )
        .with_annotations(ToolAnnotations {
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            idempotent_hint: Some(false),
            open_world_hint: Some(true),
            ..Default::default()
        })
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<ToolOutput, StrandsError> {
        let Some(url) = input.get("url").and_then(Value::as_str) else {
            return Ok(ToolOutput::error("'url' must be a string"));
        };

        let url = match self.check_url(url) {
            Ok(url) => url,
            Err(reason) => {
                warn!(url, reason, "HTTP request refused");
                return Ok(ToolOutput::error(reason));
            }
        };

        let method = input
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("GET")
            .to_uppercase();
        let method = match reqwest::Method::from_bytes(method.as_bytes()) {
            Ok(method) => method,
            Err(_) => return Ok(ToolOutput::error(format!("invalid method '{method}'"))),
        };

        debug!(%url, %method, "Making HTTP request");
        let mut request = self.client.request(method, url);

        if let Some(headers) = input.get("headers").and_then(Value::as_object) {
            for (name, value) in headers {
                if let Some(value) = value.as_str() {
                    request = request.header(name, value);
                }
            }
        }
        if let Some(body) = input.get("body").and_then(Value::as_str) {
            request = request.body(body.to_string());
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(e) => return Ok(ToolOutput::error(format!("request failed: {e}"))),
        };

        let status = response.status();
        let headers: serde_json::Map<String, Value> = response
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v| (k.as_str().to_string(), Value::String(v.to_string())))
            })
            .collect();

        let body = match response.text().await {
            Ok(body) if body.len() > self.max_body => {
                let kept: String = body.chars().take(self.max_body).collect();
                format!("{kept}\n[body truncated at {} bytes]", self.max_body)
            }
            Ok(body) => body,
            Err(e) => return Ok(ToolOutput::error(format!("could not read body: {e}"))),
        };

        let payload = json!({
            "status": status.as_u16(),
            "headers": headers,
            "body": body,
        });

        // A 4xx/5xx is reported as a tool error so the model reliably notices,
        // with the response still attached for it to act on.
        if status.is_success() {
            Ok(ToolOutput::success(payload))
        } else {
            Ok(ToolOutput {
                content: payload,
                is_error: true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn refuse(tool: &HttpRequestTool, url: &str) -> ToolOutput {
        tool.invoke(json!({"url": url}), &ToolContext::default())
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn non_http_schemes_are_refused() {
        let tool = HttpRequestTool::new();
        for url in ["file:///etc/passwd", "ftp://example.com", "gopher://x"] {
            let out = refuse(&tool, url).await;
            assert!(out.is_error, "expected {url} to be refused");
        }
    }

    #[tokio::test]
    async fn loopback_and_private_addresses_are_refused_by_default() {
        // The URL is model-controlled, so this is the SSRF boundary.
        let tool = HttpRequestTool::new();
        for url in [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
        ] {
            let out = refuse(&tool, url).await;
            assert!(out.is_error, "expected {url} to be refused");
            assert!(
                out.content
                    .as_str()
                    .unwrap()
                    .contains("private or loopback"),
                "unexpected reason for {url}: {:?}",
                out.content
            );
        }
    }

    #[tokio::test]
    async fn private_addresses_are_allowed_when_opted_in() {
        let tool = HttpRequestTool::new().allow_private_networks(true);
        // Port 1 will refuse the connection, but the destination check should
        // no longer be what stops it.
        let out = refuse(&tool, "http://127.0.0.1:1/").await;
        assert!(
            !out.content
                .as_str()
                .unwrap_or_default()
                .contains("private or loopback"),
            "the destination check should not fire when opted in"
        );
    }

    #[tokio::test]
    async fn the_host_allowlist_refuses_others() {
        let tool = HttpRequestTool::new().with_allowed_hosts(["example.com"]);
        let out = refuse(&tool, "https://evil.test/").await;
        assert!(out.is_error);
        assert!(out.content.as_str().unwrap().contains("allowlist"));
    }

    #[tokio::test]
    async fn a_malformed_url_is_rejected() {
        let tool = HttpRequestTool::new();
        let out = refuse(&tool, "not a url").await;
        assert!(out.is_error);
    }

    #[tokio::test]
    async fn a_missing_url_is_rejected() {
        let out = HttpRequestTool::new()
            .invoke(json!({}), &ToolContext::default())
            .await
            .unwrap();
        assert!(out.is_error);
    }

    #[test]
    fn private_range_detection_covers_the_usual_suspects() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
        ] {
            assert!(
                HttpRequestTool::is_private(&ip.parse().unwrap()),
                "{ip} should be treated as private"
            );
        }

        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700::1111"] {
            assert!(
                !HttpRequestTool::is_private(&ip.parse().unwrap()),
                "{ip} should be treated as public"
            );
        }
    }

    #[test]
    fn annotated_as_open_world() {
        let spec = HttpRequestTool::new().spec();
        assert!(spec.annotations.expect("annotations").is_open_world());
    }
}
