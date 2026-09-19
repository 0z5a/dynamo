// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Url;

use crate::trtllm_context_first::{ContextFirstLimits, DisaggIdNamespace};

const SIDECAR_PORT_ENV: &str = "DYN_SIDECAR_PORT";
const ADAPTER_MODE_ENV: &str = "DYN_ADAPTER_MODE";
const CONTEXT_ENGINE_URL_ENV: &str = "DYN_CONTEXT_ENGINE_URL";
const CONTEXT_FIRST_REQUEST_BYTES_ENV: &str = "DYN_CONTEXT_FIRST_REQUEST_BYTES";
const CONTEXT_FIRST_HANDOFF_BYTES_ENV: &str = "DYN_CONTEXT_FIRST_HANDOFF_BYTES";
const CONTEXT_FIRST_RESPONSE_BYTES_ENV: &str = "DYN_CONTEXT_FIRST_RESPONSE_BYTES";
const CONTEXT_FIRST_CONTEXT_DEADLINE_MS_ENV: &str = "DYN_CONTEXT_FIRST_CONTEXT_DEADLINE_MS";
const CONTEXT_FIRST_GENERATION_DEADLINE_MS_ENV: &str = "DYN_CONTEXT_FIRST_GENERATION_DEADLINE_MS";
const CONTEXT_FIRST_NODE_ID_ENV: &str = "DYN_CONTEXT_FIRST_NODE_ID";
const CONTEXT_FIRST_PROCESS_ID_ENV: &str = "DYN_CONTEXT_FIRST_PROCESS_ID";
const DECODE_ENGINE_PORT_ENV: &str = "DYN_DECODE_ENGINE_PORT";
const CONNECT_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_CONNECT_TIMEOUT_MS";
const READ_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_READ_TIMEOUT_MS";
const DRAIN_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_DRAIN_TIMEOUT_MS";

const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_READ_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;
const DEFAULT_CONTEXT_FIRST_REQUEST_BYTES: usize = 1024 * 1024;
const DEFAULT_CONTEXT_FIRST_HANDOFF_BYTES: usize = 8 * 1024 * 1024;
const DEFAULT_CONTEXT_FIRST_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_CONTEXT_FIRST_CONTEXT_DEADLINE_MS: u64 = 60_000;
const DEFAULT_CONTEXT_FIRST_GENERATION_DEADLINE_MS: u64 = 300_000;
/// Upper bound on a leg deadline, so a mistyped value fails at startup instead
/// of holding a request for hours.
const MAX_LEG_DEADLINE: Duration = Duration::from_secs(3600);

/// Which P/D adapter the sidecar runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterMode {
    /// No adapter: requests without selected-prefill metadata are proxied.
    None,
    /// The TRT-LLM context-first handoff.
    TrtllmContextFirst,
}

impl AdapterMode {
    /// Parses the configured mode.
    ///
    /// An unknown value is an error rather than a fallback, so a typo cannot
    /// silently start the sidecar with no adapter.
    pub fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "none" => Ok(Self::None),
            "trtllm-context-first" => Ok(Self::TrtllmContextFirst),
            other => bail!(
                "{ADAPTER_MODE_ENV} must be \"none\" or \"trtllm-context-first\", got {other:?}"
            ),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::TrtllmContextFirst => "trtllm-context-first",
        }
    }
}

/// Settings for the TRT-LLM context-first adapter, present only when enabled.
#[derive(Debug, Clone)]
pub struct ContextFirstConfig {
    /// Worker that runs the context (prefill) leg.
    pub context_engine_url: Url,
    pub limits: ContextFirstLimits,
    /// Namespace that makes this process's request ids unique. Configured
    /// rather than derived, because two gateways sharing one namespace would
    /// collide.
    pub namespace: DisaggIdNamespace,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub decode_engine_url: Url,
    /// Maximum time allowed to establish a connection to the decode engine.
    pub connect_timeout: Duration,
    /// Maximum idle time between reads from a streaming decode response.
    pub read_timeout: Duration,
    /// Maximum time to drain active requests before forcing their streams closed.
    pub drain_timeout: Duration,
    /// Which adapter to run.
    pub adapter_mode: AdapterMode,
    /// Present exactly when `adapter_mode` is `TrtllmContextFirst`.
    pub context_first: Option<ContextFirstConfig>,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let sidecar_port = port_from_env(SIDECAR_PORT_ENV, 8000)?;
        let decode_engine_port = port_from_env(DECODE_ENGINE_PORT_ENV, 8001)?;
        let adapter_mode = match std::env::var_os(ADAPTER_MODE_ENV) {
            Some(raw) => AdapterMode::parse(
                &raw.into_string()
                    .map_err(|_| anyhow::anyhow!("{ADAPTER_MODE_ENV} must be valid UTF-8"))?,
            )?,
            None => AdapterMode::None,
        };
        let context_first = context_first_from_env(adapter_mode)?;
        Ok(Self {
            listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), sidecar_port),
            decode_engine_url: Url::parse(&format!("http://localhost:{decode_engine_port}"))
                .context("failed to construct local decode-engine URL")?,
            connect_timeout: duration_from_env(CONNECT_TIMEOUT_MS_ENV, DEFAULT_CONNECT_TIMEOUT_MS)?,
            read_timeout: duration_from_env(READ_TIMEOUT_MS_ENV, DEFAULT_READ_TIMEOUT_MS)?,
            drain_timeout: duration_from_env(DRAIN_TIMEOUT_MS_ENV, DEFAULT_DRAIN_TIMEOUT_MS)?,
            adapter_mode,
            context_first,
        })
    }
}

/// Builds the adapter settings selected by `adapter_mode`.
fn context_first_from_env(adapter_mode: AdapterMode) -> Result<Option<ContextFirstConfig>> {
    if adapter_mode != AdapterMode::TrtllmContextFirst {
        return Ok(None);
    }
    let raw_url = required_env(CONTEXT_ENGINE_URL_ENV)?;
    let context_engine_url = Url::parse(&raw_url)
        .with_context(|| format!("{CONTEXT_ENGINE_URL_ENV} must be a valid URL"))?;
    let limits = ContextFirstLimits::new(
        bytes_from_env(
            CONTEXT_FIRST_REQUEST_BYTES_ENV,
            DEFAULT_CONTEXT_FIRST_REQUEST_BYTES,
        )?,
        bytes_from_env(
            CONTEXT_FIRST_HANDOFF_BYTES_ENV,
            DEFAULT_CONTEXT_FIRST_HANDOFF_BYTES,
        )?,
        bytes_from_env(
            CONTEXT_FIRST_RESPONSE_BYTES_ENV,
            DEFAULT_CONTEXT_FIRST_RESPONSE_BYTES,
        )?,
        duration_from_env(
            CONTEXT_FIRST_CONTEXT_DEADLINE_MS_ENV,
            DEFAULT_CONTEXT_FIRST_CONTEXT_DEADLINE_MS,
        )?,
        duration_from_env(
            CONTEXT_FIRST_GENERATION_DEADLINE_MS_ENV,
            DEFAULT_CONTEXT_FIRST_GENERATION_DEADLINE_MS,
        )?,
    )
    .context("invalid context-first limits")?;
    if limits.context_deadline > MAX_LEG_DEADLINE || limits.generation_deadline > MAX_LEG_DEADLINE {
        bail!("context-first leg deadlines must not exceed {MAX_LEG_DEADLINE:?}");
    }
    let namespace = DisaggIdNamespace::new(
        u64_from_env(CONTEXT_FIRST_NODE_ID_ENV, 0)?,
        u64_from_env(CONTEXT_FIRST_PROCESS_ID_ENV, 0)?,
    )
    .context("invalid context-first id namespace")?;
    Ok(Some(ContextFirstConfig {
        context_engine_url,
        limits,
        namespace,
    }))
}

fn required_env(name: &str) -> Result<String> {
    let Some(raw) = std::env::var_os(name) else {
        bail!("{name} must be set when the context-first adapter is enabled");
    };
    raw.into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))
}

fn u64_from_env(name: &str, default: u64) -> Result<u64> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    raw.trim()
        .parse()
        .with_context(|| format!("{name} must be a non-negative integer"))
}

fn bytes_from_env(name: &str, default: usize) -> Result<usize> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let bytes: usize = raw
        .trim()
        .parse()
        .with_context(|| format!("{name} must be a byte count"))?;
    if bytes == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(bytes)
}

fn duration_from_env(name: &str, default_ms: u64) -> Result<Duration> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(Duration::from_millis(default_ms));
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let milliseconds: u64 = raw
        .parse()
        .with_context(|| format!("{name} must be a valid duration in milliseconds"))?;
    if milliseconds == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(Duration::from_millis(milliseconds))
}

fn port_from_env(name: &str, default: u16) -> Result<u16> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let port: u16 = raw
        .parse()
        .with_context(|| format!("{name} must be a valid TCP port"))?;
    if port == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trtllm_context_first::ConfigError;

    #[test]
    fn an_unknown_adapter_mode_is_rejected_rather_than_defaulted() {
        assert!(matches!(AdapterMode::parse("none"), Ok(AdapterMode::None)));
        assert!(matches!(AdapterMode::parse(""), Ok(AdapterMode::None)));
        assert!(matches!(
            AdapterMode::parse("trtllm-context-first"),
            Ok(AdapterMode::TrtllmContextFirst)
        ));
        // Case and surrounding whitespace are tolerated.
        assert!(matches!(
            AdapterMode::parse("  TRTLLM-Context-First  "),
            Ok(AdapterMode::TrtllmContextFirst)
        ));
        // Every near-miss is an error, never a silent fallback to no adapter.
        for typo in [
            "trtllm_context_first",
            "context-first",
            "trtllm-context",
            "enabled",
            "true",
            "trtllm-context-firsts",
        ] {
            assert!(
                AdapterMode::parse(typo).is_err(),
                "{typo} must be rejected, not defaulted"
            );
        }
    }

    #[test]
    fn disabling_the_adapter_yields_no_context_first_settings() {
        assert!(context_first_from_env(AdapterMode::None).unwrap().is_none());
    }

    #[test]
    fn the_mode_round_trips_through_its_configured_spelling() {
        for mode in [AdapterMode::None, AdapterMode::TrtllmContextFirst] {
            assert_eq!(AdapterMode::parse(mode.as_str()).unwrap(), mode);
        }
    }

    /// The effective configuration must be visible, because a mode that is on
    /// without its settings, or off when the operator expected it on, is the
    /// failure this selector exists to prevent.
    #[test]
    fn the_effective_configuration_reports_the_adapter_and_its_budgets() {
        let limits = ContextFirstLimits::new(
            4096,
            8192,
            16384,
            Duration::from_secs(7),
            Duration::from_secs(11),
        )
        .expect("valid limits");
        let config = ContextFirstConfig {
            context_engine_url: Url::parse("http://context:8000/").unwrap(),
            limits,
            namespace: DisaggIdNamespace::new(3, 4).unwrap(),
        };

        // Every knob the operator can set is readable back off the config, so a
        // startup log line can state what is actually in force.
        assert_eq!(config.limits.request_body_bytes, 4096);
        assert_eq!(config.limits.handoff_body_bytes, 8192);
        assert_eq!(config.limits.response_body_bytes, 16384);
        assert_eq!(config.limits.context_deadline, Duration::from_secs(7));
        assert_eq!(config.limits.generation_deadline, Duration::from_secs(11));
        assert_eq!(config.namespace.node_id(), 3);
        assert_eq!(config.namespace.process_id(), 4);
        assert!(
            config
                .context_engine_url
                .as_str()
                .starts_with("http://context")
        );
    }

    /// A zero response cap is refused as well, so no leg runs unbounded.
    #[test]
    fn a_zero_response_cap_is_refused() {
        assert!(matches!(
            ContextFirstLimits::new(1, 1, 0, Duration::from_secs(1), Duration::from_secs(1)),
            Err(ConfigError::ResponseBodyLimit)
        ));
    }
}
