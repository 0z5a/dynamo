// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use dynamo_epp_sidecar::{
    AdapterMode, Config, ContextFirstDispatcher, ContextFirstTransport, DisaggRequestIds,
    PdAdapter, TrtllmContextFirstContract, UnavailablePdAdapter, trtllm_context_first_adapter,
};
use reqwest::Client;
use tracing_subscriber::EnvFilter;

/// Builds the configured adapter.
///
/// A mode that cannot be constructed is a startup failure rather than a
/// fallback: silently proxying when the operator asked for an adapter would
/// hide a misconfiguration until the first P/D request.
fn build_adapter(config: &Config) -> anyhow::Result<Arc<dyn PdAdapter>> {
    match config.adapter_mode {
        AdapterMode::None => Ok(Arc::new(UnavailablePdAdapter)),
        AdapterMode::TrtllmContextFirst => {
            let context_first = config.context_first.as_ref().ok_or_else(|| {
                anyhow::anyhow!("adapter mode trtllm-context-first requires its settings")
            })?;
            let limits = context_first.limits;
            let client = Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(config.connect_timeout)
                .read_timeout(config.read_timeout)
                .build()?;
            let transport = ContextFirstTransport::new(
                client,
                context_first.context_engine_url.clone(),
                config.decode_engine_url.clone(),
                limits.response_body_bytes,
            );
            let ids = Arc::new(DisaggRequestIds::new(context_first.namespace));
            let dispatcher = Arc::new(ContextFirstDispatcher::new(
                TrtllmContextFirstContract::pinned(),
                Arc::new(transport),
                ids,
                limits,
            ));
            tracing::info!(
                context_engine = %context_first.context_engine_url,
                generation_engine = %config.decode_engine_url,
                protocol_revision = dispatcher.contract().protocol_revision,
                engine_tag = dispatcher.contract().engine_tag,
                engine_revision = dispatcher.contract().engine_revision,
                request_body_bytes = limits.request_body_bytes,
                handoff_body_bytes = limits.handoff_body_bytes,
                context_deadline_ms = limits.context_deadline.as_millis(),
                generation_deadline_ms = limits.generation_deadline.as_millis(),
                node_id = context_first.namespace.node_id(),
                process_id = context_first.namespace.process_id(),
                "TRT-LLM context-first adapter enabled"
            );
            Ok(Arc::new(trtllm_context_first_adapter(
                dispatcher,
                Duration::from_secs(30),
            )))
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    let config = Config::from_env()?;
    // The effective configuration is logged once at startup, so an operator can
    // see which adapter and which budgets are actually in force.
    tracing::info!(
        listen_addr = %config.listen_addr,
        decode_engine = %config.decode_engine_url,
        adapter_mode = config.adapter_mode.as_str(),
        connect_timeout_ms = config.connect_timeout.as_millis(),
        read_timeout_ms = config.read_timeout.as_millis(),
        drain_timeout_ms = config.drain_timeout.as_millis(),
        context_first_enabled = config.context_first.is_some(),
        "effective sidecar configuration"
    );
    let adapter = build_adapter(&config)?;
    dynamo_epp_sidecar::run(config, adapter).await
}
