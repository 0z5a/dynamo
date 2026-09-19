// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `PdAdapter` binding for the TRT-LLM context-first handoff.
//!
//! This is the only place the dispatch meets the sidecar's request lifecycle.
//! Everything before the generation leg is a pre-response error and can be
//! answered with an HTTP status; everything after has already begun the
//! response and cannot be.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{HeaderValue, Request, Response, StatusCode, header};
use tokio_util::sync::CancellationToken;

use crate::error::SidecarError;
use crate::metadata::PrefillEndpoint;
use crate::server::PdAdapter;
use crate::trtllm_context_first::{
    ContextFirstDispatcher, DispatchError, DispatchOutcome, LegFailure, RequestError,
    read_request_body,
};

/// Adapts the ordered dispatcher to the sidecar's adapter interface.
pub struct ContextFirstAdapter {
    dispatcher: Arc<ContextFirstDispatcher>,
    /// Bound on how long the client may take to send its body.
    client_body_timeout: Duration,
}

impl ContextFirstAdapter {
    pub fn new(dispatcher: Arc<ContextFirstDispatcher>, client_body_timeout: Duration) -> Self {
        Self {
            dispatcher,
            client_body_timeout,
        }
    }

    pub fn dispatcher(&self) -> &Arc<ContextFirstDispatcher> {
        &self.dispatcher
    }
}

/// Maps a dispatch failure onto the sidecar's error surface.
///
/// The distinction the manual asks for is preserved here: a connect failure, a
/// read stall, a total deadline, an oversized body, a non-success status and a
/// malformed handoff each produce their own code rather than one shared
/// timeout.
fn map_dispatch_error(error: DispatchError) -> SidecarError {
    match error {
        DispatchError::Request(source) => map_request_error(&source),
        DispatchError::Handoff { source } => SidecarError::adapter(
            StatusCode::BAD_GATEWAY,
            "trtllm_context_handoff_invalid",
            source.to_string(),
        ),
        DispatchError::Cancelled { leg } => SidecarError::adapter(
            StatusCode::BAD_GATEWAY,
            "trtllm_context_cancelled",
            format!("the client cancelled the request during the {leg} leg"),
        ),
        DispatchError::Leg(failure) => map_leg_failure(failure),
    }
}

fn map_request_error(error: &RequestError) -> SidecarError {
    match error {
        RequestError::BodyTooLarge { .. }
        | RequestError::MalformedJson { .. }
        | RequestError::NotAnObject
        | RequestError::ForbiddenField { .. } => SidecarError::adapter(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        RequestError::Handoff { source } => SidecarError::adapter(
            StatusCode::BAD_GATEWAY,
            "trtllm_context_handoff_invalid",
            source.to_string(),
        ),
    }
}

fn map_leg_failure(failure: LegFailure) -> SidecarError {
    let (status, code) = match failure {
        LegFailure::Connect { .. } => (StatusCode::BAD_GATEWAY, "trtllm_context_unreachable"),
        LegFailure::Unavailable { .. } => (StatusCode::BAD_GATEWAY, "trtllm_context_unreachable"),
        LegFailure::ReadStall { leg } => (
            StatusCode::GATEWAY_TIMEOUT,
            if leg == crate::trtllm_context_first::Leg::Context {
                "trtllm_context_read_stall"
            } else {
                "trtllm_generation_read_stall"
            },
        ),
        LegFailure::Deadline { leg } => (
            StatusCode::GATEWAY_TIMEOUT,
            if leg == crate::trtllm_context_first::Leg::Context {
                "trtllm_context_deadline_exceeded"
            } else {
                "trtllm_generation_deadline_exceeded"
            },
        ),
        LegFailure::Status { .. } => (StatusCode::BAD_GATEWAY, "trtllm_worker_status"),
        LegFailure::BodyTooLarge { .. } => {
            (StatusCode::BAD_GATEWAY, "trtllm_worker_response_too_large")
        }
    };
    SidecarError::adapter(status, code, failure.to_string())
}

/// Builds the response body the client sees for one successful dispatch.
fn response_for(outcome: &DispatchOutcome) -> Response<Body> {
    let bytes = outcome.body().as_slice().to_vec();
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )
        .body(Body::from(bytes))
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

#[async_trait]
impl PdAdapter for ContextFirstAdapter {
    async fn execute(
        &self,
        request: Request<Body>,
        prefill_endpoint: PrefillEndpoint,
        cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        // The selected prefill worker is the context leg's target. It is
        // recorded rather than guessed at: #13416 excludes EPP discovery, and
        // the transport's base URL is configured.
        tracing::debug!(
            prefill = %prefill_endpoint,
            "dispatching a context-first request"
        );

        let body = match tokio::time::timeout(
            self.client_body_timeout,
            read_request_body(request, self.dispatcher.limits().request_body_bytes),
        )
        .await
        {
            Err(_elapsed) => {
                return Err(SidecarError::adapter(
                    StatusCode::REQUEST_TIMEOUT,
                    "client_body_deadline_exceeded",
                    "the client did not finish sending its request body in time",
                ));
            }
            Ok(Err(source)) => return Err(map_request_error(&source)),
            Ok(Ok(body)) => body,
        };

        // The conversation identity is gateway-owned: a client cannot supply it,
        // because a forged one would let a request be routed as another's.
        let conversation_id = format!("dyn-{}", self.dispatcher.next_conversation_sequence());

        let outcome = self
            .dispatcher
            .dispatch(&body, &conversation_id, cancellation)
            .await
            .map_err(map_dispatch_error)?;

        tracing::info!(
            %conversation_id,
            correlation_id = outcome.correlation_id(),
            context_completed = outcome.context_completed(),
            body_bytes = outcome.body().len(),
            "context-first dispatch completed"
        );
        Ok(response_for(&outcome))
    }
}

/// Convenience constructor used by the binary.
pub fn trtllm_context_first_adapter(
    dispatcher: Arc<ContextFirstDispatcher>,
    client_body_timeout: Duration,
) -> ContextFirstAdapter {
    ContextFirstAdapter::new(dispatcher, client_body_timeout)
}
