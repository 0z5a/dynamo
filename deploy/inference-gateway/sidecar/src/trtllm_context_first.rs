// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! TRT-LLM context-first handoff: protocol fixtures and the global request id.
//!
//! Pinned upstream: release tag `v1.3.0rc24`
//! (`1cef02e901be43081b1ba6d4981e94ed3bd9c1e8`). See
//! `TRTLLM_CONTEXT_FIRST_PROTOCOL.md` for the field-by-field evidence.
//!
//! This module is deliberately pure: it validates fixtures and mints
//! identifiers without performing any I/O, so the protocol can be tested at the
//! CPU layer before any engine exists.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::Request;
use bytes::Bytes;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

/// Upstream release these field semantics were read from.
pub const VERIFIED_ENGINE_TAG: &str = "v1.3.0rc24";
/// Upstream commit SHA the tag resolves to.
pub const VERIFIED_ENGINE_REVISION: &str = "1cef02e901be43081b1ba6d4981e94ed3bd9c1e8";

/// Protocol revision of the context-first handoff implemented here.
pub const PROTOCOL_REVISION: &str = "trtllm-context-first-v1";

/// Endpoint both legs are sent to. The legs differ by which worker they are
/// sent to, not by endpoint shape.
pub const CHAT_COMPLETIONS_PATH: &str = "v1/chat/completions";

/// `request_type` values the pinned release accepts.
///
/// The third value, `context_and_generation`, is *not* context-first and is
/// rejected rather than mapped to a default.
pub const REQUEST_TYPE_CONTEXT_ONLY: &str = "context_only";
pub const REQUEST_TYPE_GENERATION_ONLY: &str = "generation_only";
pub const REQUEST_TYPE_CONTEXT_AND_GENERATION: &str = "context_and_generation";

/// A generation leg is sent only when the context leg reports one of these.
///
/// Source: `_GEN_PENDING_FINISH_REASONS` in
/// `tensorrt_llm/serve/openai_disagg_service.py`.
pub const GEN_PENDING_FINISH_REASONS: [&str; 2] = ["length", "not_finished"];

// ── Snowflake layout ─────────────────────────────────────────────────────────
//
// `0(1) | timestamp_ms(39) | node_id(8) | process_id(6) | counter(10)`,
// folded into the positive int64 range. Source:
// `tensorrt_llm/llmapi/disagg_utils.py`.

pub const TIMESTAMP_BITS: u32 = 39;
pub const NODE_ID_BITS: u32 = 8;
pub const PROCESS_ID_BITS: u32 = 6;
pub const COUNTER_BITS: u32 = 10;

/// `node_id` namespace size (256).
pub const NODE_ID_SPACE: u64 = 1 << NODE_ID_BITS;
/// `process_id` namespace size (64).
pub const PROCESS_ID_SPACE: u64 = 1 << PROCESS_ID_BITS;
const COUNTER_SPACE: u64 = 1 << COUNTER_BITS;
const TIMESTAMP_SPACE: u64 = 1 << TIMESTAMP_BITS;
// Field offsets, low to high: counter | process_id | node_id | timestamp.
// Source: `disagg_utils.py:558-563`
// (`timestamp << (node+process+counter) | node << (process+counter)
//   | process << counter | counter`), i.e. counter at bit 0, timestamp at bit 24
// up to bit 62 inclusive.
const PROCESS_SHIFT: u32 = COUNTER_BITS;
const NODE_SHIFT: u32 = COUNTER_BITS + PROCESS_ID_BITS;
const TIMESTAMP_SHIFT: u32 = COUNTER_BITS + PROCESS_ID_BITS + NODE_ID_BITS;

const _: () = assert!(TIMESTAMP_SHIFT + TIMESTAMP_BITS == 63);

/// Smallest value the folded id can take. Upstream reserves `[0, MIN_GLOBAL_ID)`
/// for local (non-global) ids: *"Local ids [0, MIN_GLOBAL_ID) and global disagg
/// ids [MIN_GLOBAL_ID, 2^63) are separated"* (`disagg_utils.py:522`).
pub const MIN_GLOBAL_ID: u64 = 1 << 40;
/// `2^63 - 1`; the folded range is `[MIN_GLOBAL_ID, MAX_INT64]`.
pub const MAX_INT64: u64 = (1 << 63) - 1;
/// `MAX_INT64 - MIN_GLOBAL_ID`, written as a literal sum so the constant does
/// not underflow.
const GLOBAL_ID_SPAN: u64 = MAX_INT64 - MIN_GLOBAL_ID;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IdError {
    #[error("node_id must be in [0, 256), got {0}")]
    NodeIdOutOfRange(u64),
    #[error("process_id must be in [0, 64), got {0}")]
    ProcessIdOutOfRange(u64),
    #[error("the resulting request id does not fit a positive int64: {0}")]
    NotAPositiveInt64(u64),
}

/// The `node_id` / `process_id` pair that namespaces id allocation.
///
/// Upstream defaults `node_id` to `uuid.getnode() % 256` and documents that
/// operators must set it manually if collisions occur. That default is *not*
/// adopted here: the deployment must state its namespace, so two gateways
/// cannot silently share one allocator slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisaggIdNamespace {
    node_id: u64,
    process_id: u64,
}

impl DisaggIdNamespace {
    pub fn new(node_id: u64, process_id: u64) -> Result<Self, IdError> {
        if node_id >= NODE_ID_SPACE {
            return Err(IdError::NodeIdOutOfRange(node_id));
        }
        if process_id >= PROCESS_ID_SPACE {
            return Err(IdError::ProcessIdOutOfRange(process_id));
        }
        Ok(Self {
            node_id,
            process_id,
        })
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn process_id(&self) -> u64 {
        self.process_id
    }
}

/// Composes one id from its parts.
///
/// The counter is masked, so it **wraps every 1024 ids within the same
/// millisecond** for one `(node_id, process_id)` slot. That wrap is the
/// collision boundary the tests pin.
pub fn compose_disagg_request_id(
    namespace: DisaggIdNamespace,
    timestamp_ms: u64,
    counter: u64,
) -> u64 {
    let raw = ((timestamp_ms & (TIMESTAMP_SPACE - 1)) << TIMESTAMP_SHIFT)
        | (namespace.node_id << NODE_SHIFT)
        | (namespace.process_id << PROCESS_SHIFT)
        | (counter & (COUNTER_SPACE - 1));
    // Fold into [MIN_GLOBAL_ID, i64::MAX], matching upstream.
    raw % GLOBAL_ID_SPAN + MIN_GLOBAL_ID
}

/// Splits an id back into its parts, so tests can assert the layout instead of
/// trusting the arithmetic.
pub fn decompose_disagg_request_id(id: u64) -> Result<(u64, u64, u64, u64), IdError> {
    if !(MIN_GLOBAL_ID..=MAX_INT64).contains(&id) {
        return Err(IdError::NotAPositiveInt64(id));
    }
    let raw = id - MIN_GLOBAL_ID;
    Ok((
        (raw >> TIMESTAMP_SHIFT) & (TIMESTAMP_SPACE - 1),
        (raw >> NODE_SHIFT) & (NODE_ID_SPACE - 1),
        (raw >> PROCESS_SHIFT) & (PROCESS_ID_SPACE - 1),
        raw & (COUNTER_SPACE - 1),
    ))
}

/// Mints ids for one gateway process.
///
/// The clock is monotonic, as upstream's is, so a wall-clock rollback cannot
/// corrupt the layout. A process restart resets both the monotonic base and the
/// counter, which is why the namespace must be unique per process.
#[derive(Debug)]
pub struct DisaggRequestIds {
    namespace: DisaggIdNamespace,
    started: Instant,
    counter: AtomicU64,
}

impl DisaggRequestIds {
    pub fn new(namespace: DisaggIdNamespace) -> Self {
        Self {
            namespace,
            started: Instant::now(),
            counter: AtomicU64::new(0),
        }
    }

    pub fn namespace(&self) -> DisaggIdNamespace {
        self.namespace
    }

    /// Elapsed milliseconds since this allocator started, masked to 39 bits.
    pub fn elapsed_ms(&self) -> u64 {
        let elapsed = self.started.elapsed().as_millis();
        u64::try_from(elapsed).unwrap_or(u64::MAX) & (TIMESTAMP_SPACE - 1)
    }

    /// Mints the next id, advancing the millisecond when the counter would wrap.
    ///
    /// The layout allows only 1024 ids per millisecond per namespace. Composing
    /// blindly would therefore reuse an id under burst, which breaks the
    /// two-leg association the id exists for. Instead the millisecond is bumped
    /// once the counter's low bits return to zero, so live minting never
    /// repeats. A sustained rate above 1024/ms moves the timestamp ahead of the
    /// clock rather than colliding; that is preferred over a duplicate, but it
    /// means the timestamp is a sequence, not a wall-clock reading.
    pub fn next(&self) -> i64 {
        loop {
            let timestamp_ms = self.elapsed_ms();
            let counter = self.counter.fetch_add(1, Ordering::Relaxed);
            if counter & (COUNTER_SPACE - 1) != 0 {
                return compose_disagg_request_id(self.namespace, timestamp_ms, counter) as i64;
            }
            // The wrap would collide with an id already issued in this
            // millisecond: retry with the next one.
            if self.advance_past(timestamp_ms) {
                continue;
            }
            return compose_disagg_request_id(self.namespace, timestamp_ms + 1, counter) as i64;
        }
    }

    /// Waits for the clock to move past `timestamp_ms`, returning whether it did.
    fn advance_past(&self, timestamp_ms: u64) -> bool {
        let deadline = std::time::Duration::from_millis(2);
        let started = Instant::now();
        while self.elapsed_ms() == timestamp_ms {
            if started.elapsed() >= deadline {
                return false;
            }
            std::hint::spin_loop();
        }
        true
    }

    /// Mints an id for an explicit millisecond, so tests can pin boundaries
    /// without waiting on a clock.
    pub fn next_at_ms(&self, timestamp_ms: u64) -> i64 {
        let counter = self.counter.fetch_add(1, Ordering::Relaxed);
        compose_disagg_request_id(self.namespace, timestamp_ms, counter) as i64
    }
}

/// The parsed handoff carried on a context response.
///
/// Field names and optionality come from `DisaggregatedParams` in the pinned
/// `tensorrt_llm/serve/openai_protocol.py`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContextHandoff {
    /// `disagg_request_id` as reported by the context leg. Authoritative when
    /// present: a context retry may replace the id the orchestrator minted.
    pub disagg_request_id: Option<i64>,
    /// `ctx_request_id` is a *different* identity and must survive untouched.
    pub ctx_request_id: Option<i64>,
    /// Engine-prepended first tokens. The sidecar must neither add nor drop
    /// these; it only validates and forwards them.
    pub first_gen_tokens: Option<Vec<i64>>,
    /// Opaque state, relayed byte-for-byte.
    pub encoded_opaque_state: Option<String>,
    /// `content` of the context leg's message, kept for the EOS/no-generation
    /// path where the context leg produced the whole answer.
    pub finish_reason: Option<String>,
    /// Token representation, exactly one of the two forms.
    pub prompt_token_ids: Option<Vec<i64>>,
    /// base64 int32 buffer alternative to `prompt_token_ids`.
    pub prompt_token_ids_b64: Option<String>,
    /// Protected field: the worker rejects it unless the request is signed.
    pub ctx_info_endpoint: Option<String>,
}

impl ContextHandoff {
    /// Whether a generation leg should be sent at all.
    ///
    /// The pinned orchestrator deletes `disaggregated_params` and stops when
    /// the context leg's `finish_reason` is anything else, because the context
    /// leg already produced the complete answer. That is a terminal state, not
    /// a failure.
    pub fn needs_generation(&self) -> bool {
        self.finish_reason
            .as_deref()
            .is_some_and(|reason| GEN_PENDING_FINISH_REASONS.contains(&reason))
    }

    /// The single token representation to forward, rejecting a request that
    /// sets both forms.
    pub fn prompt_tokens(&self) -> Result<PromptTokens<'_>, HandoffError> {
        match (&self.prompt_token_ids, &self.prompt_token_ids_b64) {
            (Some(_), Some(_)) => Err(HandoffError::ConflictingPromptTokenForms),
            (None, None) => Err(HandoffError::MissingPromptTokens),
            (Some(ids), None) => Ok(PromptTokens::Ids(ids)),
            (None, Some(encoded)) => Ok(PromptTokens::Base64(encoded)),
        }
    }
}

/// The prompt-token representation chosen for the generation leg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTokens<'a> {
    Ids(&'a [i64]),
    /// Relayed verbatim: never decoded and re-encoded.
    Base64(&'a str),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HandoffError {
    #[error("context response carries no choices")]
    NoChoices,
    #[error("context response choice carries no disaggregated_params")]
    NoDisaggregatedParams,
    #[error("context disaggregated_params.request_type is {found:?}, expected \"{expected}\"")]
    UnexpectedRequestType {
        found: Option<String>,
        expected: &'static str,
    },
    #[error("context response sets both prompt_token_ids and prompt_token_ids_b64")]
    ConflictingPromptTokenForms,
    #[error("context response carries no prompt token representation")]
    MissingPromptTokens,
    #[error("context response field {field} has the wrong JSON type")]
    WrongFieldType { field: &'static str },
}

fn optional_i64(object: &Value, field: &'static str) -> Result<Option<i64>, HandoffError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or(HandoffError::WrongFieldType { field }),
    }
}

fn optional_i64_array(
    object: &Value,
    field: &'static str,
) -> Result<Option<Vec<i64>>, HandoffError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| item.as_i64().ok_or(HandoffError::WrongFieldType { field }))
            .collect::<Result<Vec<_>, _>>()
            .map(Some),
        Some(_) => Err(HandoffError::WrongFieldType { field }),
    }
}

fn optional_string(object: &Value, field: &'static str) -> Result<Option<String>, HandoffError> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(HandoffError::WrongFieldType { field }),
    }
}

/// Parses a context response body into the handoff the generation leg needs.
///
/// The handoff lives at `choices[0].disaggregated_params`; the prompt tokens
/// live on the response object itself. There is no top-level
/// `kv_transfer_params`.
pub fn parse_context_response(body: &[u8]) -> Result<ContextHandoff, HandoffError> {
    let parsed: Value =
        serde_json::from_slice(body).map_err(|_| HandoffError::NoDisaggregatedParams)?;

    let choice = parsed
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or(HandoffError::NoChoices)?;

    let params = choice
        .get("disaggregated_params")
        .filter(|value| !value.is_null())
        .ok_or(HandoffError::NoDisaggregatedParams)?;

    let request_type = optional_string(params, "request_type")?;
    if request_type.as_deref() != Some(REQUEST_TYPE_CONTEXT_ONLY) {
        return Err(HandoffError::UnexpectedRequestType {
            found: request_type,
            expected: REQUEST_TYPE_CONTEXT_ONLY,
        });
    }

    let handoff = ContextHandoff {
        disagg_request_id: optional_i64(params, "disagg_request_id")?,
        ctx_request_id: optional_i64(params, "ctx_request_id")?,
        first_gen_tokens: optional_i64_array(params, "first_gen_tokens")?,
        encoded_opaque_state: optional_string(params, "encoded_opaque_state")?,
        finish_reason: optional_string(choice, "finish_reason")?,
        prompt_token_ids: optional_i64_array(&parsed, "prompt_token_ids")?,
        prompt_token_ids_b64: optional_string(&parsed, "prompt_token_ids_b64")?,
        ctx_info_endpoint: optional_string(params, "ctx_info_endpoint")?,
    };
    Ok(handoff)
}

// ── Request construction ─────────────────────────────────────────────────────

/// Fields an EPP client must not set: they are gateway-owned orchestration
/// state, and honouring them would let a client bypass the selected worker or
/// forge the correlation id.
///
/// `prompt_token_ids_b64` is included because the pinned protocol documents it
/// as *"Not for clients"* — it is the orchestrator's relay form.
const CLIENT_FORBIDDEN_FIELDS: [&str; 4] = [
    "disaggregated_params",
    "prompt_token_ids_b64",
    "ctx_info_endpoint",
    "encoded_opaque_state",
];

#[derive(Debug, thiserror::Error)]
pub enum RequestError {
    #[error("request body exceeds the configured limit of {limit} bytes")]
    BodyTooLarge { limit: usize },
    #[error("request body is not valid JSON: {source}")]
    MalformedJson { source: serde_json::Error },
    #[error("request body must be a JSON object")]
    NotAnObject,
    #[error("client-supplied {field} is not allowed; the gateway owns orchestration state")]
    ForbiddenField { field: &'static str },
    #[error("streaming responses are not supported by the context-first adapter")]
    UnsupportedStreaming,
    #[error("handoff rejected: {source}")]
    Handoff { source: HandoffError },
}

impl From<HandoffError> for RequestError {
    fn from(source: HandoffError) -> Self {
        Self::Handoff { source }
    }
}

/// Supplies the correlation id for one request. Injected so tests can pin
/// boundaries without a clock, and so the allocator stays swappable.
pub trait RequestIds: Send + Sync {
    fn mint(&self) -> i64;
}

impl RequestIds for DisaggRequestIds {
    fn mint(&self) -> i64 {
        self.next()
    }
}

/// One constructed leg: the exact path and JSON body to send.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedLeg {
    pub path: &'static str,
    pub correlation_id: i64,
    pub body: Bytes,
    /// Header the leg must be sent with, when the protocol requires one.
    pub auth_header: Option<(&'static str, String)>,
}

impl PreparedLeg {
    /// Reads a top-level field back from the serialized body, so a
    /// serialization mistake cannot pass unnoticed.
    pub fn field(&self, name: &str) -> Option<Value> {
        serde_json::from_slice::<Value>(&self.body)
            .ok()
            .and_then(|value| value.get(name).cloned())
    }
}

/// The context leg derived from a client request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedContextRequest {
    pub leg: PreparedLeg,
    /// Echoed back on the generation leg when the context worker does not
    /// report its own `conversation_id`.
    pub conversation_id: String,
}

/// The generation leg derived from a validated context handoff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedGenerationRequest {
    pub leg: PreparedLeg,
    /// `None` when the context leg already produced the whole answer, in which
    /// case no generation leg exists at all.
    pub correlation_id: Option<i64>,
}

/// Rejects a client body that carries gateway-owned orchestration state.
pub fn reject_client_orchestration(
    object: &serde_json::Map<String, Value>,
) -> Result<(), RequestError> {
    for field in CLIENT_FORBIDDEN_FIELDS {
        if object.get(field).is_some_and(|value| !value.is_null()) {
            return Err(RequestError::ForbiddenField { field });
        }
    }
    Ok(())
}

/// Builds the context leg for one client request.
///
/// The context leg is forced non-streaming and carries `context_only`
/// orchestration state. Sampling is left exactly as the client sent it: the
/// pinned orchestrator rewrites no sampling field, so neither does this.
pub fn prepare_context_request(
    body: &[u8],
    limit: usize,
    conversation_id: &str,
    ids: &dyn RequestIds,
) -> Result<PreparedContextRequest, RequestError> {
    if body.len() > limit {
        return Err(RequestError::BodyTooLarge { limit });
    }
    let parsed: Value =
        serde_json::from_slice(body).map_err(|source| RequestError::MalformedJson { source })?;
    let Value::Object(mut object) = parsed else {
        return Err(RequestError::NotAnObject);
    };
    reject_client_orchestration(&object)?;

    // Rejected rather than silently downgraded: the adapter buffers the
    // generation leg and returns one JSON body, so answering a streaming
    // request with it would hand the client a body its SSE parser cannot read.
    if object.get("stream") == Some(&Value::Bool(true)) {
        return Err(RequestError::UnsupportedStreaming);
    }

    let correlation_id = ids.mint();
    object.insert("stream".to_string(), Value::Bool(false));
    object.insert("stream_options".to_string(), Value::Null);
    object.insert(
        "disaggregated_params".to_string(),
        serde_json::json!({
            "request_type": REQUEST_TYPE_CONTEXT_ONLY,
            "disagg_request_id": correlation_id,
            "conversation_id": conversation_id,
        }),
    );

    let encoded =
        serde_json::to_vec(&object).map_err(|source| RequestError::MalformedJson { source })?;
    Ok(PreparedContextRequest {
        leg: PreparedLeg {
            path: CHAT_COMPLETIONS_PATH,
            correlation_id,
            body: Bytes::from(encoded),
            auth_header: None,
        },
        conversation_id: conversation_id.to_string(),
    })
}

/// Builds the generation leg from a validated context handoff.
///
/// Returns `correlation_id: None` when the context leg already finished: the
/// pinned orchestrator sends no generation request in that case, and inventing
/// one would answer the client twice.
///
/// The original client body is reused so sampling, tools and response format
/// survive unchanged; only the prompt representation and the orchestration
/// state are replaced.
pub fn prepare_generation_request(
    client_body: &[u8],
    handoff: &ContextHandoff,
    fallback_id: i64,
    conversation_id: &str,
    auth_key: Option<&[u8]>,
) -> Result<PreparedGenerationRequest, RequestError> {
    let parsed: Value = serde_json::from_slice(client_body)
        .map_err(|source| RequestError::MalformedJson { source })?;
    let Value::Object(mut object) = parsed else {
        return Err(RequestError::NotAnObject);
    };

    if !handoff.needs_generation() {
        return Ok(PreparedGenerationRequest {
            leg: PreparedLeg {
                path: CHAT_COMPLETIONS_PATH,
                correlation_id: fallback_id,
                body: Bytes::new(),
                auth_header: None,
            },
            correlation_id: None,
        });
    }

    // The context worker's id is authoritative when it reports one, because a
    // context retry may have replaced the id the gateway minted.
    let correlation_id = handoff.disagg_request_id.unwrap_or(fallback_id);

    match handoff.prompt_tokens()? {
        PromptTokens::Ids(ids) => {
            object.insert(
                "prompt_token_ids".to_string(),
                serde_json::to_value(ids)
                    .map_err(|source| RequestError::MalformedJson { source })?,
            );
            object.remove("prompt_token_ids_b64");
        }
        // Relayed verbatim: never decoded and re-encoded.
        PromptTokens::Base64(encoded) => {
            object.insert(
                "prompt_token_ids_b64".to_string(),
                Value::String(encoded.to_string()),
            );
            object.remove("prompt_token_ids");
        }
    }

    let mut params = serde_json::Map::new();
    params.insert(
        "request_type".to_string(),
        Value::String(REQUEST_TYPE_GENERATION_ONLY.to_string()),
    );
    params.insert("disagg_request_id".to_string(), Value::from(correlation_id));
    params.insert(
        "conversation_id".to_string(),
        Value::String(conversation_id.to_string()),
    );
    if let Some(ctx_request_id) = handoff.ctx_request_id {
        // A different identity from `disagg_request_id`; never overwritten.
        params.insert("ctx_request_id".to_string(), Value::from(ctx_request_id));
    }
    if let Some(first_gen_tokens) = &handoff.first_gen_tokens {
        // Forwarded for validation only. The generation worker prepends these
        // itself, so the sidecar must not also prepend them.
        params.insert(
            "first_gen_tokens".to_string(),
            serde_json::to_value(first_gen_tokens)
                .map_err(|source| RequestError::MalformedJson { source })?,
        );
    }
    if let Some(opaque) = &handoff.encoded_opaque_state {
        params.insert(
            "encoded_opaque_state".to_string(),
            Value::String(opaque.clone()),
        );
    }
    object.insert("disaggregated_params".to_string(), Value::Object(params));

    // A protected handoff field makes the signature mandatory: the generation
    // worker rejects request-supplied `encoded_opaque_state` or
    // `ctx_info_endpoint` when the internal handoff is unsigned.
    let auth_header = auth_key
        .and_then(|key| {
            sign_disagg_handoff(
                key,
                handoff.encoded_opaque_state.as_deref(),
                handoff.ctx_info_endpoint.as_deref(),
            )
        })
        .map(|signature| (INTERNAL_DISAGG_AUTH_HEADER, signature));

    let encoded =
        serde_json::to_vec(&object).map_err(|source| RequestError::MalformedJson { source })?;
    Ok(PreparedGenerationRequest {
        leg: PreparedLeg {
            path: CHAT_COMPLETIONS_PATH,
            correlation_id,
            body: Bytes::from(encoded),
            auth_header,
        },
        correlation_id: Some(correlation_id),
    })
}

/// The pinned revision, as a value a caller can log and a test can assert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtllmContextFirstContract {
    pub engine_tag: &'static str,
    pub engine_revision: &'static str,
    pub protocol_revision: &'static str,
}

impl TrtllmContextFirstContract {
    pub const fn pinned() -> Self {
        Self {
            engine_tag: VERIFIED_ENGINE_TAG,
            engine_revision: VERIFIED_ENGINE_REVISION,
            protocol_revision: PROTOCOL_REVISION,
        }
    }

    /// The endpoints both legs use.
    pub const fn chat_completions_path(&self) -> &'static str {
        CHAT_COMPLETIONS_PATH
    }

    /// True when `request_type` is the combined mode this adapter does not
    /// implement and must reject rather than approximate.
    pub fn is_unsupported_request_type(&self, request_type: &str) -> bool {
        request_type == REQUEST_TYPE_CONTEXT_AND_GENERATION
    }
}

/// Reads a client request body under an explicit byte cap.
///
/// The cap is enforced while reading, so an oversized or unfinished body cannot
/// make the sidecar allocate without bound. A transport error is reported as a
/// client error, because the client is the one that failed to deliver.
pub async fn read_request_body(
    request: Request<Body>,
    limit: usize,
) -> Result<Vec<u8>, RequestError> {
    read_bounded_body(request.into_body(), limit)
        .await
        .map_err(|source| RequestError::MalformedJson { source })
}

/// Streams a body, rejecting it as soon as it exceeds `limit`.
async fn read_bounded_body(body: Body, limit: usize) -> Result<Vec<u8>, serde_json::Error> {
    let mut stream = body.into_data_stream();
    let mut collected: Vec<u8> = Vec::new();
    while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
        let chunk = chunk
            .map_err(|error| serde_json::Error::io(std::io::Error::other(error.to_string())))?;
        if collected.len() + chunk.len() > limit {
            return Err(serde_json::Error::io(std::io::Error::other(format!(
                "body exceeds the configured limit of {limit} bytes"
            ))));
        }
        collected.extend_from_slice(&chunk);
    }
    Ok(collected)
}

// ── Internal disaggregation authentication ───────────────────────────────────

/// Header the generation worker expects for protected handoff fields.
///
/// Pinned source: `INTERNAL_DISAGG_AUTH_HEADER` in
/// `tensorrt_llm/serve/disagg_auth.py`.
pub const INTERNAL_DISAGG_AUTH_HEADER: &str = "x-trtllm-disagg-auth";

type HmacSha256 = Hmac<Sha256>;

/// Handoff fields the worker refuses unless the request is signed.
///
/// Pinned source: `_INTERNAL_DISAGG_AUTH_FIELDS`.
pub const INTERNAL_DISAGG_AUTH_FIELDS: [&str; 2] = ["encoded_opaque_state", "ctx_info_endpoint"];

/// Computes the signature the generation worker validates.
///
/// The pinned scheme is HMAC-SHA256 over
/// `json.dumps({field: value}, sort_keys=True, separators=(",", ":"))` for the
/// protected fields that are present, hex-encoded and prefixed with `sha256=`.
/// `None` fields are included as JSON `null`, matching the producer, and
/// `ctx_info_endpoint` is canonicalised from a list to its first element.
///
/// Returns `None` when the handoff carries no protected field, because the
/// worker only requires a signature in that case.
pub fn sign_disagg_handoff(
    key: &[u8],
    encoded_opaque_state: Option<&str>,
    ctx_info_endpoint: Option<&str>,
) -> Option<String> {
    if encoded_opaque_state.is_none() && ctx_info_endpoint.is_none() {
        return None;
    }
    let payload = serde_json::json!({
        "encoded_opaque_state": encoded_opaque_state,
        "ctx_info_endpoint": ctx_info_endpoint,
    });
    // Sorted-key, compact separators: the two options the producer passes.
    let canonical = serde_json::to_vec(&payload).ok()?;
    let mut mac = HmacSha256::new_from_slice(key).ok()?;
    mac.update(&canonical);
    Some(format!(
        "sha256={}",
        hex_encode(&mac.finalize().into_bytes())
    ))
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, byte| {
            let _ = write!(acc, "{byte:02x}");
            acc
        })
}

// ── Dispatch (A4-C3) ─────────────────────────────────────────────────────────

/// Which leg a failure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leg {
    Context,
    Generation,
}

impl Leg {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Context => "context",
            Self::Generation => "generation",
        }
    }
}

impl std::fmt::Display for Leg {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why a leg did not produce a usable response.
///
/// These stay distinct on purpose: collapsing a connect failure, a read stall,
/// a total deadline and an oversized body into one "timeout" would make a
/// misconfigured endpoint indistinguishable from a slow engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LegFailure {
    #[error("could not connect to the {leg} worker")]
    Connect { leg: Leg },
    #[error("the {leg} worker stopped sending data before finishing")]
    ReadStall { leg: Leg },
    #[error("the {leg} leg exceeded its total deadline")]
    Deadline { leg: Leg },
    #[error("the {leg} worker returned HTTP {status}")]
    Status { leg: Leg, status: u16 },
    #[error("the {leg} response body exceeded the configured limit of {limit} bytes")]
    BodyTooLarge { leg: Leg, limit: usize },
    #[error("the {leg} worker could not be reached")]
    Unavailable { leg: Leg },
}

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error(transparent)]
    Request(#[from] RequestError),
    #[error(transparent)]
    Leg(#[from] LegFailure),
    #[error("context handoff rejected: {source}")]
    Handoff { source: HandoffError },
    #[error("the client cancelled the request during the {leg} leg")]
    Cancelled { leg: Leg },
}

impl From<HandoffError> for DispatchError {
    fn from(source: HandoffError) -> Self {
        Self::Handoff { source }
    }
}

/// A response body owned until dropped, so cancelling the request also tears
/// the stream down instead of orphaning it.
#[derive(Debug)]
pub struct OwnedBody {
    bytes: Vec<u8>,
    cancellation: CancellationToken,
}

impl OwnedBody {
    pub fn new(bytes: Vec<u8>, cancellation: CancellationToken) -> Self {
        Self {
            bytes,
            cancellation,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// The token that tears the stream down when this body is dropped.
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }
}

/// How one leg is executed. Behind a trait so ordering, error taxonomy and
/// cancellation are testable with no network and no engine.
#[async_trait::async_trait]
pub trait LegTransport: Send + Sync {
    /// Sends the context leg and returns the raw response body.
    async fn send_context(
        &self,
        leg: PreparedLeg,
        cancellation: CancellationToken,
        deadline: Duration,
    ) -> Result<Vec<u8>, LegFailure>;

    /// Sends the generation leg. The returned body is owned by the caller.
    async fn send_generation(
        &self,
        leg: PreparedLeg,
        cancellation: CancellationToken,
        deadline: Duration,
    ) -> Result<OwnedBody, LegFailure>;
}

/// How a completed dispatch ended.
#[derive(Debug)]
pub enum DispatchOutcome {
    /// The context leg produced the complete answer, so no generation ran.
    /// A terminal success: not a failure, and not a gap.
    ContextCompleted {
        body: OwnedBody,
        correlation_id: i64,
    },
    /// The generation leg ran; its body belongs to the response.
    Generated {
        body: OwnedBody,
        correlation_id: i64,
    },
}

impl DispatchOutcome {
    pub fn body(&self) -> &OwnedBody {
        match self {
            Self::ContextCompleted { body, .. } | Self::Generated { body, .. } => body,
        }
    }

    pub fn correlation_id(&self) -> i64 {
        match self {
            Self::ContextCompleted { correlation_id, .. }
            | Self::Generated { correlation_id, .. } => *correlation_id,
        }
    }

    /// True when no generation leg was sent.
    pub fn context_completed(&self) -> bool {
        matches!(self, Self::ContextCompleted { .. })
    }
}

/// Per-leg budgets and body caps.
#[derive(Debug, Clone, Copy)]
pub struct ContextFirstLimits {
    /// Maximum accepted client request body.
    pub request_body_bytes: usize,
    /// Maximum accepted context response body. Separate from the request cap
    /// because the handoff can be far larger than the request.
    pub handoff_body_bytes: usize,
    /// Maximum accepted body for any leg response. Bounds what a broken or
    /// hostile worker can make the sidecar hold.
    pub response_body_bytes: usize,
    /// Total deadline for the context leg: bounds the whole leg, not one gap.
    pub context_deadline: Duration,
    /// Total deadline for the generation leg.
    pub generation_deadline: Duration,
}

#[derive(Debug, Clone, Copy, thiserror::Error)]
pub enum ConfigError {
    #[error("request_body_bytes must be greater than zero")]
    RequestBodyLimit,
    #[error("handoff_body_bytes must be greater than zero")]
    HandoffBodyLimit,
    #[error("response_body_bytes must be greater than zero")]
    ResponseBodyLimit,
    #[error("leg deadlines must be greater than zero")]
    Deadline,
}

impl ContextFirstLimits {
    pub fn new(
        request_body_bytes: usize,
        handoff_body_bytes: usize,
        response_body_bytes: usize,
        context_deadline: Duration,
        generation_deadline: Duration,
    ) -> Result<Self, ConfigError> {
        if request_body_bytes == 0 {
            return Err(ConfigError::RequestBodyLimit);
        }
        if handoff_body_bytes == 0 {
            return Err(ConfigError::HandoffBodyLimit);
        }
        if response_body_bytes == 0 {
            return Err(ConfigError::ResponseBodyLimit);
        }
        if context_deadline.is_zero() || generation_deadline.is_zero() {
            return Err(ConfigError::Deadline);
        }
        Ok(Self {
            request_body_bytes,
            handoff_body_bytes,
            response_body_bytes,
            context_deadline,
            generation_deadline,
        })
    }
}

/// Runs the ordered context-first dispatch.
pub struct ContextFirstDispatcher {
    contract: TrtllmContextFirstContract,
    transport: Arc<dyn LegTransport>,
    ids: Arc<dyn RequestIds>,
    limits: ContextFirstLimits,
    conversations: AtomicU64,
    /// Shared secret for the internal handoff signature. `None` means unsigned
    /// handoffs, which the pinned worker only tolerates for a handoff that
    /// carries no protected field.
    internal_auth_key: Option<Vec<u8>>,
}

impl ContextFirstDispatcher {
    pub fn new(
        contract: TrtllmContextFirstContract,
        transport: Arc<dyn LegTransport>,
        ids: Arc<dyn RequestIds>,
        limits: ContextFirstLimits,
        internal_auth_key: Option<Vec<u8>>,
    ) -> Self {
        Self {
            contract,
            transport,
            ids,
            limits,
            conversations: AtomicU64::new(0),
            internal_auth_key,
        }
    }

    /// Whether handoffs are signed. Logged at startup so an operator can see
    /// that a protected handoff will be accepted.
    pub fn signs_handoffs(&self) -> bool {
        self.internal_auth_key.is_some()
    }

    /// Mints the gateway-owned conversation identity for one request.
    ///
    /// Uniqueness only has to hold within this process for the lifetime of a
    /// request, which is what the pinned context leg uses it for.
    pub fn next_conversation_sequence(&self) -> u64 {
        self.conversations.fetch_add(1, Ordering::Relaxed)
    }

    pub fn contract(&self) -> TrtllmContextFirstContract {
        self.contract
    }

    pub fn limits(&self) -> ContextFirstLimits {
        self.limits
    }

    /// Runs both legs in order for one client request.
    ///
    /// The context leg completes and its handoff validates before the
    /// generation leg exists, so a context failure can never start generation.
    /// Every failure before that point is therefore answerable with an HTTP
    /// status; a failure during the generation leg happens after the response
    /// has begun and cannot be.
    pub async fn dispatch(
        &self,
        client_body: &[u8],
        conversation_id: &str,
        cancellation: CancellationToken,
    ) -> Result<DispatchOutcome, DispatchError> {
        if cancellation.is_cancelled() {
            return Err(DispatchError::Cancelled { leg: Leg::Context });
        }

        let context = prepare_context_request(
            client_body,
            self.limits.request_body_bytes,
            conversation_id,
            self.ids.as_ref(),
        )?;
        let minted_id = context.leg.correlation_id;

        let raw_context = self
            .transport
            .send_context(
                context.leg,
                cancellation.clone(),
                self.limits.context_deadline,
            )
            .await?;

        // The handoff cap is checked before parsing, so a hostile or broken
        // worker cannot make the sidecar allocate without bound.
        if raw_context.len() > self.limits.handoff_body_bytes {
            return Err(LegFailure::BodyTooLarge {
                leg: Leg::Context,
                limit: self.limits.handoff_body_bytes,
            }
            .into());
        }

        let handoff = parse_context_response(&raw_context)?;
        let resolved_id = handoff.disagg_request_id.unwrap_or(minted_id);

        if !handoff.needs_generation() {
            // The context leg produced the whole answer: no generation leg, and
            // no added token.
            return Ok(DispatchOutcome::ContextCompleted {
                body: OwnedBody::new(raw_context, cancellation),
                correlation_id: resolved_id,
            });
        }

        let generation = prepare_generation_request(
            client_body,
            &handoff,
            resolved_id,
            conversation_id,
            self.internal_auth_key.as_deref(),
        )?;

        if cancellation.is_cancelled() {
            return Err(DispatchError::Cancelled {
                leg: Leg::Generation,
            });
        }

        let body = self
            .transport
            .send_generation(
                generation.leg,
                cancellation.clone(),
                self.limits.generation_deadline,
            )
            .await?;

        Ok(DispatchOutcome::Generated {
            body,
            correlation_id: resolved_id,
        })
    }
}

/// Reads the correlation id out of a serialized leg, so an operator can
/// correlate both legs from a log line without parsing bodies by hand.
pub fn correlation_id_of(body: &[u8]) -> Option<i64> {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("disaggregated_params")?
                .get("disagg_request_id")?
                .as_i64()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn namespace() -> DisaggIdNamespace {
        DisaggIdNamespace::new(7, 3).expect("valid namespace")
    }

    #[test]
    fn ids_are_folded_into_the_global_range() {
        // A timestamp far beyond the field width must still land in
        // [MIN_GLOBAL_ID, MAX_INT64] rather than collapsing to a small value.
        for timestamp in [0u64, 1, 1 << 38, u64::MAX] {
            let id = compose_disagg_request_id(namespace(), timestamp, 0);
            assert!(
                (MIN_GLOBAL_ID..=MAX_INT64).contains(&id),
                "id {id} left the global range for timestamp {timestamp}"
            );
        }
        assert_eq!(MIN_GLOBAL_ID, 1 << 40);
    }

    #[test]
    fn id_layout_round_trips_and_stays_a_positive_int64() {
        let id = compose_disagg_request_id(namespace(), 123_456, 42);
        assert!(id > 0, "id must be positive, got {id}");
        assert!(id <= i64::MAX as u64);
        let (timestamp, node, process, counter) = decompose_disagg_request_id(id).unwrap();
        assert_eq!(timestamp, 123_456);
        assert_eq!(node, 7);
        assert_eq!(process, 3);
        assert_eq!(counter, 42);
    }

    #[test]
    fn namespace_bounds_are_enforced_and_never_wrapped() {
        assert_eq!(
            DisaggIdNamespace::new(256, 0).unwrap_err(),
            IdError::NodeIdOutOfRange(256)
        );
        assert_eq!(
            DisaggIdNamespace::new(0, 64).unwrap_err(),
            IdError::ProcessIdOutOfRange(64)
        );
        assert!(DisaggIdNamespace::new(255, 63).is_ok());
    }

    /// Distinct namespaces in the same millisecond with the same counter must
    /// not collide; that is the whole point of the namespace pair.
    #[test]
    fn distinct_namespaces_do_not_collide_within_one_millisecond() {
        let mut seen = std::collections::HashSet::new();
        for node in 0..NODE_ID_SPACE {
            for process in 0..PROCESS_ID_SPACE {
                let ns = DisaggIdNamespace::new(node, process).unwrap();
                let id = compose_disagg_request_id(ns, 1_000, 0);
                assert!(
                    seen.insert(id),
                    "collision at node={node} process={process}"
                );
            }
        }
        assert_eq!(seen.len(), (NODE_ID_SPACE * PROCESS_ID_SPACE) as usize);
    }

    /// The documented collision boundary: the 10-bit counter wraps at 1024
    /// within a single millisecond for one namespace.
    #[test]
    fn counter_wraps_at_1024_within_a_millisecond() {
        let ns = namespace();
        // Raw composition wraps: 1024 aliases 0 in the same millisecond. This
        // is the layout's documented capacity limit, and the reason the live
        // allocator must not compose blindly.
        let first = compose_disagg_request_id(ns, 5_000, 0);
        let last = compose_disagg_request_id(ns, 5_000, 1023);
        let wrapped = compose_disagg_request_id(ns, 5_000, 1024);
        assert_ne!(first, last);
        assert_eq!(
            first, wrapped,
            "raw composition wraps at 1024 in the same millisecond"
        );
        // A different millisecond separates them again.
        assert_ne!(first, compose_disagg_request_id(ns, 5_001, 0));
    }

    /// Overflowing the 39-bit timestamp must wrap inside the field rather than
    /// bleed into node_id.
    #[test]
    fn timestamp_field_wraps_without_corrupting_the_namespace() {
        let ns = namespace();
        let id = compose_disagg_request_id(ns, TIMESTAMP_SPACE + 9, 0);
        let (timestamp, node, process, _) = decompose_disagg_request_id(id).unwrap();
        assert_eq!(timestamp, 9);
        assert_eq!(node, 7);
        assert_eq!(process, 3);
    }

    /// Bursting well past the 1024-per-millisecond capacity must still never
    /// reuse an id: the allocator advances the millisecond instead of colliding.
    #[test]
    fn allocator_never_repeats_and_never_leaves_the_positive_range() {
        let ids = DisaggRequestIds::new(namespace());
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4096 {
            let id = ids.next();
            assert!(id > 0, "id must be a positive int64, got {id}");
            assert!(seen.insert(id), "allocator repeated {id}");
        }
    }

    /// A sustained rate above the layout's capacity must not spin forever; it
    /// may exceed the true clock, but it must keep producing new ids.
    #[test]
    fn allocator_exceeding_capacity_advances_instead_of_blocking() {
        let ids = DisaggRequestIds::new(namespace());
        let started = Instant::now();
        let mut previous = ids.next();
        for _ in 0..5000 {
            let id = ids.next();
            assert!(id > previous, "ids must stay strictly increasing");
            previous = id;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "allocator stalled under burst"
        );
    }

    /// The counter is shared across calls, so an explicit millisecond pins the
    /// wrap without waiting on a clock.
    #[test]
    fn allocator_pins_the_counter_wrap_at_an_explicit_millisecond() {
        let ids = DisaggRequestIds::new(namespace());
        let first = ids.next_at_ms(9_000);
        let wrapped = ids.next_at_ms(9_000);
        assert_ne!(first, wrapped, "counter must advance within a millisecond");
        let (timestamp, node, process, counter) =
            decompose_disagg_request_id(first as u64).unwrap();
        assert_eq!((timestamp, node, process, counter), (9_000, 7, 3, 0));
        let (_, _, _, next_counter) = decompose_disagg_request_id(wrapped as u64).unwrap();
        assert_eq!(next_counter, 1);
    }

    fn context_response(params: Value, top: Value) -> Vec<u8> {
        let mut body = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "length",
                "disaggregated_params": params,
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        });
        if let (Some(object), Some(extra)) = (body.as_object_mut(), top.as_object()) {
            for (key, value) in extra {
                object.insert(key.clone(), value.clone());
            }
        }
        serde_json::to_vec(&body).unwrap()
    }

    fn full_params() -> Value {
        json!({
            "request_type": "context_only",
            "disagg_request_id": 1234567890123456789_i64,
            "ctx_request_id": 42,
            "first_gen_tokens": [15043],
            "encoded_opaque_state": "opaque-blob",
        })
    }

    #[test]
    fn handoff_parses_from_the_choice_and_not_the_top_level() {
        let body = context_response(full_params(), json!({"prompt_token_ids": [1, 2, 3]}));
        let handoff = parse_context_response(&body).expect("parses");
        assert_eq!(handoff.disagg_request_id, Some(1234567890123456789));
        assert_eq!(handoff.ctx_request_id, Some(42));
        assert_eq!(
            handoff.first_gen_tokens.as_deref(),
            Some([15043].as_slice())
        );
        assert_eq!(handoff.encoded_opaque_state.as_deref(), Some("opaque-blob"));
        assert_eq!(handoff.finish_reason.as_deref(), Some("length"));
        assert_eq!(
            handoff.prompt_token_ids.as_deref(),
            Some([1, 2, 3].as_slice())
        );
    }

    /// A top-level `kv_transfer_params` is a vLLM shape and must not be mistaken
    /// for the handoff.
    #[test]
    fn a_top_level_kv_transfer_params_is_not_the_handoff() {
        let mut body: Value = serde_json::from_slice(&context_response(
            full_params(),
            json!({"prompt_token_ids": [1]}),
        ))
        .unwrap();
        body.as_object_mut().unwrap().remove("choices");
        body.as_object_mut().unwrap().insert(
            "kv_transfer_params".to_string(),
            json!({"disagg_request_id": 1}),
        );
        assert_eq!(
            parse_context_response(&serde_json::to_vec(&body).unwrap()).unwrap_err(),
            HandoffError::NoChoices
        );
    }

    #[test]
    fn missing_disaggregated_params_is_rejected() {
        let body = json!({"choices": [{"index": 0, "finish_reason": "length"}]});
        assert_eq!(
            parse_context_response(&serde_json::to_vec(&body).unwrap()).unwrap_err(),
            HandoffError::NoDisaggregatedParams
        );
    }

    #[test]
    fn a_generation_only_response_cannot_be_used_as_a_context_handoff() {
        let mut params = full_params();
        params["request_type"] = json!("generation_only");
        let body = context_response(params, json!({"prompt_token_ids": [1]}));
        assert_eq!(
            parse_context_response(&body).unwrap_err(),
            HandoffError::UnexpectedRequestType {
                found: Some("generation_only".to_string()),
                expected: "context_only",
            }
        );
    }

    #[test]
    fn context_and_generation_is_not_accepted_as_context_first() {
        let mut params = full_params();
        params["request_type"] = json!(REQUEST_TYPE_CONTEXT_AND_GENERATION);
        let body = context_response(params, json!({"prompt_token_ids": [1]}));
        assert!(matches!(
            parse_context_response(&body).unwrap_err(),
            HandoffError::UnexpectedRequestType { .. }
        ));
    }

    #[test]
    fn the_two_prompt_token_forms_are_mutually_exclusive() {
        let both = context_response(
            full_params(),
            json!({"prompt_token_ids": [1, 2], "prompt_token_ids_b64": "AAAA"}),
        );
        let handoff = parse_context_response(&both).expect("parses");
        assert_eq!(
            handoff.prompt_tokens().unwrap_err(),
            HandoffError::ConflictingPromptTokenForms
        );

        let neither = context_response(full_params(), json!({}));
        let handoff = parse_context_response(&neither).expect("parses");
        assert_eq!(
            handoff.prompt_tokens().unwrap_err(),
            HandoffError::MissingPromptTokens
        );
    }

    #[test]
    fn the_base64_form_is_preferred_only_when_it_is_the_only_form() {
        let body = context_response(full_params(), json!({"prompt_token_ids_b64": "AAEC"}));
        let handoff = parse_context_response(&body).expect("parses");
        assert_eq!(
            handoff.prompt_tokens().unwrap(),
            PromptTokens::Base64("AAEC")
        );
    }

    #[test]
    fn wrong_field_types_are_rejected_rather_than_coerced() {
        let body = context_response(
            {
                let mut params = full_params();
                params["first_gen_tokens"] = json!(["not", "ints"]);
                params
            },
            json!({"prompt_token_ids": [1]}),
        );
        assert_eq!(
            parse_context_response(&body).unwrap_err(),
            HandoffError::WrongFieldType {
                field: "first_gen_tokens"
            }
        );

        // A fractional room-like value must not be truncated into an integer.
        let body = context_response(
            {
                let mut params = full_params();
                params["disagg_request_id"] = json!(1.5);
                params
            },
            json!({"prompt_token_ids": [1]}),
        );
        assert_eq!(
            parse_context_response(&body).unwrap_err(),
            HandoffError::WrongFieldType {
                field: "disagg_request_id"
            }
        );
    }

    #[test]
    fn the_generation_gate_follows_finish_reason() {
        for (reason, expected) in [
            ("length", true),
            ("not_finished", true),
            ("stop", false),
            ("abort", false),
            ("", false),
        ] {
            let mut body: Value = serde_json::from_slice(&context_response(
                full_params(),
                json!({"prompt_token_ids": [1]}),
            ))
            .unwrap();
            body["choices"][0]["finish_reason"] = json!(reason);
            let handoff = parse_context_response(&serde_json::to_vec(&body).unwrap()).unwrap();
            assert_eq!(
                handoff.needs_generation(),
                expected,
                "finish_reason={reason:?}"
            );
        }
    }

    // ── Builders (A4-C2) ─────────────────────────────────────────────────────

    struct FixedIds(i64);

    impl RequestIds for FixedIds {
        fn mint(&self) -> i64 {
            self.0
        }
    }

    fn client_request() -> Value {
        json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.25,
            "stream": false,
        })
    }

    fn prepare_ctx(body: &Value) -> Result<PreparedContextRequest, RequestError> {
        prepare_context_request(
            &serde_json::to_vec(body).unwrap(),
            64 * 1024,
            "conv-1",
            &FixedIds(777),
        )
    }

    #[test]
    fn context_leg_is_non_streaming_and_carries_context_only() {
        let prepared = prepare_ctx(&client_request()).expect("prepares");
        assert_eq!(prepared.leg.path, CHAT_COMPLETIONS_PATH);
        assert_eq!(prepared.leg.correlation_id, 777);
        assert_eq!(prepared.leg.field("stream"), Some(json!(false)));
        assert_eq!(prepared.leg.field("stream_options"), Some(Value::Null));
        let params = prepared.leg.field("disaggregated_params").unwrap();
        assert_eq!(params["request_type"], REQUEST_TYPE_CONTEXT_ONLY);
        assert_eq!(params["disagg_request_id"], 777);
        assert_eq!(params["conversation_id"], "conv-1");
    }

    /// A4-R08: the context leg is forced non-streaming while a non-streaming
    /// client request keeps its mode on the generation leg.
    #[test]
    fn a_non_streaming_client_request_keeps_its_mode_on_the_generation_leg() {
        let mut request = client_request();
        request["stream"] = json!(false);
        let prepared = prepare_ctx(&request).unwrap();
        // The context leg is non-streaming regardless of what the client asked.
        assert_eq!(prepared.leg.field("stream"), Some(json!(false)));

        let handoff = ContextHandoff {
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..Default::default()
        };
        let generation = prepare_generation_request(
            &serde_json::to_vec(&request).unwrap(),
            &handoff,
            prepared.leg.correlation_id,
            &prepared.conversation_id,
            None,
        )
        .expect("prepares");
        assert_eq!(generation.leg.field("stream"), Some(json!(false)));
    }

    /// A streaming request is refused before any leg runs, because the adapter
    /// buffers the generation leg and cannot produce an SSE body.
    #[test]
    fn a_streaming_request_is_refused_rather_than_downgraded() {
        let mut request = client_request();
        request["stream"] = json!(true);
        let error = prepare_ctx(&request).expect_err("must be refused");
        assert!(matches!(error, RequestError::UnsupportedStreaming));
    }

    /// A4-R16: a client must not be able to inject orchestration state.
    #[test]
    fn client_orchestration_fields_are_rejected() {
        for field in CLIENT_FORBIDDEN_FIELDS {
            let mut request = client_request();
            request[field] = json!({"request_type": "generation_only"});
            let error = prepare_ctx(&request).expect_err("must be rejected");
            assert!(
                matches!(error, RequestError::ForbiddenField { field: rejected } if rejected == field),
                "expected {field} rejected, got {error}"
            );
        }
    }

    /// A control for the mutation "stop rejecting client orchestration fields":
    /// an explicit null is absent, so a client that round-trips it is served.
    #[test]
    fn null_orchestration_fields_are_treated_as_absent() {
        let mut request = client_request();
        for field in CLIENT_FORBIDDEN_FIELDS {
            request[field] = Value::Null;
        }
        let prepared = prepare_ctx(&request).unwrap();
        assert_eq!(
            prepared.leg.field("disaggregated_params").unwrap()["request_type"],
            REQUEST_TYPE_CONTEXT_ONLY
        );
    }

    #[test]
    fn body_cap_and_shape_are_checked_before_anything_is_built() {
        let small = serde_json::to_vec(&client_request()).unwrap();
        assert!(matches!(
            prepare_context_request(&small, 4, "c", &FixedIds(1)).unwrap_err(),
            RequestError::BodyTooLarge { limit: 4 }
        ));
        assert!(matches!(
            prepare_context_request(b"[1,2,3]", 1024, "c", &FixedIds(1)).unwrap_err(),
            RequestError::NotAnObject
        ));
        assert!(matches!(
            prepare_context_request(b"not json", 1024, "c", &FixedIds(1)).unwrap_err(),
            RequestError::MalformedJson { .. }
        ));
    }

    fn generation_for(handoff: &ContextHandoff) -> Value {
        let prepared = prepare_generation_request(
            &serde_json::to_vec(&client_request()).unwrap(),
            handoff,
            777,
            "conv-1",
            None,
        )
        .expect("prepares");
        serde_json::from_slice(&prepared.leg.body).unwrap()
    }

    /// A4-R02/R04: the same trusted id on both legs, `ctx_request_id` kept
    /// distinct, and the handoff values carried with their types intact.
    #[test]
    fn generation_leg_reuses_the_id_and_preserves_the_handoff() {
        let handoff = ContextHandoff {
            disagg_request_id: Some(4242),
            ctx_request_id: Some(42),
            first_gen_tokens: Some(vec![15043]),
            encoded_opaque_state: Some("opaque-blob".to_string()),
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![1, 2, 3]),
            prompt_token_ids_b64: None,
            ctx_info_endpoint: None,
        };
        let body = generation_for(&handoff);
        let params = &body["disaggregated_params"];
        assert_eq!(params["request_type"], "generation_only");
        assert_eq!(params["disagg_request_id"], 4242);
        assert_eq!(params["ctx_request_id"], 42);
        assert_eq!(params["first_gen_tokens"], json!([15043]));
        assert_eq!(params["encoded_opaque_state"], "opaque-blob");
        assert_eq!(body["prompt_token_ids"], json!([1, 2, 3]));
        // Sampling is untouched.
        assert_eq!(body["max_tokens"], 32);
        assert_eq!(body["temperature"], 0.25);
    }

    /// The context worker's id wins when it reports one, because a retry may
    /// have replaced the id the gateway minted.
    #[test]
    fn the_context_reported_id_is_authoritative() {
        let handoff = ContextHandoff {
            disagg_request_id: Some(4242),
            finish_reason: Some("not_finished".to_string()),
            prompt_token_ids: Some(vec![1]),
            ..Default::default()
        };
        let prepared = prepare_generation_request(
            &serde_json::to_vec(&client_request()).unwrap(),
            &handoff,
            777,
            "conv-1",
            None,
        )
        .unwrap();
        assert_eq!(prepared.correlation_id, Some(4242));
        assert_eq!(prepared.leg.correlation_id, 4242);
    }

    /// A4-R05: when the context leg already finished, there is no generation
    /// leg at all — not an empty one, and never an added token.
    #[test]
    fn a_finished_context_leg_produces_no_generation_request() {
        for reason in ["stop", "abort", ""] {
            let handoff = ContextHandoff {
                finish_reason: Some(reason.to_string()),
                prompt_token_ids: Some(vec![1]),
                ..Default::default()
            };
            let prepared = prepare_generation_request(
                &serde_json::to_vec(&client_request()).unwrap(),
                &handoff,
                777,
                "conv-1",
                None,
            )
            .expect("prepares");
            assert_eq!(
                prepared.correlation_id, None,
                "finish_reason={reason:?} must not start generation"
            );
            assert!(prepared.leg.body.is_empty());
        }
    }

    /// A4-R04: the base64 form is relayed verbatim and the int form is cleared,
    /// so the two representations are never both set.
    #[test]
    fn the_base64_form_is_relayed_verbatim_and_never_decoded() {
        let handoff = ContextHandoff {
            finish_reason: Some("length".to_string()),
            prompt_token_ids_b64: Some("AAECAw==".to_string()),
            ..Default::default()
        };
        let body = generation_for(&handoff);
        assert_eq!(body["prompt_token_ids_b64"], "AAECAw==");
        assert!(body.get("prompt_token_ids").is_none() || body["prompt_token_ids"].is_null());
    }

    /// A4-R06: a large integer must round-trip exactly, with no float transit.
    #[test]
    fn large_integers_round_trip_without_precision_loss() {
        let large = 9_007_199_254_740_993_i64; // 2^53 + 1, not representable as f64
        let handoff = ContextHandoff {
            disagg_request_id: Some(large),
            ctx_request_id: Some(large - 2),
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![large]),
            ..Default::default()
        };
        let body = generation_for(&handoff);
        assert_eq!(body["disaggregated_params"]["disagg_request_id"], large);
        assert_eq!(body["disaggregated_params"]["ctx_request_id"], large - 2);
        assert_eq!(body["prompt_token_ids"][0], large);
        // Reading it back as text proves no f64 round trip happened.
        let text = String::from_utf8(generation_for(&handoff).to_string().into_bytes()).unwrap();
        assert!(
            text.contains(&large.to_string()),
            "id lost its exact digits"
        );
    }

    /// A handoff that sets both token forms is rejected rather than guessed at.
    #[test]
    fn a_handoff_with_both_token_forms_never_builds_a_generation_request() {
        let handoff = ContextHandoff {
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![1]),
            prompt_token_ids_b64: Some("AAEC".to_string()),
            ..Default::default()
        };
        let error = prepare_generation_request(
            &serde_json::to_vec(&client_request()).unwrap(),
            &handoff,
            777,
            "conv-1",
            None,
        )
        .expect_err("must be rejected");
        assert!(matches!(
            error,
            RequestError::Handoff {
                source: HandoffError::ConflictingPromptTokenForms
            }
        ));
    }

    #[test]
    fn the_contract_reports_its_pin_and_rejects_the_combined_mode() {
        let contract = TrtllmContextFirstContract::pinned();
        assert_eq!(contract.engine_tag, VERIFIED_ENGINE_TAG);
        assert_eq!(contract.engine_revision, VERIFIED_ENGINE_REVISION);
        assert_eq!(contract.protocol_revision, PROTOCOL_REVISION);
        assert_eq!(contract.chat_completions_path(), CHAT_COMPLETIONS_PATH);
        assert!(contract.is_unsupported_request_type(REQUEST_TYPE_CONTEXT_AND_GENERATION));
        assert!(!contract.is_unsupported_request_type(REQUEST_TYPE_CONTEXT_ONLY));
    }

    // ── Dispatch (A4-C3) ─────────────────────────────────────────────────────

    /// One recorded call, so tests can assert what was actually sent and in
    /// what order rather than inferring it from the outcome.
    #[derive(Debug, Clone)]
    enum Sent {
        Context(Vec<u8>),
        Generation(Vec<u8>),
    }

    #[derive(Default)]
    struct Scripted {
        sent: std::sync::Mutex<Vec<Sent>>,
        context: Option<Result<Vec<u8>, LegFailure>>,
    }

    impl Scripted {
        fn returning_context(body: Vec<u8>) -> Self {
            Self {
                sent: std::sync::Mutex::new(Vec::new()),
                context: Some(Ok(body)),
            }
        }

        fn failing_context(failure: LegFailure) -> Self {
            Self {
                sent: std::sync::Mutex::new(Vec::new()),
                context: Some(Err(failure)),
            }
        }

        fn calls(&self) -> Vec<Sent> {
            self.sent.lock().unwrap().clone()
        }

        fn generation_calls(&self) -> usize {
            self.calls()
                .iter()
                .filter(|c| matches!(c, Sent::Generation(_)))
                .count()
        }
    }

    #[async_trait::async_trait]
    impl LegTransport for Scripted {
        async fn send_context(
            &self,
            leg: PreparedLeg,
            _cancellation: CancellationToken,
            _deadline: Duration,
        ) -> Result<Vec<u8>, LegFailure> {
            self.sent
                .lock()
                .unwrap()
                .push(Sent::Context(leg.body.to_vec()));
            match &self.context {
                Some(Ok(body)) => Ok(body.clone()),
                Some(Err(failure)) => Err(*failure),
                None => Err(LegFailure::Unavailable { leg: Leg::Context }),
            }
        }

        async fn send_generation(
            &self,
            leg: PreparedLeg,
            cancellation: CancellationToken,
            _deadline: Duration,
        ) -> Result<OwnedBody, LegFailure> {
            self.sent
                .lock()
                .unwrap()
                .push(Sent::Generation(leg.body.to_vec()));
            Ok(OwnedBody::new(
                br#"{"object":"chat.completion","choices":[]}"#.to_vec(),
                cancellation,
            ))
        }
    }

    fn limits() -> ContextFirstLimits {
        ContextFirstLimits::new(
            64 * 1024,
            256 * 1024,
            512 * 1024,
            Duration::from_secs(30),
            Duration::from_secs(30),
        )
        .expect("valid limits")
    }

    fn dispatcher(transport: Arc<Scripted>, minted: i64) -> ContextFirstDispatcher {
        ContextFirstDispatcher::new(
            TrtllmContextFirstContract::pinned(),
            transport,
            Arc::new(FixedIds(minted)),
            limits(),
            None,
        )
    }

    /// A context response body with an explicit `finish_reason`, so the
    /// generation gate can be driven without re-serializing.
    fn context_body(finish_reason: &str, params: Value, top: Value) -> Vec<u8> {
        let mut parameters = json!({
            "request_type": REQUEST_TYPE_CONTEXT_ONLY,
            "ctx_request_id": 42,
            "first_gen_tokens": [15043],
        });
        if let (Some(base), Some(extra)) = (parameters.as_object_mut(), params.as_object()) {
            for (key, value) in extra {
                base.insert(key.clone(), value.clone());
            }
        }

        let mut body = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": finish_reason,
                "disaggregated_params": parameters,
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        });
        if let (Some(object), Some(extra)) = (body.as_object_mut(), top.as_object()) {
            for (key, value) in extra {
                object.insert(key.clone(), value.clone());
            }
        }
        serde_json::to_vec(&body).expect("serializes")
    }

    fn request_body() -> Vec<u8> {
        serde_json::to_vec(&client_request()).unwrap()
    }

    /// A4-R01: the generation leg is sent only after the context leg completed
    /// and its handoff validated.
    #[tokio::test]
    async fn generation_runs_only_after_a_valid_context_handoff() {
        let transport = Arc::new(Scripted::returning_context(context_body(
            "length",
            json!({}),
            json!({"prompt_token_ids": [1, 2, 3]}),
        )));
        let outcome = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect("dispatch succeeds");

        assert!(!outcome.context_completed());
        let calls = transport.calls();
        assert_eq!(calls.len(), 2, "both legs must run exactly once");
        assert!(matches!(calls[0], Sent::Context(_)));
        assert!(matches!(calls[1], Sent::Generation(_)));

        // The context leg was non-streaming and carried the minted id.
        let Sent::Context(ctx) = &calls[0] else {
            unreachable!()
        };
        let ctx: Value = serde_json::from_slice(ctx).unwrap();
        assert_eq!(ctx["stream"], json!(false));
        assert_eq!(ctx["disaggregated_params"]["disagg_request_id"], 777);
    }

    /// A4-R02: both legs carry one correlation id, and the other context
    /// identity is not overwritten by it.
    #[tokio::test]
    async fn both_legs_share_one_id_and_keep_ctx_request_id_distinct() {
        let transport = Arc::new(Scripted::returning_context(context_body(
            "length",
            json!({}),
            json!({"prompt_token_ids": [1]}),
        )));
        let outcome = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(outcome.correlation_id(), 777);

        let calls = transport.calls();
        let Sent::Generation(generation) = &calls[1] else {
            unreachable!()
        };
        let generation: Value = serde_json::from_slice(generation).unwrap();
        assert_eq!(generation["disaggregated_params"]["disagg_request_id"], 777);
        assert_eq!(generation["disaggregated_params"]["ctx_request_id"], 42);
    }

    /// A4-R05: a context leg that already finished ends the request. No
    /// generation leg is sent, and nothing is added.
    #[tokio::test]
    async fn a_finished_context_leg_runs_no_generation_and_returns_its_own_body() {
        let transport = Arc::new(Scripted::returning_context(context_body(
            "stop",
            json!({}),
            json!({"prompt_token_ids": [1]}),
        )));
        let outcome = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect("dispatch succeeds");

        assert!(outcome.context_completed());
        assert_eq!(transport.generation_calls(), 0);
        assert!(!outcome.body().is_empty());
    }

    /// A4-R03/R09/R14: every pre-generation failure produces zero generation
    /// requests and keeps its own classification.
    #[tokio::test]
    async fn no_failure_before_generation_can_start_it() {
        // Unreachable worker.
        let transport = Arc::new(Scripted::failing_context(LegFailure::Connect {
            leg: Leg::Context,
        }));
        let error = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect_err("must fail");
        assert!(matches!(
            error,
            DispatchError::Leg(LegFailure::Connect { leg: Leg::Context })
        ));
        assert_eq!(transport.generation_calls(), 0);

        // Read stall, deadline and status stay distinguishable.
        for failure in [
            LegFailure::ReadStall { leg: Leg::Context },
            LegFailure::Deadline { leg: Leg::Context },
            LegFailure::Status {
                leg: Leg::Context,
                status: 503,
            },
        ] {
            let transport = Arc::new(Scripted::failing_context(failure));
            let error = dispatcher(transport.clone(), 777)
                .dispatch(&request_body(), "conv-1", CancellationToken::new())
                .await
                .expect_err("must fail");
            assert_eq!(error.to_string(), failure.to_string());
            assert_eq!(transport.generation_calls(), 0);
        }

        // Malformed context body.
        let transport = Arc::new(Scripted::returning_context(b"not json".to_vec()));
        let error = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect_err("must fail");
        assert!(matches!(error, DispatchError::Handoff { .. }));
        assert_eq!(transport.generation_calls(), 0);

        // Context response with no handoff at all.
        let transport = Arc::new(Scripted::returning_context(
            serde_json::to_vec(&json!({"choices": [{"index": 0}]})).unwrap(),
        ));
        let error = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect_err("must fail");
        assert!(matches!(
            error,
            DispatchError::Handoff {
                source: HandoffError::NoDisaggregatedParams
            }
        ));
        assert_eq!(transport.generation_calls(), 0);
    }

    /// A4-R14: an oversized handoff body is rejected on size, before parsing.
    #[tokio::test]
    async fn an_oversized_handoff_is_rejected_by_size_not_by_parsing() {
        let mut body = context_body("length", json!({}), json!({"prompt_token_ids": [1]}));
        body.extend(std::iter::repeat_n(b' ', 512 * 1024));
        let transport = Arc::new(Scripted::returning_context(body));
        let error = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", CancellationToken::new())
            .await
            .expect_err("must fail");
        match error {
            DispatchError::Leg(LegFailure::BodyTooLarge { leg, limit }) => {
                assert_eq!(leg, Leg::Context);
                assert_eq!(limit, 256 * 1024);
            }
            other => panic!("expected a size rejection, got {other}"),
        }
        assert_eq!(transport.generation_calls(), 0);
    }

    /// A4-R11: cancelling while the context leg is in flight must not start the
    /// generation leg.
    #[tokio::test]
    async fn cancellation_before_the_context_leg_starts_nothing() {
        let transport = Arc::new(Scripted::returning_context(context_body(
            "length",
            json!({}),
            json!({"prompt_token_ids": [1]}),
        )));
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let error = dispatcher(transport.clone(), 777)
            .dispatch(&request_body(), "conv-1", cancellation)
            .await
            .expect_err("must fail");
        assert!(matches!(
            error,
            DispatchError::Cancelled { leg: Leg::Context }
        ));
        assert!(transport.calls().is_empty(), "no leg may be sent");
    }

    /// A4-R16: a client cannot inject orchestration state into either leg.
    #[tokio::test]
    async fn client_supplied_orchestration_state_never_reaches_a_leg() {
        let mut request = client_request();
        request["disaggregated_params"] = json!({
            "request_type": "generation_only",
            "disagg_request_id": 1,
        });
        let transport = Arc::new(Scripted::returning_context(context_body(
            "length",
            json!({}),
            json!({"prompt_token_ids": [1]}),
        )));
        let error = dispatcher(transport.clone(), 777)
            .dispatch(
                &serde_json::to_vec(&request).unwrap(),
                "conv-1",
                CancellationToken::new(),
            )
            .await
            .expect_err("must fail");
        assert!(matches!(
            error,
            DispatchError::Request(RequestError::ForbiddenField { .. })
        ));
        assert!(transport.calls().is_empty());
    }

    /// A4-R18 control: the limits are validated at construction rather than
    /// silently accepting a zero budget.
    #[test]
    fn limits_reject_zero_budgets() {
        assert!(matches!(
            ContextFirstLimits::new(0, 1, 1, Duration::from_secs(1), Duration::from_secs(1)),
            Err(ConfigError::RequestBodyLimit)
        ));
        assert!(matches!(
            ContextFirstLimits::new(1, 0, 1, Duration::from_secs(1), Duration::from_secs(1)),
            Err(ConfigError::HandoffBodyLimit)
        ));
        assert!(matches!(
            ContextFirstLimits::new(1, 1, 1, Duration::ZERO, Duration::from_secs(1)),
            Err(ConfigError::Deadline)
        ));
    }

    /// The correlation id is readable from a serialized leg, for logging.
    #[test]
    fn correlation_id_is_readable_from_a_leg_body() {
        let prepared = prepare_ctx(&client_request()).unwrap();
        assert_eq!(correlation_id_of(&prepared.leg.body), Some(777));
        assert_eq!(correlation_id_of(b"not json"), None);
    }

    /// The generation body is owned by the response, so the stream is bound to
    /// the response's lifetime rather than the request's.
    #[tokio::test]
    async fn the_generation_body_carries_the_response_cancellation() {
        let transport = Arc::new(Scripted::returning_context(context_body(
            "length",
            json!({}),
            json!({"prompt_token_ids": [1]}),
        )));
        let cancellation = CancellationToken::new();
        let outcome = dispatcher(transport, 777)
            .dispatch(&request_body(), "conv-1", cancellation.clone())
            .await
            .unwrap();
        assert!(!outcome.body().cancellation().is_cancelled());
        cancellation.cancel();
        assert!(outcome.body().cancellation().is_cancelled());
    }

    // ── Internal handoff signature ───────────────────────────────────────────

    /// The signature must match the pinned producer byte-for-byte, so it is
    /// checked against an independently computed HMAC rather than a round trip
    /// through our own code.
    #[test]
    fn the_handoff_signature_matches_an_independent_hmac() {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;

        let key = b"shared-secret";
        let opaque = "opaque-blob";
        let signature = sign_disagg_handoff(key, Some(opaque), None).expect("signed");
        assert!(signature.starts_with("sha256="), "got {signature}");

        // The producer canonicalises with sorted keys and compact separators,
        // including nulls, so the payload is exactly this JSON.
        let payload = br#"{"ctx_info_endpoint":null,"encoded_opaque_state":"opaque-blob"}"#;
        let mut mac = Hmac::<Sha256>::new_from_slice(key).unwrap();
        mac.update(payload);
        let expected = mac
            .finalize()
            .into_bytes()
            .iter()
            .fold(String::new(), |mut acc, b| {
                use std::fmt::Write as _;
                let _ = write!(acc, "{b:02x}");
                acc
            });
        assert_eq!(signature, format!("sha256={expected}"));
    }

    /// A handoff with no protected field needs no signature, because the worker
    /// only demands one when such a field is present.
    #[test]
    fn an_unprotected_handoff_needs_no_signature() {
        assert_eq!(sign_disagg_handoff(b"k", None, None), None);
        assert!(sign_disagg_handoff(b"k", Some("state"), None).is_some());
        assert!(sign_disagg_handoff(b"k", None, Some("host:1")).is_some());
    }

    /// Every protected field changes the signature, so a tampered value cannot
    /// reuse another handoff's signature.
    #[test]
    fn changing_a_protected_field_changes_the_signature() {
        let a = sign_disagg_handoff(b"k", Some("state-a"), None).unwrap();
        let b = sign_disagg_handoff(b"k", Some("state-b"), None).unwrap();
        let c = sign_disagg_handoff(b"k", Some("state-a"), Some("h:1")).unwrap();
        assert_ne!(a, b, "a changed opaque state must change the signature");
        assert_ne!(a, c, "a changed endpoint must change the signature");
    }

    /// The generation leg carries the header exactly when a protected field is
    /// present and a key is configured.
    #[test]
    fn the_generation_leg_is_signed_only_when_it_needs_to_be() {
        let signed = |handoff: &ContextHandoff, key: Option<&[u8]>| {
            prepare_generation_request(
                &serde_json::to_vec(&client_request()).unwrap(),
                handoff,
                1,
                "conv-1",
                key,
            )
            .expect("prepares")
            .leg
            .auth_header
        };

        let with_opaque = ContextHandoff {
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![1]),
            encoded_opaque_state: Some("blob".to_string()),
            ..Default::default()
        };
        let header = signed(&with_opaque, Some(b"k")).expect("must be signed");
        assert_eq!(header.0, INTERNAL_DISAGG_AUTH_HEADER);
        assert!(header.1.starts_with("sha256="));
        // No key configured: unsigned, which the worker only tolerates for an
        // unprotected handoff.
        assert!(signed(&with_opaque, None).is_none());

        let without_opaque = ContextHandoff {
            finish_reason: Some("length".to_string()),
            prompt_token_ids: Some(vec![1]),
            ..Default::default()
        };
        assert!(
            signed(&without_opaque, Some(b"k")).is_none(),
            "an unprotected handoff must not be signed"
        );
    }
}
