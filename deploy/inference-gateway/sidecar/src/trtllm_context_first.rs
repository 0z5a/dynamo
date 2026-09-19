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

use bytes::Bytes;
use serde_json::Value;

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

    let encoded =
        serde_json::to_vec(&object).map_err(|source| RequestError::MalformedJson { source })?;
    Ok(PreparedGenerationRequest {
        leg: PreparedLeg {
            path: CHAT_COMPLETIONS_PATH,
            correlation_id,
            body: Bytes::from(encoded),
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
            "stream": true,
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

    /// A4-R08: forcing the context leg non-streaming must not change what the
    /// client asked for on the generation leg.
    #[test]
    fn the_client_streaming_mode_survives_to_the_generation_leg() {
        let request = client_request();
        let prepared = prepare_ctx(&request).unwrap();
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
        )
        .expect("prepares");
        assert_eq!(generation.leg.field("stream"), Some(json!(true)));
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
}
