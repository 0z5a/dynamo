// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang P/D request contract: trusted bootstrap parameters and the two-leg
//! request builder.
//!
//! This module is the protocol layer only. It validates a client
//! `/v1/chat/completions` body and derives the prefill/decode request bodies
//! that an SGLang P/D deployment requires. It does not dispatch either leg;
//! concurrency, response-body lifetime and termination are separate work.
//!
//! The bootstrap parameters are an explicit typed input. SGLang serves the
//! cross-engine KV bootstrap on a listener separate from its HTTP server
//! (`--disaggregation-bootstrap-port`, default 8998, versus `--port`, default
//! 30000), and that port is not part of the selected-prefill metadata the
//! sidecar receives. See `SGLANG_PD_PROTOCOL.md`.

use bytes::Bytes;
use serde_json::{Map, Value};

use crate::error::SidecarError;

/// Protocol revision this module implements.
///
/// Bump only with a matching upstream SGLang release pin recorded in
/// `SGLANG_PD_PROTOCOL.md`.
pub const PROTOCOL_REVISION: &str = "sglang-pd-v1";

/// Upstream engine revision these field semantics were verified against.
pub const VERIFIED_ENGINE_REVISION: &str =
    "sglang f4e0ac382e4e5d644f2fbe4a15c20da53500bbca (2026-07-29)";

/// Request fields this sidecar owns. A client-supplied value is never
/// respected for these; the trusted tuple replaces it.
const BOOTSTRAP_FIELDS: [&str; 3] = ["bootstrap_host", "bootstrap_port", "bootstrap_room"];

/// Rank-routing extensions that could bypass the worker selected by the
/// gateway. They are rejected rather than forwarded, so a deployment never
/// silently routes against EPP selection.
const RANK_ROUTING_FIELDS: [&str; 2] = ["routed_dp_rank", "disagg_prefill_dp_rank"];

/// SGLang types `bootstrap_room` as a signed 64-bit integer, and the decode
/// side indexes an array with `room % dp_size`, so the value must be
/// non-negative. Zero is the engine's "no room" sentinel.
const MAX_ROOM: i64 = i64::MAX;
const MIN_ROOM: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    #[error("unsupported SGLang P/D protocol revision: {revision}")]
    UnsupportedRevision { revision: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TrustError {
    #[error("trusted bootstrap host is empty")]
    EmptyBootstrapHost,
    #[error("trusted bootstrap host contains a scheme, path, port or permission separator")]
    MalformedBootstrapHost,
    #[error("trusted bootstrap port must be greater than zero")]
    ZeroBootstrapPort,
    #[error("trusted bootstrap room must be between 1 and i64::MAX")]
    BootstrapRoomOutOfRange,
}

#[derive(Debug, thiserror::Error)]
pub enum RequestError {
    #[error("request body exceeds the configured limit of {limit} bytes")]
    BodyTooLarge { limit: usize },
    #[error("request body is not valid JSON: {source}")]
    MalformedJson { source: serde_json::Error },
    #[error("request body must be a JSON object")]
    NotAnObject,
    #[error("client-supplied {field} is not allowed; the gateway assigns rank routing")]
    RankRoutingField { field: &'static str },
    #[error("trusted bootstrap rejected: {source}")]
    Bootstrap { source: TrustError },
    #[error("unsupported SGLang P/D protocol revision: {revision}")]
    UnsupportedRevision { revision: String },
    #[error("failed to serialize the {leg} request body: {source}")]
    Serialize {
        leg: &'static str,
        source: serde_json::Error,
    },
}

impl From<TrustError> for RequestError {
    fn from(source: TrustError) -> Self {
        Self::Bootstrap { source }
    }
}

impl From<RequestError> for SidecarError {
    fn from(error: RequestError) -> Self {
        // A malformed client body is the client's problem (400). Everything
        // else here is a sidecar configuration or contract failure and must
        // not be reported as a bad request.
        match error {
            RequestError::BodyTooLarge { .. }
            | RequestError::MalformedJson { .. }
            | RequestError::NotAnObject
            | RequestError::RankRoutingField { .. } => SidecarError::adapter(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request",
                error.to_string(),
            ),
            _ => SidecarError::adapter(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "pd_contract_error",
                error.to_string(),
            ),
        }
    }
}

/// Protocol features this build implements and has verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SglangPdContract {
    pub revision: &'static str,
    /// Verified against upstream source; see `PROTOCOL_REVISION`.
    pub engine_revision: &'static str,
}

impl SglangPdContract {
    pub const fn pinned() -> Self {
        Self {
            revision: PROTOCOL_REVISION,
            engine_revision: VERIFIED_ENGINE_REVISION,
        }
    }

    /// Accepts only the revision this build implements. A declared revision
    /// says nothing about the engine image actually deployed; that remains a
    /// deployment precondition.
    pub fn validate_revision(self, declared: &str) -> Result<Self, ContractError> {
        if declared == self.revision {
            Ok(self)
        } else {
            Err(ContractError::UnsupportedRevision {
                revision: declared.to_string(),
            })
        }
    }
}

/// The trusted bootstrap tuple. Constructed by the gateway from its own
/// discovery state, never from a client request body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedBootstrap {
    host: String,
    port: u16,
    room: i64,
}

impl TrustedBootstrap {
    pub fn new(host: impl Into<String>, port: u16, room: i64) -> Result<Self, TrustError> {
        let host = host.into();
        if host.is_empty() {
            return Err(TrustError::EmptyBootstrapHost);
        }
        if host.contains("://")
            || host.contains('/')
            || host.contains('@')
            || host.contains(':')
            // A comma-joined list is the same separator the shared EPP metadata
            // parser rejects, so a comma-joined host is never a single target.
            || host.contains(',')
            || host.trim() != host
        {
            return Err(TrustError::MalformedBootstrapHost);
        }
        if port == 0 {
            return Err(TrustError::ZeroBootstrapPort);
        }
        if !(MIN_ROOM..=MAX_ROOM).contains(&room) {
            return Err(TrustError::BootstrapRoomOutOfRange);
        }
        Ok(Self { host, port, room })
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn room(&self) -> i64 {
        self.room
    }
}

/// Draws `bootstrap_room` values for independently bootstrapped requests.
pub trait RoomGenerator: Send + Sync {
    fn next_room(&self) -> i64;
}

/// Random 63-bit rooms, matching the value shape the native SGLang sidecar
/// uses. Room uniqueness is probabilistic, not coordinated: two gateways can
/// collide, which is why the room is only a demultiplexing key for an
/// in-flight KV handoff.
#[derive(Debug, Default, Clone, Copy)]
pub struct RandomRoomGenerator;

impl RoomGenerator for RandomRoomGenerator {
    fn next_room(&self) -> i64 {
        let drawn = rand::random::<u64>() & (MAX_ROOM as u64);
        // `bootstrap_room` must be non-negative, and zero is the engine's
        // "absent" sentinel, so force a non-zero room.
        (drawn | 1) as i64
    }
}

/// One leg's fully prepared request: the serialized JSON body plus the exact
/// authority and path that body must be sent to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedLeg {
    pub authority: String,
    pub path: &'static str,
    pub body: Bytes,
}

impl PreparedLeg {
    /// The `bootstrap_room` actually present in the serialized body. Read back
    /// from the wire form so a serialization bug cannot pass unnoticed.
    pub fn room(&self) -> Result<i64, serde_json::Error> {
        let value: Value = serde_json::from_slice(&self.body)?;
        Ok(value
            .get("bootstrap_room")
            .and_then(Value::as_i64)
            .expect("a prepared leg always carries an integer bootstrap_room"))
    }
}

/// The two request bodies derived from one client request. Both legs share the
/// same trusted bootstrap tuple and are semantically equivalent otherwise;
/// engine mode (not request content) decides which leg computes prefill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSglangRequests {
    pub prefill: PreparedLeg,
    pub decode: PreparedLeg,
}

/// Rejects a body that is larger than the deployment limit before any parsing
/// work happens. This is the whole cap mechanism: the caller is responsible
/// for stopping the read at this many bytes, including for a client that never
/// finishes sending.
pub fn enforce_body_limit(len: usize, limit: usize) -> Result<(), RequestError> {
    if len > limit {
        Err(RequestError::BodyTooLarge { limit })
    } else {
        Ok(())
    }
}

/// Builds both legs using the production room generator.
pub fn prepare_pd_requests_with_random_room(
    contract: SglangPdContract,
    declared_revision: &str,
    body: &[u8],
    bootstrap: &TrustedBootstrap,
) -> Result<PreparedSglangRequests, RequestError> {
    prepare_pd_requests(
        contract,
        declared_revision,
        body,
        bootstrap,
        &RandomRoomGenerator,
    )
}

/// Builds the prefill and decode request bodies for one client request.
///
/// `contract` gates the declared protocol revision. `bootstrap` is the trusted
/// tuple; any bootstrap field the client sent is discarded, including extra
/// values when the client sent an array. `room_generator` supplies the room,
/// so callers share one generator across requests.
pub fn prepare_pd_requests(
    contract: SglangPdContract,
    declared_revision: &str,
    body: &[u8],
    bootstrap: &TrustedBootstrap,
    room_generator: &dyn RoomGenerator,
) -> Result<PreparedSglangRequests, RequestError> {
    contract.validate_revision(declared_revision).map_err(
        |ContractError::UnsupportedRevision { revision }| RequestError::UnsupportedRevision {
            revision,
        },
    )?;

    let parsed: Value =
        serde_json::from_slice(body).map_err(|source| RequestError::MalformedJson { source })?;
    let Value::Object(mut object) = parsed else {
        return Err(RequestError::NotAnObject);
    };

    reject_rank_routing(&object)?;

    let room = room_generator.next_room();
    // Re-validate the drawn room through the same constructor the trusted path
    // uses, so a broken generator surfaces as an error instead of a request
    // the engine will reject or misroute.
    let bootstrap = TrustedBootstrap::new(bootstrap.host(), bootstrap.port(), room)?;

    let (host, port, room) = (
        Value::String(bootstrap.host().to_string()),
        Value::from(bootstrap.port()),
        Value::from(bootstrap.room()),
    );
    for (field, value) in BOOTSTRAP_FIELDS.into_iter().zip([host, port, room]) {
        object.insert(field.to_string(), value);
    }

    let authority = format!("{}:{}", bootstrap.host(), bootstrap.port());
    Ok(PreparedSglangRequests {
        prefill: serialize_leg(&object, &authority, "prefill")?,
        decode: serialize_leg(&object, &authority, "decode")?,
    })
}

fn reject_rank_routing(object: &Map<String, Value>) -> Result<(), RequestError> {
    for field in RANK_ROUTING_FIELDS {
        if object.get(field).is_some_and(|value| !value.is_null()) {
            return Err(RequestError::RankRoutingField { field });
        }
    }
    Ok(())
}

fn serialize_leg(
    object: &Map<String, Value>,
    authority: &str,
    leg: &'static str,
) -> Result<PreparedLeg, RequestError> {
    let body =
        serde_json::to_vec(object).map_err(|source| RequestError::Serialize { leg, source })?;
    Ok(PreparedLeg {
        authority: authority.to_string(),
        path: "/v1/chat/completions",
        body: Bytes::from(body),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> SglangPdContract {
        SglangPdContract::pinned()
    }

    fn bootstrap() -> TrustedBootstrap {
        TrustedBootstrap::new("10.0.0.7", 8998, 42).expect("valid bootstrap")
    }

    struct FixedRoom(i64);

    impl RoomGenerator for FixedRoom {
        fn next_room(&self) -> i64 {
            self.0
        }
    }

    fn prepare_with_room(body: &str, room: i64) -> Result<PreparedSglangRequests, RequestError> {
        prepare_pd_requests(
            contract(),
            PROTOCOL_REVISION,
            body.as_bytes(),
            &bootstrap(),
            &FixedRoom(room),
        )
    }

    /// The default path, exercised through the production generator.
    fn prepare(body: &str) -> Result<PreparedSglangRequests, RequestError> {
        prepare_pd_requests_with_random_room(
            contract(),
            PROTOCOL_REVISION,
            body.as_bytes(),
            &bootstrap(),
        )
    }

    fn json(leg: &PreparedLeg) -> Value {
        serde_json::from_slice(&leg.body).expect("prepared leg is JSON")
    }

    #[test]
    fn both_legs_carry_the_trusted_bootstrap_tuple() {
        let prepared = prepare_with_room(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            42,
        )
        .expect("prepare succeeds");
        for leg in [&prepared.prefill, &prepared.decode] {
            assert_eq!(leg.authority, "10.0.0.7:8998");
            assert_eq!(leg.path, "/v1/chat/completions");
            let value = json(leg);
            assert_eq!(value["bootstrap_host"], "10.0.0.7");
            assert_eq!(value["bootstrap_port"], 8998);
        }
        // Both legs share one room for the request they belong to.
        assert_eq!(
            prepared.prefill.room().unwrap(),
            prepared.decode.room().unwrap()
        );
        assert_eq!(prepared.prefill.room().unwrap(), 42);
    }

    #[test]
    fn client_forged_bootstrap_is_replaced_including_extra_array_values() {
        let prepared = prepare_with_room(
            r#"{
                "model":"m",
                "bootstrap_host":["evil.example","evil2.example"],
                "bootstrap_port":[1,2,3],
                "bootstrap_room":999999,
                "messages":[{"role":"user","content":"bootstrap_host is a word in my prompt"}]
            }"#,
            42,
        )
        .expect("prepare succeeds");
        let value = json(&prepared.prefill);
        assert_eq!(value["bootstrap_host"], "10.0.0.7");
        assert_eq!(value["bootstrap_port"], 8998);
        assert_eq!(value["bootstrap_room"], 42);
        // Prose that merely mentions a bootstrap field name is untouched.
        assert_eq!(
            value["messages"][0]["content"],
            "bootstrap_host is a word in my prompt"
        );
    }

    #[test]
    fn unknown_extension_fields_are_preserved() {
        let prepared = prepare(
            r#"{"model":"m","stream":true,"seed":7,"tools":[{"type":"function"}],
                "x-vendor-extension":{"keep":true},"messages":[]}"#,
        )
        .expect("prepare succeeds");
        let value = json(&prepared.decode);
        assert_eq!(value["stream"], true);
        assert_eq!(value["seed"], 7);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["x-vendor-extension"]["keep"], true);
    }

    #[test]
    fn client_cannot_bypass_epp_worker_selection() {
        for field in RANK_ROUTING_FIELDS {
            let body = format!(r#"{{"model":"m","messages":[],"{field}":3}}"#);
            let error = prepare(&body).expect_err("rank routing must be rejected");
            assert!(
                matches!(error, RequestError::RankRoutingField { field: rejected } if rejected == field),
                "expected {field} to be rejected, got {error}"
            );
        }
    }

    #[test]
    fn non_object_and_malformed_bodies_are_rejected_before_any_request_exists() {
        assert!(matches!(
            prepare("[1,2,3]").unwrap_err(),
            RequestError::NotAnObject
        ));
        assert!(matches!(
            prepare("not json").unwrap_err(),
            RequestError::MalformedJson { .. }
        ));
    }

    #[test]
    fn unsupported_revision_produces_no_requests() {
        let error = prepare_pd_requests_with_random_room(
            contract(),
            "sglang-pd-v2",
            br#"{"model":"m","messages":[]}"#,
            &bootstrap(),
        )
        .expect_err("unsupported revision must fail");
        assert!(
            matches!(
                error,
                RequestError::UnsupportedRevision { ref revision } if revision == "sglang-pd-v2"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn body_limit_is_enforced_before_parsing() {
        assert!(enforce_body_limit(10, 10).is_ok());
        assert!(matches!(
            enforce_body_limit(11, 10).unwrap_err(),
            RequestError::BodyTooLarge { limit: 10 }
        ));
    }

    #[test]
    fn trusted_bootstrap_rejects_port_and_host_that_are_not_usable_targets() {
        assert_eq!(
            TrustedBootstrap::new("10.0.0.7", 0, 1).unwrap_err(),
            TrustError::ZeroBootstrapPort
        );
        assert_eq!(
            TrustedBootstrap::new("http://10.0.0.7", 8998, 1).unwrap_err(),
            TrustError::MalformedBootstrapHost
        );
        assert_eq!(
            TrustedBootstrap::new("10.0.0.7:8998", 8998, 1).unwrap_err(),
            TrustError::MalformedBootstrapHost
        );
        assert_eq!(
            TrustedBootstrap::new("10.0.0.7,10.0.0.8", 8998, 1).unwrap_err(),
            TrustError::MalformedBootstrapHost
        );
        assert_eq!(
            TrustedBootstrap::new("", 8998, 1).unwrap_err(),
            TrustError::EmptyBootstrapHost
        );
    }

    #[test]
    fn bootstrap_room_is_always_a_usable_non_zero_int64() {
        assert_eq!(
            TrustedBootstrap::new("h", 1, 0).unwrap_err(),
            TrustError::BootstrapRoomOutOfRange
        );
        assert_eq!(
            TrustedBootstrap::new("h", 1, -1).unwrap_err(),
            TrustError::BootstrapRoomOutOfRange
        );
        let generator = RandomRoomGenerator;
        for _ in 0..1000 {
            let room = generator.next_room();
            assert!(
                (MIN_ROOM..=MAX_ROOM).contains(&room),
                "room {room} unusable"
            );
        }
    }

    /// A stub generator pins room boundaries deterministically instead of
    /// relying on probabilistic pressure.
    #[test]
    fn room_generator_boundaries_are_revalidated() {
        for drawn in [0i64, -5, i64::MIN] {
            let error = prepare_pd_requests(
                contract(),
                PROTOCOL_REVISION,
                br#"{"model":"m","messages":[]}"#,
                &bootstrap(),
                &FixedRoom(drawn),
            )
            .expect_err("unusable room must be rejected");
            assert!(
                matches!(
                    error,
                    RequestError::Bootstrap {
                        source: TrustError::BootstrapRoomOutOfRange
                    }
                ),
                "room {drawn} must be rejected"
            );
        }

        let prepared = prepare_pd_requests(
            contract(),
            PROTOCOL_REVISION,
            br#"{"model":"m","messages":[]}"#,
            &bootstrap(),
            &FixedRoom(i64::MAX),
        )
        .expect("i64::MAX is a legal room");
        assert_eq!(prepared.prefill.room().unwrap(), i64::MAX);
    }
}
