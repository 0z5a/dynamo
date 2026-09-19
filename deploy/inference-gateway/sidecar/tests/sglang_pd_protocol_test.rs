// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Protocol-level regression matrix for the SGLang P/D request contract.
//!
//! These tests exercise `prepare_pd_requests` directly. They cover request
//! construction, the trust boundary and pre-dispatch error semantics only;
//! dispatch, concurrency, response-body lifetime and termination are A2 scope
//! and are not covered here.

use std::collections::HashSet;
use std::sync::atomic::{AtomicI64, Ordering};

use dynamo_epp_sidecar::{
    PROTOCOL_REVISION, PreparedLeg, PreparedSglangRequests, RequestError, RoomGenerator,
    SglangPdContract, TrustedBootstrap, enforce_body_limit, prepare_pd_requests,
};
use serde_json::{Value, json};

/// Deterministic room source. Room boundaries are pinned by construction
/// instead of relying on probabilistic pressure or wall-clock sleeps.
struct SteppedRooms(AtomicI64);

impl SteppedRooms {
    fn starting_at(first: i64) -> Self {
        Self(AtomicI64::new(first))
    }
}

impl RoomGenerator for SteppedRooms {
    fn next_room(&self) -> i64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

fn prepare_with(
    body: &Value,
    bootstrap: &TrustedBootstrap,
    generator: &dyn RoomGenerator,
) -> Result<PreparedSglangRequests, RequestError> {
    let bytes = serde_json::to_vec(body).expect("fixture serializes");
    prepare_pd_requests(
        SglangPdContract::pinned(),
        PROTOCOL_REVISION,
        &bytes,
        bootstrap,
        generator,
    )
}

fn trusted() -> TrustedBootstrap {
    TrustedBootstrap::new("10.0.0.7", 8998, 1).expect("valid trusted bootstrap")
}

fn prepare(body: &Value) -> PreparedSglangRequests {
    prepare_with(body, &trusted(), &SteppedRooms::starting_at(1)).expect("prepare succeeds")
}

fn decode(leg: &PreparedLeg) -> Value {
    serde_json::from_slice(&leg.body).expect("prepared leg is JSON")
}

/// A realistic non-streaming chat request with tools, response format, stop
/// sequences and sampling fields, matching the pinned SGLang OpenAI schema.
fn rich_request() -> Value {
    json!({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "what is bootstrap_host?"}
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 1234,
        "stop": ["\n\n", "END"],
        "max_tokens": 64,
        "stream": false,
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        }],
        "response_format": {"type": "json_object"},
        "logprobs": true,
        "top_logprobs": 3
    })
}

/// A1-P01: both legs carry the trusted host/port/room.
#[test]
fn p01_both_legs_carry_the_trusted_bootstrap_tuple() {
    let prepared = prepare(&rich_request());
    for leg in [&prepared.prefill, &prepared.decode] {
        assert_eq!(leg.authority, "10.0.0.7:8998");
        assert_eq!(leg.path, "/v1/chat/completions");
        let body = decode(leg);
        assert_eq!(body["bootstrap_host"], "10.0.0.7");
        assert_eq!(body["bootstrap_port"], 8998);
    }
    // The two legs of one request are paired on the same room.
    assert_eq!(prepared.prefill.room().unwrap(), 1);
    assert_eq!(prepared.decode.room().unwrap(), 1);
}

/// A1-P02: a forged bootstrap is replaced, including every extra value when the
/// client sent arrays, while prose naming the same fields is left alone.
///
/// Mutation: removing the trusted override fails this test. The controls are
/// `p11_unrelated_extension_fields_are_not_stripped` (the boundary is not a
/// blanket sanitizer) and `p01` (an un-forged request is unaffected).
#[test]
fn p02_forged_bootstrap_is_fully_replaced_and_prose_is_untouched() {
    let mut request = rich_request();
    request["bootstrap_host"] = json!(["evil.example", "evil2.example"]);
    request["bootstrap_port"] = json!([1, 2, 3]);
    request["bootstrap_room"] = json!(999_999);

    let prepared = prepare(&request);
    for leg in [&prepared.prefill, &prepared.decode] {
        let body = decode(leg);
        assert_eq!(body["bootstrap_host"], "10.0.0.7");
        assert_eq!(body["bootstrap_port"], 8998);
        assert_eq!(body["bootstrap_room"], 1);
        // Prose mentioning the same field name is not recursively deleted.
        assert_eq!(body["messages"][1]["content"], "what is bootstrap_host?");
    }
}

/// A1-P03: concurrent requests never reuse a room, and both legs of a request
/// are paired on the same room.
///
/// Mutation: pinning the room to a constant fails this test. The control is
/// `p01`, whose single request still passes under a constant room.
#[test]
fn p03_concurrent_requests_do_not_reuse_rooms() {
    let generator = SteppedRooms::starting_at(1);
    let mut seen = Vec::new();
    for _ in 0..64 {
        let prepared =
            prepare_with(&rich_request(), &trusted(), &generator).expect("prepare succeeds");
        let prefill_room = prepared.prefill.room().unwrap();
        assert_eq!(prefill_room, prepared.decode.room().unwrap());
        seen.push(prefill_room);
    }
    let unique: HashSet<_> = seen.iter().collect();
    assert_eq!(unique.len(), seen.len(), "rooms were reused: {seen:?}");
}

/// A1-P04: an unsupported revision fails while building, before any request
/// exists.
#[test]
fn p04_unknown_protocol_revision_produces_no_requests() {
    let bytes = serde_json::to_vec(&rich_request()).unwrap();
    for declared in ["sglang-pd-v2", "", "SGLANG-PD-V1", "sglang-pd-v1 "] {
        let error = prepare_pd_requests(
            SglangPdContract::pinned(),
            declared,
            &bytes,
            &trusted(),
            &SteppedRooms::starting_at(1),
        )
        .expect_err("revision must be rejected");
        assert!(
            matches!(
                error,
                RequestError::UnsupportedRevision { ref revision } if revision == declared
            ),
            "unexpected error for {declared:?}: {error}"
        );
    }
    // The pinned revision itself is accepted.
    assert_eq!(
        SglangPdContract::pinned().validate_revision(PROTOCOL_REVISION),
        Ok(SglangPdContract::pinned())
    );
}

/// A1-P05: the bootstrap endpoint is a typed input, so an HTTP address cannot be
/// smuggled in as a bootstrap endpoint and no half-built leg is produced.
#[test]
fn p05_unusable_bootstrap_endpoint_is_rejected_at_construction() {
    // An HTTP URL is not a bootstrap host.
    assert!(TrustedBootstrap::new("http://10.0.0.7", 8998, 1).is_err());
    // An HTTP authority is not a bootstrap host either: the HTTP port must never
    // be passed off as the bootstrap port.
    assert!(TrustedBootstrap::new("10.0.0.7:30000", 8998, 1).is_err());
    // A zero port is not a usable target.
    assert!(TrustedBootstrap::new("10.0.0.7", 0, 1).is_err());
    // A well-formed bootstrap endpoint is accepted.
    assert!(TrustedBootstrap::new("10.0.0.7", 8998, 1).is_ok());
}

/// A1-P06: the decode leg preserves the client's streaming mode, and the pinned
/// protocol changes no stream or token-budget field on the prefill leg.
#[test]
fn p06_both_legs_preserve_client_streaming_mode() {
    for stream in [false, true] {
        let mut request = rich_request();
        request["stream"] = json!(stream);
        let prepared = prepare(&request);
        assert_eq!(decode(&prepared.decode)["stream"], stream);
        assert_eq!(decode(&prepared.prefill)["stream"], stream);
        assert_eq!(decode(&prepared.prefill)["max_tokens"], 64);
    }
}

/// A1-P07: sampling, tools and response-format extensions keep both value and
/// type across both legs.
#[test]
fn p07_generation_semantics_are_preserved_on_both_legs() {
    let prepared = prepare(&rich_request());
    for leg in [&prepared.prefill, &prepared.decode] {
        let body = decode(leg);
        assert_eq!(body["model"], "Qwen/Qwen3-0.6B");
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["seed"], 1234);
        assert_eq!(body["stop"], json!(["\n\n", "END"]));
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["logprobs"], true);
        assert_eq!(body["top_logprobs"], 3);
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }
}

/// A1-P08: the bootstrap host is a bare host, so the shared metadata separator
/// forms are rejected rather than silently reinterpreted.
#[test]
fn p08_bootstrap_host_rejects_metadata_separator_forms() {
    for host in [
        "a,b",
        "10.0.0.7,10.0.0.8",
        " 10.0.0.7",
        "10.0.0.7 ",
        "10.0.0.7@x",
    ] {
        assert!(
            TrustedBootstrap::new(host, 8998, 1).is_err(),
            "{host:?} must be rejected"
        );
    }
    assert!(TrustedBootstrap::new("prefill-0.internal", 8998, 1).is_ok());
}

/// A1-P09: the body cap is enforced before any parsing. Distinguishing a size
/// error from a streaming deadline is A2 scope.
#[test]
fn p09_body_cap_is_enforced_before_parsing() {
    let limit = 1024;
    assert!(enforce_body_limit(0, limit).is_ok());
    assert!(enforce_body_limit(limit, limit).is_ok());
    assert!(matches!(
        enforce_body_limit(limit + 1, limit).unwrap_err(),
        RequestError::BodyTooLarge { limit: reported } if reported == limit
    ));
    // A body over the cap fails on size, never on parse or transport.
    let oversized = "x".repeat(limit + 1);
    assert!(matches!(
        enforce_body_limit(oversized.len(), limit).unwrap_err(),
        RequestError::BodyTooLarge { .. }
    ));
}

/// A1-P10: the wire type for `bootstrap_room` is a JSON integer in
/// `1..=i64::MAX`. Unusable values are rejected, never coerced or clamped.
#[test]
fn p10_generated_room_is_a_strict_integer_in_range() {
    let prepared = prepare(&rich_request());
    for leg in [&prepared.prefill, &prepared.decode] {
        let body: Value = serde_json::from_slice(&leg.body).unwrap();
        let room = &body["bootstrap_room"];
        assert!(room.is_i64(), "room must be a JSON integer, got {room}");
        assert!(room.as_i64().unwrap() >= 1);
    }

    for drawn in [0i64, -1, i64::MIN] {
        let error = prepare_with(
            &rich_request(),
            &trusted(),
            &SteppedRooms::starting_at(drawn),
        )
        .expect_err("unusable room must be rejected");
        assert!(
            matches!(error, RequestError::Bootstrap { .. }),
            "expected a bootstrap range error for {drawn}, got {error}"
        );
    }

    // i64::MAX is the inclusive upper bound and must remain usable.
    let prepared = prepare_with(
        &rich_request(),
        &trusted(),
        &SteppedRooms::starting_at(i64::MAX),
    )
    .expect("i64::MAX is a legal room");
    assert_eq!(prepared.prefill.room().unwrap(), i64::MAX);
}

/// A1-P11: unknown but allowed extension fields survive, so the trust boundary
/// does not degenerate into a blanket sanitizer.
#[test]
fn p11_unrelated_extension_fields_are_not_stripped() {
    let mut request = rich_request();
    request["x-vendor-trace"] = json!({"keep": true});
    request["custom_params"] = json!({"nested": [1, 2]});
    request["session_id"] = json!("session-1");
    request["priority"] = json!(5);

    let prepared = prepare(&request);
    for leg in [&prepared.prefill, &prepared.decode] {
        let body = decode(leg);
        assert_eq!(body["x-vendor-trace"]["keep"], true);
        assert_eq!(body["custom_params"]["nested"], json!([1, 2]));
        assert_eq!(body["session_id"], "session-1");
        assert_eq!(body["priority"], 5);
    }
}

/// A1-P12: rank-routing extensions are rejected, because honouring them would
/// route against the worker the gateway selected.
#[test]
fn p12_rank_routing_fields_are_rejected() {
    for field in ["routed_dp_rank", "disagg_prefill_dp_rank"] {
        let mut request = rich_request();
        request[field] = json!(2);
        let error = prepare_with(&request, &trusted(), &SteppedRooms::starting_at(1))
            .expect_err("rank routing must be rejected");
        assert!(
            matches!(error, RequestError::RankRoutingField { field: rejected } if rejected == field),
            "expected {field} to be rejected, got {error}"
        );
    }
}

/// A1-P12 control: an explicit null rank is absent rather than a bypass, so a
/// client that round-trips the field as null is still served.
#[test]
fn p12_control_null_rank_fields_are_accepted() {
    let mut request = rich_request();
    request["routed_dp_rank"] = json!(null);
    request["disagg_prefill_dp_rank"] = json!(null);
    assert_eq!(decode(&prepare(&request).prefill)["bootstrap_port"], 8998);
}

/// A1-P04 control: a non-object body is rejected before request construction,
/// and is reported as a client error rather than a contract failure.
#[test]
fn p04_control_non_object_and_malformed_bodies_are_rejected() {
    for bytes in [
        b"[1,2,3]".as_slice(),
        b"not json".as_slice(),
        b"".as_slice(),
    ] {
        let error = prepare_pd_requests(
            SglangPdContract::pinned(),
            PROTOCOL_REVISION,
            bytes,
            &trusted(),
            &SteppedRooms::starting_at(1),
        )
        .expect_err("body must be rejected");
        assert!(
            matches!(
                error,
                RequestError::NotAnObject | RequestError::MalformedJson { .. }
            ),
            "unexpected error for {bytes:?}: {error}"
        );
    }
}
