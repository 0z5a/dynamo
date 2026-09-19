# SGLang P/D request contract (protocol lock)

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

This document pins the upstream SGLang protocol facts the EPP decode sidecar's
SGLang P/D adapter depends on, and records which of them are verified and which
are still assumptions. It exists so a reviewer can check every claim against a
revision instead of trusting a summary.

Scope: request construction and the trust boundary for `/v1/chat/completions`.
Dispatch, concurrency, response-body lifetime and termination belong to the
concurrent-dispatch work and are deliberately absent here.

## 1. Verified engine revision

| Item | Value |
|---|---|
| Revision verified | `f4e0ac382e4e5d644f2fbe4a15c20da53500bbca` |
| Commit date | 2026-07-29T21:17:16-07:00 |
| Subject | `[misc] Remove unused multi_layer_draft_forward_cg module (#32881)` |
| Checkout | `python/sglang` subtree, clean working tree |
| Release / tag | **not pinned** — see §8 |
| Container digest | **not recorded** — no SGLang image was present to digest |

`PROTOCOL_REVISION` in `src/sglang_pd.rs` names the wire contract this module
implements (`sglang-pd-v1`); `VERIFIED_ENGINE_REVISION` names the engine revision
the field semantics were read from. A passing revision check says nothing about
the engine image actually deployed — that remains a deployment precondition,
because SGLang exposes no version-negotiation endpoint.

## 2. Request fields

`ChatCompletionRequest` (`python/sglang/srt/entrypoints/openai/protocol.py:720`)
declares, under `# For PD disaggregation` at lines 369–372:

| Field | Wire type |
|---|---|
| `bootstrap_host` | `str` or `List[str]` |
| `bootstrap_port` | `int` or `List[int \| null]` |
| `bootstrap_room` | `int` or `List[int]` |

The same triplet appears on the second request model at lines 833–835.

The call chain was traced past the schema, because a schema that accepts a field
does not prove the value reaches the transfer layer:

| Hop | Location |
|---|---|
| OpenAI handler | `serving_chat.py:761` |
| Internal request | `managers/io_struct.py:250–252` |
| Normalization | `io_struct.py:674–696` |
| Tokenizer manager | `managers/tokenizer_manager.py:1258–1281` |
| Scheduler | `managers/scheduler.py:2174–2175` |
| KV receiver | `disaggregation/decode.py:592` |

For `n > 1` a scalar `bootstrap_room` is expanded to `room + i` per choice
(`io_struct.py:694`). The sidecar always sends a scalar, one per request.

## 3. The bootstrap endpoint is not the HTTP endpoint

This is the fact most likely to be got wrong, so it is recorded with its
evidence. SGLang serves the cross-engine KV handshake on a listener separate
from its HTTP server, and a third, unrelated bootstrap server also exists.

| Listener | Default port | Source |
|---|---|---|
| HTTP server | `30000` | `server_args.py:1204` — `"The port of the HTTP server."` |
| **Disaggregation KV bootstrap** | **`8998`** | `server_args.py:2954` — `"Bootstrap server port on the prefill server."` |
| Engine-info bootstrap (unrelated) | `6789` | `server_args.py:3099` |

Startup evidence that the KV bootstrap server is a distinct listener, started
only on prefill (`managers/disagg_service.py:18–33`):

```python
if disagg_mode == DisaggregationMode.PREFILL:
    # only start bootstrap server on prefill tm
    bootstrap_server = kv_bootstrap_server_class(
        host=server_args.host,
        port=server_args.disaggregation_bootstrap_port,
    )
```

Concrete server class: `disaggregation/nixl/conn.py:2891` `NixlKVBootstrapServer`
(also moonrise/mooncake/mori/ascend variants). The connector binds the configured
value at `disaggregation/common/conn.py:164–165`.

Upstream's own router treats the HTTP address and the bootstrap port as
**separate fields** (`experimental/sgl-router/src/workers/worker.rs:90–99`),
carrying `bootstrap_port: Option<u16>` beside the worker URL and documenting it as
*"Set via `--disaggregation-bootstrap-port` at worker startup"*.

**Therefore the HTTP port must never be passed off as the bootstrap port.**
Doing so would make the decode leg dial an HTTP server and never reach the KV
bootstrap server.

## 4. Where the bootstrap endpoint comes from — and why this slice takes it as input

The EPP path gives the sidecar exactly one routing input
(`deploy/inference-gateway/sidecar/src/metadata.rs:8`):

```rust
pub const PREFILLER_HOST_PORT: &str = "x-prefiller-host-port";
pub struct PrefillEndpoint(Authority);   // HTTP host + port only
```

That authority is the selected prefill worker's **HTTP** address. No bootstrap
port — and no bootstrap host distinct from the HTTP host — appears anywhere in
`deploy/inference-gateway/` or `lib/kv-router/`.

The value does exist upstream and is published by the engine: the native SGLang
sidecar reads it from discovery, not from the HTTP endpoint
(`lib/sidecar/sglang/src/engine.rs:557–576`):

```rust
fn discovery_bootstrap_port(discovery: &Discovery) -> Result<Option<u16>, DynamoError> {
    client::json_u64(&discovery.server_info, "disaggregation_bootstrap_port")
    // rejects port == 0 with:
    // "prefill SGLang server did not report disaggregation_bootstrap_port"
}
```

`server_info` is the parsed `GetServerInfo.json_info` payload — a separate
discovery channel.

Because that channel is not part of the EPP contract, `prepare_pd_requests`
takes the bootstrap endpoint as an **explicit typed input**
(`TrustedBootstrap`) rather than deriving it. Deriving it from
`x-prefiller-host-port` would be the exact substitution this document forbids.
Supplying the real value is a separate, minimal contract decision
(see §8, open item 1).

## 5. `bootstrap_room`: type, range and uniqueness

| Property | Value | Evidence |
|---|---|---|
| Declared wire type | signed 64-bit integer | `protocol.py:371`; enforced by `lib/sidecar/sglang/src/protocol.rs:313` (`i64::try_from`) |
| Internal metadata buffer | `torch.uint64` (CUDA), `int64` (NPU) | `disaggregation/utils.py:280–286` |
| Non-negative required | yes — used as `room % dp_size` | `disaggregation/decode.py:575` |
| `0` means "absent" | yes | `decode.py:1798` logs `(bootstrap_room=0)` as the no-room case |
| Valid range used here | `1..=i64::MAX` | `src/sglang_pd.rs` `MIN_ROOM`/`MAX_ROOM` |

Zero and negatives are unusable: the value indexes an array modulo the prefill
data-parallel size, so a negative room is a meaningless index and `0` collides
with the engine's own "no room" sentinel. Rooms are therefore drawn in
`1..=i64::MAX` and re-validated through `TrustedBootstrap` after drawing, so a
broken generator surfaces as an error rather than a request the engine
misroutes.

**Uniqueness is probabilistic, not coordinated.** `RandomRoomGenerator` draws a
random 63-bit value, matching the shape used by the native SGLang sidecar
(`lib/sidecar/sglang/src/protocol.rs:279`,
`rand::random::<u64>() & (i64::MAX as u64)`). Two independent gateways can draw
the same room. The room is a demultiplexing key for an in-flight KV handoff
between a paired prefill and decode worker, so a cross-gateway collision only
matters if both requests reach the same prefill worker concurrently. This
limitation is recorded rather than hidden; a coordinated allocator would need
shared state the sidecar does not have.

Note that the native sidecar's mask can yield `0`, which this module avoids by
forcing the low bit.

## 6. Supported fields on each leg

Both legs carry **the same** trusted bootstrap tuple and are otherwise
semantically identical. Engine mode, not request content, determines which leg
computes prefill, so the sidecar does not change `stream`, `max_tokens` or any
other generation field per leg. There is no vLLM-style `max_tokens = 1` prefill
rewrite, and none is invented here.

Preserved unchanged on both legs: `model`, `messages`, `temperature`, `top_p`,
`seed`, `stop`, `max_tokens`, `stream`, `tools`, `tool_choice`,
`response_format`, `logprobs`, `top_logprobs`, `session_id`, `priority`,
`custom_params`, and any unknown vendor extension.

Owned and overwritten by the sidecar: `bootstrap_host`, `bootstrap_port`,
`bootstrap_room`. A client value is discarded entirely — including every extra
value when the client sends an array — and is never merged or truncated
per-element.

Rejected rather than forwarded: `routed_dp_rank`, `disagg_prefill_dp_rank`.
Both steer the engine to a specific data-parallel worker and so can bypass the
worker the gateway selected. An explicit `null` is treated as absent, because a
client that round-trips the field as `null` is not attempting a bypass. Proper
support for an explicit rank requires a gateway-owned selection field (EPP
metadata) and is out of scope here.

Text that merely mentions a bootstrap field name inside `messages` or a tool
schema is untouched: the trust boundary applies to structured protocol fields,
not to arbitrary strings.

## 7. Error semantics before dispatch

All of the following are raised while building the request, so no engine request
is ever sent:

| Condition | Reported as | HTTP |
|---|---|---|
| Protocol revision not supported | `UnsupportedRevision` | 500 `pd_contract_error` |
| Body over the configured cap | `BodyTooLarge` | 400 `invalid_request` |
| Body is not valid JSON | `MalformedJson` | 400 `invalid_request` |
| Body is JSON but not an object | `NotAnObject` | 400 `invalid_request` |
| Client supplied a rank-routing field | `RankRoutingField` | 400 `invalid_request` |
| Trusted bootstrap unusable | `Bootstrap { .. }` | 500 `pd_contract_error` |
| Leg failed to serialize | `Serialize { .. }` | 500 `pd_contract_error` |

A malformed client body is the client's problem and is reported as such; a
sidecar configuration or contract failure is not disguised as a bad request.

Distinguishing a **size** failure from a **streaming deadline** failure is part
of this taxonomy: `enforce_body_limit` owns only the size decision. Enforcing the
cap against a client that never finishes sending, and mapping that to a distinct
deadline error, belongs to the caller that owns the body read — it is not
implemented here and is not claimed.

## 8. Open items and deployment preconditions

These are **not** verified by this document.

1. **Bootstrap endpoint source (blocking for dispatch).** `TrustedBootstrap` must
   be supplied from a gateway-owned source. The engine already publishes the
   value; the EPP contract does not yet carry it. Until that is decided, no
   adapter is wired to `PdAdapter::execute`, and the HTTP port is never used as
   a substitute.
2. **Release pin.** The verified revision is a commit on `main`, not a tagged
   release, and the second available checkout differs. A release/tag plus a
   container digest must be pinned before the adapter is enabled for serving.
3. **Transfer backend and device limits.** The bootstrap server class is
   backend-specific (NIXL, mooncake, mori, ascend). This document does not claim
   that any particular connector works on any particular hardware, and no
   cross-node or RDMA capability is implied.
4. **Engine cancellation.** Whether closing the HTTP connection aborts an
   in-flight KV transfer is not established here. Do not assume partial-write
   limits.
5. **Engine-side verification.** Every fact above is read from source. No claim
   is made that a live engine accepted these requests; that requires the L2/L3
   validation layers and a pinned image.

## 9. Interface for the concurrent-dispatch work

`prepare_pd_requests` is pure and synchronous: it takes bytes plus trusted
inputs and returns two prepared legs. It performs no I/O, so the caller owns the
body read (including the byte cap and the client-body deadline), both legs'
dispatch, response-body lifetime and termination. That split is deliberate: the
concurrency and termination semantics are defined separately, and no sequential
fallback exists that could become a wrong default.
