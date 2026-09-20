<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# EPP Decode Sidecar

The EPP decode sidecar is the pod-local HTTP data plane for standalone
disaggregated routing. It accepts the original OpenAI-compatible request from
Gateway and selects one of two paths:

- Without `x-prefiller-host-port`, it proxies the request to the local decode
  engine.
- With exactly one valid `x-prefiller-host-port`, it removes that header and
  invokes the configured backend P/D adapter.

Empty, malformed, repeated, or comma-separated prefill endpoint values return
`502 Bad Gateway` with the OpenAI-style error code `invalid_epp_metadata`.

The binary listens on `0.0.0.0:8000` and proxies to
`http://localhost:8001` by default. Set `DYN_SIDECAR_PORT` and
`DYN_DECODE_ENGINE_PORT` to change the ports. Upstream connections time out
after 10 seconds by default, and stalled response reads time out after 300
seconds without imposing a deadline on the full response stream. Configure
these values in milliseconds with `DYN_SIDECAR_CONNECT_TIMEOUT_MS` and
`DYN_SIDECAR_READ_TIMEOUT_MS`. Active requests drain for 30 seconds during
shutdown before their streams are forced closed. Configure this deadline with
`DYN_SIDECAR_DRAIN_TIMEOUT_MS`.

`GET /health` remains live while requests drain. `GET /ready` returns `200 OK`
while the sidecar accepts requests and `503 Service Unavailable` once draining
starts. Readiness does not indicate that EPP endpoint propagation is complete.

## P/D adapters

Backend-specific P/D execution is selected with `DYN_ADAPTER_MODE`:

| Value | Effect |
|---|---|
| `none` (default) | No adapter. Requests carrying a valid prefill endpoint return `501 Not Implemented`; decode-only passthrough is available. |
| `trtllm-context-first` | The TRT-LLM context-first handoff. |

An unrecognised value fails at startup rather than falling back, so a typo
cannot leave the sidecar running with no adapter. The effective mode, both
engine URLs, every budget and the request-id namespace are logged once at
startup.

### `trtllm-context-first`

The context (prefill) leg runs first and must produce a valid handoff before a
generation leg exists, so a context failure can never start generation. A
context leg that already carries a terminal `finish_reason` ends the request
there and is returned as the response; no generation request is sent and no
token is added. The protocol lock, including the pinned upstream revision, is in
`TRTLLM_CONTEXT_FIRST_PROTOCOL.md`.

| Variable | Default | Meaning |
|---|---|---|
| `DYN_CONTEXT_ENGINE_URL` | required | Worker that runs the context leg |
| `DYN_CONTEXT_FIRST_REQUEST_BYTES` | 1048576 | Maximum accepted client request body |
| `DYN_CONTEXT_FIRST_HANDOFF_BYTES` | 8388608 | Maximum accepted context response body |
| `DYN_CONTEXT_FIRST_RESPONSE_BYTES` | 16777216 | Maximum accepted body for any leg response |
| `DYN_CONTEXT_FIRST_CONTEXT_DEADLINE_MS` | 60000 | Total deadline for the context leg |
| `DYN_CONTEXT_FIRST_GENERATION_DEADLINE_MS` | 300000 | Total deadline for the generation leg |
| `DYN_CONTEXT_FIRST_NODE_ID` | 0 | Request-id namespace node, `[0, 256)` |
| `DYN_CONTEXT_FIRST_PROCESS_ID` | 0 | Request-id namespace process, `[0, 64)` |

The generation leg targets `DYN_DECODE_ENGINE_PORT`. Both the namespace and the
context engine URL are required configuration rather than derived values: two
gateways sharing an id namespace would collide, and no context endpoint is
carried by the selected-prefill metadata.

### Error codes

The failure taxonomy keeps distinct causes distinct, so a misconfigured endpoint
is not reported as a slow engine:

| Code | Status | Cause |
|---|---|---|
| `trtllm_context_unreachable` | 502 | The worker could not be connected to |
| `trtllm_context_read_stall` | 504 | The worker stopped sending during the context leg |
| `trtllm_generation_read_stall` | 504 | The worker stopped sending during the generation leg |
| `trtllm_context_deadline_exceeded` | 504 | The context leg exceeded its total deadline |
| `trtllm_generation_deadline_exceeded` | 504 | The generation leg exceeded its total deadline |
| `trtllm_worker_status` | 502 | The worker returned a non-success status |
| `trtllm_worker_response_too_large` | 502 | A leg response exceeded its cap |
| `trtllm_context_handoff_invalid` | 502 | The context response carried no usable handoff |
| `client_body_deadline_exceeded` | 408 | The client did not finish sending its body |
| `invalid_request` | 400 | The client body was rejected before any leg ran |
| `streaming_not_supported` | 501 | The request asked for a streaming response |

**Streaming is not supported.** The context-first adapter buffers the generation
leg and returns one JSON body, so a streaming request is refused with
`501 Not Implemented` rather than answered with a body an SSE parser cannot read.
The context leg is always non-streaming in any case.

Every failure above the generation leg is a pre-response failure and is
answerable with a status; a transport failure during the generation leg is not,
because the response has already begun.
