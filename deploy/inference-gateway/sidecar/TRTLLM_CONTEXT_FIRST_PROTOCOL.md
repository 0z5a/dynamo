# TRT-LLM context-first handoff (protocol lock)

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Pins the upstream TRT-LLM protocol facts the EPP decode sidecar's context-first
adapter depends on. Every claim carries a file and line reference against the
pinned revision; nothing here is inferred from a summary.

Scope: the context leg request, the handoff validation, and the generation leg
request. Dispatch lifetime, cancellation and the response-body ownership rules
are a separate concern.

## 1. Pin

| Item | Value |
|---|---|
| Release tag | `v1.3.0rc24` |
| Commit SHA | `1cef02e901be43081b1ba6d4981e94ed3bd9c1e8` |
| Commit date | 2026-08-09T03:44:40-07:00 |
| Container digest | **UNSET** — deployment precondition, see §8 |
| Model revision | **UNSET** — deployment precondition, see §8 |

A development checkout at `7e78fdbaac61d96b607ba3bb6480184a4ffbfb16` sits 673
commits past this tag on a different line and differs in protocol fields
(`resolved_thinking` exists there and **not** here; `conversation_id` exists here
and not there). Reading the development checkout would have produced an adapter
that reads a field this release never sends.

## 2. Routing

| Request type | Method and path |
|---|---|
| `CompletionRequest` | `POST v1/completions` |
| `ChatCompletionRequest` | `POST v1/chat/completions` |

Source: `tensorrt_llm/serve/openai_client.py:76`.

The context leg is forced **non-streaming**; the generation leg keeps the
client's streaming mode.

## 3. `request_type`

`DisaggregatedParams.request_type` is a required `str` inside
`disaggregated_params`; there is no top-level field. The complete set is three
values (`tensorrt_llm/disaggregated_params.py:70`):

| Wire value | Meaning | In scope |
|---|---|---|
| `context_only` | prefill only, produces the handoff | yes |
| `generation_only` | decode using the handoff | yes |
| `context_and_generation` | combined; not context-first | **no — reject explicitly** |

`__post_init__` lowercases and validates the value, and an unknown value raises
`ValueError`. The adapter must reject the third value rather than ignore it or
map it to a default.

## 4. Fixtures

These four fixtures are derived from the pinned producer and consumer code. They
are illustrative of the wire shape, not captured from a live engine — §8 records
the layer that would capture them.

### 4.1 Client request (to the sidecar)

```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "Name one colour."}
  ],
  "max_tokens": 32,
  "stream": false
}
```

The client sends no `disaggregated_params`; that field is owned by the gateway.

### 4.2 Derived context request

Built by the adapter's equivalent of `_get_ctx_request`
(`openai_disagg_service.py`). Note `stream` is forced `false` and
`stream_options` cleared.

```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "messages": [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "Name one colour."}
  ],
  "max_tokens": 32,
  "stream": false,
  "disaggregated_params": {
    "request_type": "context_only",
    "disagg_request_id": 1234567890123456789,
    "schedule_style": "context_first",
    "conversation_id": "conv-8f21",
    "return_prompt_token_ids_b64": false
  }
}
```

### 4.3 Context response

`POST v1/chat/completions` returns a non-streaming completion. The handoff lives
at `choices[0].disaggregated_params`; the prompt tokens are on the **response
object**, and `finish_reason` decides whether a generation leg is sent at all.

```json
{
  "id": "chatcmpl-3f1c...",
  "object": "chat.completion",
  "created": 1757000000,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt_token_ids": [1, 9038, 2501, 29871, 13],
  "prompt_token_ids_b64": null,
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": ""},
      "finish_reason": "length",
      "disaggregated_params": {
        "request_type": "context_only",
        "disagg_request_id": 1234567890123456789,
        "ctx_request_id": 42,
        "conversation_id": "conv-8f21",
        "first_gen_tokens": [15043],
        "encoded_opaque_state": "<opaque string, relayed verbatim>",
        "ctx_usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
      }
    }
  ],
  "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
}
```

### 4.4 Derived generation request

Built by the adapter's equivalent of `_get_gen_request`. The request type
changes, the string prompt is replaced by the token representation, and the
handoff fields are carried forward.

```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "prompt_token_ids": [1, 9038, 2501, 29871, 13],
  "max_tokens": 32,
  "stream": false,
  "disaggregated_params": {
    "request_type": "generation_only",
    "disagg_request_id": 1234567890123456789,
    "ctx_request_id": 42,
    "conversation_id": "conv-8f21",
    "first_gen_tokens": [15043],
    "encoded_opaque_state": "<opaque string, relayed verbatim>",
    "ctx_dp_rank": null,
    "ctx_info_endpoint": null,
    "schedule_style": "context_first",
    "ctx_usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
  }
}
```

## 5. Handoff rules

| Rule | Source |
|---|---|
| Handoff is `choices[0].disaggregated_params`, not a top-level `kv_transfer_params` | `openai_protocol.py:774`, `:266` |
| Prompt tokens are on the response object | `openai_protocol.py:785+` |
| `prompt_token_ids_b64` wins when present, else `prompt_token_ids` | `_get_gen_request` |
| The two token forms are mutually exclusive per request | same |
| base64 payload is relayed verbatim; never decoded and re-encoded | same |
| `encoded_opaque_state` is opaque; relay byte-for-byte, trim no sub-field | `openai_protocol.py:200` |
| `prompt_token_ids_b64` is internal (`"Not for clients"`) — reject from clients | `openai_protocol.py:849` |
| `ctx_request_id` and `disagg_request_id` are different identities | `openai_protocol.py:199`, `:202` |
| `conversation_id` exists in this pin | `openai_protocol.py:206` |
| Multimodal E/PD over this protocol is unsupported | `TODO(TRTLLM-12407)`, `:208` |

### 5.1 The gate that must be modelled

```python
_GEN_PENDING_FINISH_REASONS = ("length", "not_finished")
```

Source: `tensorrt_llm/serve/openai_disagg_service.py:41`.

Generation is sent **only** when the context leg's `finish_reason` is one of
those two. Otherwise upstream deletes `disaggregated_params` and stops: the
context leg produced the whole answer, and no KV transfer happens. This is a
distinct terminal state — not a failure, not a normal success, and never a reason
to add a token.

## 6. First token

The generation worker replays the first token itself
(`req.add_new_token(first_gen_tokens[beam], beam)` in
`_torch/pyexecutor/py_executor.py`), validating `len(first_gen_tokens) >= beam_width`
and comparing against the EOS id.

**The sidecar must not prepend `first_gen_tokens`.** Prepending duplicates a
token; dropping them loses one. Both directions need a test, and neither can be
proven against a mock that agrees with itself.

## 7. Global request ID

Layout (`tensorrt_llm/llmapi/disagg_utils.py:532`):
`0(1) | timestamp_ms(39) | node_id(8) | process_id(6) | counter(10)`, folded into
the positive int64 range.

| Property | Value |
|---|---|
| Clock | `time.monotonic()` (not wall clock) |
| `node_id` | `[0, 256)`; default `uuid.getnode() % 256` |
| `process_id` | `[0, 64)`; from `TRTLLM_DISAGG_WORKER_PROCESS_ID`, default `0` |
| Counter | 10 bits, module-global from 0; **wraps at 1024 per millisecond** |

Two rules follow:

- **Do not adopt `uuid.getnode() % 256`.** The default carries upstream's own
  comment that operators must set `node_id` manually if collisions occur, and
  `uuid.getnode()` can return a random value on hosts without a usable MAC.
  The deployment must configure the namespace explicitly.
- **The identifier is revisable.** A context retry may replace
  `disagg_request_id`, and the context response may carry its own authoritative
  value which then wins. "Mint once and put it on both legs" is therefore wrong.

The 1024-per-millisecond counter wrap is the collision boundary to test.

## 8. Sampling: neither leg rewrites it

Verified against the pin. The context leg changes exactly three things and no
sampling field (`openai_disagg_service.py:203`):

```python
ctx_request = request.model_copy(
    update={
        "disaggregated_params": DisaggregatedParams(...),
        "stream": False,
        "stream_options": None,
    }
)
```

`max_tokens` appears **nowhere** in `openai_disagg_service.py` — neither in
`_get_ctx_request` nor in `_get_gen_request`. So the client's sampling settings
are carried unchanged on both legs, and there is **no vLLM-style `max_tokens = 1`
prefill rewrite**. The adapter must not invent one.

### 8.1 `conversation_id` is required on the context request in this pin

`_get_ctx_request` sets `conversation_id=self._get_conversation_id(request)`
(`:206`, `:213`), and `DisaggregatedParams.conversation_id` exists in this pin
(`openai_protocol.py:206`). Any adapter written against the newer development
line, which lacks this field, would omit it here.

## 9. Open items and deployment preconditions

Not established here; each is required before the adapter can be enabled.

1. **Container digest.** No TRT-LLM image is present on the host and `nvcr.io`
   returns 401 without credentials, so no digest could be recorded. The tag+SHA
   pins the *protocol*; the digest pins the *deployment*.
2. **Model revision.** No engine has been started, so no model identity is pinned.
3. **Cancellation semantics.** Whether closing the HTTP connection aborts an
   in-flight transfer, and what reclaims the reservation, is not established.
   Do not assume partial-write limits.
4. **Engine acceptance.** Every fact here is read from source. No live engine has
   accepted these requests; that requires the native two-GPU baseline, which is
   `NOT_RUN` because no engine image is available to this account.
