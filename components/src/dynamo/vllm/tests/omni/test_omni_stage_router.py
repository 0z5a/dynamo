# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for OmniStageRouter."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dynamo.common.storage import get_fs
from dynamo.common.utils.output_modalities import RequestType

try:
    from dynamo.vllm.omni import stage_router
    from dynamo.vllm.omni.output_formatter import OutputFormatter
    from PIL import Image
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
]

_MEMORY_FS = get_fs("memory://a3-router-tests")


def _image(size=(4, 4)):
    """A real small CPU image; no diffusion model is involved."""
    return Image.new("RGB", size, color="red")


def _audio_stage(samples, sample_rate=24000):
    return SimpleNamespace(
        final_output_type="audio",
        multimodal_output={
            "audio": np.asarray(samples, dtype=np.float32),
            "sr": sample_rate,
        },
    )


class _Chunk:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class _StageClient:
    def __init__(self, handler):
        self._handler = handler

    async def round_robin(self, request):
        async def _gen():
            payload = await self._handler(request)
            yield _Chunk(payload)

        return _gen()


class _SpyFormatter(OutputFormatter):
    """A real formatter that records what the router handed it."""

    def __init__(self, **kwargs):
        super().__init__(model_name="test-model", **kwargs)
        self.calls = []

    async def format(self, context, stage_output, session):
        self.calls.append((context, stage_output, session))
        return await super().format(context, stage_output, session)


def _make_stage_cfg(stage_id: int):
    return SimpleNamespace(
        stage_id=stage_id,
        engine_args=SimpleNamespace(model_stage=f"stage{stage_id}"),
    )


def _make_router(
    stage_configs,
    stage_clients,
    formatter=None,
    output_modalities=None,
    media_fs=_MEMORY_FS,
):
    router = stage_router.OmniStageRouter.__new__(stage_router.OmniStageRouter)
    router.config = SimpleNamespace(
        output_modalities=output_modalities,
        model="test-model",
        served_model_name=None,
    )
    router.stage_configs = stage_configs
    router.stage_clients = stage_clients
    router._formatter = (
        formatter if formatter is not None else _SpyFormatter(media_fs=media_fs)
    )
    return router


def _stage_handler(payload, seen=None):
    async def _handler(request):
        if seen is not None:
            seen.update(request)
        return payload

    return _handler


async def _generate(router, request, request_type, request_id="req-1"):
    """Drive ``generate`` with the parse and id patches its tests need."""
    with (
        patch(
            "dynamo.vllm.omni.stage_router.parse_request_type",
            return_value=(None, request_type),
        ),
        patch("dynamo.vllm.omni.stage_router.uuid.uuid4", return_value=request_id),
    ):
        return [c async for c in router.generate(request, None)]


def test_router_loads_stage_configs_from_model_deploy_config():
    config = SimpleNamespace(
        model="zai-org/GLM-Image",
        served_model_name=None,
        media_output_fs_url=None,
        media_output_http_url=None,
        default_video_fps=16,
    )
    stage_configs = [_make_stage_cfg(0)]

    with (
        patch(
            "dynamo.vllm.omni.stage_router.resolve_stage_configs",
            return_value=("/deploy/glm_image.yaml", stage_configs),
        ) as resolve_stage_configs,
        patch("dynamo.vllm.omni.stage_router.OutputFormatter") as output_formatter,
    ):
        router = stage_router.OmniStageRouter(config, "/deploy/glm_image.yaml")

    resolve_stage_configs.assert_called_once_with(
        config.model,
        trust_remote_code=False,
        deploy_config_path="/deploy/glm_image.yaml",
    )
    output_formatter.assert_called_once()
    assert router.stage_configs == stage_configs


# ── issue-004: opaque router ──────────────────────────────


@pytest.mark.asyncio
async def test_generate_passes_stage_connector_refs_opaquely():
    """Router must pass stage_connector_refs from stage output to next stage unchanged."""
    stage1_received = {}

    async def stage0_handler(request):
        return {
            "original_prompt": {"prompt": "hi"},
            "stage_connector_refs": {"0": {"shm_name": "abc", "size": 42}},
            "finished": True,
        }

    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(
                _stage_handler({"shm_meta": {"x": 1}}, stage1_received)
            ),
        },
    )

    with patch.object(stage_router, "shm_deserialize", return_value=SimpleNamespace()):
        await _generate(router, {"prompt": "x"}, RequestType.CHAT_COMPLETION)

    # Router must forward stage_connector_refs and original_prompt verbatim — never inspect them.
    assert stage1_received["stage_connector_refs"] == {
        "0": {"shm_name": "abc", "size": 42}
    }
    assert stage1_received["original_prompt"] == {"prompt": "hi"}
    assert stage1_received["request_id"] == "req-1"
    # 'finished' must be stripped — it is a router signal, not a stage protocol field.
    assert "finished" not in stage1_received


@pytest.mark.asyncio
async def test_generate_concurrent_requests_have_independent_connector_refs():
    """Concurrent requests must carry independent stage_connector_refs (no cross-leakage)."""
    stage1_refs_by_request: dict = {}
    event = asyncio.Event()

    async def stage0_handler(request):
        rid = request["request_id"]
        return {
            "original_prompt": {"prompt": "x"},
            "stage_connector_refs": {"0": f"ref-for-{rid}"},
            "finished": True,
        }

    async def stage1_handler(request):
        rid = request["request_id"]
        if rid == "req-A":
            await event.wait()
        else:
            event.set()
        stage1_refs_by_request[rid] = request.get("stage_connector_refs")
        return {"shm_meta": {"x": 1}, "finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(stage1_handler),
        },
    )

    async def run_one(request_id):
        with patch.object(
            stage_router, "shm_deserialize", return_value=SimpleNamespace()
        ):
            return await _generate(
                router, {"prompt": "x"}, RequestType.CHAT_COMPLETION, request_id
            )

    await asyncio.gather(run_one("req-A"), run_one("req-B"))

    assert stage1_refs_by_request["req-A"] == {"0": "ref-for-req-A"}
    assert stage1_refs_by_request["req-B"] == {"0": "ref-for-req-B"}


@pytest.mark.asyncio
async def test_generate_stage_error_stops_pipeline():
    """Error from any stage must immediately stop the pipeline; later stages must not run."""
    stage1_called = False

    async def stage0_handler(request):
        return {"error": "thinker exploded", "finished": True}

    async def stage1_handler(request):
        nonlocal stage1_called
        stage1_called = True
        return {"shm_meta": {"x": 1}, "finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0), _make_stage_cfg(1)],
        stage_clients={
            "stage0": _StageClient(stage0_handler),
            "stage1": _StageClient(stage1_handler),
        },
    )

    chunks = await _generate(router, {"prompt": "x"}, RequestType.CHAT_COMPLETION)

    assert chunks == [{"error": "thinker exploded", "finished": True}]
    assert not stage1_called


# ── existing tests (formatting + error paths) ────────────


@pytest.mark.asyncio
async def test_generate_delegates_formatting_to_output_formatter():
    """Final stage output should be deserialized and passed to OutputFormatter."""
    formatter = _SpyFormatter(media_fs=_MEMORY_FS)
    fake_result = SimpleNamespace(final_output_type="image", images=[_image()])
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={
            "stage0": _StageClient(_stage_handler({"shm_meta": {"some": "meta"}}))
        },
        formatter=formatter,
    )

    request = {"prompt": "x", "response_format": "b64_json"}
    with patch.object(stage_router, "shm_deserialize", return_value=fake_result):
        chunks = await _generate(
            router, request, RequestType.IMAGE_GENERATION, "req-fmt"
        )

    assert len(chunks) == 1
    assert chunks[0]["data"][0]["b64_json"] is not None
    context, stage_output, session = formatter.calls[0]
    assert stage_output is fake_result
    assert context.response_format == "b64_json"
    assert session.request_id == "req-fmt"


@pytest.mark.asyncio
async def test_generate_yields_error_when_no_shm_meta():
    """When final stage returns no shm_meta, generate yields an error."""
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"finished": True}))},
    )

    chunks = await _generate(router, {"prompt": "x"}, RequestType.CHAT_COMPLETION, "r")

    assert chunks == [{"error": "No SHM output from final stage", "finished": True}]


# ── issue-007: router forwards raw request to stage 0 ────────────


@pytest.mark.asyncio
async def test_generate_forwards_raw_request_to_stage0():
    """Stage 0 must receive the raw request fields + request_id (no router parsing)."""
    stage0_received = {}
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={
            "stage0": _StageClient(
                _stage_handler({"shm_meta": {"x": 1}}, stage0_received)
            )
        },
    )

    request = {
        "prompt": "a dog",
        "size": "832x480",
        "nvext": {"num_inference_steps": 30},
    }
    with patch.object(stage_router, "shm_deserialize", return_value=SimpleNamespace()):
        await _generate(router, request, RequestType.VIDEO_GENERATION, "req-raw")

    assert stage0_received["request_id"] == "req-raw"
    assert stage0_received["prompt"] == "a dog"
    assert stage0_received["size"] == "832x480"
    assert stage0_received["nvext"] == {"num_inference_steps": 30}


# ── A3-T05: a legal gap is not a router error ─────────────


@pytest.mark.asyncio
async def test_a3_t05_buffered_audio_stage_is_not_reported_as_an_error():
    """The aggregated path calls the same no-emission step buffering."""
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
    )

    request = {"input": "hi", "data_source": "b64_json"}
    with patch.object(
        stage_router, "shm_deserialize", return_value=_audio_stage(np.zeros(2400))
    ):
        chunks = await _generate(router, request, RequestType.AUDIO_GENERATION)

    assert len(chunks) == 1
    assert chunks[0]["status"] == "completed"
    assert chunks[0]["error"] is None


@pytest.mark.asyncio
async def test_a3_t05_router_does_not_error_when_a_text_stage_has_no_payload():
    """A text stage that produced nothing mid-stream is a gap, not a failure.

    The router formats one stage, so the gap is only observable while the
    request is still building; it must not be turned into a pipeline error.
    """
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
    )
    empty_text = SimpleNamespace(final_output_type="text", request_output=None)

    with patch.object(stage_router, "shm_deserialize", return_value=empty_text):
        chunks = await _generate(router, {"prompt": "x"}, RequestType.CHAT_COMPLETION)

    # Nothing was emitted, so the request end reports one explicit failure body
    # rather than a stream of router errors.
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["finish_reason"] == "error"
    assert chunks[0]["choices"][0]["delta"]["content"].startswith("Error:")


# ── A3-T06: unknown final modality ────────────────────────


@pytest.mark.asyncio
async def test_a3_t06_unknown_final_modality_is_an_explicit_failure():
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
    )

    with patch.object(
        stage_router,
        "shm_deserialize",
        return_value=SimpleNamespace(final_output_type="hologram"),
    ):
        chunks = await _generate(router, {"prompt": "x"}, RequestType.CHAT_COMPLETION)

    assert len(chunks) == 1
    assert "hologram" in chunks[0]["choices"][0]["delta"]["content"]


# ── A3-T13: URL without storage ───────────────────────────


@pytest.mark.asyncio
async def test_a3_t13_url_without_storage_fails_before_any_stage_runs():
    stage0_called = []

    async def stage0_handler(request):
        stage0_called.append(request)
        return {"shm_meta": {"x": 1}, "finished": True}

    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(stage0_handler)},
        media_fs=None,
    )

    chunks = await _generate(
        router,
        {"prompt": "a drone", "response_format": "url"},
        RequestType.VIDEO_GENERATION,
    )

    assert stage0_called == []
    assert len(chunks) == 1
    assert "media storage" in chunks[0]["error"]
    assert chunks[0]["finished"] is True


# ── A3-T14: a failed upload maps to the modality failure body ─────────────


@pytest.mark.asyncio
async def test_a3_t14_upload_failure_yields_the_audio_failure_body(monkeypatch):
    async def _boom(*args, **kwargs):
        raise OSError("storage backend refused the upload")

    monkeypatch.setattr("dynamo.vllm.omni.output_formatter.upload_to_fs", _boom)
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
    )
    request = {"input": "hi", "data_source": "url"}

    with patch.object(
        stage_router, "shm_deserialize", return_value=_audio_stage(np.zeros(2400))
    ):
        chunks = await _generate(router, request, RequestType.AUDIO_GENERATION)

    assert len(chunks) == 1
    assert chunks[0]["status"] == "failed"
    assert "storage backend refused" in chunks[0]["error"]


# ── A3-T16: connector and SHM read paths agree ────────────


@pytest.mark.asyncio
async def test_a3_t16_connector_and_shm_paths_format_identically():
    result = _audio_stage(np.zeros(2400))
    connector = MagicMock()
    connector.get.return_value = (result, 10)

    # Connector path (multi-node).
    routed = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={},
    )
    routed.connectors = {stage_router._connector_key(0, "router"): connector}
    stage_output = SimpleNamespace(
        stage_connector_refs={"0": {"rdma": "meta"}}, shm_meta=None
    )
    context = routed._formatter.request_context(
        {"input": "hi", "data_source": "b64_json"}, RequestType.AUDIO_GENERATION
    )
    session = stage_router.FormatSession.for_request(
        "req-1", RequestType.AUDIO_GENERATION
    )
    connector_chunks = [
        c async for c in routed._format_output(stage_output, context, session, 0)
    ]

    # SHM path (single-node legacy) over the same object.
    shm_router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={},
    )
    context = shm_router._formatter.request_context(
        {"input": "hi", "data_source": "b64_json"}, RequestType.AUDIO_GENERATION
    )
    session = stage_router.FormatSession.for_request(
        "req-1", RequestType.AUDIO_GENERATION
    )
    with patch.object(stage_router, "shm_deserialize", return_value=result) as shm_read:
        shm_chunks = [
            c
            async for c in shm_router._format_output(
                SimpleNamespace(stage_connector_refs=None, shm_meta={"x": 1}),
                context,
                session,
                0,
            )
        ]

    assert connector.get.call_count == 1
    assert shm_read.call_count == 1
    assert len(connector_chunks) == len(shm_chunks) == 1
    assert connector_chunks[0]["status"] == shm_chunks[0]["status"] == "completed"
    assert base64.b64decode(connector_chunks[0]["data"][0]["b64_json"]) == (
        base64.b64decode(shm_chunks[0]["data"][0]["b64_json"])
    )


# ── Context normalization: audio data_source vs non-audio ─────────────────


class TestStageRouterContextNormalization:
    """generate() normalizes audio data_source/response_format into one context."""

    async def _capture(self, request, request_type, output_modalities=None):
        formatter = _SpyFormatter(media_fs=_MEMORY_FS)
        router = _make_router(
            stage_configs=[_make_stage_cfg(0)],
            stage_clients={
                "stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))
            },
            formatter=formatter,
            output_modalities=output_modalities,
        )
        with patch.object(
            stage_router, "shm_deserialize", return_value=SimpleNamespace()
        ):
            await _generate(router, request, request_type)
        assert len(formatter.calls) == 1
        return formatter.calls[0][0]

    @pytest.mark.asyncio
    async def test_audio_request_maps_data_source_to_response_format(self):
        """data_source present: context carries response_format=data_source, output_format=response_format."""
        context = await self._capture(
            {"input": "hi", "data_source": "url", "response_format": "mp3"},
            RequestType.AUDIO_GENERATION,
            output_modalities=["audio"],
        )

        assert context.response_format == "url"  # data_source
        assert context.output_format == "mp3"  # response_format (codec)

    @pytest.mark.asyncio
    async def test_audio_request_b64_json_maps_correctly(self):
        context = await self._capture(
            {"input": "hi", "data_source": "b64_json", "response_format": "opus"},
            RequestType.AUDIO_GENERATION,
            output_modalities=["audio"],
        )

        assert context.response_format == "b64_json"
        assert context.output_format == "opus"

    @pytest.mark.asyncio
    async def test_non_audio_request_passes_through_unchanged(self):
        """No data_source: response_format and output_format passed as-is."""
        context = await self._capture(
            {"prompt": "cat", "response_format": "url", "output_format": "mp4"},
            RequestType.CHAT_COMPLETION,
        )

        assert context.response_format == "url"
        assert context.output_format == "mp4"

    @pytest.mark.asyncio
    async def test_no_format_fields_omitted_from_context(self):
        """Fields not present in request are absent from the validated context."""
        context = await self._capture({"prompt": "cat"}, RequestType.CHAT_COMPLETION)

        assert context.response_format is None
        assert context.output_format is None
        assert context.fps is None


# ── A3-T08: streaming audio keeps its layout across chunks ────────────────


@pytest.mark.asyncio
async def test_a3_t08_streamed_audio_keeps_chunk_state_and_layout():
    """A request-local stream session carries channel layout between chunks."""
    from dynamo.vllm.omni.format_contract import FormatSession
    from dynamo.vllm.omni.output_formatter import AudioFormatter

    formatter = AudioFormatter("test-model", _MEMORY_FS, None)
    session = FormatSession.for_request(
        "req-1", RequestType.AUDIO_GENERATION, stream_audio=True
    )
    assert session.audio_stream_state is not None

    first = await formatter.run(
        {"audio": np.zeros((8, 2), dtype=np.float32), "sr": 24000},
        "req-1",
        output_format="wav",
        audio_stream_state=session.audio_stream_state,
    )
    second = await formatter.run(
        {"audio": np.zeros((4, 2), dtype=np.float32), "sr": 24000},
        "req-1",
        output_format="wav",
        audio_stream_state=session.audio_stream_state,
    )

    first_bytes = base64.b64decode(first.payload["data"][0]["b64_json"])
    second_bytes = base64.b64decode(second.payload["data"][0]["b64_json"])
    assert first_bytes[:4] == b"RIFF"  # header once, on the first chunk
    assert second_bytes[:4] != b"RIFF"
    assert session.audio_stream_state.num_channels == 2
    assert session.audio_stream_state.channel_axis == 1


# ── A3-T03/T04: the router never re-derives the modality ──────────────────


@pytest.mark.asyncio
async def test_a3_t04_router_keeps_the_audio_stage_audio():
    formatter = _SpyFormatter(media_fs=_MEMORY_FS)
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
        formatter=formatter,
    )

    with patch.object(
        stage_router, "shm_deserialize", return_value=_audio_stage(np.zeros(2400))
    ):
        chunks = await _generate(
            router,
            {"input": "hi", "data_source": "b64_json"},
            RequestType.AUDIO_GENERATION,
        )

    assert chunks[0]["object"] == "audio.speech"
    assert formatter.calls[0][0].request_type is RequestType.AUDIO_GENERATION


@pytest.mark.asyncio
async def test_a3_t03_router_promotes_an_image_labelled_video_stage(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.encode_to_video_bytes",
        lambda frames, fps, output_format: b"mp4-bytes",
    )
    result = SimpleNamespace(final_output_type="image", images=[np.zeros((2, 4, 4, 3))])
    router = _make_router(
        stage_configs=[_make_stage_cfg(0)],
        stage_clients={"stage0": _StageClient(_stage_handler({"shm_meta": {"x": 1}}))},
    )

    with patch.object(stage_router, "shm_deserialize", return_value=result):
        chunks = await _generate(
            router,
            {"prompt": "a drone", "response_format": "b64_json"},
            RequestType.VIDEO_GENERATION,
        )

    assert len(chunks) == 1
    assert chunks[0]["object"] == "video"
    assert chunks[0]["data"][0]["output_format"] == "mp4"


def test_router_resolves_every_stage_through_the_contract():
    """The router must not pick a modality of its own."""
    import inspect

    source = inspect.getsource(stage_router.OmniStageRouter._format_output)
    assert "final_output_type" not in source
