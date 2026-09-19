# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the typed Omni formatting contract.

Each test names the regression it pins (``A3-T*``). Together they cover the
four questions the contract answers: option validation at the boundary, the
single modality resolver, per-request session state, and the three-way outcome
that replaced ``None``.
"""

import base64
import inspect
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import soundfile as sf
    from dynamo.common.utils.output_modalities import RequestType
    from dynamo.vllm.omni import omni_handler, stage_router
    from dynamo.vllm.omni.format_contract import (
        Emitted,
        Failed,
        FormatContext,
        FormatContractError,
        FormatSession,
        Modality,
        NoEmission,
        NoEmissionReason,
        ResponseSchema,
        resolve_dispatch,
        serialize_outcome,
    )
    from dynamo.vllm.omni.output_formatter import (
        AudioFormatter,
        DiffusionFormatter,
        OutputFormatter,
        TextFormatter,
    )
except ImportError:
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
]


def _text_output(text, finish_reason=None):
    """A vLLM request output, spelled out exactly rather than auto-mocked."""
    return SimpleNamespace(
        outputs=[
            SimpleNamespace(text=text, finish_reason=finish_reason, token_ids=[1, 2, 3])
        ],
        prompt_token_ids=[10, 20, 30],
        num_cached_tokens=None,
    )


def _text_stage(text, finish_reason=None):
    return SimpleNamespace(
        final_output_type="text", request_output=_text_output(text, finish_reason)
    )


def _audio_stage(samples, sample_rate=24000):
    return SimpleNamespace(
        final_output_type="audio",
        multimodal_output={
            "audio": np.asarray(samples, dtype=np.float32),
            "sr": sample_rate,
        },
    )


def _decode_audio(payload):
    """Read a response's inline audio back as (samples, sample_rate)."""
    return sf.read(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"])))


# ── A3-T01: a misspelled option is rejected, not ignored ───────────────────


def test_a3_t01_misspelled_context_field_is_rejected():
    """``fpps`` used to land in ``**ctx`` and vanish; it must fail instead."""
    with pytest.raises(FormatContractError) as excinfo:
        FormatContext.from_options(RequestType.VIDEO_GENERATION, {"fpps": 30})

    assert "fpps" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a3_t01_formatter_has_no_untyped_option_bag_left():
    """A stray keyword is a TypeError, not a silently dropped option."""
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.CHAT_COMPLETION)
    session = FormatSession.for_request("req-1", RequestType.CHAT_COMPLETION)

    with pytest.raises(TypeError):
        await formatter.format(context, _text_stage("hello"), session, fpps=30)


# ── A3-T02: wrong type, illegal value, foreign option ──────────────────────


@pytest.mark.parametrize(
    ("request_type", "options", "expected"),
    [
        (RequestType.VIDEO_GENERATION, {"fps": "fast"}, "fps"),
        (RequestType.VIDEO_GENERATION, {"fps": 0}, "fps"),
        (RequestType.VIDEO_GENERATION, {"fps": -4}, "fps"),
        (RequestType.IMAGE_GENERATION, {"response_format": "png"}, "response_format"),
        (RequestType.VIDEO_GENERATION, {"output_format": "webm"}, "output_format"),
        (RequestType.AUDIO_GENERATION, {"output_format": "mp4"}, "output_format"),
        (RequestType.AUDIO_GENERATION, {"speed": 9.0}, "speed"),
        (RequestType.IMAGE_GENERATION, {"speed": 2.0}, "speed"),
        (RequestType.VIDEO_GENERATION, {"speed": 2.0}, "speed"),
        (RequestType.AUDIO_GENERATION, {"fps": 24}, "fps"),
    ],
)
def test_a3_t02_invalid_options_are_rejected(request_type, options, expected):
    with pytest.raises(FormatContractError) as excinfo:
        FormatContext.from_options(request_type, options)

    assert expected in str(excinfo.value)


def test_a3_t02_no_encoding_starts_for_an_invalid_option(monkeypatch):
    """Validation happens before the codec, so a bad option costs no encode."""
    started = []

    def _record(*args, **kwargs):
        started.append(args)
        return {}

    monkeypatch.setattr(DiffusionFormatter, "_encode_video", _record)
    formatter = OutputFormatter(model_name="test-model")

    with pytest.raises(FormatContractError):
        formatter.engine_context(RequestType.VIDEO_GENERATION, output_format="webm")

    assert started == []


def test_a3_t02_chat_request_may_hold_several_option_groups():
    """A multi-modal chat request is not a single-modality endpoint.

    Its text stage ignores the audio options and its audio stage ignores the
    frame rate; neither may be rejected just because the other stage exists.
    """
    context = FormatContext.from_options(
        RequestType.CHAT_COMPLETION,
        {"response_format": "b64_json", "output_format": "wav", "fps": 8, "speed": 1.5},
    )

    assert (context.fps, context.speed) == (8, 1.5)
    assert (context.response_format, context.output_format) == ("b64_json", "wav")


# ── A3-T03: video request with an image-labelled diffusion stage ───────────


def test_a3_t03_image_labelled_stage_of_a_video_request_is_video():
    context = FormatContext.from_options(RequestType.VIDEO_GENERATION, {"fps": 8})

    dispatch = resolve_dispatch(context, SimpleNamespace(final_output_type="image"))

    assert dispatch.modality is Modality.VIDEO
    assert dispatch.schema is ResponseSchema.MEDIA


def test_a3_t03_ordinary_image_request_stays_image():
    context = FormatContext.from_options(RequestType.IMAGE_GENERATION, {})

    dispatch = resolve_dispatch(context, SimpleNamespace(final_output_type="image"))

    assert dispatch.modality is Modality.IMAGE


def test_a3_t03_ordinary_text_request_stays_text():
    context = FormatContext.from_options(RequestType.CHAT_COMPLETION, {})

    dispatch = resolve_dispatch(context, SimpleNamespace(final_output_type="text"))

    assert dispatch.modality is Modality.TEXT
    assert dispatch.schema is ResponseSchema.CHAT_CHUNK


@pytest.mark.asyncio
async def test_a3_t03_image_labelled_video_stage_keeps_its_audio_track(monkeypatch):
    """Promoting the stage to video must not drop the muxed audio track."""
    muxed = []
    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.mux_video_audio_bytes",
        lambda frames, audio, **kwargs: muxed.append(audio.shape) or b"mp4-bytes",
    )
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(
        RequestType.VIDEO_GENERATION, response_format="b64_json"
    )
    session = FormatSession.for_request("req-1", RequestType.VIDEO_GENERATION)
    stage = SimpleNamespace(
        final_output_type="image",
        images=np.zeros((2, 4, 4, 3), dtype=np.float32),
        multimodal_output={
            "audio": np.zeros((1, 2, 1334), dtype=np.float32),
            "fps": 16,
            "audio_sample_rate": 16000,
        },
    )

    outcome = await formatter.format(context, stage, session)

    assert isinstance(outcome, Emitted)
    assert outcome.payload["data"][0]["audio_sample_rate"] == 16000
    assert muxed == [(2, 1334)]


# ── A3-T04: a legal text -> audio multi-stage chat request ─────────────────


def test_a3_t04_each_stage_of_a_text_to_audio_chat_keeps_its_modality():
    context = FormatContext.from_options(RequestType.CHAT_COMPLETION, {})

    text = resolve_dispatch(context, SimpleNamespace(final_output_type="text"))
    audio = resolve_dispatch(context, SimpleNamespace(final_output_type="audio"))

    assert text.modality is Modality.TEXT
    assert audio.modality is Modality.AUDIO


@pytest.mark.asyncio
async def test_a3_t04_later_audio_stage_is_neither_text_nor_an_error():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.CHAT_COMPLETION)
    session = FormatSession.for_request("req-1", RequestType.CHAT_COMPLETION)

    first = await formatter.format(context, _text_stage("hello"), session)
    second = await formatter.format(context, _audio_stage(np.zeros(2400)), session)

    assert isinstance(first, Emitted)
    assert first.payload["choices"][0]["delta"]["content"] == "hello"
    assert isinstance(second, Emitted)
    assert second.payload["status"] == "completed"


# ── A3-T06: unknown modality and missing required payload ──────────────────


def test_a3_t06_unknown_final_modality_is_an_explicit_failure():
    context = FormatContext.from_options(RequestType.CHAT_COMPLETION, {})

    outcome = resolve_dispatch(context, SimpleNamespace(final_output_type="hologram"))

    assert isinstance(outcome, Failed)
    assert "hologram" in outcome.error


def test_a3_t06_missing_final_output_type_is_an_explicit_failure():
    context = FormatContext.from_options(RequestType.CHAT_COMPLETION, {})

    outcome = resolve_dispatch(context, SimpleNamespace(final_output_type=None))

    assert isinstance(outcome, Failed)
    assert "final_output_type" in outcome.error


@pytest.mark.asyncio
async def test_a3_t06_unknown_modality_carries_the_request_failure_body():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.IMAGE_GENERATION)
    session = FormatSession.for_request("req-1", RequestType.IMAGE_GENERATION)

    outcome = await formatter.format(
        context, SimpleNamespace(final_output_type="hologram"), session
    )

    assert isinstance(outcome, Failed)
    assert outcome.compatible_payload["choices"][0]["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_a3_t06_empty_image_payload_is_a_failure_not_a_gap():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.IMAGE_GENERATION)
    session = FormatSession.for_request("req-1", RequestType.IMAGE_GENERATION)
    stage = SimpleNamespace(final_output_type="image", images=[])

    outcome = await formatter.format(context, stage, session)

    assert isinstance(outcome, Failed)
    assert outcome.compatible_payload is not None


# ── A3-T10: finish is idempotent ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_a3_t10_repeated_finish_encodes_and_emits_once(monkeypatch):
    encoded = []
    original = AudioFormatter._encode_audio

    def _counting(self, *args, **kwargs):
        encoded.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AudioFormatter, "_encode_audio", _counting)
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.AUDIO_GENERATION)
    session = FormatSession.for_request("req-1", RequestType.AUDIO_GENERATION)
    await formatter.format(context, _audio_stage(np.zeros(2400)), session)

    first = await formatter.finish(context, session)
    second = await formatter.finish(context, session)

    assert isinstance(first, Emitted)
    assert second is first
    assert len(encoded) == 1


@pytest.mark.asyncio
async def test_a3_t10_single_finish_of_a_pure_text_request_adds_nothing():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.CHAT_COMPLETION)
    session = FormatSession.for_request("req-1", RequestType.CHAT_COMPLETION)
    emitted = await formatter.format(context, _text_stage("hi"), session)
    assert isinstance(emitted, Emitted)

    outcome = await formatter.finish(context, session)

    assert isinstance(outcome, NoEmission)
    assert outcome.reason is NoEmissionReason.COMPLETE


# ── A3-T11: interleaved requests never share state ─────────────────────────


@pytest.mark.asyncio
async def test_a3_t11_interleaved_text_requests_keep_their_own_previous_text():
    """One formatter, two requests: the delta base may not cross over."""
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.CHAT_COMPLETION)
    session_a = FormatSession.for_request("req-A", RequestType.CHAT_COMPLETION)
    session_b = FormatSession.for_request("req-B", RequestType.CHAT_COMPLETION)

    await formatter.format(context, _text_stage("Hello"), session_a)
    await formatter.format(context, _text_stage("World"), session_b)
    second_a = await formatter.format(context, _text_stage("Hello there"), session_a)
    second_b = await formatter.format(context, _text_stage("World peace"), session_b)

    assert second_a.payload["id"] == "req-A"
    assert second_a.payload["choices"][0]["delta"]["content"] == " there"
    assert second_b.payload["id"] == "req-B"
    assert second_b.payload["choices"][0]["delta"]["content"] == " peace"


@pytest.mark.asyncio
async def test_a3_t11_interleaved_audio_requests_keep_their_own_buffers():
    """Sample rate, buffered samples and request id are request-local."""
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(
        RequestType.AUDIO_GENERATION, response_format="b64_json"
    )
    session_a = FormatSession.for_request("req-A", RequestType.AUDIO_GENERATION)
    session_b = FormatSession.for_request("req-B", RequestType.AUDIO_GENERATION)

    await formatter.format(context, _audio_stage(np.zeros(1200), 24000), session_a)
    await formatter.format(context, _audio_stage(np.zeros(800), 48000), session_b)

    first = await formatter.finish(context, session_a)
    second = await formatter.finish(context, session_b)

    assert isinstance(first, Emitted)
    assert isinstance(second, Emitted)
    samples_a, rate_a = _decode_audio(first.payload)
    samples_b, rate_b = _decode_audio(second.payload)
    assert (len(samples_a), rate_a) == (1200, 24000)
    assert (len(samples_b), rate_b) == (800, 48000)
    assert first.payload["id"] == "req-A"
    assert second.payload["id"] == "req-B"


# ── A3-T12: cancellation releases buffers and is not a success ─────────────


@pytest.mark.asyncio
async def test_a3_t12_abort_then_finish_emits_no_successful_audio():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.AUDIO_GENERATION)
    session = FormatSession.for_request("req-1", RequestType.AUDIO_GENERATION)
    await formatter.format(context, _audio_stage(np.zeros(2400)), session)
    assert session.audio_aggregate_state.chunks

    session.abort()

    assert session.audio_aggregate_state.chunks == []
    outcome = await formatter.finish(context, session)
    assert isinstance(outcome, NoEmission)
    assert outcome.reason is NoEmissionReason.ABORTED
    assert serialize_outcome(outcome) is None


@pytest.mark.asyncio
async def test_a3_t12_abort_does_not_wrap_itself_as_a_failure_either():
    """A cancelled request is neither a success nor an error to report."""
    session = FormatSession.for_request("req-1", RequestType.AUDIO_GENERATION)
    session.abort()

    assert session.aborted is True
    assert session.final_outcome is None


# ── A3-T13: URL requested without storage ──────────────────────────────────


def test_a3_t13_url_without_media_storage_fails_when_the_context_is_built():
    formatter = OutputFormatter(model_name="test-model", media_fs=None)

    with pytest.raises(FormatContractError, match="media storage"):
        formatter.engine_context(RequestType.VIDEO_GENERATION, response_format="url")


def test_a3_t13_default_video_url_also_needs_storage():
    """The video path defaults to a URL, so an unset format still needs one."""
    formatter = OutputFormatter(model_name="test-model", media_fs=None)

    with pytest.raises(FormatContractError, match="media storage"):
        formatter.engine_context(RequestType.VIDEO_GENERATION)


def test_a3_t13_url_with_storage_is_accepted():
    formatter = OutputFormatter(model_name="test-model", media_fs=object())

    context = formatter.engine_context(
        RequestType.VIDEO_GENERATION, response_format="url"
    )

    assert context.response_format == "url"
    assert context.requires_media_storage is True


@pytest.mark.asyncio
async def test_a3_t13_no_upload_is_attempted_and_no_url_is_faked(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        "dynamo.vllm.omni.output_formatter.upload_to_fs",
        lambda *args, **kwargs: uploads.append(args) or "http://fabricated/url",
    )
    formatter = OutputFormatter(model_name="test-model", media_fs=None)

    with pytest.raises(FormatContractError):
        formatter.request_context(
            {"prompt": "x", "response_format": "url"}, RequestType.IMAGE_GENERATION
        )

    assert uploads == []


# ── A3-T18: a request that produced nothing is not a silent success ────────


@pytest.mark.asyncio
async def test_a3_t18_buffered_request_with_nothing_emittable_fails_at_finish():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.AUDIO_GENERATION)
    session = FormatSession.for_request("req-1", RequestType.AUDIO_GENERATION)

    gap = await formatter.format(
        context, _audio_stage(np.zeros(0, dtype=np.float32)), session
    )
    assert isinstance(gap, NoEmission)
    assert gap.reason is NoEmissionReason.STREAM_GAP

    outcome = await formatter.finish(context, session)
    assert isinstance(outcome, Failed)
    assert outcome.compatible_payload["status"] == "failed"


@pytest.mark.asyncio
async def test_a3_t18_text_request_that_never_emitted_fails_at_finish():
    formatter = OutputFormatter(model_name="test-model")
    context = formatter.engine_context(RequestType.CHAT_COMPLETION)
    session = FormatSession.for_request("req-1", RequestType.CHAT_COMPLETION)
    gap = await formatter.format(
        context, SimpleNamespace(final_output_type="text", request_output=None), session
    )
    assert isinstance(gap, NoEmission)

    outcome = await formatter.finish(context, session)

    assert isinstance(outcome, Failed)
    assert outcome.compatible_payload is not None


# ── A3-T17: no legacy untyped call site survives ───────────────────────────


def test_a3_t17_no_caller_passes_an_untyped_option_bag():
    """The old ``**ctx`` escape hatch must not survive anywhere on the path."""
    for module in (omni_handler, stage_router):
        source = Path(module.__file__).read_text()
        assert "**ctx" not in source, module.__name__
        assert "formatted_chunk" not in source, module.__name__

    for owner in (
        OutputFormatter.format,
        OutputFormatter.finish,
        TextFormatter.format,
        TextFormatter.run,
        DiffusionFormatter.format,
        DiffusionFormatter.run,
        AudioFormatter.format,
        AudioFormatter.run,
        AudioFormatter.finish_aggregate,
        AudioFormatter.finalize_aggregate,
    ):
        kinds = {
            parameter.kind for parameter in inspect.signature(owner).parameters.values()
        }
        assert inspect.Parameter.VAR_KEYWORD not in kinds, owner.__qualname__
