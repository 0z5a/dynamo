# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert vLLM-Omni stage outputs into Dynamo protocol responses.

``OutputFormatter`` is the caller entry point. Construct it once for a model,
open one ``FormatSession`` per request, and hand every stage to
``await format(context, stage_output, session)`` and then
``await finish(context, session)``. The aggregated ``OmniHandler`` and the
disaggregated ``OmniStageRouter`` use exactly that sequence; the questions it
answers live in ``format_contract``:

* what the request asked for -- ``FormatContext``, validated before any encode;
* which modality this stage is -- ``resolve_dispatch``, the only place that
  reads ``final_output_type`` against request intent;
* what the request has accumulated -- ``FormatSession``, one per request;
* whether this step emitted, legally emitted nothing, or failed --
  ``FormattingOutcome``.

A typical request is::

    context = formatter.engine_context(inputs.request_type, fps=inputs.fps)
    session = FormatSession.for_request(request_id, inputs.request_type)
    async for stage_output in engine.generate(...):
        payload = serialize_outcome(
            await formatter.format(context, stage_output, session)
        )
        if payload is not None:
            yield payload
    payload = serialize_outcome(await formatter.finish(context, session))

The modality formatters below own the codecs and the storage calls. Their
``run()`` methods take a resolved dispatch and return an outcome; their
``format()`` methods keep returning the serialized payload for direct callers.

URL responses require a writable ``media_fs``; ``media_http_url`` optionally
rewrites the returned public URL. A context that asks for a URL while the
worker has no media filesystem is rejected where the context is built, so the
request fails before any frame is encoded.

Audio state belongs to the request: an audio request that streams encodes each
chunk as it arrives through ``session.audio_stream_state``, and one that does
not buffers into ``session.audio_aggregate_state`` and encodes once in
``finish()``. The formatter itself keeps no request state.

The formatters are independent of engine construction and model loading, so
aggregated handlers, disaggregated routers, and tests can share them.
"""

import asyncio
import base64
import logging
import struct
import time
import uuid
from collections.abc import Mapping
from io import BytesIO
from typing import Any

import numpy as np
import soundfile as sf
import torch

try:
    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
except (ImportError, OSError):
    mux_video_audio_bytes = None  # type: ignore[assignment]

from dynamo.common.protocols.audio_protocol import AudioData, NvAudioSpeechResponse
from dynamo.common.protocols.image_protocol import ImageData, NvImagesResponse
from dynamo.common.protocols.video_protocol import NvVideosResponse, VideoData
from dynamo.common.storage import upload_to_fs
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.output_modalities import RequestType
from dynamo.common.utils.video_utils import (
    encode_to_video_bytes,
    frames_to_numpy,
    normalize_video_frames,
)
from dynamo.vllm.handlers import build_prompt_tokens_details
from dynamo.vllm.omni.format_contract import (
    AudioAggregateState,
    AudioStreamState,
    Emitted,
    Failed,
    FormatContext,
    FormatSession,
    FormattingOutcome,
    Modality,
    NoEmission,
    NoEmissionReason,
    ResponseSchema,
    StageDispatch,
    resolve_dispatch,
    serialize_outcome,
)
from dynamo.vllm.omni.utils import is_empty_payload

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_SAMPLE_RATE = 24000


def _failure_payload(
    modality: Modality, request_id: str, model_name: str, error: str
) -> dict[str, Any]:
    """Build the body a modality already uses to report a failed serialization.

    Args:
        modality: Modality whose response shape the body must take.
        request_id: Identifier included in the response.
        model_name: Model identifier included in the response.
        error: User-facing error description.

    Returns:
        Dict[str, Any]: Serialized failure body for that modality.
    """
    if modality is Modality.VIDEO:
        return NvVideosResponse(
            id=request_id,
            object="video",
            model=model_name,
            status="failed",
            progress=0,
            created=int(time.time()),
            data=[],
            error=error,
        ).model_dump()
    if modality is Modality.AUDIO:
        return NvAudioSpeechResponse(
            id=request_id,
            model=model_name,
            status="failed",
            created=int(time.time()),
            error=error,
        ).model_dump()
    # Text and image responses both report a failed serialization as an error
    # chunk; an image endpoint has no failure field of its own.
    return _error_chunk(request_id, model_name, error)


class TextFormatter:
    """Formats LLM text output as OpenAI chat completion chunks."""

    def __init__(self, model_name: str) -> None:
        """Initialize a text formatter.

        Args:
            model_name: Model identifier included in response chunks.

        Returns:
            None.
        """
        self._model_name = model_name

    def run(
        self,
        request_output: Any,
        request_id: str,
        *,
        previous_text: str = "",
    ) -> FormattingOutcome:
        """Format the next text delta as a chat-completion chunk.

        Args:
            request_output: vLLM request output containing generated choices.
            request_id: Identifier included in the response chunk.
            previous_text: Text already emitted for this request.

        Returns:
            FormattingOutcome: ``Emitted`` with the chunk, or ``Failed`` when
                the engine produced no choice to read.

        Raises:
            AttributeError: If the request output lacks required choice fields.
        """
        if not request_output.outputs:
            error = "No outputs from engine"
            return Failed(
                error=error,
                compatible_payload=_error_chunk(request_id, self._model_name, error),
            )

        output = request_output.outputs[0]
        delta_text = output.text[len(previous_text) :]

        chunk: dict[str, Any] = {
            "id": request_id,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": delta_text},
                    "finish_reason": (
                        normalize_finish_reason(output.finish_reason)
                        if output.finish_reason
                        else None
                    ),
                }
            ],
        }

        if output.finish_reason:
            chunk["usage"] = _build_completion_usage(request_output)

        return Emitted(chunk)

    def format(
        self,
        request_output: Any,
        request_id: str,
        *,
        previous_text: str = "",
    ) -> dict[str, Any] | None:
        """Serialize the next text delta; ``run`` is the typed contract.

        Args:
            request_output: vLLM request output containing generated choices.
            request_id: Identifier included in the response chunk.
            previous_text: Text already emitted for this request.

        Returns:
            Dict[str, Any] | None: Formatted chunk or an engine-output error chunk.
        """
        return serialize_outcome(
            self.run(request_output, request_id, previous_text=previous_text)
        )


class DiffusionFormatter:
    """Formats diffusion output (images/video frames) for the frontend.

    Handles both image and video output. The modality is decided once, by
    ``format_contract.resolve_dispatch``, because vLLM-Omni reports
    final_output_type="image" for all diffusion outputs including video frames.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any,
        media_http_url: str | None,
        default_fps: int = 16,
    ) -> None:
        """Initialize a diffusion formatter.

        Args:
            model_name: Model identifier included in responses.
            media_fs: Storage backend used for URL responses.
            media_http_url: Public base URL for stored media.
            default_fps: Frame rate used when output metadata omits one.

        Returns:
            None.
        """
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._default_fps = default_fps

    async def run(
        self,
        stage_output: Any,
        request_id: str,
        *,
        dispatch: StageDispatch,
        fps: int | None = None,
        response_format: str | None = None,
        output_format: str | None = None,
    ) -> FormattingOutcome:
        """Format one diffusion stage under its resolved dispatch.

        Args:
            stage_output: vLLM-Omni stage output containing generated media.
            request_id: Identifier included in the response.
            dispatch: The modality and response schema resolved for this stage.
            fps: Frame-rate override; the formatter default applies when unset.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Video container format; currently only ``"mp4"``.

        Returns:
            FormattingOutcome: The emitted response, or an explicit failure
                carrying that modality's failure body.

        Raises:
            ValueError: If an image or video response option is unsupported.
        """
        images = (
            stage_output.images if hasattr(stage_output, "images") else stage_output
        )

        if dispatch.modality is Modality.VIDEO:
            response = await self._encode_video(
                images,
                request_id,
                multimodal_output=self._extract_multimodal_output(stage_output),
                fps=self._default_fps if fps is None else fps,
                response_format=response_format,
                output_format=output_format,
            )
            return self._video_outcome(response, request_id)

        if is_empty_payload(images):
            error = "No images generated"
            return Failed(
                error=error,
                compatible_payload=_error_chunk(request_id, self._model_name, error),
            )
        try:
            response = await self._encode_image(
                images,
                request_id,
                schema=dispatch.schema,
                response_format=response_format,
            )
        except Exception as e:
            # Encoding and upload failures are a result of the request, not an
            # error for one caller to raise while the other reports it.
            logger.error("Failed to encode image for request %s: %s", request_id, e)
            return Failed(
                error=str(e),
                compatible_payload=_failure_payload(
                    Modality.IMAGE, request_id, self._model_name, str(e)
                ),
            )
        return Emitted(response)

    async def format(
        self,
        stage_output: Any,
        request_id: str,
        *,
        dispatch: StageDispatch,
        fps: int | None = None,
        response_format: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any] | None:
        """Serialize one diffusion stage; ``run`` is the typed contract.

        Args:
            stage_output: vLLM-Omni stage output containing generated media.
            request_id: Identifier included in the response.
            dispatch: The modality and response schema resolved for this stage.
            fps: Frame-rate override; the formatter default applies when unset.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Video container format; currently only ``"mp4"``.

        Returns:
            Dict[str, Any] | None: Formatted response, or its failure body.
        """
        return serialize_outcome(
            await self.run(
                stage_output,
                request_id,
                dispatch=dispatch,
                fps=fps,
                response_format=response_format,
                output_format=output_format,
            )
        )

    def _video_outcome(
        self, response: dict[str, Any] | None, request_id: str
    ) -> FormattingOutcome:
        """Read an encoded video response as an outcome.

        ``_encode_video`` keeps returning its serialized model for direct
        callers, and ``status`` is where it records that the encode failed.
        """
        if response is None:
            error = "No video outputs found in generation result"
            return Failed(
                error=error,
                compatible_payload=_failure_payload(
                    Modality.VIDEO, request_id, self._model_name, error
                ),
            )
        if response.get("status") == "failed":
            return Failed(
                error=str(response.get("error", "")), compatible_payload=response
            )
        return Emitted(response)

    async def _encode_video(
        self,
        images: Any,
        request_id: str,
        fps: int,
        multimodal_output: dict[str, Any] | None = None,
        response_format: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any] | None:
        """Encode generated frames and optional audio as video responses.

        Args:
            images: Primary generated video-frame payload.
            request_id: Identifier included in the response and storage path.
            fps: Positive fallback frame rate when metadata does not provide one.
            multimodal_output: Optional video, audio, and encoding metadata.
            response_format: ``"url"`` or ``"b64_json"`` output representation.
            output_format: Video container format; currently only ``"mp4"``.

        Returns:
            Dict[str, Any] | None: Completed response, or a failed response for
                invalid media metadata, mismatched durations, or encoding errors.

        Raises:
            ValueError: If the response or output format is unsupported.
        """
        output_format = output_format or "mp4"
        response_format = response_format or "url"
        if response_format not in ("url", "b64_json"):
            raise ValueError(
                f"Unsupported response_format: {response_format!r}; expected 'url' or 'b64_json'"
            )
        if output_format != "mp4":
            raise ValueError(
                f"Unsupported output_format: {output_format!r}; only 'mp4' is supported"
            )
        try:
            start_time = time.time()
            multimodal_output = multimodal_output or {}
            videos = self._split_video_outputs(images, multimodal_output)
            if not videos:
                raise ValueError("No video outputs found in generation result")
            resolved_fps = self._resolve_int_metadata(multimodal_output, "fps", "video")
            if resolved_fps is None:
                resolved_fps = self._coerce_positive_int(fps)
            if resolved_fps is None:
                raise ValueError(f"Video fps must be greater than zero, got {fps!r}")
            audio_sample_rate = self._resolve_audio_sample_rate(multimodal_output)
            audio_outputs = self._split_audio_outputs(
                multimodal_output.get("audio"), len(videos)
            )

            data = []
            for index, (video, audio) in enumerate(
                zip(videos, audio_outputs, strict=True)
            ):
                frames_np = self._video_to_numpy_frames(video)
                if audio is None:
                    # The codec-compliant standard image retains its existing
                    # royalty-free VP9 path for silent video models.
                    video_bytes = await asyncio.to_thread(
                        encode_to_video_bytes,
                        frames_np,
                        fps=resolved_fps,
                        output_format=output_format,
                    )
                else:
                    if mux_video_audio_bytes is None:
                        raise RuntimeError(
                            "Generated audio requires PyAV with H.264 and AAC encoders; "
                            "use the video-audio codec overlay"
                        )
                    audio_np = self._audio_to_numpy(audio)
                    if audio_np.ndim in (1, 2):
                        # Match the muxer's channel-first/channel-last normalization.
                        audio_sample_count = max(audio_np.shape)
                        video_duration_s = frames_np.shape[0] / resolved_fps
                        audio_duration_s = audio_sample_count / audio_sample_rate
                        duration_tolerance_s = (
                            1.0 / resolved_fps + 1.0 / audio_sample_rate
                        )
                        if (
                            abs(video_duration_s - audio_duration_s)
                            > duration_tolerance_s
                        ):
                            raise ValueError(
                                "Audio/video duration mismatch: "
                                f"video={video_duration_s:.3f}s, "
                                f"audio={audio_duration_s:.3f}s, "
                                f"tolerance={duration_tolerance_s:.3f}s"
                            )
                    video_bytes = await asyncio.to_thread(
                        mux_video_audio_bytes,
                        frames_np,
                        audio_np,
                        fps=float(resolved_fps),
                        audio_sample_rate=audio_sample_rate,
                    )

                video_data = VideoData(
                    output_format=output_format,
                    fps=resolved_fps,
                    audio_sample_rate=audio_sample_rate if audio is not None else None,
                )
                if response_format == "b64_json":
                    video_data.b64_json = base64.b64encode(video_bytes).decode("utf-8")
                else:
                    filename = (
                        f"videos/{request_id}.{output_format}"
                        if len(videos) == 1
                        else f"videos/{request_id}/{index}.{output_format}"
                    )
                    video_data.url = await upload_to_fs(
                        self._media_fs,
                        filename,
                        video_bytes,
                        self._media_http_url,
                    )
                data.append(video_data)

            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="completed",
                progress=100,
                created=int(time.time()),
                data=data,
                inference_time_s=time.time() - start_time,
            ).model_dump()
        except Exception as e:
            logger.error("Failed to encode video for request %s: %s", request_id, e)
            return NvVideosResponse(
                id=request_id,
                object="video",
                model=self._model_name,
                status="failed",
                progress=0,
                created=int(time.time()),
                data=[],
                error=str(e),
            ).model_dump()

    @staticmethod
    def _extract_multimodal_output(stage_output: Any) -> dict[str, Any]:
        """Extract multimodal metadata from a stage output.

        Args:
            stage_output: vLLM-Omni stage result or compatibility wrapper.

        Returns:
            dict[str, Any]: A shallow metadata copy, or an empty dictionary.
        """
        multimodal_output = getattr(stage_output, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            return dict(multimodal_output)

        request_output = getattr(stage_output, "request_output", None)
        if isinstance(request_output, dict):
            multimodal_output = request_output.get("multimodal_output")
            if multimodal_output is None:
                multimodal_output = request_output.get("_multimodal_output")
        elif request_output is not None:
            multimodal_output = getattr(request_output, "multimodal_output", None)
            if multimodal_output is None:
                multimodal_output = getattr(request_output, "_multimodal_output", None)
        return dict(multimodal_output) if isinstance(multimodal_output, Mapping) else {}

    @staticmethod
    def _split_video_outputs(
        images: Any, multimodal_output: dict[str, Any]
    ) -> list[Any]:
        """Separate a batched diffusion payload into individual videos.

        Args:
            images: Primary video-frame payload from the stage output.
            multimodal_output: Metadata that may contain a fallback video payload.

        Returns:
            list[Any]: Individual video payloads, or an empty list when absent.

        Raises:
            ValueError: If a list mixes incompatible batched video payloads.
        """
        videos = images
        if is_empty_payload(videos):
            videos = multimodal_output.get("video")
        if videos is None:
            return []
        if isinstance(videos, (np.ndarray, torch.Tensor)):
            if videos.ndim == 5:
                return [videos[index] for index in range(videos.shape[0])]
            return [videos]
        if isinstance(videos, (list, tuple)):
            videos = list(videos)
            if not videos:
                return []
            first = videos[0]
            if isinstance(first, (np.ndarray, torch.Tensor)) and first.ndim == 5:
                flattened: list[Any] = []
                for batch in videos:
                    if (
                        not isinstance(batch, (np.ndarray, torch.Tensor))
                        or batch.ndim != 5
                    ):
                        raise ValueError("Video output batches must all be 5-D")
                    flattened.extend(batch[index] for index in range(batch.shape[0]))
                return flattened
            if isinstance(first, (np.ndarray, torch.Tensor)) and first.ndim == 4:
                return videos
            if isinstance(first, list):
                return videos
            return [videos]
        return [videos]

    @staticmethod
    def _video_to_numpy_frames(video: Any) -> np.ndarray:
        """Normalize one video to uint8 ``(frames, height, width, channels)``.

        Args:
            video: Video tensor, array, or sequence of image frames.

        Returns:
            np.ndarray: Contiguous FHWC frames with uint8 RGB values.

        Raises:
            ValueError: If frames are empty, inconsistent, or use an unsupported layout.
        """
        if isinstance(video, torch.Tensor):
            video = video.detach().float().cpu().numpy()

        if isinstance(video, np.ndarray):
            if video.ndim == 3:
                video = video[None, ...]
            if video.ndim != 4:
                raise ValueError(
                    f"Expected a 4-D video tensor, got shape {video.shape}"
                )
            if video.shape[-1] in (1, 3, 4):
                frames = video
            else:
                is_cfhw = video.shape[0] in (1, 3, 4)
                is_fchw = video.shape[1] in (1, 3, 4)
                if is_cfhw and is_fchw:
                    raise ValueError(
                        "Ambiguous channel-first video tensor; expected an "
                        "unambiguous CFHW or FCHW shape, "
                        f"got {video.shape}"
                    )
                if is_cfhw:
                    frames = video.transpose(1, 2, 3, 0)
                elif is_fchw:
                    frames = video.transpose(0, 2, 3, 1)
                else:
                    raise ValueError(
                        "Video tensor must use CFHW, FCHW, or FHWC channel layout, "
                        f"got shape {video.shape}"
                    )
            if frames.shape[-1] == 1:
                frames = np.repeat(frames, 3, axis=-1)
            elif frames.shape[-1] == 4:
                frames = frames[..., :3]
            if np.issubdtype(frames.dtype, np.floating) and frames.min(initial=0.0) < 0:
                frames = (frames + 1.0) / 2.0
            return frames_to_numpy(list(np.ascontiguousarray(frames)))

        frames = normalize_video_frames(video if isinstance(video, list) else [video])
        normalized = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().float().cpu().numpy()
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                if frame.shape[-1] in (1, 3, 4):
                    pass
                elif frame.shape[0] in (1, 3, 4):
                    frame = frame.transpose(1, 2, 0)
                else:
                    raise ValueError(
                        "Video frame must use CHW or HWC channel layout, "
                        f"got shape {frame.shape}"
                    )
                if frame.shape[-1] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[..., :3]
                if (
                    np.issubdtype(frame.dtype, np.floating)
                    and frame.min(initial=0.0) < 0
                ):
                    frame = (frame + 1.0) / 2.0
            normalized.append(frame)
        return frames_to_numpy(normalized)

    @staticmethod
    def _split_audio_outputs(audio: Any, expected_count: int) -> list[Any | None]:
        """Align generated audio payloads with their corresponding videos.

        Args:
            audio: Batched or per-video audio payload.
            expected_count: Number of video outputs requiring audio entries.

        Returns:
            list[Any | None]: One audio payload or ``None`` for each video.

        Raises:
            ValueError: If the audio payload count does not match the video count.
        """
        if audio is None:
            return [None] * expected_count
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            if audio.ndim >= 3:
                if audio.shape[0] != expected_count:
                    raise ValueError(
                        f"Expected {expected_count} audio output(s) for "
                        f"{expected_count} videos"
                    )
                return [audio[index] for index in range(expected_count)]
            if expected_count == 1:
                return [audio]
        if isinstance(audio, (list, tuple)):
            if len(audio) == expected_count:
                return list(audio)
            if expected_count == 1:
                return [audio]
        raise ValueError(
            f"Expected {expected_count} audio output(s) for {expected_count} videos"
        )

    @staticmethod
    def _audio_to_numpy(audio: Any) -> np.ndarray:
        """Convert an audio payload to a float32 NumPy array.

        Args:
            audio: Tensor, array, or array-like audio samples.

        Returns:
            np.ndarray: Audio samples represented as float32 values.

        Raises:
            TypeError: If the payload cannot be interpreted as an array.
            ValueError: If the payload cannot be converted to float32 values.
        """
        if isinstance(audio, torch.Tensor):
            return audio.detach().float().cpu().numpy()
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32, copy=False)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _resolve_int_metadata(
        multimodal_output: dict[str, Any],
        key: str,
        metadata_section: str,
        metadata_key: str | None = None,
    ) -> int | None:
        """Resolve positive numeric metadata rounded to the nearest integer.

        Args:
            multimodal_output: Multimodal payload containing optional metadata.
            key: Top-level metadata key to inspect first.
            metadata_section: Nested section under the ``metadata`` field.
            metadata_key: Nested key, defaulting to the top-level key.

        Returns:
            int | None: Positive resolved value, or ``None`` when absent or invalid.

        Raises:
            RuntimeError: If tensor metadata contains more than one value.
            OverflowError: If a non-finite numeric value cannot be converted.
        """
        value = multimodal_output.get(key)
        if value is None:
            metadata = multimodal_output.get("metadata")
            if isinstance(metadata, Mapping):
                section = metadata.get(metadata_section)
                if isinstance(section, Mapping):
                    value = section.get(metadata_key or key)
        if value is None:
            return None
        return DiffusionFormatter._coerce_positive_int(value)

    def _resolve_audio_sample_rate(self, multimodal_output: dict[str, Any]) -> int:
        """Resolve an audio sample rate from known metadata aliases.

        Args:
            multimodal_output: Multimodal payload containing audio metadata.

        Returns:
            int: Positive sample rate, or the default when none is valid.

        Raises:
            RuntimeError: If tensor metadata contains more than one value.
            OverflowError: If a non-finite numeric value cannot be converted.
        """
        for key in ("audio_sample_rate", "sample_rate", "sampling_rate", "sr"):
            sample_rate = self._coerce_positive_int(multimodal_output.get(key))
            if sample_rate is not None:
                return sample_rate

        metadata = multimodal_output.get("metadata")
        audio_metadata = (
            metadata.get("audio") if isinstance(metadata, Mapping) else None
        )
        if isinstance(audio_metadata, Mapping):
            for key in (
                "audio_sample_rate",
                "sample_rate",
                "sampling_rate",
                "sr",
            ):
                sample_rate = self._coerce_positive_int(audio_metadata.get(key))
                if sample_rate is not None:
                    return sample_rate

        return DEFAULT_AUDIO_SAMPLE_RATE

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        """Round a scalar value to a positive integer when possible.

        Args:
            value: Scalar-like value to convert.

        Returns:
            int | None: Positive integer value, or ``None`` when invalid.

        Raises:
            RuntimeError: If a tensor contains more than one value.
            OverflowError: If a non-finite numeric value cannot be converted.
        """
        if value is None:
            return None
        try:
            scalar = value.item() if hasattr(value, "item") else value
            resolved = round(float(scalar))
        except (TypeError, ValueError):
            return None
        return resolved if resolved > 0 else None

    async def _encode_image(
        self,
        images: list,
        request_id: str,
        *,
        schema: ResponseSchema,
        response_format: str | None = None,
    ) -> dict[str, Any]:
        """Encode generated images for a chat or media response.

        Args:
            images: Generated image objects to encode.
            request_id: Identifier included in the response and storage path.
            schema: Response shape resolved for this dispatch.
            response_format: ``"url"`` or ``"b64_json"`` output representation.

        Returns:
            Dict[str, Any]: Formatted response.

        Raises:
            ValueError: If the response format is unsupported.
            OSError: If an image cannot be encoded or uploaded.
        """
        if is_empty_payload(images):
            return _error_chunk(request_id, self._model_name, "No images generated")

        data_urls = await self._prepare_images(images, request_id, response_format)

        if schema is ResponseSchema.CHAT_CHUNK:
            return {
                "id": request_id,
                "created": int(time.time()),
                "object": "chat.completion.chunk",
                "model": self._model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": [
                                {"type": "image_url", "image_url": {"url": u}}
                                for u in data_urls
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

        image_data_list = []
        for data_url in data_urls:
            if response_format == "url":
                image_data_list.append(ImageData(url=data_url))
            elif response_format == "b64_json" or response_format is None:
                b64 = (
                    data_url.split(",", 1)[1]
                    if data_url.startswith("data:")
                    else data_url
                )
                image_data_list.append(ImageData(b64_json=b64))
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return NvImagesResponse(
            created=int(time.time()), data=image_data_list
        ).model_dump()

    async def _prepare_images(
        self, images: list, request_id: str, response_format: str | None = None
    ) -> list:
        """Serialize images as data URLs or upload-backed URLs.

        Args:
            images: Generated image objects supporting PNG serialization.
            request_id: Identifier used to construct storage paths.
            response_format: ``"url"`` or ``"b64_json"`` output representation.

        Returns:
            list: Encoded data URLs or uploaded media URLs.

        Raises:
            ValueError: If the response format is unsupported.
            OSError: If an image cannot be encoded or uploaded.
        """
        outlist = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            if response_format == "url":
                url = await upload_to_fs(
                    self._media_fs,
                    f"images/{request_id}/{uuid.uuid4()}.png",
                    image_bytes,
                    self._media_http_url,
                )
                outlist.append(url)
            elif response_format == "b64_json" or response_format is None:
                outlist.append(
                    f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                )
            else:
                raise ValueError(f"Invalid response format: {response_format}")
        return outlist


class AudioFormatter:
    """Formats audio multimodal_output → NvAudioSpeechResponse."""

    def __init__(
        self, model_name: str, media_fs: Any, media_http_url: str | None
    ) -> None:
        """Initialize an audio formatter.

        Args:
            model_name: Model identifier included in responses.
            media_fs: Storage backend used for URL responses.
            media_http_url: Public base URL for stored media.

        Returns:
            None.
        """
        self._model_name = model_name
        self._media_fs = media_fs
        self._media_http_url = media_http_url
        self._AudioData = AudioData  # stored for use in format()

    async def run(
        self,
        stage_output: Any,
        request_id: str,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        speed: float = 1.0,
        audio_stream_state: AudioStreamState | None = None,
        audio_aggregate_state: AudioAggregateState | None = None,
    ) -> FormattingOutcome:
        """Format complete, streaming, or aggregate audio output.

        Args:
            stage_output: Stage output or multimodal audio mapping.
            request_id: Identifier included in the response and storage path.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Audio codec; WAV when unset.
            speed: Playback-speed multiplier applied to the whole file.
            audio_stream_state: Request-local state for incremental output.
            audio_aggregate_state: Request-local buffer for one final encode.

        Returns:
            FormattingOutcome: ``Emitted`` with the audio response,
                ``NoEmission`` while the request buffers or has no audio yet,
                or ``Failed`` with the failed audio body.
        """
        mm_output = (
            stage_output.multimodal_output
            if hasattr(stage_output, "multimodal_output")
            else stage_output
        )
        if is_empty_payload(mm_output):
            if audio_stream_state is not None or audio_aggregate_state is not None:
                return NoEmission(NoEmissionReason.STREAM_GAP)
            return self._failed(request_id, "No audio generated")

        try:
            start_time = time.time()
            # A cumulative payload already carries the earlier frames, so it is
            # taken whole and de-duplicated in _append_audio_chunk; tracking the
            # newly appended entries would keep only the first snapshot.
            chunk_state: AudioStreamState | AudioAggregateState | None
            if audio_stream_state is not None:
                chunk_state = audio_stream_state
            elif (
                audio_aggregate_state is not None
                and not audio_aggregate_state.cumulative
            ):
                chunk_state = audio_aggregate_state
            else:
                chunk_state = None

            audio_np, sample_rate = self._extract_audio_tensor(
                mm_output, chunk_state=chunk_state
            )
            if audio_np.size == 0:
                return NoEmission(NoEmissionReason.STREAM_GAP)

            if audio_aggregate_state is not None:
                self._append_audio_chunk(audio_aggregate_state, audio_np, sample_rate)
                return NoEmission(NoEmissionReason.BUFFERED)

            encode_fmt = (output_format or "wav").lower()
            if audio_stream_state is not None:
                audio_bytes, _ = await asyncio.to_thread(
                    self._encode_audio_chunk,
                    audio_np,
                    sample_rate,
                    encode_fmt,
                    audio_stream_state,
                )
            else:
                audio_bytes, _ = await asyncio.to_thread(
                    self._encode_audio, audio_np, sample_rate, encode_fmt, speed
                )

            logger.debug(
                "Audio encoded for request %s: %d samples, sr=%d, %d bytes %s",
                request_id,
                audio_np.shape[-1],
                sample_rate,
                len(audio_bytes),
                encode_fmt,
            )

            if response_format == "url":
                ext = encode_fmt if encode_fmt != "opus" else "ogg"
                url = await upload_to_fs(
                    self._media_fs,
                    f"audios/{request_id}/{uuid.uuid4()}.{ext}",
                    audio_bytes,
                    self._media_http_url,
                )
                audio_data_obj = self._AudioData(output_format=encode_fmt, url=url)
            else:
                audio_data_obj = self._AudioData(
                    output_format=encode_fmt,
                    b64_json=base64.b64encode(audio_bytes).decode(),
                )

            return Emitted(
                NvAudioSpeechResponse(
                    id=request_id,
                    object="audio.speech",
                    model=self._model_name,
                    status="completed",
                    progress=100,
                    created=int(time.time()),
                    data=[audio_data_obj],
                    inference_time_s=time.time() - start_time,
                ).model_dump()
            )

        except Exception as e:
            logger.error("Failed to process audio for request %s: %s", request_id, e)
            return self._failed(request_id, str(e))

    async def format(
        self,
        stage_output: Any,
        request_id: str,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        speed: float = 1.0,
        audio_stream_state: AudioStreamState | None = None,
        audio_aggregate_state: AudioAggregateState | None = None,
    ) -> dict[str, Any] | None:
        """Serialize one audio payload; ``run`` is the typed contract.

        Args:
            stage_output: Stage output or multimodal audio mapping.
            request_id: Identifier included in the response and storage path.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Audio codec; WAV when unset.
            speed: Playback-speed multiplier applied to the whole file.
            audio_stream_state: Request-local state for incremental output.
            audio_aggregate_state: Request-local buffer for one final encode.

        Returns:
            Dict[str, Any] | None: Audio response, failure response, or no streaming chunk.
        """
        return serialize_outcome(
            await self.run(
                stage_output,
                request_id,
                response_format=response_format,
                output_format=output_format,
                speed=speed,
                audio_stream_state=audio_stream_state,
                audio_aggregate_state=audio_aggregate_state,
            )
        )

    async def finalize_aggregate(
        self,
        request_id: str,
        aggregate_state: AudioAggregateState,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        speed: float = 1.0,
    ) -> FormattingOutcome:
        """Encode all buffered raw chunks as one complete audio file.

        Args:
            request_id: Identifier included in the response and storage path.
            aggregate_state: Buffered audio chunks and their shared metadata.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Audio codec; WAV when unset.
            speed: Playback-speed multiplier applied to the whole file.

        Returns:
            FormattingOutcome: ``Emitted`` with the single audio response, or
                ``Failed`` when the request buffered nothing emittable.

        Raises:
            ValueError: If buffered chunks cannot be concatenated.
        """
        if not aggregate_state.chunks or aggregate_state.sample_rate is None:
            return self._failed(request_id, "No audio generated")

        audio_np = np.concatenate(aggregate_state.chunks, axis=-1)
        outcome = await self.run(
            {"audio": audio_np, "sr": aggregate_state.sample_rate},
            request_id,
            response_format=response_format,
            output_format=output_format,
            speed=speed,
        )
        if isinstance(outcome, NoEmission):
            return self._failed(request_id, "No audio generated")
        return outcome

    async def finish_aggregate(
        self,
        request_id: str,
        aggregate_state: AudioAggregateState,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        speed: float = 1.0,
    ) -> dict[str, Any]:
        """Serialize the buffered aggregate; ``finalize_aggregate`` is typed.

        Args:
            request_id: Identifier included in the response and storage path.
            aggregate_state: Buffered audio chunks and their shared metadata.
            response_format: ``"url"`` or ``"b64_json"`` representation.
            output_format: Audio codec; WAV when unset.
            speed: Playback-speed multiplier applied to the whole file.

        Returns:
            Dict[str, Any]: Completed or failed audio response.

        Raises:
            ValueError: If buffered chunks cannot be concatenated.
        """
        return serialize_outcome(
            await self.finalize_aggregate(
                request_id,
                aggregate_state,
                response_format=response_format,
                output_format=output_format,
                speed=speed,
            )
        )

    def _failed(self, request_id: str, error: str) -> Failed:
        """Build a failure outcome carrying the failed audio response body."""
        return Failed(
            error=error,
            compatible_payload=self._error_response(request_id, error),
        )

    def _append_audio_chunk(
        self,
        state: AudioAggregateState,
        audio_np: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Normalize and append one chunk to aggregate audio state.

        Args:
            state: Aggregate state receiving the normalized chunk.
            audio_np: Raw mono or stereo audio samples.
            sample_rate: Chunk sample rate in hertz.

        Returns:
            None.

        Raises:
            ValueError: If the chunk shape or metadata conflicts with prior chunks.
        """
        audio_np, num_channels, channel_axis = self._channel_first_audio(
            audio_np,
            expected_channels=state.num_channels,
            expected_channel_axis=state.channel_axis,
        )
        self._validate_audio_metadata(state, sample_rate, num_channels, channel_axis)
        if not state.cumulative:
            state.chunks.append(audio_np)
            return

        # Snapshots only grow, but keep the longest rather than the latest so a
        # truncated trailing payload cannot drop already-decoded audio.
        if state.chunks and state.chunks[0].shape[-1] >= audio_np.shape[-1]:
            return
        state.chunks = [audio_np]

    @staticmethod
    def _validate_audio_metadata(
        state: AudioStreamState | AudioAggregateState,
        sample_rate: int,
        num_channels: int,
        channel_axis: int | None,
    ) -> None:
        """Validate and record stable audio-stream metadata.

        Args:
            state: Streaming or aggregate state to validate and update.
            sample_rate: Current chunk sample rate in hertz.
            num_channels: Current chunk channel count.
            channel_axis: Original channel axis, or ``None`` for mono samples.

        Returns:
            None.

        Raises:
            ValueError: If sample rate, channel count, or layout changes.
        """
        if state.sample_rate is not None and state.sample_rate != sample_rate:
            raise ValueError(
                f"Audio sample rate changed from {state.sample_rate} to {sample_rate}"
            )
        if state.num_channels is not None and state.num_channels != num_channels:
            raise ValueError("Audio channel count changed while generating")
        if (
            state.channel_axis is not None
            and channel_axis is not None
            and state.channel_axis != channel_axis
        ):
            raise ValueError("Audio channel layout changed while generating")
        state.sample_rate = sample_rate
        state.num_channels = num_channels
        if channel_axis is not None:
            state.channel_axis = channel_axis

    def _extract_audio_tensor(
        self,
        mm_output: dict[str, Any],
        *,
        chunk_state: AudioStreamState | AudioAggregateState | None = None,
    ) -> tuple[np.ndarray, int]:
        """Extract new float32 audio samples and their sample rate.

        Args:
            mm_output: Multimodal mapping containing audio and sample-rate data.
            chunk_state: State used to skip chunks emitted previously.

        Returns:
            tuple[np.ndarray, int]: Audio samples and sample rate in hertz.

        Raises:
            ValueError: If audio data or sample-rate metadata is invalid.
            RuntimeError: If listed tensors cannot be concatenated.
            TypeError: If audio samples cannot be converted to a NumPy array.
        """
        audio_key = "audio" if "audio" in mm_output else "model_outputs"
        audio_val = mm_output.get(audio_key)
        if audio_val is None:
            raise ValueError(
                f"No audio data in multimodal_output. Keys: {list(mm_output.keys())}"
            )

        if isinstance(audio_val, list):
            if chunk_state is not None:
                # Slicing by a running count assumes each list *extends* the
                # previous one. That holds only while the engine does not drain
                # what it emits: a draining (``DELTA``) stage restarts its list
                # at index 0, and this would then keep just the tail. Audex
                # takes the delta path but yields bare tensors, which skip this
                # branch entirely. If a stage ever emits multi-entry lists under
                # DELTA, the count has to be per-payload rather than running.
                new_audio = audio_val[chunk_state.emitted_chunks :]
                chunk_state.emitted_chunks = len(audio_val)
                audio_val = new_audio
                if not audio_val:
                    return np.empty(0, dtype=np.float32), self._sample_rate(mm_output)
            audio_val = torch.cat(audio_val, dim=-1)

        if hasattr(audio_val, "float"):
            audio_np = audio_val.float().detach().cpu().numpy()
        elif isinstance(audio_val, np.ndarray):
            audio_np = audio_val.astype(np.float32)
        else:
            audio_np = np.array(audio_val, dtype=np.float32)

        return audio_np, self._sample_rate(mm_output)

    @staticmethod
    def _sample_rate(mm_output: dict[str, Any]) -> int:
        """Resolve the latest sample rate from multimodal output.

        Args:
            mm_output: Multimodal mapping containing optional ``sr`` metadata.

        Returns:
            int: Resolved sample rate, defaulting to 24000 Hz.

        Raises:
            TypeError: If sample-rate metadata is not integer-compatible.
            ValueError: If sample-rate metadata cannot be converted to an integer.
            RuntimeError: If tensor metadata contains more than one value.
        """
        sr_raw = mm_output.get("sr", 24000)
        if isinstance(sr_raw, list):
            sr_raw = sr_raw[-1] if sr_raw else 24000
        return sr_raw.item() if hasattr(sr_raw, "item") else int(sr_raw)

    def _encode_audio_chunk(
        self,
        audio_np: np.ndarray,
        sample_rate: int,
        fmt: str,
        stream_state: AudioStreamState,
    ) -> tuple[bytes, str]:
        """Encode one incremental audio chunk as PCM or streaming WAV.

        Args:
            audio_np: Raw mono or stereo audio samples.
            sample_rate: Chunk sample rate in hertz.
            fmt: Requested streaming format, ``"wav"`` or raw PCM.
            stream_state: State tracking established stream metadata.

        Returns:
            tuple[bytes, str]: Encoded bytes and their media type.

        Raises:
            ValueError: If audio layout or metadata is invalid or changes.
            sf.LibsndfileError: If PCM encoding fails.
        """
        audio_np, num_channels, channel_axis = self._normalize_audio_layout(
            audio_np,
            expected_channels=stream_state.num_channels,
            expected_channel_axis=stream_state.channel_axis,
        )
        first_chunk = stream_state.sample_rate is None
        self._validate_audio_metadata(
            stream_state, sample_rate, num_channels, channel_axis
        )
        pcm_bytes, _ = self._write_audio(audio_np, sample_rate, "pcm")
        if fmt == "wav" and first_chunk:
            pcm_bytes = self._wav_stream_header(sample_rate, num_channels) + pcm_bytes
        return pcm_bytes, "audio/wav" if fmt == "wav" else "audio/pcm"

    @staticmethod
    def _channel_first_audio(
        audio_np: np.ndarray,
        *,
        expected_channels: int | None = None,
        expected_channel_axis: int | None = None,
    ) -> tuple[np.ndarray, int, int | None]:
        """Normalize mono or stereo audio to vLLM-Omni's channel-first layout.

        An established channel axis determines the layout, including for square
        chunks. Otherwise, channel count can disambiguate the axes. Shapes that
        remain ambiguous follow vLLM-Omni's channel-first contract.

        Args:
            audio_np: Audio samples in mono, batched, or two-dimensional layout.
            expected_channels: Channel count established by earlier chunks.
            expected_channel_axis: Channel axis established by earlier chunks.

        Returns:
            tuple[np.ndarray, int, int | None]: Channel-first samples, channel
                count, and the original channel axis.

        Raises:
            ValueError: If the shape is unsupported or conflicts with prior layout.
        """
        if audio_np.ndim == 3:
            if audio_np.shape[0] != 1:
                raise ValueError(
                    f"Expected one audio batch, got shape {audio_np.shape}"
                )
            audio_np = audio_np[0]

        if audio_np.ndim == 1:
            return audio_np, 1, None

        if audio_np.ndim != 2:
            raise ValueError(f"Unexpected audio shape {audio_np.shape}")

        if expected_channel_axis is not None:
            if expected_channel_axis not in (0, 1):
                raise ValueError(f"Invalid audio channel axis {expected_channel_axis}")
            num_channels = int(audio_np.shape[expected_channel_axis])
            if num_channels not in (1, 2):
                raise ValueError(
                    "Audio channel layout changed while generating: "
                    f"expected channel axis {expected_channel_axis}, "
                    f"got shape {audio_np.shape}"
                )
            if expected_channel_axis == 0:
                return audio_np, num_channels, 0
            return audio_np.T, num_channels, 1

        channel_first_possible = audio_np.shape[0] in (1, 2)
        frame_major_possible = audio_np.shape[1] in (1, 2)
        channel_axis = None

        if expected_channels is not None:
            channel_first_matches = audio_np.shape[0] == expected_channels
            frame_major_matches = audio_np.shape[1] == expected_channels
            if channel_first_matches and not frame_major_matches:
                channel_axis = 0
            elif frame_major_matches and not channel_first_matches:
                channel_axis = 1

        if channel_axis is None:
            if channel_first_possible and frame_major_possible:
                logger.warning(
                    "Ambiguous audio shape %s without an established channel layout; "
                    "assuming vLLM-Omni's channel-first layout",
                    audio_np.shape,
                )
                channel_axis = 0
            elif channel_first_possible:
                channel_axis = 0
            elif frame_major_possible:
                channel_axis = 1
            else:
                raise ValueError(
                    f"Expected mono or stereo audio, got shape {audio_np.shape}"
                )

        num_channels = int(audio_np.shape[channel_axis])
        if channel_axis == 0:
            return audio_np, num_channels, channel_axis
        return audio_np.T, num_channels, channel_axis

    @classmethod
    def _normalize_audio_layout(
        cls,
        audio_np: np.ndarray,
        *,
        expected_channels: int | None = None,
        expected_channel_axis: int | None = None,
    ) -> tuple[np.ndarray, int, int | None]:
        """Convert supported audio layouts to soundfile's frame-major layout.

        Args:
            audio_np: Raw mono or stereo audio samples.
            expected_channels: Channel count established by earlier chunks.
            expected_channel_axis: Channel axis established by earlier chunks.

        Returns:
            tuple[np.ndarray, int, int | None]: Frame-major samples, channel
                count, and the original channel axis.

        Raises:
            ValueError: If the shape is unsupported or conflicts with prior layout.
        """
        audio_np, num_channels, channel_axis = cls._channel_first_audio(
            audio_np,
            expected_channels=expected_channels,
            expected_channel_axis=expected_channel_axis,
        )
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        return audio_np, num_channels, channel_axis

    @staticmethod
    def _wav_stream_header(
        sample_rate: int, num_channels: int = 1, bits_per_sample: int = 16
    ) -> bytes:
        """Build a PCM WAV header whose payload length is not known yet.

        Args:
            sample_rate: Audio sample rate in hertz.
            num_channels: Number of interleaved audio channels.
            bits_per_sample: PCM bit depth for each sample.

        Returns:
            bytes: WAV header with placeholder RIFF and data lengths.

        Raises:
            struct.error: If a numeric field does not fit the WAV header layout.
        """
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        placeholder_size = 0xFFFFFFFF

        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            placeholder_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            placeholder_size,
        )

    def _encode_audio(
        self, audio_np: Any, sample_rate: int, fmt: str = "wav", speed: float = 1.0
    ) -> tuple[bytes, str]:
        """Normalize, optionally retime, and encode complete audio.

        Args:
            audio_np: Raw mono or stereo audio samples.
            sample_rate: Audio sample rate in hertz.
            fmt: Requested output format.
            speed: Playback-speed multiplier applied when librosa is available.

        Returns:
            tuple[bytes, str]: Encoded audio bytes and their media type.

        Raises:
            ValueError: If the audio layout or speed is invalid.
            sf.LibsndfileError: If audio encoding fails.
        """
        audio_np, _, _ = self._channel_first_audio(audio_np)
        if speed != 1.0:
            try:
                import librosa

                audio_np = librosa.effects.time_stretch(y=audio_np, rate=speed)
            except ImportError:
                logger.warning("librosa not installed, ignoring speed adjustment")

        if audio_np.ndim == 2:
            audio_np = audio_np.T
        return self._write_audio(audio_np, sample_rate, fmt)

    @staticmethod
    def _write_audio(
        audio_np: np.ndarray, sample_rate: int, fmt: str
    ) -> tuple[bytes, str]:
        """Encode frame-major audio with soundfile.

        Args:
            audio_np: Frame-major mono or stereo samples.
            sample_rate: Audio sample rate in hertz.
            fmt: Requested output format; unsupported values fall back to WAV.

        Returns:
            tuple[bytes, str]: Encoded audio bytes and their media type.

        Raises:
            sf.LibsndfileError: If the selected codec cannot encode the audio.
            TypeError: If samples or format parameters are incompatible.
        """
        fmt = (fmt or "wav").lower()
        format_map = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "aac": ("AAC", "audio/aac", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if fmt not in format_map:
            logger.warning("Unsupported format '%s', defaulting to wav", fmt)
            fmt = "wav"

        sf_format, media_type, kwargs = format_map[fmt]

        buf = BytesIO()
        sf.write(buf, audio_np, sample_rate, format=sf_format, **kwargs)
        return buf.getvalue(), media_type

    def _error_response(self, request_id: str, error: str) -> dict[str, Any]:
        """Build a failed audio-speech response.

        Args:
            request_id: Identifier included in the response.
            error: User-facing error description.

        Returns:
            Dict[str, Any]: Serialized failed audio response.
        """
        return NvAudioSpeechResponse(
            id=request_id,
            model=self._model_name,
            status="failed",
            created=int(time.time()),
            error=error,
        ).model_dump()


def _error_chunk(
    request_id: str, model_name: str, error_message: str
) -> dict[str, Any]:
    """Build an OpenAI chat-completion error chunk.

    Args:
        request_id: Identifier included in the response chunk.
        model_name: Model identifier included in the response chunk.
        error_message: User-facing error description.

    Returns:
        Dict[str, Any]: Serialized chat-completion error chunk.
    """
    return {
        "id": request_id,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": f"Error: {error_message}"},
                "finish_reason": "error",
            }
        ],
    }


def _build_completion_usage(request_output: Any) -> dict[str, Any]:
    """Build token-usage statistics from a vLLM request output.

    Args:
        request_output: Request output containing prompt and completion token IDs.

    Returns:
        Dict[str, Any]: Prompt, completion, total, and cached-token statistics.

    Raises:
        AttributeError: If required output token fields are missing.
        IndexError: If the request output contains no choices.
        TypeError: If token identifiers do not provide a length.
    """
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    prompt_tokens = (
        len(prompt_token_ids)
        if prompt_token_ids is not None and not is_empty_payload(prompt_token_ids)
        else None
    )
    completion_tokens = len(request_output.outputs[0].token_ids)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            prompt_tokens + completion_tokens if prompt_tokens is not None else None
        ),
        "prompt_tokens_details": build_prompt_tokens_details(
            getattr(request_output, "num_cached_tokens", None)
        ),
    }


class OutputFormatter:
    """Dispatches raw engine output to modality-specific formatters.

    Shared by OmniHandler (aggregated) and OmniStageRouter (disaggregated).
    Both call ``format()`` once per stage with a per-request
    ``FormatSession`` and then ``finish()`` once, so neither knows a concrete
    formatter's parameters and both read the same outcome type.
    """

    def __init__(
        self,
        model_name: str,
        media_fs: Any = None,
        media_http_url: str | None = None,
        default_fps: int = 16,
    ) -> None:
        """Initialize modality-specific output formatters.

        Args:
            model_name: Model identifier included in responses.
            media_fs: Storage backend used for URL responses.
            media_http_url: Public base URL for stored media.
            default_fps: Frame rate used when video metadata omits one.

        Returns:
            None.
        """
        self._model_name = model_name
        self._media_fs = media_fs
        self._text = TextFormatter(model_name)
        self._diffusion = DiffusionFormatter(
            model_name, media_fs, media_http_url, default_fps
        )
        self._audio = AudioFormatter(model_name, media_fs, media_http_url)

    def engine_context(
        self,
        request_type: RequestType,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        fps: int = 0,
        speed: float = 1.0,
    ) -> FormatContext:
        """Validate the aggregated handler's request intent before any stage runs.

        Args:
            request_type: Request kind parsed at the entry point.
            response_format: Requested representation, if any.
            output_format: Requested container or codec, if any.
            fps: Requested frame rate; ``0`` means unset.
            speed: Requested audio retiming; ``1.0`` means unset.

        Returns:
            FormatContext: The validated context for this request.

        Raises:
            FormatContractError: If the intent is unusable on this deployment.
        """
        return FormatContext.from_engine_inputs(
            request_type,
            response_format=response_format,
            output_format=output_format,
            fps=fps,
            speed=speed,
            media_storage_available=self._media_fs is not None,
        )

    def request_context(
        self, request: Mapping, request_type: RequestType
    ) -> FormatContext:
        """Validate the disaggregated router's raw request intent.

        Args:
            request: Raw frontend request mapping.
            request_type: Request kind parsed from that request.

        Returns:
            FormatContext: The validated context for this request.

        Raises:
            FormatContractError: If the intent is unusable on this deployment.
        """
        return FormatContext.from_request(
            request,
            request_type,
            media_storage_available=self._media_fs is not None,
        )

    async def format(
        self,
        context: FormatContext,
        stage_output: Any,
        session: FormatSession,
    ) -> FormattingOutcome:
        """Format one stage output under a request's validated context.

        Args:
            context: Validated request intent and options.
            stage_output: vLLM-Omni output carrying ``final_output_type``.
            session: The request's own state.

        Returns:
            FormattingOutcome: What this stage contributed; a legal gap and a
                failure are different values, never a shared ``None``.

        Raises:
            AttributeError: If a text output lacks required generation fields.
            ValueError: If a diffusion response option is unsupported.
        """
        dispatch = resolve_dispatch(context, stage_output)
        if isinstance(dispatch, Failed):
            outcome = self._failure(session, dispatch.error)
        else:
            outcome = await self._run_dispatch(dispatch, context, stage_output, session)
        session.advance_text(stage_output)
        if isinstance(outcome, Emitted):
            session.note_emitted()
        elif isinstance(outcome, Failed):
            session.note_failed()
        return outcome

    async def finish(
        self, context: FormatContext, session: FormatSession
    ) -> FormattingOutcome:
        """Finalize a request exactly once.

        Encodes whatever the request buffered and reports whether the request
        produced anything at all. The result is latched on the session, so a
        repeated finish re-encodes nothing and re-uploads nothing.

        Args:
            context: Validated request intent and options.
            session: The request's own state.

        Returns:
            FormattingOutcome: ``Emitted`` with the buffered result, or an
                explicit ``NoEmission``/``Failed``; never a silent empty
                success for a request that produced nothing.
        """
        if session.final_outcome is not None:
            return session.final_outcome

        if session.aborted:
            outcome: FormattingOutcome = NoEmission(NoEmissionReason.ABORTED)
        elif session.reported_failure:
            # The failure body is already on its way; adding a second one at
            # the request end would answer one broken step with two responses.
            outcome = NoEmission(NoEmissionReason.COMPLETE)
        elif session.audio_aggregate_state is not None:
            outcome = await self._audio.finalize_aggregate(
                session.request_id,
                session.audio_aggregate_state,
                response_format=context.response_format,
                output_format=context.output_format,
                speed=context.speed if context.speed is not None else 1.0,
            )
        elif session.emitted:
            outcome = NoEmission(NoEmissionReason.COMPLETE)
        else:
            outcome = self._failure(
                session,
                f"No {session.modality.value} output was produced for this request",
            )
        session.final_outcome = outcome
        session.release_audio()
        return outcome

    async def _run_dispatch(
        self,
        dispatch: StageDispatch,
        context: FormatContext,
        stage_output: Any,
        session: FormatSession,
    ) -> FormattingOutcome:
        """Send one resolved stage to the formatter that owns its modality."""
        if dispatch.modality is Modality.TEXT:
            request_output = getattr(stage_output, "request_output", None)
            if request_output is None:
                # The stage carries no engine output on this step, so there is
                # no delta to serialize. Legal mid-stream, not a failure.
                return NoEmission(NoEmissionReason.STREAM_GAP)
            return self._text.run(
                request_output,
                session.request_id,
                previous_text=session.previous_text,
            )
        if dispatch.modality is Modality.AUDIO:
            return await self._audio.run(
                stage_output,
                session.request_id,
                response_format=context.response_format,
                output_format=context.output_format,
                speed=context.speed if context.speed is not None else 1.0,
                audio_stream_state=session.audio_stream_state,
                audio_aggregate_state=session.audio_aggregate_state,
            )
        return await self._diffusion.run(
            stage_output,
            session.request_id,
            dispatch=dispatch,
            fps=context.fps,
            response_format=context.response_format,
            output_format=context.output_format,
        )

    def _failure(
        self,
        session: FormatSession,
        error: str,
        modality: Modality | None = None,
    ) -> Failed:
        """Build a failure outcome carrying this modality's failure body."""
        return Failed(
            error=error,
            compatible_payload=_failure_payload(
                modality or session.modality,
                session.request_id,
                self._model_name,
                error,
            ),
        )
