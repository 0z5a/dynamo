# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed formatting contract shared by every Omni caller.

Before this module, ``OutputFormatter`` accepted an untyped ``**ctx`` bag and
decided the output modality itself, so the aggregated ``OmniHandler`` and the
disaggregated ``OmniStageRouter`` each had their own reading of what a stage
meant and each interpreted a missing payload on its own. Three things are
defined here and nowhere else:

* :class:`FormatContext` -- validated request intent and options. An unknown
  keyword or an ill-typed value raises before any encoder or uploader runs.
* :func:`resolve_dispatch` -- the one place that turns request intent plus one
  stage's metadata into the modality that stage is formatted as.
* :class:`FormatSession` -- per-request state (previous text, audio buffers,
  finished/aborted status). Model-scoped formatters hold only request-free
  configuration, so nothing here can cross between requests.
* :class:`FormattingOutcome` and :func:`serialize_outcome` -- a legal streaming
  gap, a buffered chunk and a real failure are separate values, and the one
  mapping onto a caller-visible payload lives here rather than in each caller.

Options are validated with Pydantic, the same type system the request
protocols in ``dynamo.common.protocols`` already use; no second schema
framework is introduced.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union

import numpy as np
from dynamo.common.utils.output_modalities import RequestType
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)


class FormatContractError(ValueError):
    """A request's formatting intent is invalid.

    Raised at the boundary, before any encoder or storage backend is touched,
    so an unusable request cannot start a pointless encode. It is a
    ``ValueError`` because that is what the media request paths already map
    onto an HTTP 400 at the binding boundary.
    """


class Modality(Enum):
    """Normalized output modality of one dispatched stage."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ResponseSchema(Enum):
    """Response shape a serializer must produce for one dispatch."""

    CHAT_CHUNK = "chat_completion"
    """OpenAI ``chat.completion.chunk``: the request went to a chat endpoint."""

    MEDIA = "media"
    """The modality's own response model: the request went to a media endpoint."""


_REQUEST_MODALITY = {
    RequestType.CHAT_COMPLETION: Modality.TEXT,
    RequestType.IMAGE_GENERATION: Modality.IMAGE,
    RequestType.VIDEO_GENERATION: Modality.VIDEO,
    RequestType.AUDIO_GENERATION: Modality.AUDIO,
}


def request_modality(request_type: RequestType) -> Modality:
    """Return the modality a request produces when no stage narrows it further."""
    return _REQUEST_MODALITY[request_type]


_MEDIA_OUTPUT_FORMATS = frozenset({"mp4"})
_AUDIO_OUTPUT_FORMATS = frozenset({"wav", "pcm", "flac", "mp3", "aac", "opus"})


class FormatContext(BaseModel):
    """Validated formatting intent for one request.

    Holds request intent and options only. Execution state -- previous text,
    decoded audio, finish status -- belongs to :class:`FormatSession`; engine
    and storage handles belong to the model-scoped formatter.

    ``extra="forbid"`` is the point of the model: a misspelled option is a
    request error rather than a silently ignored keyword, which is what the old
    ``**ctx`` bag made impossible to detect.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_type: RequestType
    """Request kind, parsed once at the entry point and never re-derived below."""

    response_format: Literal["url", "b64_json"] | None = None
    """Response representation; ``None`` keeps the modality's own default."""

    output_format: str | None = None
    """Video container or audio codec; ``None`` keeps the modality's default."""

    fps: int | None = Field(default=None, gt=0)
    """Frame rate for a video result. Only video stages read it."""

    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    """Playback-speed multiplier for audio. Only audio stages read it."""

    @field_validator("output_format", mode="before")
    @classmethod
    def _normalize_output_format(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _reject_foreign_options(self) -> "FormatContext":
        """Reject options that cannot apply to a single-modality endpoint.

        An image, video or audio endpoint describes exactly one output, so an
        option owned by a different modality can only be a mistake. A chat
        completion is genuinely multi-modal -- one request holds a text stage
        and an audio stage -- so option groups coexist there and none is
        rejected for being unused by the stage that happens to be dispatching.
        """
        if self.request_type is RequestType.CHAT_COMPLETION:
            return self
        if self.request_type is RequestType.IMAGE_GENERATION and self.speed is not None:
            raise ValueError(_foreign_option("speed", "audio", "image"))
        if self.request_type is RequestType.VIDEO_GENERATION and self.speed is not None:
            raise ValueError(_foreign_option("speed", "audio", "video"))
        if self.request_type is RequestType.AUDIO_GENERATION and self.fps is not None:
            raise ValueError(_foreign_option("fps", "video", "audio"))
        return self

    @model_validator(mode="after")
    def _check_output_format(self) -> "FormatContext":
        if self.output_format is None:
            return self
        if not self.output_format:
            raise ValueError("output_format must not be blank")
        if self.request_type is RequestType.AUDIO_GENERATION:
            allowed = _AUDIO_OUTPUT_FORMATS
        elif self.request_type is RequestType.VIDEO_GENERATION:
            allowed = _MEDIA_OUTPUT_FORMATS
        else:
            allowed = _MEDIA_OUTPUT_FORMATS | _AUDIO_OUTPUT_FORMATS
        if self.output_format not in allowed:
            raise ValueError(
                f"Unsupported output_format {self.output_format!r} for a "
                f"{self.request_type.value} request; expected one of "
                f"{', '.join(sorted(allowed))}"
            )
        return self

    @property
    def requires_media_storage(self) -> bool:
        """Whether the response must be uploaded to a media filesystem.

        The video path defaults to ``url`` when the request does not choose a
        representation, so an unset ``response_format`` on a video request still
        needs storage. Images and audio default to an inline payload.
        """
        if self.response_format is not None:
            return self.response_format == "url"
        return self.request_type is RequestType.VIDEO_GENERATION

    @classmethod
    def from_options(
        cls,
        request_type: RequestType,
        options: Mapping[str, Any],
        *,
        media_storage_available: bool = True,
    ) -> "FormatContext":
        """Validate one caller's option mapping.

        Args:
            request_type: Request kind parsed at the entry point.
            options: Formatting options as named by the request vocabulary.
                Unknown names are rejected, so a misspelled option fails here.
            media_storage_available: Whether this deployment has a media
                filesystem configured.

        Returns:
            FormatContext: The validated context.

        Raises:
            FormatContractError: If an option is unknown, ill-typed, out of
                range, foreign to the endpoint, or asks for a URL that this
                deployment cannot store.
        """
        try:
            context = cls(request_type=request_type, **dict(options))
        except ValidationError as exc:
            raise FormatContractError(
                f"Invalid formatting options for a {request_type.value} request: "
                f"{_describe_validation_error(exc)}"
            ) from exc
        if context.requires_media_storage and not media_storage_available:
            raise FormatContractError(
                "response_format='url' requires media storage, but this worker has "
                "none configured; set media_output_fs_url or request 'b64_json'"
            )
        return context

    @classmethod
    def from_request(
        cls,
        request: Mapping[str, Any],
        request_type: RequestType,
        *,
        media_storage_available: bool = True,
    ) -> "FormatContext":
        """Build a context from a raw frontend request.

        Args:
            request: Raw request mapping as received by the router.
            request_type: Request kind parsed from that request.
            media_storage_available: Whether this deployment has a media
                filesystem configured.

        Returns:
            FormatContext: The validated context.

        Raises:
            FormatContractError: If the carried options are unusable.

        Note:
            The audio endpoint spells the response representation
            ``data_source`` and the codec ``response_format``; every other
            modality uses ``response_format`` and ``output_format``. That alias
            is resolved here, once, so no caller translates option names on its
            own.
        """
        nvext = request.get("nvext")
        nvext = nvext if isinstance(nvext, Mapping) else {}

        options: dict[str, Any] = {}
        if request_type is RequestType.AUDIO_GENERATION:
            options["response_format"] = request.get("data_source")
            options["output_format"] = request.get("response_format")
        else:
            options["response_format"] = request.get("response_format")
            options["output_format"] = request.get("output_format")
            options["fps"] = nvext.get("fps")
        speed = request.get("speed")
        options["speed"] = nvext.get("speed") if speed is None else speed

        return cls.from_options(
            request_type,
            {name: value for name, value in options.items() if value is not None},
            media_storage_available=media_storage_available,
        )

    @classmethod
    def from_engine_inputs(
        cls,
        request_type: RequestType,
        *,
        response_format: str | None = None,
        output_format: str | None = None,
        fps: int = 0,
        speed: float = 1.0,
        media_storage_available: bool = True,
    ) -> "FormatContext":
        """Build a context from the aggregated handler's engine inputs.

        ``fps=0`` and ``speed=1.0`` are ``EngineInputs``' "not set" values
        rather than request options: zero is not a frame rate and 1.0 is the
        identity retiming, so both become "unset" here instead of being
        validated as an explicit choice.
        """
        options: dict[str, Any] = {
            "response_format": response_format,
            "output_format": output_format,
            "fps": fps if fps else None,
            "speed": None if speed == 1.0 else speed,
        }
        return cls.from_options(
            request_type,
            {name: value for name, value in options.items() if value is not None},
            media_storage_available=media_storage_available,
        )


def _foreign_option(option: str, owner: str, endpoint: str) -> str:
    return (
        f"{option!r} applies to {owner} generation but this is a {endpoint} "
        f"request; send it to the {owner} endpoint"
    )


def _describe_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<options>'}: {error['msg']}"
        for error in exc.errors()
    )


@dataclass(frozen=True)
class StageDispatch:
    """One stage normalized against the request's intent."""

    modality: Modality
    """Formatter that must serialize this stage."""

    schema: ResponseSchema
    """Response shape the serializer must produce."""

    final_output_type: str
    """The stage's own label, kept for diagnostics."""


def resolve_dispatch(
    context: FormatContext, stage_output: Any
) -> Union[StageDispatch, "Failed"]:
    """Decide, once per stage, which formatter handles it.

    The stage's own ``final_output_type`` is authoritative, so a chat request
    whose pipeline runs text and then audio keeps a text stage text and an
    audio stage audio. The one promotion is the diffusion representation:
    vLLM-Omni labels every diffusion stage ``image``, so under a video request
    that payload is the frame sequence of a video result and video semantics --
    container, frame rate, muxed audio track -- win.

    Args:
        context: The request's validated formatting intent.
        stage_output: One engine stage output carrying ``final_output_type``.

    Returns:
        StageDispatch | Failed: The normalized dispatch, or an explicit failure
            when the stage names no modality this build can serialize.
    """
    final_output_type = getattr(stage_output, "final_output_type", None)
    if not isinstance(final_output_type, str) or not final_output_type:
        return Failed(
            error="Stage output carries no final_output_type; cannot choose a formatter"
        )

    try:
        modality = Modality(final_output_type)
    except ValueError:
        known = ", ".join(m.value for m in Modality)
        return Failed(
            error=(
                f"Unsupported final_output_type {final_output_type!r}; "
                f"expected one of {known}"
            )
        )

    if (
        modality is Modality.IMAGE
        and context.request_type is RequestType.VIDEO_GENERATION
    ):
        modality = Modality.VIDEO

    schema = (
        ResponseSchema.CHAT_CHUNK
        if context.request_type is RequestType.CHAT_COMPLETION
        else ResponseSchema.MEDIA
    )
    return StageDispatch(
        modality=modality, schema=schema, final_output_type=final_output_type
    )


class NoEmissionReason(Enum):
    """Why a dispatch produced nothing to send.

    These are not interchangeable: ``STREAM_GAP`` and ``BUFFERED`` are normal
    parts of a healthy request, ``COMPLETE`` says the response is already fully
    sent, and ``ABORTED`` says the request was cancelled and its buffers were
    released.
    """

    STREAM_GAP = "stream_gap"
    """A stage legally carries no payload for the response being built."""

    BUFFERED = "buffered"
    """The payload was retained in request-local audio state for ``finish()``."""

    COMPLETE = "complete"
    """Nothing is left to send; the response was already emitted."""

    ABORTED = "aborted"
    """The request was cancelled; nothing final is emitted."""


@dataclass(frozen=True)
class Emitted:
    """One payload that must be sent to the client."""

    payload: dict[str, Any]


@dataclass(frozen=True)
class NoEmission:
    """Nothing to send, with the reason that makes it legal or terminal."""

    reason: NoEmissionReason


@dataclass(frozen=True)
class Failed:
    """A serialization step failed.

    ``compatible_payload`` is the body this modality already uses to report a
    failure. The boundary fills it in, so both callers send the same thing and
    neither has to invent one.
    """

    error: str
    compatible_payload: dict[str, Any] | None = None


FormattingOutcome = Union[Emitted, NoEmission, Failed]


def serialize_outcome(outcome: FormattingOutcome) -> dict[str, Any] | None:
    """Map an outcome onto the payload a caller sends.

    Args:
        outcome: One formatting outcome.

    Returns:
        dict | None: The payload, or ``None`` when there is nothing to send.
    """
    match outcome:
        case Emitted(payload):
            return payload
        case NoEmission():
            return None
        case Failed(compatible_payload=payload):
            return payload
    raise TypeError(f"Unknown formatting outcome: {outcome!r}")


@dataclass
class AudioStreamState:
    """Request-local state for incremental audio output."""

    emitted_chunks: int = 0
    sample_rate: int | None = None
    num_channels: int | None = None
    channel_axis: int | None = None


@dataclass
class AudioAggregateState:
    """Request-local raw audio accumulated for one final encode."""

    chunks: list[np.ndarray] = field(default_factory=list)
    sample_rate: int | None = None
    emitted_chunks: int = 0
    num_channels: int | None = None
    channel_axis: int | None = None
    cumulative: bool = False
    """Each payload is a snapshot of the whole waveform decoded so far.

    Set from the output kind the engine was actually given, not from the model:
    ``RequestOutputKind.CUMULATIVE`` consolidates the accumulated audio on every
    step and drains nothing, so the snapshots must be de-duplicated to the
    longest one rather than concatenated, while ``DELTA`` drains what it emits
    and yields disjoint pieces that must all be kept. See
    ``utils.audio_output_is_cumulative``, which the handler uses to fill this in.
    """


class FormatSession:
    """Everything one request accumulates while it is being formatted.

    One instance per request. Because the formatter itself is model-scoped and
    holds only request-free configuration, previous text, audio buffers, sample
    rate and finish state cannot leak from one request into another.
    """

    def __init__(self, request_id: str, *, request_type: RequestType) -> None:
        self.request_id = request_id
        self.modality = request_modality(request_type)
        self.previous_text = ""
        self.audio_stream_state: AudioStreamState | None = None
        self.audio_aggregate_state: AudioAggregateState | None = None
        self.emitted = False
        self.reported_failure = False
        self.aborted = False
        self.final_outcome: FormattingOutcome | None = None
        """Latched result of ``finish()``, so a repeated finish re-encodes nothing."""

    @classmethod
    def for_request(
        cls,
        request_id: str,
        request_type: RequestType,
        *,
        stream_audio: bool = False,
        cumulative_audio: bool = False,
    ) -> "FormatSession":
        """Open the request-local session for one request.

        An audio request that streams keeps an :class:`AudioStreamState` and
        encodes every chunk as it arrives. Any other audio request buffers into
        an :class:`AudioAggregateState` and encodes once in ``finish()``, which
        is what makes "emit exactly one file" and "a repeated finish does
        nothing" properties of the request instead of habits of each caller.

        Args:
            request_id: Identifier shared by every payload of this request.
            request_type: Request kind parsed at the entry point.
            stream_audio: Whether the engine streams audio for this request.
            cumulative_audio: Whether each audio payload repeats the whole
                waveform decoded so far.

        Returns:
            FormatSession: An empty session for this request.
        """
        session = cls(request_id, request_type=request_type)
        if request_type is RequestType.AUDIO_GENERATION:
            if stream_audio:
                session.audio_stream_state = AudioStreamState()
            else:
                session.audio_aggregate_state = AudioAggregateState(
                    cumulative=cumulative_audio
                )
        return session

    def advance_text(self, stage_output: Any) -> None:
        """Record how much text this request has produced.

        The only place previous text advances: the dispatcher calls it after
        formatting a text stage, so no caller keeps a second copy that could
        advance the same stage twice and truncate the next delta.
        """
        if getattr(stage_output, "final_output_type", None) != "text":
            return
        request_output = getattr(stage_output, "request_output", None)
        if request_output is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        self.previous_text = outputs[0].text

    def note_emitted(self) -> None:
        """Record that at least one payload has been sent for this request."""
        self.emitted = True

    def note_failed(self) -> None:
        """Record that a failure body has been sent for this request.

        ``finish()`` reads this so one broken step produces one response body
        instead of an error at the stage and a second one at the request end.
        """
        self.reported_failure = True

    def abort(self) -> None:
        """Mark the request cancelled and release what it buffered."""
        self.aborted = True
        self.release_audio()

    def release_audio(self) -> None:
        """Drop the buffered waveform; the encoded payload, if any, is kept."""
        if self.audio_aggregate_state is not None:
            self.audio_aggregate_state.chunks.clear()
