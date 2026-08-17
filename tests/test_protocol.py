from __future__ import annotations

from typing import Any, cast

import msgspec
import pytest
import vapoursynth as vs

from vsremote.exceptions import (
    MalformedMessageError,
    UnknownCommandError,
    UnknownStatusCodeError,
    UnsupportedFormatError,
)
from vsremote.protocol import (
    ClipInfo,
    Command,
    FrameHeader,
    FrameRequest,
    OutputIndexRequest,
    PlaneInfo,
    RemoteLogRecord,
    RequestEnvelope,
    ResponseEnvelope,
    StatusCode,
    StreamOutputEvent,
    StreamSubscribeRequest,
    pack_payload,
    sanitize_props,
    unpack_payload,
)


def test_plane_info_serialization() -> None:
    p = PlaneInfo(width=1920, height=1080, bytes_per_sample=1, size_bytes=1920 * 1080)
    direct_data = pack_payload(p)
    p2 = unpack_payload(direct_data, PlaneInfo)
    assert p2 == p


@pytest.mark.vpy("initial-core")
def test_plane_info_from_clip() -> None:
    with pytest.raises(UnsupportedFormatError, match="Variable format clips are not supported by vs-remote"):
        PlaneInfo.from_clip(vs.core.std.BlankClip(varformat=True))


@pytest.mark.vpy("initial-core")
def test_clip_info_serialization() -> None:
    core = vs.core
    clip = core.std.BlankClip(width=1920, height=1080, format=vs.YUV420P8, length=100, fpsnum=24, fpsden=1)
    info = ClipInfo.from_clip(clip, name="Main")

    direct_payload = pack_payload(info)
    info2 = unpack_payload(direct_payload, ClipInfo)

    assert info2.width == info.width
    assert info2.height == info.height
    assert info2.num_frames == info.num_frames
    assert info2.format_id == info.format_id
    assert info2.format_name == info.format_name
    assert info2.planes == info.planes
    assert info2.name == "Main"
    assert info2 == info


def test_clip_info_no_format_error() -> None:
    class FakeClipNoFormat:
        format = None

    with pytest.raises(UnsupportedFormatError, match="Variable format clips are not supported by vs-remote"):
        ClipInfo.from_clip(FakeClipNoFormat())  # type: ignore[arg-type]


def test_frame_header_serialization() -> None:
    header = FrameHeader(
        status=StatusCode.OK,
        request_id=42,
        n=15,
        output_index=0,
        compression="zstd",
        plane_sizes=[1024, 256, 256],
        props={"_SARNum": 1, "_SARDen": 1, "test": "val", "bytes_prop": b"hello"},
    )

    data = pack_payload(header)
    header2 = unpack_payload(data, FrameHeader)

    assert header2.status == header.status
    assert header2.request_id == header.request_id
    assert header2.n == header.n
    assert header2.props["_SARNum"] == 1
    assert header2.props["bytes_prop"] == b"hello"
    assert header2 == header


def test_sanitize_props() -> None:
    custom_obj = object()
    nested_bad_list = [custom_obj]
    raw = {
        "int": 10,
        "float": 3.14,
        "str": "test",
        "bytes": b"binary",
        "list": [1, 2, "three", b"bytes"],
        "custom": custom_obj,
        "bad_list": nested_bad_list,
    }
    clean = sanitize_props(raw)
    assert clean["list"] == [1, 2, "three", b"bytes"]
    assert clean["custom"] == repr(custom_obj)
    assert clean["bad_list"] == repr(nested_bad_list)

    # Verify MsgPack can pack and unpack it cleanly
    packed = pack_payload(clean)
    unpacked = unpack_payload(packed)
    assert unpacked["bytes"] == b"binary"
    assert unpacked["custom"] == repr(custom_obj)


def test_request_envelope_from_frames() -> None:
    # Standard 4-frame request
    frames = [b"client-id-1", (42).to_bytes(4, "big"), bytes([Command.GET_FRAME.value]), pack_payload({"n": 5})]
    req = RequestEnvelope.from_frames(frames)
    assert req.identity == b"client-id-1"
    assert req.request_id == 42
    assert req.request_id_bytes == (42).to_bytes(4, "big")
    assert req.command == Command.GET_FRAME
    frame_req = unpack_payload(req.payload_bytes, FrameRequest)
    assert frame_req.n == 5
    assert req.extra_frames == []

    # 3-frame request (e.g. PING)
    frames_ping = [b"client-id-2", (1).to_bytes(4, "big"), bytes([Command.PING.value])]
    req_ping = RequestEnvelope.from_frames(frames_ping)
    assert req_ping.identity == b"client-id-2"
    assert req_ping.request_id == 1
    assert req_ping.command == Command.PING
    assert req_ping.payload_bytes == b""
    assert req_ping.extra_frames == []

    # Multipart with extra frames and auth token
    frames_extra = [
        b"client-id-3",
        (2).to_bytes(4, "big"),
        bytes([Command.CLOSE.value]),
        b"",
        b"secret_token_123",
        b"extra2",
    ]
    req_extra = RequestEnvelope.from_frames(frames_extra)
    assert req_extra.extra_frames == [b"secret_token_123", b"extra2"]
    assert req_extra.auth_token == "secret_token_123"

    # Extra frame with invalid UTF-8 (fails auth token decode gracefully)
    frames_invalid_utf8 = [
        b"client-id-4",
        (3).to_bytes(4, "big"),
        bytes([Command.PING.value]),
        b"",
        b"\xff\xfe\xfd",
    ]
    req_invalid_utf8 = RequestEnvelope.from_frames(frames_invalid_utf8)
    assert req_invalid_utf8.auth_token is None

    # Malformed: less than 3 frames
    with pytest.raises(MalformedMessageError, match="Malformed multipart request"):
        RequestEnvelope.from_frames([b"id", b"req_id"])

    # Invalid command byte
    with pytest.raises(UnknownCommandError, match="Unknown command byte"):
        RequestEnvelope.from_frames([b"id", (1).to_bytes(4, "big"), bytes([255])])


def test_response_envelope_from_frames() -> None:
    # 2-frame response
    frames_2 = [bytes([StatusCode.OK]), pack_payload({"status": "healthy"})]
    resp_2 = ResponseEnvelope.from_frames(frames_2)
    assert resp_2.status == StatusCode.OK
    assert resp_2.payload == pack_payload({"status": "healthy"})

    # Single frame response (status byte only)
    frames_single = [bytes([StatusCode.OK])]
    resp_single = ResponseEnvelope.from_frames(frames_single)
    assert resp_single.payload_bytes == b""
    assert resp_single.extra_frames == []

    # Empty frames raises MalformedMessageError
    with pytest.raises(MalformedMessageError, match="Malformed multipart response"):
        ResponseEnvelope.from_frames([])

    # Invalid status code byte
    with pytest.raises(UnknownStatusCodeError, match="Unknown status code byte"):
        ResponseEnvelope.from_frames([bytes([250])])

    # Error status with typed target falling back to untyped payload decoding
    err_frames = [bytes([StatusCode.ERROR]), pack_payload({"error": "Failed to load script"})]
    err_resp = ResponseEnvelope.from_frames(err_frames, ClipInfo)
    assert err_resp.status == StatusCode.ERROR
    assert cast(dict[str, Any], err_resp.payload) == {"error": "Failed to load script"}


def test_unpack_payload_typed_and_empty() -> None:
    # Empty byte buffer with default dataclass
    req = unpack_payload(b"", OutputIndexRequest)
    assert isinstance(req, OutputIndexRequest)
    assert req.output_index == 0

    frame_req = unpack_payload(b"", FrameRequest)
    assert isinstance(frame_req, FrameRequest)
    assert frame_req.n == 0
    assert frame_req.output_index == 0
    assert frame_req.compression == "zstd"

    # Struct with required fields: calling Target() raises TypeError, fallback decodes empty msgpack map
    class StructWithRequired(msgspec.Struct):
        val: int

    with pytest.raises(msgspec.ValidationError):
        unpack_payload(b"", StructWithRequired)

    # Empty byte buffer with untyped default
    assert unpack_payload(b"") == {}


def test_stream_events_serialization() -> None:
    out_evt = StreamOutputEvent(stream="stdout", text="hello stdout")
    packed_out = pack_payload(out_evt)
    unpacked_out = unpack_payload(packed_out, StreamOutputEvent)
    assert isinstance(unpacked_out, StreamOutputEvent)
    assert unpacked_out.stream == "stdout"
    assert unpacked_out.text == "hello stdout"

    log_rec = RemoteLogRecord(
        name="test.logger",
        levelno=30,
        levelname="WARNING",
        msg="A warning was emitted: %s",
        args=("some_val",),
        filename="test.py",
        lineno=42,
        funcName="do_work",
        created=1700000000.0,
        exc_text="Traceback ...",
        stack_info=None,
    )
    packed_log = pack_payload(log_rec)
    unpacked_log = unpack_payload(packed_log, RemoteLogRecord)
    assert isinstance(unpacked_log, RemoteLogRecord)
    assert unpacked_log.name == "test.logger"
    assert unpacked_log.levelno == 30
    assert unpacked_log.levelname == "WARNING"
    assert unpacked_log.msg == "A warning was emitted: %s"
    assert unpacked_log.args == ("some_val",)
    assert unpacked_log.filename == "test.py"
    assert unpacked_log.lineno == 42
    assert unpacked_log.funcName == "do_work"
    assert unpacked_log.created == 1700000000.0
    assert unpacked_log.exc_text == "Traceback ..."

    sub_req = StreamSubscribeRequest(replay_history=True)
    packed_sub = pack_payload(sub_req)
    unpacked_sub = unpack_payload(packed_sub, StreamSubscribeRequest)
    assert unpacked_sub.replay_history is True
