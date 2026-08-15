from __future__ import annotations

import pytest
import vapoursynth as vs

from vsremote.protocol import (
    ClipInfo,
    Command,
    FrameHeader,
    FrameRequest,
    LoadCodeRequest,
    LoadScriptRequest,
    OutputIndexRequest,
    PlaneInfo,
    ReloadRequest,
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
    raw = {
        "int": 10,
        "float": 3.14,
        "str": "test",
        "bytes": b"binary",
        "list": [1, 2, "three", b"bytes"],
    }
    clean = sanitize_props(raw)
    assert clean["list"] == [1, 2, "three", b"bytes"]

    # Verify MsgPack can pack and unpack it cleanly
    packed = pack_payload(clean)
    assert unpack_payload(packed)["bytes"] == b"binary"


def test_request_envelope_from_frames() -> None:
    # 1. Standard 4-frame request
    frames = [b"client-id-1", (42).to_bytes(4, "big"), bytes([Command.GET_FRAME.value]), pack_payload({"n": 5})]
    req = RequestEnvelope.from_frames(frames)
    assert req.identity == b"client-id-1"
    assert req.request_id == 42
    assert req.request_id_bytes == (42).to_bytes(4, "big")
    assert req.command == Command.GET_FRAME
    frame_req = unpack_payload(req.payload_bytes, FrameRequest)
    assert frame_req.n == 5
    assert req.extra_frames == []

    # 2. 3-frame request (e.g. PING)
    frames_ping = [b"client-id-2", (1).to_bytes(4, "big"), bytes([Command.PING.value])]
    req_ping = RequestEnvelope.from_frames(frames_ping)
    assert req_ping.identity == b"client-id-2"
    assert req_ping.request_id == 1
    assert req_ping.command == Command.PING
    assert req_ping.payload_bytes == b""
    assert req_ping.extra_frames == []

    # 3. Multipart with extra frames and auth token
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

    # 4. Malformed: less than 3 frames
    with pytest.raises(ValueError, match="Malformed multipart request"):
        RequestEnvelope.from_frames([b"id", b"req_id"])

    # 5. Invalid command byte
    with pytest.raises(ValueError, match="Unknown command byte"):
        RequestEnvelope.from_frames([b"id", (1).to_bytes(4, "big"), bytes([255])])


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

    # Empty byte buffer with untyped default
    assert unpack_payload(b"") == {}


def test_stream_events_serialization() -> None:
    # 1. StreamOutputEvent
    out_evt = StreamOutputEvent(stream="stdout", text="hello stdout")
    packed_out = pack_payload(out_evt)
    unpacked_out = unpack_payload(packed_out, StreamOutputEvent)
    assert isinstance(unpacked_out, StreamOutputEvent)
    assert unpacked_out.stream == "stdout"
    assert unpacked_out.text == "hello stdout"

    # 2. RemoteLogRecord
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

    # 3. StreamSubscribeRequest
    sub_req = StreamSubscribeRequest(replay_history=True)
    packed_sub = pack_payload(sub_req)
    unpacked_sub = unpack_payload(packed_sub, StreamSubscribeRequest)
    assert unpacked_sub.replay_history is True


def test_response_envelope_from_frames() -> None:
    # 1. Standard typed response
    p = PlaneInfo(width=1920, height=1080, bytes_per_sample=1, size_bytes=1920 * 1080)
    frames_ok = [bytes([StatusCode.OK]), pack_payload(p)]
    resp = ResponseEnvelope.from_frames(frames_ok, PlaneInfo)
    assert resp.is_ok is True
    assert resp.status == StatusCode.OK
    assert resp.payload == p
    assert resp.extra_frames == []

    # 2. Raw bytes payload mode (no target_type)
    frames_raw = [bytes([StatusCode.OK]), b"PONG"]
    resp_raw = ResponseEnvelope.from_frames(frames_raw)
    assert resp_raw.is_ok is True
    assert resp_raw.payload == b"PONG"
    assert resp_raw.payload_bytes == b"PONG"

    # 3. Response with extra frames (FrameHeader + planes)
    header = FrameHeader(
        status=StatusCode.OK,
        request_id=1,
        n=0,
        output_index=0,
        compression="none",
        plane_sizes=[10, 20],
    )
    frames_extra = [bytes([StatusCode.OK]), pack_payload(header), b"plane_0_bytes", b"plane_1_bytes"]
    resp_extra = ResponseEnvelope.from_frames(frames_extra, FrameHeader)
    assert resp_extra.is_ok is True
    assert resp_extra.payload == header
    assert resp_extra.extra_frames == [b"plane_0_bytes", b"plane_1_bytes"]

    # 4. Single-part response (status only)
    frames_single = [bytes([StatusCode.OK])]
    resp_single = ResponseEnvelope.from_frames(frames_single)
    assert resp_single.is_ok is True
    assert resp_single.payload == b""
    assert resp_single.extra_frames == []

    # 5. Error status response with fallback decoding
    frames_err = [bytes([StatusCode.NOT_FOUND]), pack_payload({"error": "Output not found"})]
    resp_err = ResponseEnvelope.from_frames(frames_err, PlaneInfo)
    assert resp_err.is_ok is False
    assert resp_err.status == StatusCode.NOT_FOUND

    # 6. Malformed response: empty list
    with pytest.raises(ValueError, match="Malformed multipart response"):
        ResponseEnvelope.from_frames([])

    # 7. Invalid status byte
    with pytest.raises(ValueError, match="Unknown status code byte"):
        ResponseEnvelope.from_frames([bytes([255])])


def test_reload_and_load_requests_serialization() -> None:
    # 1. ReloadRequest
    rel_req = ReloadRequest(chdir=False)
    packed_rel = pack_payload(rel_req)
    unpacked_rel = unpack_payload(packed_rel, ReloadRequest)
    assert unpacked_rel.chdir is False

    # 2. LoadCodeRequest
    code_req = LoadCodeRequest(code="import vapoursynth as vs", filename="test.py")
    packed_code = pack_payload(code_req)
    unpacked_code = unpack_payload(packed_code, LoadCodeRequest)
    assert unpacked_code.code == "import vapoursynth as vs"
    assert unpacked_code.filename == "test.py"

    # 3. LoadScriptRequest
    script_req = LoadScriptRequest(script_path="/path/to/script.vpy", chdir=True)
    packed_script = pack_payload(script_req)
    unpacked_script = unpack_payload(packed_script, LoadScriptRequest)
    assert unpacked_script.script_path == "/path/to/script.vpy"
    assert unpacked_script.chdir is True
