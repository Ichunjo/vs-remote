from __future__ import annotations

import os

import pytest
import vapoursynth as vs

from vsremote.protocol.codec import compress_plane, decompress_plane

core = vs.core


def test_zstd_compression_roundtrip() -> None:
    # 64 KB of repeating + random data
    original = (b"VapourSynthRemoteFrameData1234567890" * 2000)[:65536]
    assert len(original) == 65536

    compressed = compress_plane(original, "zstd")
    assert len(compressed) < len(original)

    decompressed = decompress_plane(compressed, len(original), "zstd")
    assert decompressed == original


def test_none_compression_roundtrip() -> None:
    original = os.urandom(4096)
    compressed = compress_plane(original, "none")
    assert compressed == original

    decompressed = decompress_plane(compressed, len(original), "none")
    assert decompressed == original


def test_memoryview_compression() -> None:
    original = bytearray(b"0123456789" * 500)
    mv = memoryview(original)

    compressed = compress_plane(mv, "zstd")
    decompressed = decompress_plane(compressed, len(original), "zstd")

    assert decompressed == bytes(original)


@pytest.mark.vpy("initial-core")
def test_non_contiguous_plane_compression() -> None:
    # A clip with non-aligned width (e.g. 130) produces strided 2D memoryview where stride (192) != width (130)
    clip = core.std.BlankClip(width=130, height=100, format=vs.YUV420P8)
    frame = clip.get_frame(0)
    p0 = frame[0]
    assert not p0.c_contiguous

    compressed = compress_plane(p0, "zstd")
    decompressed = decompress_plane(compressed, p0.nbytes, "zstd")
    assert decompressed == p0.tobytes()

    uncompressed = compress_plane(p0, "none")
    assert uncompressed == p0.tobytes()
