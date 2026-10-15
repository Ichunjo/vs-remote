from __future__ import annotations

import sys
import threading
from collections.abc import Mapping
from typing import Any

import msgspec
from typing_extensions import TypeForm

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

from .constants import Compression

_ZSTD_LOCAL = threading.local()


def _get_zstd_compressor() -> zstd.ZstdCompressor:
    compressor = getattr(_ZSTD_LOCAL, "compressor", None)
    if compressor is None:
        compressor = zstd.ZstdCompressor(level=-1)
        _ZSTD_LOCAL.compressor = compressor
    return compressor


def sanitize_props(props: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict[str, Any]()

    for key, value in props.items():
        match value:
            case int() | float() | str() | bytes():
                clean[key] = value
            case list() if all(isinstance(v, (int, float, str, bytes)) for v in value):
                clean[key] = value
            case _:
                clean[key] = repr(value)
    return clean


def unpack_payload[T](data: bytes, target_type: TypeForm[T]) -> T:
    """
    Deserialize MsgPack bytes into the specified target type.

    Args:
        data: MsgPack encoded byte buffer.
        target_type: Target struct or type to decode into.

    Returns:
        The decoded target instance.

    Raises:
        msgspec.DecodeError: If data is not valid MsgPack.
        msgspec.ValidationError: If decoded data does not conform to target_type.
    """
    return msgspec.msgpack.decode(data or b"\x80", type=target_type)


def compress_plane(data: bytes | memoryview, mode: Compression) -> bytes:
    """
    Compress planar image bytes using the specified compression mode.

    Args:
        data: Raw plane memory buffer.
        mode: The compression algorithm to use.

    Returns:
        Compressed plane bytes.
    """
    if isinstance(data, memoryview) and not data.c_contiguous:
        data = data.tobytes()

    match mode:
        case "zstd":
            return _get_zstd_compressor().compress(data, mode=zstd.ZstdCompressor.FLUSH_FRAME)
        case "none":
            return bytes(data) if isinstance(data, memoryview) else data


def decompress_plane(data: bytes, mode: Compression) -> bytes:
    """
    Decompress planar image bytes using the specified compression mode.

    Args:
        data: Compressed plane bytes.
        uncompressed_size: Expected uncompressed byte length of the plane.
        mode: The compression algorithm used.

    Returns:
        Raw uncompressed plane bytes.
    """
    match mode:
        case "zstd":
            return zstd.decompress(data)
        case "none":
            return data
