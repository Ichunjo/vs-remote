from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, overload

import msgspec
import zstandard as zstd
from typing_extensions import TypeForm

from .constants import Compression

_ZSTD_LOCAL = threading.local()


def _get_zstd_compressor() -> zstd.ZstdCompressor:
    compressor = getattr(_ZSTD_LOCAL, "compressor", None)
    if compressor is None:
        compressor = zstd.ZstdCompressor(level=-1, threads=0)
        _ZSTD_LOCAL.compressor = compressor
    return compressor


def _get_zstd_decompressor() -> zstd.ZstdDecompressor:
    decompressor = getattr(_ZSTD_LOCAL, "decompressor", None)
    if decompressor is None:
        decompressor = zstd.ZstdDecompressor()
        _ZSTD_LOCAL.decompressor = decompressor
    return decompressor


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


def pack_payload(data: Any) -> bytes:
    """Serialize a Python dict, primitive, or dataclass to MsgPack bytes."""
    return msgspec.msgpack.encode(data)


@overload
def unpack_payload[T](data: bytes, target_type: TypeForm[T]) -> T: ...
@overload
def unpack_payload(data: bytes, target_type: None = None) -> Any: ...
def unpack_payload(data: bytes, target_type: Any = None) -> Any:
    """
    Deserialize MsgPack bytes into Python objects or the specified target type.

    Args:
        data: MsgPack encoded byte buffer.
        target_type: Optional target struct or type to decode into.

    Returns:
        The decoded Python object or target instance.
    """
    if not data:
        if target_type is None:
            return {}
        try:
            return target_type()
        except TypeError:
            return msgspec.msgpack.decode(b"\x80", type=target_type)

    return msgspec.msgpack.decode(data) if target_type is None else msgspec.msgpack.decode(data, type=target_type)


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
            return _get_zstd_compressor().compress(data)
        case "none":
            return bytes(data) if isinstance(data, memoryview) else data


def decompress_plane(data: bytes, uncompressed_size: int, mode: Compression) -> bytes:
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
            return _get_zstd_decompressor().decompress(data, max_output_size=uncompressed_size)
        case "none":
            return data
