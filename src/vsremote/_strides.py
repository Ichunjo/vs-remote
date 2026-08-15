from __future__ import annotations

import ctypes

from mypy_extensions import i32, i64

_PyBytes_AsString = ctypes.pythonapi.PyBytes_AsString
_PyBytes_AsString.argtypes = [ctypes.py_object]
_PyBytes_AsString.restype = ctypes.c_void_p

_memmove = ctypes.memmove


def copy_plane_strided(dst_addr: i64, decompressed: bytes, width: i32, height: i32, bps: i32, stride: i64) -> None:
    """
    Copy uncompressed planar bytes to a strided frame buffer line by line.

    Args:
        dst_addr: Base memory address of destination frame plane.
        decompressed: Raw planar byte string.
        width: Plane width in pixels.
        height: Plane height in lines.
        bps: Number of bytes per sample (e.g. 1 for 8-bit, 2 for 16-bit, 4 for float).
        stride: Destination plane stride in bytes.
    """
    row_size: i64 = i64(width) * i64(bps)
    src: i64 = i64(_PyBytes_AsString(decompressed))
    dst: i64 = dst_addr
    h: i64 = i64(height)

    for _ in range(h):
        _memmove(dst, src, row_size)
        dst += stride
        src += row_size
