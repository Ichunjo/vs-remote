from __future__ import annotations

from collections.abc import Sequence

import zmq
import zmq.utils.z85


def z85_encode(key: bytes) -> bytes:
    """Encode a 32-byte binary key into 40-byte Z85 ascii bytes."""
    return zmq.utils.z85.encode(key)  # type: ignore[no-untyped-call]


def z85_decode(key: str | bytes) -> bytes:
    """Decode a 40-byte or 40-char Z85 key into 32-byte raw binary bytes."""
    raw = key.encode("ascii") if isinstance(key, str) else key
    return zmq.utils.z85.decode(raw)  # type: ignore[no-untyped-call]


def validate_curve_key(curve_key: str | bytes | None, name: str = "curve_key") -> bytes | None:
    """
    Validate and normalize a CurveZMQ public or secret key.

    Returns:
        The normalized key as bytes (either 40-byte ASCII Z85 or 32-byte binary), or None.

    Raises:
        ValueError: If curve_key length does not match 40 Z85 characters or 32/40 bytes.
    """
    if (isinstance(curve_key, str) and len(curve_key) != 40) or (
        isinstance(curve_key, bytes) and len(curve_key) not in (32, 40)
    ):
        raise ValueError(f"Invalid {name} length. Expected 40 Z85 characters or 32 or 40 bytes")

    return curve_key.encode("ascii") if isinstance(curve_key, str) else curve_key


def validate_curve_allowed_keys(curve_allowed_keys: Sequence[str | bytes] | None) -> set[bytes]:
    """
    Validate and decode a sequence of authorized CurveZMQ client public keys into 32-byte binary keys.

    Returns:
        Set of 32-byte binary public keys.

    Raises:
        ValueError: If key length or Z85 encoding is invalid.
    """
    keys = set[bytes]()

    for item in curve_allowed_keys or []:
        if (isinstance(item, bytes) and len(item) not in (40, 32)) or (isinstance(item, str) and len(item) != 40):
            raise ValueError(
                f"Invalid Curve key length ({len(item)} bytes). Expected 40 Z85 characters or 32 or 40 bytes."
            )

        raw = item.encode("ascii") if isinstance(item, str) else item
        if len(raw) == 40:
            try:
                decoded = z85_decode(raw)
            except Exception as exc:
                raise ValueError(f"Invalid Z85 Curve key: {item!r}") from exc
            keys.add(decoded)
        else:
            keys.add(raw)

    return keys
