"""
Output registration API for vsremote.
"""

from __future__ import annotations

import sys
from logging import getLogger
from typing import Any, overload

import vapoursynth as vs

_logger = getLogger(__name__)
_output_metadata = dict[int, str]()


@overload
def set_output(node: vs.RawNode, index: int = ..., /) -> None: ...
@overload
def set_output(node: vs.RawNode, name: str | bool | None = ..., /) -> None: ...
@overload
def set_output(node: vs.RawNode, index: int = ..., name: str | bool | None = ..., /) -> None: ...
def set_output(node: vs.RawNode, index_or_name: int | str | bool | None = None, name: str | bool | None = None) -> None:
    """
    Register one or more VapourSynth nodes as outputs for preview.

    If no index is provided, outputs are assigned to the next available indices.

    Args:
        node: A VideoNode, AudioNode, or iterable of nodes to output.
        index_or_name: Either:

               - An int specifying output index
               - A str to use as the output name
               - True/None to auto-detect the variable name
               - False to disable name detection

        name: Explicit name override. If provided when index_or_name is an int,
            this sets the display name for the output.
        **kwargs: Additional metadata for custom configuration of this output.
    """
    if isinstance(index_or_name, (str, bool)):
        index = None
        name = index_or_name
    else:
        index = index_or_name

    outputs = vs.get_outputs()
    index = index if index is not None else max(outputs, default=-1) + 1

    if index in outputs:
        _logger.warning("Output index %d already in use; overwriting.", index)

    node.set_output(index)

    if not sys.modules.get("__vsremote__"):
        return

    effective_name: str | None

    match name:
        case True | None:
            effective_name = _resolve_var_name(node, frame_depth=2)
        case False:
            effective_name = None
        case str():
            effective_name = name

    if not effective_name:
        match node:
            case vs.VideoNode():
                title = "Clip"
            case vs.AudioNode():
                title = "Audio"
            case _:
                raise NotImplementedError
        effective_name = f"{title} {index}"

    _output_metadata[index] = effective_name


def _resolve_var_name(obj: Any, *, frame_depth: int = 1) -> str | None:
    try:
        frame = sys._getframe(frame_depth)
    except ValueError:
        return None

    try:
        obj_id = id(obj)

        for var_name, value in reversed(list(frame.f_locals.items())):
            if id(value) == obj_id:
                return var_name

        return None
    finally:
        del frame
