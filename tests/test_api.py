from __future__ import annotations

import logging
import sys
import types

import pytest
import vapoursynth as vs

from vsremote.api.info import is_preview
from vsremote.api.output import _output_metadata, _resolve_var_name, set_output

core = vs.core


def test_is_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "__vsremote__", raising=False)
    assert is_preview() is False

    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    assert is_preview() is True


@pytest.mark.vpy("initial-core")
def test_set_output_without_vsremote_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "__vsremote__", raising=False)
    _output_metadata.clear()
    vs.clear_outputs()

    clip = core.std.BlankClip(width=100, height=100)
    set_output(clip, 0)

    # set_output sets output on VapourSynth, but does not populate _output_metadata when not in __vsremote__
    outputs = vs.get_outputs()
    assert 0 in outputs
    assert 0 not in _output_metadata


@pytest.mark.vpy("initial-core")
def test_set_output_auto_name_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    my_test_clip = core.std.BlankClip(width=160, height=120)
    set_output(my_test_clip)

    outputs = vs.get_outputs()
    assert 0 in outputs
    assert _output_metadata.get(0) == "my_test_clip"


@pytest.mark.vpy("initial-core")
def test_set_output_explicit_name_and_index(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    clip_a = core.std.BlankClip(width=100, height=100)
    set_output(clip_a, 2, "Custom Track Name")

    assert _output_metadata.get(2) == "Custom Track Name"

    # Positional string name without explicit index
    clip_b = core.std.BlankClip(width=100, height=100)
    set_output(clip_b, "Positional Name Only")
    # Next available index after 2 should be 3
    assert _output_metadata.get(3) == "Positional Name Only"


@pytest.mark.vpy("initial-core")
def test_set_output_disabled_name_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    named_var_clip = core.std.BlankClip(width=100, height=100)
    # Pass False to disable name detection
    set_output(named_var_clip, False)

    assert _output_metadata.get(0) == "Clip 0"


@pytest.mark.vpy("initial-core")
def test_set_output_audio_node(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    audio_var = core.std.BlankAudio()
    set_output(audio_var, False)
    assert _output_metadata.get(0) == "Audio 0"


@pytest.mark.vpy("initial-core")
def test_set_output_overwrite_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    clip1 = core.std.BlankClip(width=100, height=100)
    clip2 = core.std.BlankClip(width=200, height=200)

    set_output(clip1, 0, "Clip One")
    with caplog.at_level(logging.WARNING):
        set_output(clip2, 0, "Clip Two")

    assert "Output index 0 already in use; overwriting." in caplog.text
    assert _output_metadata.get(0) == "Clip Two"


@pytest.mark.vpy("initial-core")
def test_set_output_unsupported_node_type(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_module = types.ModuleType("__vsremote__")
    monkeypatch.setitem(sys.modules, "__vsremote__", dummy_module)
    _output_metadata.clear()
    vs.clear_outputs()

    class FakeRawNode:
        def set_output(self, index: int) -> None: ...

    fake = FakeRawNode()
    with pytest.raises(NotImplementedError):
        set_output(fake, False)  # type: ignore[call-overload]


def test_resolve_var_name_edge_cases() -> None:
    assert _resolve_var_name("test_obj", frame_depth=99999) is None

    def helper() -> str | None:
        return _resolve_var_name(test_is_preview, frame_depth=1)

    assert helper() is None
