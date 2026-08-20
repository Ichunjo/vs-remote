from __future__ import annotations

import os
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from logging import getLogger
from pathlib import Path
from typing import Self

import vapoursynth as vs
from vsengine.policy import ContextVarStore, ManagedEnvironment, Policy
from vsengine.vpy import ExecutionError, Script, load_code, load_script

from ..api.output import _output_metadata
from ..exceptions import EnvironmentNotSetError, OutputNotFoundError, ScriptNotLoadedError
from ..protocol import ClipInfo, OutputItem, StreamEvent
from ..utils import gc_collect
from .policy import RemotePolicy
from .redirect import capture_streams

logger = getLogger(__name__)


class ScriptRunner:
    """Manages VapourSynth script execution, dynamic reload, and output caching."""

    def __init__(self, environment: Policy | ManagedEnvironment | vs.Environment | None = None) -> None:
        self._rlock = threading.RLock()
        self._script: Script[ManagedEnvironment] | Script[vs.Environment] | None = None
        self._environment: vs.Environment | ManagedEnvironment | None = None
        self._policy: Policy | None = None
        self._registered_policy = False
        self._script_path: Path | None = None
        self._chdir: bool = True
        self._clips = dict[int, vs.VideoNode]()
        self._clip_infos = dict[int, ClipInfo]()
        self._output_items = list[OutputItem]()
        self._startup_events = deque[StreamEvent](maxlen=1000)

        self._ensure_policy(environment)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()

    @property
    def startup_events(self) -> list[StreamEvent]:
        """Get copy of startup log records and stream events captured during script load."""
        with self._rlock:
            return list(self._startup_events)

    @property
    def script_path(self) -> Path | None:
        """Get the active script path, if loaded from a file."""
        return self._script_path

    @property
    def environment(self) -> vs.Environment | ManagedEnvironment:
        """Get the active VapourSynth environment for this runner, if any."""
        with self._rlock:
            if not self._environment:
                raise EnvironmentNotSetError("No environment has been passed to this ScriptRunner instance")
            return self._environment

    def load_script(
        self,
        script_path: str | os.PathLike[str],
        *,
        environment: Policy | ManagedEnvironment | None = None,
        chdir: bool = True,
    ) -> list[OutputItem]:
        """
        Load and execute a .vpy script file within an isolated environment.

        Args:
            script_path: Path to the script file.
            environment: Optional Policy or ManagedEnvironment.
            chdir: Whether to change directory to the script's parent while loading.

        Returns:
            List of OutputItem describing available video outputs.
        """
        with self._rlock:
            path = Path(script_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Script not found: {path}")

            self._teardown_environment()
            self._script_path = path
            self._chdir = chdir

            env = self._ensure_policy(environment)
            working_dir = path.parent if chdir else None
            logger.info("Loading script: %s (cwd: %s)", path, working_dir)

            with capture_streams(self._startup_events.append):
                self._script = load_script(path, env, module="__vsremote__", chdir=working_dir)
                self._environment = self._script.environment
                try:
                    self._script.result()
                except ExecutionError as error:
                    self._script.dispose()
                    raise error from None

            return self._extract_outputs()

    def load_code(
        self,
        code: str,
        *,
        filename: str | None = None,
        chdir: str | os.PathLike[str] | None = None,
        environment: Policy | ManagedEnvironment | None = None,
    ) -> list[OutputItem]:
        """
        Execute raw Python / VapourSynth code string within an isolated environment.

        Args:
            code: Python code string.
            filename: Virtual filename for traceback reporting.
            chdir: Optional working directory while executing the code.
            environment: Optional Policy or ManagedEnvironment.

        Returns:
            List of OutputItem describing available video outputs.
        """
        with self._rlock:
            self._teardown_environment()
            self._script_path = None

            env = self._ensure_policy(environment)
            fn = filename or "<remote_code>"
            logger.info("Loading code string (filename: %s, cwd: %s)", fn, chdir)

            with capture_streams(self._startup_events.append):
                self._script = load_code(code, env, module="__vsremote__", filename=fn, chdir=chdir)
                self._environment = self._script.environment
                try:
                    self._script.result()
                except ExecutionError as error:
                    self._script.dispose()
                    raise error from None

            return self._extract_outputs()

    def reload(self, *, chdir: bool | None = None) -> list[OutputItem]:
        """
        Reload the currently loaded script file from disk.

        Args:
            chdir: Optional override for changing working directory during reload.

        Returns:
            List of OutputItem describing available video outputs.
        """
        with self._rlock:
            if self._script_path is None:
                raise ScriptNotLoadedError("No script file is associated with this ScriptRunner instance")

            c_dir = self._chdir if chdir is None else chdir
            return self.load_script(self._script_path, chdir=c_dir)

    def get_clip(self, index: int) -> vs.VideoNode:
        """Retrieve the VideoNode at the given output index."""
        with self._rlock:
            if index not in self._clips:
                raise OutputNotFoundError(f"Output index {index} not found. Available: {list(self._clips.keys())}")
            return self._clips[index]

    def get_clip_info(self, index: int) -> ClipInfo:
        """Retrieve static metadata for the VideoNode at index."""
        with self._rlock:
            if index not in self._clip_infos:
                raise OutputNotFoundError(f"Output index {index} not found. Available: {list(self._clip_infos.keys())}")
            return self._clip_infos[index]

    def list_outputs(self) -> list[OutputItem]:
        """List all available outputs."""
        with self._rlock:
            return list(self._output_items)

    def close(self) -> None:
        """Clean up the environment and release VapourSynth resources."""
        with self._rlock:
            self._teardown_environment()

            if self._policy and self._registered_policy:
                if self._policy.is_registered:
                    self._policy.unregister()
                self._policy = None
                self._registered_policy = False

    @classmethod
    def from_script(
        cls,
        script_path: str | os.PathLike[str],
        *,
        environment: Policy | ManagedEnvironment | None = None,
        chdir: bool = True,
    ) -> ScriptRunner:
        """
        Create a runner from a script file path.

        Args:
            script_path: Path to the script file.
            environment: Optional Policy or ManagedEnvironment.
            chdir: Whether to change directory to the script's parent while loading.
        """
        self = cls(environment=environment)
        self.load_script(script_path, environment=environment, chdir=chdir)
        return self

    @classmethod
    def from_clips(
        cls,
        clips: Mapping[int, vs.VideoNode] | Sequence[vs.VideoNode],
        *,
        environment: vs.Environment | ManagedEnvironment | None = None,
    ) -> ScriptRunner:
        """Create a runner from existing VideoNodes (useful for tests/embedded usage)."""
        self = cls(environment=environment)
        with self._rlock:
            self._environment = environment
            if self._environment is None and vs.has_policy():
                self._environment = vs.get_current_environment()

            if isinstance(clips, Sequence):
                clips = dict(enumerate(clips))

            for idx, clip in clips.items():
                info = ClipInfo.from_clip(clip, name=f"Output {idx}")
                self._clips[idx] = clip
                self._clip_infos[idx] = info
                self._output_items.append(OutputItem(index=idx, name=info.name, info=info))

        return self

    def _ensure_policy(
        self, environment: Policy | ManagedEnvironment | vs.Environment | None = None
    ) -> Policy | ManagedEnvironment | vs.Environment | None:
        if environment is not None:
            if isinstance(environment, Policy):
                self._policy = environment
            return environment

        if self._policy is not None:
            return self._policy

        if not vs.has_policy():
            self._policy = RemotePolicy(store=ContextVarStore())
            self._policy.register()
            self._registered_policy = True
            return self._policy

        return None

    def _teardown_environment(self) -> None:
        self._clips.clear()
        self._clip_infos.clear()
        self._output_items.clear()
        self._startup_events.clear()

        if self._script:
            self._script.dispose()
            self._script = None

        if isinstance(self._environment, ManagedEnvironment):
            self._environment.dispose()

        self._environment = None
        gc_collect()

    def _extract_outputs(self) -> list[OutputItem]:
        if not self._script:
            raise ScriptNotLoadedError("Script doesn't exist")

        with self._script.environment.use():
            outputs = vs.get_outputs()

        for idx, output in outputs.items():
            if isinstance(output, vs.VideoOutputTuple):
                clip = output.clip
                name = _output_metadata.get(idx) or f"Output {idx}"
            else:
                continue

            info = ClipInfo.from_clip(clip, name=name)
            self._clips[idx] = clip
            self._clip_infos[idx] = info
            self._output_items.append(OutputItem(index=idx, name=name, info=info))

        logger.info("Found %d video output(s)", len(self._clips))
        return list(self._output_items)
