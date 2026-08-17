from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, override

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from mypyc.build import mypycify
from setuptools import Distribution
from setuptools.command.build_ext import build_ext


class CustomBuildHook(BuildHookInterface[Any]):
    PLUGIN_NAME = "custom"

    @override
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable" or os.environ.get("VSREMOTE_BUILD_PURE_PYTHON") == "1":
            return

        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        root = Path(self.root)
        source_file = root / "src" / "vsremote" / "_strides.py"

        if not source_file.is_file():
            raise RuntimeError(f"File {source_file} doesn't exist or isn't a file.")

        with tempfile.TemporaryDirectory(prefix="mypyc_build_", ignore_cleanup_errors=True) as tmp_dir:
            tmp_source = os.path.join(tmp_dir, "_strides.py")
            shutil.copy(source_file, tmp_source)

            ext_modules = mypycify([tmp_source], target_dir=tmp_dir, opt_level="3")
            for ext in ext_modules:
                ext.name = "vsremote._strides"

            dist = Distribution({"name": "vsremote", "ext_modules": ext_modules})
            cmd = build_ext(dist)
            cmd.inplace = False
            cmd.build_lib = os.path.join(self.root, "src")
            cmd.ensure_finalized()
            cmd.run()
