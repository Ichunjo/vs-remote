from __future__ import annotations

from logging import DEBUG, ERROR, FATAL, INFO, WARNING, getLogger
from typing import override

from vapoursynth import CoreCreationFlags, MessageType
from vsengine.policy import ContextVarStore, EnvironmentStore, ManagedEnvironment, Policy

logger = getLogger(__name__)
vslogger = getLogger("vapoursynth")

VS_LOG_LEVEL_MAP: dict[MessageType, int] = {
    MessageType.MESSAGE_TYPE_DEBUG: DEBUG,
    MessageType.MESSAGE_TYPE_INFORMATION: INFO,
    MessageType.MESSAGE_TYPE_WARNING: WARNING,
    MessageType.MESSAGE_TYPE_CRITICAL: ERROR,
    MessageType.MESSAGE_TYPE_FATAL: FATAL,
}


class RemotePolicy(Policy):
    def __init__(
        self,
        store: EnvironmentStore | None = None,
        flags_creation: CoreCreationFlags | int | None = None,
    ) -> None:
        super().__init__(store or ContextVarStore(), flags_creation)

    @override
    def new_environment(self, flags_creation: int | None = None) -> ManagedEnvironment:
        logger.debug("Creating new VapourSynth environment in RemotePolicy")
        data = self.api.create_environment(flags_creation if flags_creation is not None else self.flags_creation)

        self.api.set_logger(
            data,
            lambda mt, msg: vslogger.log(
                VS_LOG_LEVEL_MAP[MessageType(mt)],
                msg,
                exc_info=mt >= MessageType.MESSAGE_TYPE_CRITICAL,
                stacklevel=2,
            ),
        )

        env = self.api.wrap_environment(data)
        return ManagedEnvironment(env, data, self)
