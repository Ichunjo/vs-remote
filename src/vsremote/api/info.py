import sys


def is_preview() -> bool:
    """Check if the current script is running in a preview environment (VSRemote only)."""
    return bool(sys.modules.get("__vsremote__"))
