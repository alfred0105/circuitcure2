"""Local executable capability checks used to enable dashboard actions."""

from __future__ import annotations

import shutil


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None
