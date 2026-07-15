"""
_bootstrap.py — Synthetic package registration for the standalone module.

The standalone module files (agent.py, tools.py, etc.) live in a directory
with spaces/hyphens in its name and use relative imports (from .agent import ...).
These relative imports only work when the module is loaded as part of a package.

This module creates a synthetic ``memory_module`` package using importlib,
exactly matching what conftest.py does for the test suite.

Usage in any demo script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from _bootstrap import ensure_memory_module
    ensure_memory_module()
    from memory_module import MemoryAgent   # now works
"""

import os
import sys
import types
import importlib.util
from pathlib import Path


_PKG_NAME = "memory_module"


def ensure_memory_module(standalone_root: str | Path | None = None):
    """
    Register the standalone directory as the ``memory_module`` package.

    Call this once at the top of any script that imports from memory_module
    before any ``from memory_module import ...`` statements.

    Parameters
    ----------
    standalone_root : str or Path, optional
        Path to the standalone module's root directory.
        If None, uses the caller's parent directory (assumes _bootstrap.py
        is in the same parent or a sibling of the standalone root).
    """
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]

    if standalone_root is None:
        # Default: _bootstrap.py lives in a subdirectory of the standalone root
        standalone_root = Path(__file__).resolve().parent.parent

    root = str(standalone_root)

    # Ensure root is on sys.path
    if root not in sys.path:
        sys.path.insert(0, root)

    # Create synthetic package
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [root]
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg

    # Load __init__.py
    init_path = os.path.join(root, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        init_path,
        submodule_search_locations=[root],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = mod
    spec.loader.exec_module(mod)
    return mod
