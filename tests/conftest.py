"""
conftest.py — Import setup for the standalone memory_module package.

The standalone module lives in a directory with spaces/hyphens in its name,
so we use importlib to register it as a valid Python package at import time.
"""
import sys
import types
import importlib.util
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configure embedding backend BEFORE importing memory_module.
# Use llama-cpp-python with nomic-embed-text GGUF (avoids PyTorch/sentence-
# transformers segfaults when many agents are instantiated in tests).
# ---------------------------------------------------------------------------
_NOMIC_GGUF = os.environ.get(
    "MUNINN_EMBEDDING_MODEL",
    r"M:\Dev\projects\models\nomic-embed-text-v1.5.Q8_0.gguf",
)
if os.path.isfile(_NOMIC_GGUF):
    os.environ["MUNINN_EMBEDDING_MODEL"] = _NOMIC_GGUF

# ---------------------------------------------------------------------------
# Locate the standalone package root (one level up from tests/)
# ---------------------------------------------------------------------------
_STANDALONE_ROOT = str(Path(__file__).resolve().parent.parent)
_PKG_NAME = "memory_module"


def _ensure_standalone_package():
    """Register the standalone directory as an importable package."""
    if _PKG_NAME in sys.modules:
        return sys.modules[_PKG_NAME]

    # Create a stub package module
    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [_STANDALONE_ROOT]
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg

    # Load __init__.py as the package itself
    init_path = os.path.join(_STANDALONE_ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        _PKG_NAME,
        init_path,
        submodule_search_locations=[_STANDALONE_ROOT],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# Execute on import so the package is available to all test modules
_ensure_standalone_package()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_agent():
    """Fresh MemoryAgent with an in-memory database and small STM window."""
    from memory_module import MemoryAgent
    agent = MemoryAgent(":memory:", max_stm_segments=5)
    return agent


@pytest.fixture
def db():
    """Standalone in-memory Database instance."""
    from memory_module.db import Database
    return Database(":memory:")


@pytest.fixture
def stm(db):
    """STMManager wired to an in-memory database."""
    from memory_module.stm import STMManager
    return STMManager(db, max_segments=5)


@pytest.fixture
def ltm(db):
    """LTMManager wired to an in-memory database."""
    from memory_module.ltm import LTMManager
    return LTMManager(db)


@pytest.fixture
def entities(db):
    """EntityManager wired to an in-memory database."""
    from memory_module.entities import EntityManager
    return EntityManager(db)


@pytest.fixture
def sources(db):
    """SourceManager wired to an in-memory database."""
    from memory_module.sources import SourceManager
    return SourceManager(db)


@pytest.fixture
def recall_engine(db, ltm, entities, sources):
    """RecallEngine wired to an in-memory database with all sub-managers."""
    from memory_module.recall import RecallEngine
    return RecallEngine(db, ltm, entities, source_mgr=sources)


@pytest.fixture
def forgetting(db, ltm):
    """ForgettingEngine wired to an in-memory database."""
    from memory_module.forgetting import ForgettingEngine
    return ForgettingEngine(db, ltm)
