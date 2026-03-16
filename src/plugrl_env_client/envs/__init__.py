from importlib import import_module
from pathlib import Path

from loguru import logger

from .dummy_env import DummyEnv

__all__ = ["DummyEnv"]


def _iter_env_module_names():
    """Yield fully-qualified module names for *_env.py files under this package."""
    env_dir = Path(__file__).resolve().parent
    prefix = f"{__name__}."
    for py_file in env_dir.rglob("*_env.py"):
        rel_parts = py_file.relative_to(env_dir).with_suffix("").parts
        module_name = prefix + ".".join(rel_parts)
        yield module_name


def _load_env_modules() -> None:
    """Import env modules so their registration side effects run."""
    for module_name in _iter_env_module_names():
        try:
            import_module(module_name)
        except ImportError as exc:
            logger.warning(f"Skip loading env module {module_name}: {exc}")


_load_env_modules()
