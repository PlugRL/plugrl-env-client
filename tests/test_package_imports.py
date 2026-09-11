"""Import-time contracts for the package.

Two invariants are checked here:

1. Every module that is not behind an optional extra imports on a bare install.
   A module that quietly depends on a package missing from pyproject.toml only
   fails once someone imports it, far from where the mistake was made.

2. Every module that *is* behind an optional extra fails with a message naming
   that extra. A raw ModuleNotFoundError leaves the user guessing which extra
   they were supposed to install.
"""

import importlib
import pkgutil

import pytest

import plugrl_env_client

# Modules guarded by an optional extra, mapped to the extra that provides them.
# The value is the string the error message must mention.
OPTIONAL_ENV_MODULES = {
    "plugrl_env_client.envs.atari.atari_env": "atari",
    "plugrl_env_client.envs.atari.atari_wrappers": "atari",
    "plugrl_env_client.envs.classic.classic_env": "classic",
    "plugrl_env_client.envs.d4rl.d4rl_env": "d4rl",
    "plugrl_env_client.envs.libero.libero_env": "libero",
    "plugrl_env_client.envs.robomimic.robomimic_env": "robomimic",
}


def _all_module_names():
    return sorted(
        module_info.name
        for module_info in pkgutil.walk_packages(
            plugrl_env_client.__path__, prefix="plugrl_env_client."
        )
    )


def _core_module_names():
    return [name for name in _all_module_names() if name not in OPTIONAL_ENV_MODULES]


@pytest.mark.parametrize("module_name", _core_module_names())
def test_core_modules_import_on_a_bare_install(module_name):
    module = importlib.import_module(module_name)

    # The import not raising is the whole test, but saying so out loud keeps
    # it from reading as an empty test body that someone later deletes.
    assert module.__name__ == module_name


@pytest.mark.parametrize("module_name,extra", sorted(OPTIONAL_ENV_MODULES.items()))
def test_optional_modules_name_their_extra_when_missing(module_name, extra):
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        assert extra in str(exc), (
            f"{module_name} failed to import without naming the '{extra}' extra. "
            f"An unguarded top-level import is the usual cause. Got: {exc}"
        )
    else:
        pytest.skip(f"the '{extra}' extra is installed, nothing to assert")
