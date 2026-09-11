"""What `import plugrl_env_client.envs` does to the process.

That import is not neutral: `_load_env_modules()` imports every `*_env.py`
under the package so their registration decorators run, and it downgrades an
ImportError to a warning so a missing extra does not stop the others. The
consequence is easy to miss - a module that fails to import can still have
run its first few lines, and anything it did to global state stays done.

`robomimic_env.py` set `MUJOCO_GL=egl` on line 3 and failed to import on
line 10. On Linux that was invisible. On Windows `egl` is not a legal value,
so `mujoco` raised `RuntimeError: invalid value for environment variable
MUJOCO_GL: egl` and every other MuJoCo-based environment in the process died
- because of a family that was not even installed.

These run in a subprocess: by the time a normal test runs, the package has
already been imported and the side effect has already happened.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

VALID_BY_PLATFORM = {
    "win32": {"wgl", "glfw", "osmesa"},
    "darwin": {"cgl", "glfw", "osmesa"},
    "linux": {"egl", "glfw", "osmesa"},
}


def _import_envs_and_report(expression):
    """Import the env package in a clean subprocess, then evaluate expression."""
    code = f"import plugrl_env_client.envs; print(repr({expression}))"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "MUJOCO_GL"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_importing_the_package_does_not_choose_a_gl_backend_for_the_caller():
    """An env family that is not installed must not set MUJOCO_GL.

    robomimic legitimately needs a backend chosen before robosuite is
    imported, so when robomimic IS installed a value here is correct. What is
    not correct - and what used to happen - is setting one on a machine that
    has no robomimic at all, purely because the package import walks every
    `*_env.py` and swallows the resulting ImportError.
    """
    if importlib.util.find_spec("robomimic") is not None:
        pytest.skip("robomimic is installed, so setting a backend is correct")

    value = _import_envs_and_report("__import__('os').environ.get('MUJOCO_GL')")

    assert value == "None", (
        f"importing plugrl_env_client.envs set MUJOCO_GL={value} on a machine "
        "with no robomimic; an env module that is not installed must not pick "
        "a rendering backend for the whole process"
    )


def test_if_a_gl_backend_is_ever_set_it_is_legal_on_this_platform():
    """The narrower guarantee, in case the rule above is ever relaxed.

    Setting a default is defensible. Setting one that the platform rejects
    is not, and is what actually broke things.
    """
    value = _import_envs_and_report("__import__('os').environ.get('MUJOCO_GL')")
    if value == "None":
        pytest.skip("nothing was set, which the previous test already asserts")

    platform = next(
        (p for p in VALID_BY_PLATFORM if sys.platform.startswith(p)), "linux"
    )
    assert value.strip("'\"") in VALID_BY_PLATFORM[platform]


def test_a_caller_who_chose_a_backend_keeps_it():
    """`os.environ[...] = ` would take this away; `setdefault` does not."""
    code = (
        "import os; os.environ['MUJOCO_GL'] = 'glfw';"
        "import plugrl_env_client.envs;"
        "print(os.environ['MUJOCO_GL'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "glfw"
