"""Every env-client command in the README must survive its own parser.

An adversarial audit found flags documented on the project's site that do not
exist at all - `--num-workers` for what is really `--num-procs`,
`--use-real-time` and `--fps` for nothing, and `--use-env-lock` for
`--runner.use-env-lock`. They had been there since the pages were written,
because documentation is not executed.

This executes the README's commands, as far as the parser. Parsing is a floor
rather than a guarantee: a flag can parse and still be ignored. It is the floor
that was missing.

Environments behind an optional extra are reported as skipped, decided from the
registry rather than from the error text, so a genuine parse failure can never
be mistaken for a missing extra.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import sys

import pytest

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:bash|sh|console)\n(.*?)```", text, re.S)


def _commands(text: str) -> list[str]:
    out: list[str] = []
    for block in _bash_blocks(text):
        joined = re.sub(r"\\s*\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # The README writes invocations three ways, and one block is a
            # usage sketch with a placeholder rather than a command.
            if not re.match(
                r"^(uv run )?(plugrl-run-env-client|python -m plugrl_env_client\.cli)",
                line,
            ):
                continue
            if "<" in line or ">" in line or "[OPTIONS]" in line:
                continue
            out.append(line)
    return out


def _args(command: str) -> list[str]:
    parts = shlex.split(command)
    if parts[0] == "uv":  # uv run plugrl-run-env-client ...
        parts = parts[2:]
    if parts[0] == "python":  # python -m plugrl_env_client.cli ...
        return parts[3:]
    return parts[1:]


COMMANDS = _commands(README.read_text(encoding="utf-8"))


def test_the_readme_contains_commands_to_check():
    """A regex that silently matched nothing would make every test below pass."""
    assert len(COMMANDS) >= 3, f"only found {len(COMMANDS)} commands in {README}"


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: c[:60])
def test_documented_command_parses(command, monkeypatch, capsys):
    args = _args(command)
    if "--help" in args:
        pytest.skip("--help exits by design")

    import plugrl_env_client.envs  # noqa: F401 - registers the env modules
    from plugrl_env_client.cli import cli
    from plugrl_env_client.utils.registration import REGISTERED_ENV_CONFIGS

    # tyro renders the subcommand from the registry key, lowercased: the
    # registry holds "MuJoCo-v1" and the README writes "mujoco-v1".
    registered = {k.lower() for k in REGISTERED_ENV_CONFIGS}
    if args and not args[0].startswith("-") and args[0].lower() not in registered:
        pytest.skip(f"env '{args[0]}' needs an extra this install lacks")

    monkeypatch.setattr(sys, "argv", ["prog", *args])
    try:
        cli()
    except SystemExit as exc:
        out = capsys.readouterr()
        tail = (out.out + out.err)[-1500:]
        pytest.fail(
            f"README command failed to parse (exit {exc.code}):\n  {command}\n\n{tail}"
        )
