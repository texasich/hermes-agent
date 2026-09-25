"""Tests for the strict gateway command-line matcher.

Regression guard for the Windows ``hermes gateway restart`` silent-outage bug:
the previous loose substring match (``"... gateway" in cmdline``) false-matched
``gateway status``/``dashboard`` siblings and unrelated processes such as
``python -m tui_gateway``, which let ``restart()`` race a still-draining old
process and ``status``/``start`` report false positives.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from gateway.status import _record_looks_like_gateway
from gateway.status import (
    _command_line_belongs_to_profile,
    hermes_home_assignments,
    looks_like_gateway_command_line as matches,
    looks_like_gateway_runtime_command_line as matches_runtime,
    profile_flag_value,
)
from hermes_cli.venv_sync import relaunch_command


ACCEPT = [
    "pythonw.exe -m hermes_cli.main gateway run",
    r"C:\Users\me\hermes\venv\Scripts\pythonw.exe -m hermes_cli.main gateway run",
    "python -m hermes_cli.main --profile work gateway run",
    "python -m hermes_cli.main gateway run --replace",
    "python -m hermes_cli/main.py gateway run",
    "python gateway/run.py",
    "hermes-gateway.exe",
    "hermes gateway",          # bare `hermes gateway` defaults to run
    "hermes gateway run",
    # profile selector AFTER the `gateway` token (argv is profile-position
    # agnostic — _apply_profile_override strips --profile/-p anywhere)
    "hermes gateway --profile work run",
    "python -m hermes_cli.main gateway -p work run",
    "hermes gateway --profile=work run",
    # a profile literally NAMED "gateway"
    "hermes -p gateway gateway run",
    "python -m hermes_cli.main --profile gateway gateway run",
    # quoted Windows paths with spaces (shlex-aware tokenization)
    r'"C:\Program Files\Hermes\hermes-gateway.exe"',
    r'"C:\Program Files\Hermes\gateway\run.py" run',
    r'"C:\Program Files\Py\pythonw.exe" -m hermes_cli.main gateway run',
]

REJECT = [
    "python -m tui_gateway",                              # unrelated module
    "python -m hermes_cli.main gateway status",           # other subcommand
    "python -m hermes_cli.main gateway restart",
    "python -m hermes_cli.main gateway stop",
    "python -m hermes_cli.main --profile x dashboard",    # non-gateway subcommand
    "some random python -m mygateway thing",
    "",
    None,
]


@pytest.mark.parametrize("cmd", ACCEPT)
def test_accepts_real_gateway_run(cmd):
    assert matches(cmd) is True


@pytest.mark.parametrize("cmd", REJECT)
def test_rejects_non_gateway_command(cmd):
    assert matches(cmd) is False


# The managed shim relaunch (``hermes_cli.venv_sync.relaunch_command``) re-enters the entrypoint on
# the store interpreter as ``python -I -c "<bootstrap>"``: the real argv lives INSIDE the payload
# (``sys.argv = [...]``), so the command line itself carries no ``gateway`` token. Built from the
# real builder, so the fixture cannot drift from the launcher the installs actually run.
STORE_PYTHON = Path("/opt/hermes/.hermes/tools/python-3.14/bin/python3")
STORE_ROOT = Path("/opt/hermes")


def shim_relaunch(*args: str, module: str = "hermes_cli.main") -> str:
    """Command line of ``python -m <module> <args>`` after the shim re-execs it."""
    entry = STORE_ROOT / (module.replace(".", "/") + ".py")
    return " ".join(
        relaunch_command(
            STORE_PYTHON, STORE_ROOT, [str(entry), *args], ["python", "-m", module, *args], module,
        )
    )


def shim_relaunch_on(python: Path) -> str:
    """The same relaunch shape on another store interpreter (argv[0] as the launcher passes it)."""
    return " ".join(
        relaunch_command(
            python,
            STORE_ROOT,
            [str(STORE_ROOT / "hermes_cli/main.py"), "gateway", "run"],
            ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "hermes_cli.main",
        )
    )


@pytest.mark.parametrize(
    "args",
    [
        ("gateway", "run"),
        ("gateway",),                          # bare `gateway` defaults to run
        ("--profile", "work", "gateway", "run"),
        ("-p", "gateway", "gateway", "run"),   # a profile literally NAMED "gateway"
    ],
)
def test_accepts_managed_shim_relaunch(args):
    assert matches(shim_relaunch(*args)) is True


@pytest.mark.parametrize("args", [("gateway", "status"), ("gateway", "stop"), ("dashboard",)])
def test_rejects_managed_shim_non_run(args):
    assert matches(shim_relaunch(*args)) is False


def test_managed_shim_gateway_restart_is_a_runtime_form_only():
    command = shim_relaunch("gateway", "restart")
    assert matches(command) is False
    assert matches_runtime(command) is True


def test_managed_shim_relaunch_of_another_module_is_rejected():
    assert matches(shim_relaunch("serve", module="hermes_cli.web_server")) is False


def test_rejects_unparsable_shim_argv_payload():
    assert matches("python -I -c \"import sys; sys.argv = ['/opt/hermes/hermes_cli/main.py',") is False
    assert matches("python -I -c \"import sys; sys.argv = nope\"") is False


def test_pid_record_with_shim_argv_reads_as_gateway():
    """A record written by a shim-launched gateway must validate when its cmdline is unreadable."""
    record = {"kind": "hermes-gateway", "argv": shim_relaunch("gateway", "run").split()}
    assert _record_looks_like_gateway(record) is True
    stopped = {"kind": "hermes-gateway", "argv": shim_relaunch("gateway", "stop").split()}
    assert _record_looks_like_gateway(stopped) is False


def test_payload_argv_only_counts_after_an_interpreter_c():
    """A ``sys.argv`` literal in a trailing argument is that argument's text, not this argv.

    Reading any literal on the line would flip both directions: a ``gateway status`` invocation
    carrying a decoy payload would read as a running gateway (and be SIGTERMed), while a real
    ``gateway run`` carrying an unrelated one would read as foreign -- the outage this change fixes.
    """
    assert (
        matches("python -m hermes_cli.main gateway status -c \"import sys; sys.argv = ['gateway', 'run']\"")
        is False
    )
    assert matches("python -m hermes_cli.main gateway run -c \"import sys; sys.argv = ['x']\"") is True


def test_unterminated_payload_candidates_stay_bounded():
    """Every scan pays for an unterminated literal that runs to end-of-line: cap the candidates."""
    decoys = " ".join("sys.argv = [" for _ in range(2000))
    assert matches(f"python -I -c \"{decoys}\"") is False


@pytest.mark.parametrize(
    "cmd",
    [
        f"sudo {shim_relaunch('gateway', 'run')}",
        f"sh -c \"{shim_relaunch('gateway', 'run')}\"",
    ],
)
def test_wrapped_shim_launcher_matches(cmd):
    """A wrapper that still shows the interpreter invocation on the command line matches -- the same
    class as a visible-argv ``sh -c python -m hermes_cli.main gateway run``, which has always
    matched. A wrapper that hides the whole line in a quoted argument does not, for the plain form
    either; the shim form is reachable through its payload literal."""
    assert matches(cmd) is True


def test_shim_relaunch_profile_is_attributed_to_its_profile():
    """Profile checks are argv checks, and the shim hides argv in the payload.

    Without reading the payload back, a live shim-launched named-profile gateway is unattributable:
    every profile-scoped probe reports it stopped, and the default-profile rule ("no other profile's
    flags") accepts it -- so the default profile's scan can claim another profile's live gateway.
    """
    work = shim_relaunch("--profile", "work", "gateway", "run")
    assert profile_flag_value(work) == "work"
    assert _command_line_belongs_to_profile(work, STORE_ROOT / "profiles" / "work") is True
    assert _command_line_belongs_to_profile(work, STORE_ROOT) is False
    bare = shim_relaunch("gateway", "run")
    assert profile_flag_value(bare) is None
    assert _command_line_belongs_to_profile(bare, STORE_ROOT / "profiles" / "work") is False
    assert _command_line_belongs_to_profile(bare, STORE_ROOT) is True


def test_shim_relaunch_keeps_the_home_its_wrapper_spelled_out():
    """A wrapper's explicit HERMES_HOME still disqualifies the default profile.

    Reading the recovered argv instead of the outer line would drop the assignment and let the
    default profile claim another profile's -- or another install's -- live gateway.
    """
    command = f"env HERMES_HOME={STORE_ROOT}/profiles/work " + shim_relaunch("gateway", "run")
    # Callers normalize before the argv checks reach these readers (_command_line_belongs_to_profile).
    command_lc = command.lower().replace("\\", "/")
    assert hermes_home_assignments(command_lc) == [f"{STORE_ROOT}/profiles/work"]
    assert _command_line_belongs_to_profile(command_lc, STORE_ROOT) is False
    assert _command_line_belongs_to_profile(command_lc, STORE_ROOT / "profiles" / "work") is True


def test_payload_window_is_the_quoted_argument():
    """A quoted payload is one argument: a later command's literal on the same line is not our argv."""
    assert (
        matches("python -I -c \"print(1)\" \"sys.argv = ['/opt/hermes/hermes_cli/main.py', 'gateway', 'run']\"")
        is False
    )


@pytest.mark.parametrize(
    "original",
    [
        ["python", "-W", "ignore", "-m", "hermes_cli.main", "gateway", "run"],
        ["python", "-X", "dev", "-I", "-m", "hermes_cli.main", "gateway", "run"],
    ],
)
def test_accepts_shim_relaunch_with_interpreter_options(original):
    """``relaunch_command`` preserves ``-W``/``-X`` with their value from the original invocation, so
    those spellings reach installs and must resolve like any other relaunch."""
    command = " ".join(
        relaunch_command(
            STORE_PYTHON,
            STORE_ROOT,
            [str(STORE_ROOT / "hermes_cli/main.py"), "gateway", "run"],
            original,
            "hermes_cli.main",
        )
    )
    assert ("-W ignore" in command) or ("-X dev" in command)
    assert matches(command) is True


def test_accepts_shim_relaunch_on_a_free_threaded_interpreter():
    """A free-threaded store interpreter (``python3.14t``) is still an interpreter."""
    python = Path("/opt/hermes/.hermes/tools/python-3.14t/bin/python3.14t")
    assert matches(shim_relaunch_on(python)) is True


def test_option_run_scan_stays_linear():
    """A long run of option-shaped words before the payload must not backtrack.

    The interpreter-option scan is walked token by token for this reason: a pattern that consumes
    those options takes seconds on a short line of ``-W -X`` pairs, and every process scan pays it
    (``hermes gateway status``/``stop``, the venv-blocker scan, the update paths).
    """
    command = "x -c sys.argv python " + " ".join(["-W", "-X"] * 400)
    started = time.monotonic()
    assert matches(command) is False
    assert time.monotonic() - started < 2.0


def test_accepts_windows_escaped_shim_payload():
    """Windows renders an argv element with cmd.exe quoting, escaping every embedded double quote.

    A launcher argv carrying an apostrophe is repr'd with double quotes, so the payload reaches a raw
    Windows command line full of ``\\"`` sequences. Those are quoting, not Python source: read raw,
    the literal never terminates and the live gateway reads as a foreign process.
    """
    python = Path("C:/Users/O'Brien/.hermes/tools/python-3.14/python.exe")
    root = Path("C:/Users/O'Brien/hermes")
    command = subprocess.list2cmdline(
        relaunch_command(
            python,
            root,
            [str(root / "hermes_cli" / "main.py"), "gateway", "run"],
            ["python", "-m", "hermes_cli.main", "gateway", "run"],
            "hermes_cli.main",
        )
    )
    assert '\\"' in command
    assert matches(command) is True


