"""Seatbelt policy construction and the fail-closed launch contract (no learner code runs here)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dogma.core.errors import ErrorCode, RunnerError
from dogma.runner import policy
from dogma.runner import sandbox as sandbox_module


@pytest.fixture
def darwin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pretend to be a supported host with a stand-in sandbox-exec that exists and is executable."""
    wrapper = tmp_path / "sandbox-exec"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setattr(policy, "_current_platform", lambda: "darwin")
    monkeypatch.setattr(policy, "SANDBOX_EXEC", wrapper)
    return wrapper


def test_profile_denies_network_and_writes_and_then_allows_only_the_run_dir():
    text = policy.PROFILE
    assert "(deny network*)" in text
    assert "(deny file-write*)" in text
    allow = '(allow file-write* (subpath (param "RUN_DIR")))'
    assert allow in text
    # Seatbelt applies the last matching rule, so the allowance must follow the denial.
    assert text.index(allow) > text.index("(deny file-write*)")


def test_profile_text_never_contains_a_path(darwin, tmp_path):
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    assert str(run_dir) not in policy.PROFILE
    assert str(run_dir.resolve()) not in policy.PROFILE


def test_command_is_wrapped_once_with_the_fixed_executable_and_run_dir_parameter(darwin, tmp_path):
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    argv = policy.sandbox_command([sys.executable, "-c", "pass"], run_dir)
    assert argv[:5] == [str(darwin), "-D", f"RUN_DIR={run_dir.resolve()}", "-p", policy.PROFILE]
    assert argv[5:] == [sys.executable, "-c", "pass"]
    assert argv.count(str(darwin)) == 1, "exactly one wrapper, never a nested or raw fallback"


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_hosts_without_seatbelt_refuse_before_any_command_is_built(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(policy, "_current_platform", lambda: platform)
    with pytest.raises(RunnerError) as info:
        policy.sandbox_command([sys.executable], tmp_path)
    assert info.value.code is ErrorCode.RUNNER_CRASHED
    assert "no supported OS sandbox" in info.value.message
    assert "without OS enforcement" in info.value.message


def test_missing_sandbox_exec_refuses_instead_of_running_unwrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, "_current_platform", lambda: "darwin")
    monkeypatch.setattr(policy, "SANDBOX_EXEC", tmp_path / "absent-sandbox-exec")
    with pytest.raises(RunnerError) as info:
        policy.sandbox_command([sys.executable], tmp_path)
    assert info.value.code is ErrorCode.RUNNER_CRASHED
    assert "sandbox-exec" in info.value.message


def test_non_executable_sandbox_exec_refuses(monkeypatch, tmp_path):
    impostor = tmp_path / "sandbox-exec"
    impostor.write_text("#!/bin/sh\n", encoding="utf-8")
    impostor.chmod(0o644)
    monkeypatch.setattr(policy, "_current_platform", lambda: "darwin")
    monkeypatch.setattr(policy, "SANDBOX_EXEC", impostor)
    with pytest.raises(RunnerError) as info:
        policy.ensure_supported()
    assert info.value.code is ErrorCode.RUNNER_CRASHED


def test_run_dir_with_unusual_but_plain_characters_is_passed_as_one_parameter(darwin, tmp_path):
    odd = tmp_path / 'sp ace "quoted" (paren) ünï; $HOME'
    odd.mkdir()
    argv = policy.sandbox_command(["harness"], odd)
    assert argv[2] == f"RUN_DIR={odd.resolve()}"
    assert "$HOME" in argv[2], "no shell expansion happens on the way in"


def test_run_dir_with_a_newline_is_refused(darwin, tmp_path):
    bad = tmp_path / "two\nlines"
    bad.mkdir()
    with pytest.raises(RunnerError) as info:
        policy.sandbox_command(["harness"], bad)
    assert info.value.code is ErrorCode.RUNNER_CRASHED


@pytest.fixture
def refusing_wrapper(darwin: Path) -> Path:
    """A wrapper that fails the way sandbox-exec does when the profile cannot be applied."""
    darwin.write_text(
        "#!/bin/sh\necho 'sandbox_apply: Operation not permitted' >&2\nexit 71\n",
        encoding="utf-8",
    )
    return darwin


def test_refused_wrapper_is_an_infrastructure_error_with_exactly_one_spawn(
    refusing_wrapper, monkeypatch, tmp_path
):
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    spawned: list[list[str]] = []
    real_popen = sandbox_module.subprocess.Popen

    def record(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        spawned.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", record)
    with pytest.raises(RunnerError) as info:
        sandbox_module.execute(
            run_dir, tests=("test_public.py",), timeout_s=10, memory_limit_mb=512
        )
    assert info.value.code is ErrorCode.RUNNER_CRASHED
    assert "nothing was executed" in info.value.message
    assert len(spawned) == 1
    assert spawned[0][0] == str(refusing_wrapper), "no retry without the wrapper"
    assert not (run_dir / sandbox_module.REPORT_NAME).exists()


def test_a_learner_exit_code_that_is_not_a_wrapper_refusal_stays_data(darwin, tmp_path):
    darwin.write_text("#!/bin/sh\nexit 71\n", encoding="utf-8")  # no sandbox text: learner data
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    result = sandbox_module.execute(run_dir, tests=("t.py",), timeout_s=10, memory_limit_mb=512)
    assert result.exit_code == 71
    assert result.kill_reason is None


def test_harness_gets_the_tests_dir_only_when_one_is_supplied(darwin, monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def refuse(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        raise OSError("argv recorded, nothing started")

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", refuse)
    run_dir = tmp_path / "attempt"
    run_dir.mkdir()
    bundle = tmp_path / "bundle"
    for tests_dir in (None, bundle):
        with pytest.raises(RunnerError):
            sandbox_module.execute(
                run_dir,
                tests=("test_public.py",),
                timeout_s=10,
                memory_limit_mb=512,
                tests_dir=tests_dir,
            )
    without, with_dir = seen
    assert "--tests-dir" not in without
    assert with_dir[with_dir.index("--tests-dir") + 1] == str(bundle)
    assert "-P" in with_dir
    assert "-s" in with_dir
