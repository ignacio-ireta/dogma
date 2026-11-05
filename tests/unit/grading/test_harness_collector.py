"""The harness's pytest collector, exercised in-process through pytester."""

from __future__ import annotations

import pytest

from dogma_harness.collector import Collector, _BoundedBuffer, describe_exception

pytest_plugins = ["pytester"]

ARGS = ("-q", "-p", "no:cacheprovider", "--capture=no", "--tb=short")


def _run(pytester: pytest.Pytester, *args: str, exit_code_override: int | None = None):
    collector = Collector()
    run = pytester.inline_run(*ARGS, *args, plugins=[collector])
    exit_code = run.ret if exit_code_override is None else exit_code_override
    return collector.result(int(exit_code))


def _by_name(result):
    return {t.name: t for t in result.tests}


def test_all_passing_tests_give_passed_and_split_public_hidden(pytester):
    pytester.makepyfile(
        test_public="def test_a():\n    assert 1 + 1 == 2\n",
        test_hidden="def test_b():\n    assert True\n",
    )
    result = _run(pytester, "test_public.py", "test_hidden.py")
    assert result.status == "passed"
    assert {(t.name, t.file, t.outcome) for t in result.tests} == {
        ("test_a", "public", "pass"),
        ("test_b", "hidden", "pass"),
    }


def test_failing_assertion_reports_the_custom_message(pytester):
    pytester.makepyfile(
        test_public=("def test_sum():\n    assert 2 + 2 == 5, 'the sum should be 4'\n"),
    )
    result = _run(pytester, "test_public.py")
    assert result.status == "failed"
    (test,) = result.tests
    assert test.outcome == "fail"
    assert test.message.startswith("the sum should be 4")


def test_non_assertion_exception_is_named_in_the_message(pytester):
    pytester.makepyfile(test_public="def test_boom():\n    raise ValueError('bad input')\n")
    (test,) = _run(pytester, "test_public.py").tests
    assert test.outcome == "fail", "an exception in the call is a failure, not an error"
    assert test.message == "ValueError: bad input"


def test_failing_setup_is_an_error_not_a_failure(pytester):
    pytester.makepyfile(
        test_public=(
            "import pytest\n"
            "@pytest.fixture\n"
            "def broken():\n"
            "    raise RuntimeError('fixture exploded')\n"
            "def test_uses_it(broken):\n"
            "    pass\n"
        ),
    )
    result = _run(pytester, "test_public.py")
    (test,) = result.tests
    assert test.outcome == "error"
    assert "fixture exploded" in test.message
    assert result.status == "failed"


def test_failing_teardown_is_an_error_even_when_the_call_passed(pytester):
    pytester.makepyfile(
        test_public=(
            "import pytest\n"
            "@pytest.fixture\n"
            "def leaky():\n"
            "    yield\n"
            "    raise RuntimeError('cleanup failed')\n"
            "def test_fine(leaky):\n"
            "    assert True\n"
        ),
    )
    (test,) = _run(pytester, "test_public.py").tests
    assert test.outcome == "error"
    assert "cleanup failed" in test.message


def test_skipped_test_is_recorded_as_skip_and_not_a_pass(pytester):
    pytester.makepyfile(
        test_public=(
            "import pytest\n"
            "@pytest.mark.skip(reason='not yet')\n"
            "def test_later():\n"
            "    assert False\n"
            "def test_now():\n"
            "    assert True\n"
        ),
    )
    result = _run(pytester, "test_public.py")
    assert _by_name(result)["test_later"].outcome == "skip"
    assert result.status == "failed", "a skipped test means not everything passed"


def test_hypothesis_style_notes_are_kept_in_the_message(pytester):
    pytester.makepyfile(
        test_hidden=(
            "def test_note():\n"
            "    err = AssertionError('mismatch')\n"
            "    err.add_note('Falsifying example: test_note(a=3)')\n"
            "    raise err\n"
        ),
    )
    (test,) = _run(pytester, "test_hidden.py").tests
    assert test.message.splitlines() == ["mismatch", "Falsifying example: test_note(a=3)"]


def test_collection_error_is_reported_with_the_exception_line(pytester):
    pytester.makepyfile(
        test_public=(
            "raise RuntimeError('cannot import the submission')\n\ndef test_x():\n    pass\n"
        ),
    )
    result = _run(pytester, "test_public.py")
    assert result.status == "error"
    assert result.tests == ()
    assert result.collection_error is not None
    assert "cannot import the submission" in result.collection_error


def test_file_with_no_tests_is_an_error(pytester):
    pytester.makepyfile(test_public="X = 1\n")
    result = _run(pytester, "test_public.py")
    assert result.status == "error"
    assert result.collection_error == "no tests were collected"


def test_unexpected_pytest_exit_code_is_an_error_even_with_results(pytester):
    pytester.makepyfile(test_public="def test_a():\n    assert True\n")
    result = _run(pytester, "test_public.py", exit_code_override=3)
    assert result.status == "error"
    assert "exit code 3" in (result.collection_error or "")
    assert [t.outcome for t in result.tests] == ["pass"]


def test_captured_stdout_from_every_test_is_kept_in_order(pytester):
    pytester.makepyfile(
        test_public=("def test_one():\n    print('first')\ndef test_two():\n    print('second')\n"),
    )
    result = _run(pytester, "test_public.py")
    assert result.stdout.index("first") < result.stdout.index("second")


def test_stdout_is_truncated_to_64_kb(pytester):
    pytester.makepyfile(test_public="def test_loud():\n    print('x' * 1_000_000)\n")
    result = _run(pytester, "test_public.py")
    assert len(result.stdout.encode("utf-8")) <= 64 * 1024
    assert "truncated" in result.stdout


def test_describe_exception_keeps_plain_assertion_text():
    assert describe_exception(AssertionError("shape mismatch")) == "shape mismatch"


def test_describe_exception_without_text_uses_the_type_name():
    assert describe_exception(TypeError()) == "TypeError"


def test_bounded_buffer_keeps_the_head_and_counts_what_it_dropped():
    buffer = _BoundedBuffer(limit=10)
    assert buffer.write(b"abcdef") == 6
    assert buffer.write(b"ghijklmnop") == 10, "a write reports every byte as accepted"
    assert buffer.getvalue() == b"abcdefghij"
    assert buffer.dropped == 6


def test_endless_print_inside_a_test_does_not_grow_memory_past_the_cap(pytester):
    pytester.makepyfile(
        test_public="def test_loop():\n    for _ in range(2000):\n        print('x' * 1000)\n"
    )
    result = _run(pytester, "test_public.py")
    assert result.status == "passed"
    assert len(result.stdout.encode("utf-8")) <= 64 * 1024
    assert "truncated" in result.stdout or "discarded" in result.stdout
