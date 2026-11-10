"""Tests for the hidden-test lint. The sources are strings: nothing is imported or run."""

from __future__ import annotations

import pytest

from dogma.content.testlint import RULES, LintFinding, id_leaks, lint_hidden_tests

PARAM_MESSAGE = "parametrize needs ids=: default ids print the hidden inputs in the test name"
RAISES_MESSAGE = "use a helper that names the mistake instead of pytest.raises"
GIVEN_MESSAGE = "@given needs @settings(deadline=None, report_multiple_bugs=False)"
ASSERT_MESSAGE = "assert needs a message that names the kind of mistake"

RULE_TABLE = [
    pytest.param(
        "HIDDEN_PARAM_IDS",
        '@pytest.mark.parametrize("x", [1, 2])\ndef test_a(x):\n    assert x, "m"\n',
        1,
        PARAM_MESSAGE,
        '@pytest.mark.parametrize("x", [1, 2], ids=["a", "b"])\n'
        'def test_a(x):\n    assert x, "m"\n',
        id="param-ids",
    ),
    pytest.param(
        "HIDDEN_BARE_RAISES",
        'def test_a():\n    with pytest.raises(ValueError, match="m"):\n        f()\n',
        2,
        RAISES_MESSAGE,
        'def test_a():\n    with check_error(ValueError, "bad shape"):\n        f()\n',
        id="bare-raises",
    ),
    pytest.param(
        "HIDDEN_GIVEN_SETTINGS",
        '@given(st.integers())\ndef test_a(x):\n    assert x, "m"\n',
        2,
        GIVEN_MESSAGE,
        "@given(st.integers())\n"
        "@settings(deadline=None, report_multiple_bugs=False, max_examples=50)\n"
        'def test_a(x):\n    assert x, "m"\n',
        id="given-settings",
    ),
    pytest.param(
        "HIDDEN_ASSERT_MESSAGE",
        "def test_a():\n    assert x\n",
        2,
        ASSERT_MESSAGE,
        'def test_a():\n    assert x, "shape mismatch: rows differ"\n',
        id="assert-message",
    ),
]


@pytest.mark.parametrize(("rule", "bad", "line", "message", "good"), RULE_TABLE)
def test_rule_reports_its_site_once_and_stays_quiet_on_the_fix(
    rule: str, bad: str, line: int, message: str, good: str
) -> None:
    assert lint_hidden_tests(bad) == [LintFinding(rule, line, message)]
    assert lint_hidden_tests(good) == []


def test_rules_tuple_lists_the_four_rules_in_order() -> None:
    assert RULES == (
        "HIDDEN_PARAM_IDS",
        "HIDDEN_BARE_RAISES",
        "HIDDEN_GIVEN_SETTINGS",
        "HIDDEN_ASSERT_MESSAGE",
    )


def test_bare_name_raises_is_reported() -> None:
    source = "def test_a():\n    with raises(ValueError):\n        f()\n"
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_BARE_RAISES", 2, RAISES_MESSAGE)]


def test_parametrize_with_param_ids_on_every_case_passes() -> None:
    source = (
        "@pytest.mark.parametrize(\n"
        '    "x", [pytest.param(1, id="one"), pytest.param(2, id="two")]\n'
        ")\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == []


def test_parametrize_with_one_param_missing_id_is_reported() -> None:
    source = (
        "@pytest.mark.parametrize(\n"
        '    "x", [pytest.param(1, id="one"), pytest.param(2)]\n'
        ")\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_PARAM_IDS", 1, PARAM_MESSAGE)]


def test_parametrize_on_a_class_without_ids_is_reported() -> None:
    source = (
        '@pytest.mark.parametrize("x", [1])\n'
        "class TestA:\n"
        "    def test_a(self, x):\n"
        '        assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_PARAM_IDS", 1, PARAM_MESSAGE)]


def test_settings_without_report_multiple_bugs_is_reported_on_the_def_line() -> None:
    source = (
        "@settings(max_examples=50, deadline=None)\n"
        "@given(st.integers())\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_GIVEN_SETTINGS", 3, GIVEN_MESSAGE)]


def test_a_module_level_settings_alias_with_both_flags_is_accepted() -> None:
    source = (
        "SETTINGS = settings(max_examples=60, deadline=None, report_multiple_bugs=False)\n"
        "@SETTINGS\n"
        "@given(st.integers())\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == []


def test_a_settings_alias_missing_a_flag_is_reported() -> None:
    source = (
        "SETTINGS = settings(max_examples=60, deadline=None)\n"
        "@SETTINGS\n"
        "@given(st.integers())\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_GIVEN_SETTINGS", 4, GIVEN_MESSAGE)]


def test_a_settings_alias_rebound_to_an_unclean_value_is_reported() -> None:
    source = (
        "SETTINGS = settings(deadline=None, report_multiple_bugs=False)\n"
        "SETTINGS = settings(deadline=None)\n"
        "@SETTINGS\n"
        "@given(st.integers())\n"
        "def test_a(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_GIVEN_SETTINGS", 5, GIVEN_MESSAGE)]


def test_a_bare_decorator_with_no_module_binding_is_reported() -> None:
    source = '@SETTINGS\n@given(st.integers())\ndef test_a(x):\n    assert x, "m"\n'
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_GIVEN_SETTINGS", 3, GIVEN_MESSAGE)]


def test_given_on_a_helper_not_named_test_is_ignored() -> None:
    source = '@given(st.integers())\ndef helper(x):\n    assert x, "m"\n'
    assert lint_hidden_tests(source) == []


def test_assert_inside_a_helper_function_is_reported() -> None:
    source = 'def check(a):\n    assert a > 0, "m"\n    assert a < 9\n'
    assert lint_hidden_tests(source) == [LintFinding("HIDDEN_ASSERT_MESSAGE", 3, ASSERT_MESSAGE)]


def test_numpy_assert_with_err_msg_passes() -> None:
    source = 'np.testing.assert_allclose(a, b, err_msg="x")\n'
    assert lint_hidden_tests(source) == []


def test_torch_assert_with_msg_passes() -> None:
    source = 'torch.testing.assert_close(a, b, msg="x")\n'
    assert lint_hidden_tests(source) == []


def test_numpy_assert_without_message_is_reported_with_its_name() -> None:
    source = "np.testing.assert_allclose(a, b)\n"
    assert lint_hidden_tests(source) == [
        LintFinding("HIDDEN_ASSERT_MESSAGE", 1, "assert_allclose needs err_msg= (msg= for torch)")
    ]


def test_syntax_error_gives_no_findings() -> None:
    assert lint_hidden_tests("def test_a(:\n    pytest.raises(ValueError)\n") == []


def test_findings_come_back_sorted_by_line() -> None:
    source = (
        "def test_a():\n"
        "    assert x\n"
        "    with pytest.raises(ValueError):\n"
        "        f()\n"
        "\n"
        '@pytest.mark.parametrize("x", [1])\n'
        "def test_b(x):\n"
        '    assert x, "m"\n'
    )
    assert lint_hidden_tests(source) == [
        LintFinding("HIDDEN_ASSERT_MESSAGE", 2, ASSERT_MESSAGE),
        LintFinding("HIDDEN_BARE_RAISES", 3, RAISES_MESSAGE),
        LintFinding("HIDDEN_PARAM_IDS", 6, PARAM_MESSAGE),
    ]


@pytest.mark.parametrize(
    ("name", "leaks"),
    [
        pytest.param("test_a[180.0]", True, id="float-value"),
        pytest.param("test_a[seed0]", True, id="digit-in-word"),
        pytest.param("test_a[float32-n2]", True, id="dtype-then-digit"),
        pytest.param("test_a[empty]", False, id="word"),
        pytest.param("test_a[float32]", False, id="dtype-only"),
        pytest.param("test_a[int8-uint16]", False, id="two-dtypes"),
        pytest.param("test_a[inference-fp16-none-False]", False, id="precision-name"),
        pytest.param("test_a[bf16]", False, id="brain-float-name"),
        pytest.param("test_a[fp16-n2]", True, id="precision-then-digit"),
        pytest.param("test_a[fp7]", True, id="not-a-precision-width"),
        pytest.param("test_a[case-a]", False, id="word-pair"),
        pytest.param("test_b", False, id="no-brackets"),
    ],
)
def test_id_leaks_flags_digits_outside_dtype_names(name: str, leaks: bool) -> None:
    assert id_leaks([name]) == ([name] if leaks else [])


def test_id_leaks_keeps_input_order() -> None:
    names = ["test_a[seed1]", "test_b", "test_c[empty]", "test_d[2]"]
    assert id_leaks(names) == ["test_a[seed1]", "test_d[2]"]
