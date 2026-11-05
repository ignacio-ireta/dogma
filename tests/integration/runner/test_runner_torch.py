"""Actual CPU tensors/autograd through the optional runtime and the isolated harness."""

from __future__ import annotations

import importlib.util

import pytest

from dogma.core.config import Settings
from dogma.grading.models import RunStatus
from dogma.runner import run

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="optional torch group absent")
def test_torch_cpu_autograd_passes_public_and_hidden_tests_repeatably(tmp_path):
    problem = tmp_path / "tensor-gradient"
    problem.mkdir()
    (problem / "meta.yaml").write_text(
        "id: fixtures/torch/p01-gradient\nversion: 1\nlanguage: python\n"
        "runtime: py-torch\nentry: square_grad\ntimeout_s: 30\n"
    )
    (problem / "test_public.py").write_text(
        "import pytest\nfrom submission import square_grad\n"
        "@pytest.mark.parametrize('values', [[1.0, 2.0], [0.0], [-2.5, 3.5]])\n"
        "def test_gradient(values):\n"
        "    assert square_grad(values) == [2 * value for value in values]\n"
    )
    (problem / "test_hidden.py").write_text(
        "import pytest\nfrom submission import square_grad\n"
        "@pytest.mark.parametrize('values', [[], [1e-10, -1e-10], list(range(20))])\n"
        "def test_other_shapes_and_values(values):\n"
        "    assert square_grad(values) == [2 * value for value in values]\n"
    )
    submission = tmp_path / "answer.py"
    submission.write_text(
        "import torch\n"
        "def square_grad(values):\n"
        "    x = torch.tensor(values, dtype=torch.float64, device='cpu', requires_grad=True)\n"
        "    (x * x).sum().backward()\n"
        "    return x.grad.tolist()\n"
    )
    settings = Settings(repo_root=tmp_path / "scratch", _env_file=None, record_attempts=False)
    for attempt in ("first", "repeat"):
        report = run(
            problem, submission, include_hidden=True, settings=settings, attempt_id=attempt
        )
        assert report.status is RunStatus.PASSED, report.message or report.tests
        assert report.runtime == "py-torch"
        assert report.counts("public") == (3, 3)
        assert report.counts("hidden") == (3, 3)
