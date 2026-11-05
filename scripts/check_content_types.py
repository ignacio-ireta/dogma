"""Check content solutions and tests with plain non-strict mypy (§5.2).

Each problem gets its own invocation so repeated submission/test module names
cannot collide. Staging mirrors the runner without executing any content code.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mypy import api

REPO = Path(__file__).resolve().parent.parent
TEST_FILES = ("test_public.py", "test_hidden.py")
CONFIG = "[mypy]\npython_version = 3.12\nignore_missing_imports = True\nshow_error_codes = True\n"


def discover_problems(root: Path) -> list[Path]:
    """Find authored problem directories and reject incomplete packages."""
    problems = sorted(path for path in root.glob("content/*/*/problems/*") if path.is_dir())
    if not problems:
        raise ValueError("no problem packages found under content/")
    for problem in problems:
        required = list(TEST_FILES)
        if not (problem / "solution.py").is_file() and not (problem / "solution.sql").is_file():
            required.append("solution.py or solution.sql")
        missing = [name for name in required if not (problem / name).is_file()]
        if missing:
            raise ValueError(f"{problem.relative_to(root)}: missing {', '.join(missing)}")
    return problems


def check_problem(problem: Path, root: Path, stage: Path, config: Path, cache: Path) -> int:
    """Stage and type-check one package, reporting original repository paths."""
    stage.mkdir()
    originals = {name: problem / name for name in TEST_FILES}
    if (problem / "solution.py").is_file():
        originals["submission.py"] = problem / "solution.py"
    for name, original in originals.items():
        shutil.copyfile(original, stage / name)
    out, err, code = api.run(
        [
            "--config-file",
            str(config),
            "--cache-dir",
            str(cache),
            *(str(stage / name) for name in originals),
        ]
    )
    for name, original in originals.items():
        out = out.replace(str(stage / name), original.relative_to(root).as_posix())
        err = err.replace(str(stage / name), original.relative_to(root).as_posix())
    if code:
        print(out, end="")
    if err:
        print(err, end="", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    """Return mypy's status, or 2 for invalid input or staging errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        problems = discover_problems(root)
        with TemporaryDirectory(prefix="dogma-content-types-") as temporary:
            workspace = Path(temporary)
            config = workspace / "mypy.ini"
            config.write_text(CONFIG, encoding="utf-8")
            status = 0
            for index, problem in enumerate(problems):
                status = max(
                    status,
                    check_problem(
                        problem, root, workspace / f"p{index}", config, workspace / "cache"
                    ),
                )
    except (OSError, ValueError) as exc:
        print(f"content-types: {exc}", file=sys.stderr)
        return 2
    print(f"content-types: checked {len(problems)} problem package(s)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
