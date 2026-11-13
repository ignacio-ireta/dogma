"""Probe what the tests of each problem never try (ROADMAP I-34).

The reject sweep of I-32 wrote one wrong solution per Contract bullet by hand, eight at most.
This tool asks three questions of every Python problem by machine, with the reference solution
inside a wrapper, so that the only thing wrong with a probe is the one thing its name says:

    record    Run the tests on the reference and note what each argument was given and what
              came back: container, size, dtype, sign, ties, order. Plain pytest, no sandbox.
    scribble  The right result, and one argument changed in place afterwards (sorted, centred,
              rescaled, a nested cell written). Graded on the learner's real path.
    returns   The right values in a laxer form (a list for an array, float32 for float64, a
              float for an int, a one-element array for a float, rows in reverse order).
    report    Survivors of both families and the input classes no passing call was given.
    write     One probe as a file, for ``exercise_gate.py add-test --wrong``.

A probe that survives is a candidate, not a finding: the Contract decides. ``report`` prints the
bullets next to each survivor so that the decision can be made from the statement.

Usage (the torch group is needed for the py-torch runtime)::

    uv run --frozen --group torch python scripts/exercise_probe.py record   --out DIR
    uv run --frozen --group torch python scripts/exercise_probe.py scribble --out DIR
    uv run --frozen --group torch python scripts/exercise_probe.py returns  --out DIR
    uv run --frozen --group torch python scripts/exercise_probe.py report   --out DIR
    uv run --frozen --group torch python scripts/exercise_probe.py write    --out DIR ID PROBE FILE

``--only PREFIX`` (repeatable) limits a command to problems whose id starts with PREFIX. SQL
problems and problems whose entry is a class are skipped: a wrapper has no single call to wrap.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from dogma.core.config import Settings

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exercise_audit as audit

_TEST_FILES = ("test_public.py", "test_hidden.py")
_MUTATION_WORDS = re.compile(
    r"mutat|unchanged|in.?place|not.?modif|untouched|intact|left.?alone|does.?not.?change"
    r"|not.?change|preserv|caller|same.?object|not.?alter|leaves",
    re.IGNORECASE,
)

# --- the recorder -------------------------------------------------------------------------------

_RECORDER = r"""


import atexit as _probe_atexit
import json as _probe_json
import os as _probe_os

_probe_reference = {entry}
_PROBE_PARAMS = {params!r}
_probe_ok = {{}}
_probe_raised = {{}}
_probe_first = {{}}
_probe_calls = [0, 0]
_probe_visible = {{}}
_PROBE_CHANGES = {{}}
{changes}


def _probe_array(a, np):
    feats = ["ndim%d" % a.ndim, "dtype_" + a.dtype.kind]
    if a.dtype == np.float32:
        feats.append("float32")
    if not a.flags.writeable:
        feats.append("readonly")
    if not a.flags.c_contiguous:
        feats.append("noncontiguous")
    if a.size == 0:
        feats.append("empty")
        return feats
    if a.ndim >= 1 and a.shape[0] == 1:
        feats.append("one_row")
    if a.ndim >= 2 and a.shape[1] == 1:
        feats.append("one_column")
    if a.dtype.kind not in "fiu":
        return feats
    x = a.astype(np.float64)
    finite = x[np.isfinite(x)]
    if np.isnan(x).any():
        feats.append("nan")
    if np.isinf(x).any():
        feats.append("inf")
    if finite.size:
        if (finite < 0).any():
            feats.append("negative")
        if (finite == 0).any():
            feats.append("zero")
        if np.abs(finite).max() > 1e100:
            feats.append("huge")
        if a.dtype.kind == "f":
            feats.append("whole" if (finite == np.round(finite)).all() else "fractional")
    if a.ndim == 1 and a.size >= 2:
        feats.append("ties" if np.unique(x).size < x.size else "distinct")
    if a.ndim == 1 and a.size >= 3 and np.isfinite(x).all():
        steps = np.diff(x)
        if (steps >= 0).all():
            feats.append("ascending")
        elif (steps <= 0).all():
            feats.append("descending")
        else:
            feats.append("unsorted")
    return feats


def _probe_scalar(v):
    feats = []
    try:
        if v != v:
            return ["nan"]
        if v in (float("inf"), float("-inf")):
            return ["inf"]
        if v == 0:
            feats.append("zero")
        elif v < 0:
            feats.append("negative")
        else:
            feats.append("positive")
    except Exception:
        pass
    return feats


def _probe_sequence(v):
    feats = []
    if len(v) == 0:
        return ["empty"]
    if len(v) == 1:
        feats.append("one_item")
    kinds = set(type(item).__name__ for item in v)
    if kinds <= {{"int"}}:
        feats.append("ints_only")
    if kinds <= {{"int", "float"}}:
        import numpy as np

        feats.extend(f for f in _probe_array(np.asarray(v, dtype=float), np) if f[:4] != "ndim"
                     and f[:6] != "dtype_" and f not in ("whole", "fractional"))
    elif kinds <= {{"str"}}:
        feats.append("ties" if len(set(v)) < len(v) else "distinct")
        if "" in v:
            feats.append("empty_string")
    if kinds & {{"list", "tuple"}}:
        feats.append("nested")
        try:
            import numpy as np

            # A rectangular list of whole numbers reaches the solution as an integer array.
            if np.asarray(v).dtype.kind in "iu":
                feats.append("ints_only")
        except Exception:
            pass
        if any(len(item) == 0 for item in v if isinstance(item, (list, tuple))):
            feats.append("empty_inner")
    if "bool" in kinds:
        feats.append("has_bool")
    if "NoneType" in kinds:
        feats.append("has_none")
    return feats


def _probe_features(v):
    name = type(v).__name__
    feats = ["type_" + name]
    try:
        if v is None or isinstance(v, bool):
            return feats
        if isinstance(v, (int, float)):
            return feats + _probe_scalar(v)
        if name == "ndarray":
            import numpy as np

            return feats + _probe_array(v, np)
        if name in ("Tensor", "Parameter"):
            import numpy as np

            if v.requires_grad:
                feats.append("requires_grad")
            if not v.is_contiguous():
                feats.append("noncontiguous")
            feats.append("torch_" + str(v.dtype).replace("torch.", ""))
            data = v.detach().cpu()
            if data.dtype.is_floating_point and data.dtype.itemsize < 4:
                data = data.float()
            if str(data.dtype) == "torch.bfloat16":
                data = data.float()
            return feats + [f for f in _probe_array(data.numpy(), np) if f != "noncontiguous"]
        if name == "DataFrame":
            if len(v) == 0:
                return feats + ["empty"]
            if len(v) == 1:
                feats.append("one_row")
            if bool(v.isna().any().any()):
                feats.append("nan")
            if bool(v.duplicated().any()):
                feats.append("duplicate_rows")
            if list(v.index) != list(range(len(v))):
                feats.append("index_not_range")
            return feats
        if hasattr(v, "dtype") and hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
            return feats + ["numpy_scalar"] + _probe_scalar(v.item())
        if isinstance(v, (list, tuple)):
            return feats + _probe_sequence(v)
        if isinstance(v, (dict, set, frozenset, str, bytes)):
            if len(v) == 0:
                feats.append("empty")
            elif len(v) == 1:
                feats.append("one_item")
            return feats
    except Exception as exc:
        feats.append("unreadable_" + type(exc).__name__)
    return feats


def _probe_leaves(value, depth=0):
    # What a result is made of, three levels down: the types a laxer return form could change.
    name = type(value).__name__
    if isinstance(value, (list, tuple)) and depth < 3:
        found = set()
        for item in value[:50]:
            found |= _probe_leaves(item, depth + 1)
        return found
    if isinstance(value, dict) and depth < 3:
        found = set()
        for item in list(value.values())[:50]:
            found |= _probe_leaves(item, depth + 1)
        return found
    if name == "ndarray":
        return {{"leaf_ndarray_" + value.dtype.kind}}
    if name in ("Tensor", "Parameter"):
        grad = "_grad" if value.requires_grad else ""
        return {{"leaf_tensor_" + str(value.dtype).replace("torch.", "") + grad}}
    return {{"leaf_" + name}}


def _probe_note(table, key, value):
    test = _probe_os.environ.get("PYTEST_CURRENT_TEST", "?").split("::")[-1].split(" ")[0]
    seen = table.setdefault(key, {{}})
    for feat in _probe_features(value):
        seen[feat] = seen.get(feat, 0) + 1
        if table is _probe_ok:
            _probe_first.setdefault(key, {{}}).setdefault(feat, test)


def _probe_same(a, b):
    name = type(a).__name__
    if name == "ndarray":
        import numpy as np

        return a.dtype == b.dtype and a.shape == b.shape and bool(
            np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        )
    if name in ("Tensor", "Parameter"):
        import torch

        return a.dtype == b.dtype and a.shape == b.shape and bool(
            torch.equal(torch.nan_to_num(a.detach().float()), torch.nan_to_num(b.detach().float()))
            if a.dtype.is_floating_point
            else torch.equal(a, b)
        )
    if name == "DataFrame":
        return a.equals(b) and list(a.index) == list(b.index)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(
            type(x) is type(y) and _probe_same(x, y) for x, y in zip(a, b)
        )
    if isinstance(a, dict):
        return list(a) == list(b) and all(_probe_same(a[k], b[k]) for k in a)
    try:
        return bool(a == b) or (a != a and b != b)
    except Exception:
        return True


def _probe_visible_changes(key, value):
    # For each in-place change that applies to this type: would it show on this very input?
    import copy

    name = type(value).__name__
    name = "Tensor" if name == "Parameter" else name
    for (type_name, kind), change in _PROBE_CHANGES.items():
        if type_name != name:
            continue
        try:
            trial = value.detach().clone() if name == "Tensor" else copy.deepcopy(value)
            change(trial)
            shows = not _probe_same(value, trial)
        except Exception:
            shows = True  # the change raises on this input, a read-only array for instance
        if shows:
            seen = _probe_visible.setdefault(key, {{}})
            seen[kind] = seen.get(kind, 0) + 1


def _probe_bound(args, kwargs):
    bound = dict(zip(_PROBE_PARAMS, args))
    bound.update(kwargs)
    return bound


def {entry}(*args, **kwargs):
    bound = _probe_bound(args, kwargs)
    try:
        result = _probe_reference(*args, **kwargs)
    except Exception:
        _probe_calls[1] += 1
        for key, value in bound.items():
            _probe_note(_probe_raised, key, value)
        raise
    _probe_calls[0] += 1
    for key, value in bound.items():
        _probe_note(_probe_ok, key, value)
        if _probe_calls[0] <= 400:
            _probe_visible_changes(key, value)
    _probe_note(_probe_ok, "return", result)
    leaves = _probe_ok.setdefault("return", {{}})
    for leaf in _probe_leaves(result):
        leaves[leaf] = leaves.get(leaf, 0) + 1
    if type(result) is tuple:
        for index, member in enumerate(result):
            _probe_note(_probe_ok, "return.%d" % index, member)
    return result


{entry}.__wrapped__ = _probe_reference
{entry}.__doc__ = _probe_reference.__doc__


def _probe_dump():
    path = _probe_os.environ.get("DOGMA_PROBE_OUT")
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            _probe_json.dump(
                {{
                    "params": _PROBE_PARAMS,
                    "ok": _probe_ok,
                    "raised": _probe_raised,
                    "first": _probe_first,
                    "calls": _probe_calls,
                    "visible": _probe_visible,
                }},
                handle,
            )


_probe_atexit.register(_probe_dump)
"""

# --- scribbles: the right result, one argument changed in place ----------------------------------

# Each body changes ``v`` in place and is written without imports, so that the probe passes the
# static checks of every runtime. Keys are (type name of the argument, probe name).
_SCRIBBLES: dict[tuple[str, str], str] = {
    ("ndarray", "any"): (
        "if v.dtype.kind == 'b':\n    v[...] = ~v\nelif v.dtype.kind in 'fiuc':\n    v[...] = v + 1"
    ),
    ("ndarray", "sort"): "if v.ndim >= 1 and v.dtype.kind in 'fiub':\n    v.sort(axis=-1)",
    ("ndarray", "abs"): "if v.dtype.kind in 'fi':\n    v[...] = abs(v)",
    ("ndarray", "centre"): "if v.dtype.kind == 'f' and v.size:\n    v -= v.mean()",
    ("ndarray", "scale"): (
        "if v.dtype.kind == 'f' and v.size:\n"
        "    m = abs(v).max()\n"
        "    if m > 0 and m == m and m != float('inf'):\n"
        "        v /= m"
    ),
    ("ndarray", "round"): "if v.dtype.kind == 'f':\n    v[...] = v.round(6)",
    ("ndarray", "float32"): "if v.dtype.kind == 'f':\n    v[...] = v.astype('float32')",
    ("Tensor", "any"): (
        "if str(v.dtype) == 'torch.bool':\n    v.data.logical_not_()\nelse:\n    v.data.add_(1)"
    ),
    ("Tensor", "sort"): (
        "if v.dim() >= 1 and str(v.dtype) != 'torch.bool':\n"
        "    v.data.copy_(v.data.sort(dim=-1).values)"
    ),
    ("Tensor", "abs"): "if str(v.dtype) != 'torch.bool':\n    v.data.abs_()",
    ("Tensor", "centre"): (
        "if v.is_floating_point() and v.numel():\n    v.data.sub_(v.data.mean())"
    ),
    ("Tensor", "scale"): (
        "if v.is_floating_point() and v.numel():\n"
        "    m = v.data.abs().max()\n"
        "    if 0 < float(m) < float('inf'):\n"
        "        v.data.div_(m)"
    ),
    ("list", "any"): "v.append(v[0] if v else 0)",
    ("list", "sort"): "try:\n    v.sort()\nexcept TypeError:\n    pass",
    ("list", "reverse"): "v.reverse()",
    ("list", "min_first"): (
        "try:\n"
        "    if len(v) > 1:\n"
        "        i = v.index(min(v))\n"
        "        v[0], v[i] = v[i], v[0]\n"
        "except TypeError:\n"
        "    pass"
    ),
    ("list", "inner"): (
        "for item in v:\n"
        "    kind = type(item).__name__\n"
        "    if kind == 'list' and item:\n"
        "        cell = item[0]\n"
        "        if isinstance(cell, bool):\n"
        "            item[0] = not cell\n"
        "        elif isinstance(cell, (int, float)):\n"
        "            item[0] = cell + 1\n"
        "        elif isinstance(cell, str):\n"
        "            item[0] = cell + '#'\n"
        "        else:\n"
        "            item.append(cell)\n"
        "        break\n"
        "    if kind == 'list':\n"
        "        item.append(0)\n"
        "        break\n"
        "    if kind == 'dict':\n"
        "        item['probe'] = 0\n"
        "        break\n"
        "    if kind == 'set':\n"
        "        item.add('probe')\n"
        "        break\n"
        "    if kind == 'ndarray' and item.size and item.dtype.kind in 'fiu':\n"
        "        item[...] = item + 1\n"
        "        break\n"
        "    if kind in ('Tensor', 'Parameter') and item.is_floating_point() and item.numel():\n"
        "        item.data.add_(1)\n"
        "        break"
    ),
    ("dict", "any"): "v['probe'] = 0",
    ("dict", "inner"): (
        "for item in v.values():\n"
        "    kind = type(item).__name__\n"
        "    if kind == 'list':\n"
        "        item.append(item[0] if item else 0)\n"
        "        break\n"
        "    if kind == 'dict':\n"
        "        item['probe'] = 0\n"
        "        break\n"
        "    if kind == 'set':\n"
        "        item.add('probe')\n"
        "        break\n"
        "    if kind == 'ndarray' and item.size and item.dtype.kind in 'fiu':\n"
        "        item[...] = item + 1\n"
        "        break\n"
        "    if kind in ('Tensor', 'Parameter') and item.is_floating_point() and item.numel():\n"
        "        item.data.add_(1)\n"
        "        break"
    ),
    ("set", "any"): "v.add('probe')",
    ("DataFrame", "any"): "v['probe'] = 0",
    ("DataFrame", "sort"): (
        "try:\n"
        "    v.sort_values(list(v.columns), inplace=True, kind='stable')\n"
        "except TypeError:\n"
        "    pass"
    ),
    ("DataFrame", "reset_index"): "v.reset_index(drop=True, inplace=True)",
    ("DataFrame", "dropna"): "v.dropna(inplace=True)",
    ("DataFrame", "drop_duplicates"): (
        "try:\n    v.drop_duplicates(inplace=True)\nexcept TypeError:\n    pass"
    ),
    ("DataFrame", "strings"): (
        "for column in v.columns:\n"
        "    values = v[column]\n"
        "    if values.dtype == object and all(isinstance(x, str) for x in values.dropna()):\n"
        "        v[column] = values.str.strip().str.lower()"
    ),
}
_SCRIBBLE_WORDS = {
    "any": "overwrites",
    "sort": "sorts",
    "abs": "takes the absolute value of",
    "centre": "subtracts the mean from",
    "scale": "rescales",
    "round": "rounds",
    "float32": "rounds to float32 the values of",
    "reverse": "reverses",
    "min_first": "moves the smallest item to the front of",
    "inner": "writes into an item nested in",
    "reset_index": "resets the index of",
    "dropna": "drops rows with a missing value from",
    "drop_duplicates": "drops repeated rows from",
    "strings": "strips and lower-cases the text of",
}
# A list or dict handed to a NumPy or torch function is copied by the first ``asarray``, so a
# change to the caller's list is only a realistic mistake where the argument is never an array.
_ARRAY_TYPES = ("ndarray", "Tensor", "Parameter", "DataFrame")

_SCRIBBLE_WRAPPER = '''


_correct = {entry}


def _change_in_place(v):
{body}


def {entry}(*args, **kwargs):
    """Wrong on purpose: the result is right, but this {words} `{name}` in place."""
    result = _correct(*args, **kwargs)
    target = args[{index}] if len(args) > {index} else kwargs.get({name!r})
    if type(target).__name__ in {types!r}:
        _change_in_place(target)
    return result


{entry}.__wrapped__ = _correct
'''

# --- forms: the reference, but it trips on one valid form of an argument -------------------------

# Each body reads ``target`` the way a common slip does, and so fails on the form its name says.
# No grading run is needed to know these survive: ``report`` lists the forms no test gives.
_FORMS: dict[str, tuple[str, str]] = {
    "needs_array": (
        "reads an array attribute before converting, so a list fails for",
        "if isinstance(target, (list, tuple)):\n    target.shape",
    ),
    "needs_float": (
        "centres a copy that keeps the integer dtype, so whole numbers fail for",
        "import numpy\n"
        "\n"
        "copy = numpy.array(target)\n"
        "if copy.dtype.kind in 'iu' and copy.size:\n"
        "    copy -= copy.mean()",
    ),
}

_FORM_WRAPPER = '''


_correct = {entry}


def {entry}(*args, **kwargs):
    """Wrong on purpose: this {words} `{name}`."""
    target = args[{index}] if len(args) > {index} else kwargs.get({name!r})
{body}
    return _correct(*args, **kwargs)


{entry}.__wrapped__ = _correct
'''

# --- returns: the right values in a laxer form ----------------------------------------------------

_RETURNS: dict[str, str] = {
    "array_as_list": (
        "if type(r).__name__ == 'ndarray' and r.ndim >= 1:\n    return r.tolist()\nreturn r"
    ),
    "float64_as_float32": (
        "if type(r).__name__ == 'ndarray' and str(r.dtype) == 'float64':\n"
        "    return r.astype('float32')\n"
        "return r"
    ),
    "int_as_float": (
        "if isinstance(r, int) and not isinstance(r, bool):\n"
        "    return float(r)\n"
        "if type(r).__name__ == 'ndarray' and r.dtype.kind in 'iu':\n"
        "    return r.astype('float64')\n"
        "return r"
    ),
    "float_as_int": (
        "if type(r) is float and r == r and abs(r) < 1e15 and r == int(r):\n"
        "    return int(r)\n"
        "return r"
    ),
    "float_as_one_element_array": (
        "if type(r) is float or (hasattr(r, 'dtype') and getattr(r, 'ndim', 1) == 0\n"
        "                        and r.dtype.kind == 'f' and type(r).__name__ != 'Tensor'):\n"
        "    import numpy\n"
        "\n"
        "    return numpy.array([float(r)])\n"
        "return r"
    ),
    "tensor_float32_as_float64": (
        "if type(r).__name__ == 'Tensor' and str(r.dtype) == 'torch.float32':\n"
        "    return r.double()\n"
        "return r"
    ),
    "tensor_detached": (
        "if type(r).__name__ == 'Tensor' and r.requires_grad:\n    return r.detach()\nreturn r"
    ),
    "frame_index_kept": (
        "if type(r).__name__ == 'DataFrame' and len(r):\n"
        "    r = r.copy()\n"
        "    r.index = range(1, len(r) + 1)\n"
        "return r"
    ),
    "frame_int_as_float": (
        "if type(r).__name__ == 'DataFrame':\n"
        "    r = r.copy()\n"
        "    for column in r.columns:\n"
        "        if r[column].dtype.kind in 'iu':\n"
        "            r[column] = r[column].astype('float64')\n"
        "return r"
    ),
    "frame_columns_reversed": (
        "if type(r).__name__ == 'DataFrame' and len(r.columns) > 1:\n"
        "    return r[list(r.columns[::-1])]\n"
        "return r"
    ),
    "frame_rows_reversed": (
        "if type(r).__name__ == 'DataFrame' and len(r) > 1:\n"
        "    return r.iloc[::-1].reset_index(drop=True)\n"
        "return r"
    ),
}
# Probes that act on the whole result only. The others reach into lists, tuples and dicts.
_TOP_ONLY: dict[str, str] = {
    "tuple_as_list": "return list(r) if type(r) is tuple else r",
    "list_as_tuple": "return tuple(r) if type(r) is list else r",
    "list_reversed": "return r[::-1] if type(r) is list and len(r) > 1 else r",
}
_RETURN_WORDS = {
    "array_as_list": "returns a nested list where an array is promised",
    "float64_as_float32": "returns float32 arrays where float64 is promised",
    "int_as_float": "returns floats where integers are promised",
    "float_as_int": "returns an int when the float result is a whole number",
    "float_as_one_element_array": "returns a one-element array where a float is promised",
    "tensor_float32_as_float64": "returns float64 tensors for float32 input",
    "tensor_detached": "returns a result cut off from the graph",
    "frame_index_kept": "returns the frame with an index that does not start at zero",
    "frame_int_as_float": "returns float columns where integer columns are promised",
    "frame_columns_reversed": "returns the columns in the opposite order",
    "frame_rows_reversed": "returns the rows in the opposite order",
    "tuple_as_list": "returns a list where a tuple is promised",
    "list_as_tuple": "returns a tuple where a list is promised",
    "list_reversed": "returns the list in the opposite order",
}

_RETURN_WRAPPER = '''


_correct = {entry}


def _laxer(r):
{body}


def {entry}(*args, **kwargs):
    """Wrong on purpose: {words}."""
    result = _correct(*args, **kwargs)
{apply}


{entry}.__wrapped__ = _correct
'''
_APPLY_MEMBERS = (
    "    return _everywhere(result)\n"
    "\n"
    "\n"
    "def _everywhere(r):\n"
    "    if type(r) is list:\n"
    "        return [_everywhere(item) for item in r]\n"
    "    if type(r) is tuple:\n"
    "        return tuple(_everywhere(item) for item in r)\n"
    "    if type(r) is dict:\n"
    "        return {key: _everywhere(item) for key, item in r.items()}\n"
    "    return _laxer(r)"
)
_APPLY_TOP = "    return _laxer(result)"


def _indent(body: str) -> str:
    return "\n".join("    " + line if line else line for line in body.splitlines())


# --- sources -----------------------------------------------------------------------------------


def entry_params(problem: audit.Problem) -> list[str] | None:
    """The parameter names of a function entry, in order. ``None`` for SQL and class entries."""
    if problem.language == "sql":
        return None
    tree = ast.parse(problem.solution.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == problem.entry:
            args = node.args
            return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return None


def _recorder_changes() -> str:
    """The scribble bodies as functions, registered by (type name, probe) for the recorder."""
    parts = []
    for (type_name, kind), body in _SCRIBBLES.items():
        name = f"_probe_change_{type_name}_{kind}"
        parts.append(f"\n\ndef {name}(v):\n{_indent(body)}\n\n")
        parts.append(f"\n_PROBE_CHANGES[({type_name!r}, {kind!r})] = {name}\n")
    return "".join(parts)


def recorder_source(problem: audit.Problem, params: Sequence[str]) -> str:
    source = problem.solution.read_text(encoding="utf-8")
    return source + _RECORDER.format(
        entry=problem.entry, params=list(params), changes=_recorder_changes()
    )


def scribble_source(
    problem: audit.Problem, params: Sequence[str], name: str, kind: str, type_name: str
) -> str:
    """The reference, followed by a wrapper that changes argument ``name`` in place."""
    types = ("Tensor", "Parameter") if type_name == "Tensor" else (type_name,)
    return problem.solution.read_text(encoding="utf-8") + _SCRIBBLE_WRAPPER.format(
        entry=problem.entry,
        body=_indent(_SCRIBBLES[(type_name, kind)]),
        words=_SCRIBBLE_WORDS[kind],
        name=name,
        index=list(params).index(name),
        types=types,
    )


def form_source(problem: audit.Problem, params: Sequence[str], name: str, kind: str) -> str:
    """The reference, behind a wrapper that fails on one valid form of argument ``name``."""
    words, body = _FORMS[kind]
    return problem.solution.read_text(encoding="utf-8") + _FORM_WRAPPER.format(
        entry=problem.entry,
        words=words,
        name=name,
        index=list(params).index(name),
        body=_indent(body),
    )


def returns_source(problem: audit.Problem, kind: str) -> str:
    """The reference, followed by a wrapper that hands its result back in a laxer form."""
    top = kind in _TOP_ONLY
    return problem.solution.read_text(encoding="utf-8") + _RETURN_WRAPPER.format(
        entry=problem.entry,
        body=_indent(_TOP_ONLY[kind] if top else _RETURNS[kind]),
        words=_RETURN_WORDS[kind],
        apply=_APPLY_TOP if top else _APPLY_MEMBERS,
    )


# --- record -------------------------------------------------------------------------------------


def record_problem(problem: audit.Problem, params: Sequence[str]) -> dict[str, Any]:
    """Plain pytest on both test files with the recorder as the submission."""
    with tempfile.TemporaryDirectory(prefix="probe-record-") as tmp:
        root = Path(tmp)
        (root / "submission.py").write_text(recorder_source(problem, params), encoding="utf-8")
        for name in _TEST_FILES:
            if (problem.path / name).is_file():
                shutil.copy(problem.path / name, root / name)
        if (problem.path / "fixtures").is_dir():
            shutil.copytree(problem.path / "fixtures", root / "fixtures")
        names = [name for name in _TEST_FILES if (root / name).is_file()]
        out = root / "probe.json"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=line",
            "-p",
            "no:cacheprovider",
            "--hypothesis-seed=0",
            *names,
        ]
        env = {**os.environ, "PYTHONHASHSEED": "0", "DOGMA_PROBE_OUT": str(out)}
        try:
            done = subprocess.run(
                command,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=problem.timeout_s * 6,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"problem": problem.id, "status": "timeout"}
        if not out.is_file():
            return {"problem": problem.id, "status": "no-record", "tail": done.stdout[-400:]}
        data = json.loads(out.read_text(encoding="utf-8"))
    status = "passed" if done.returncode == 0 else "failed"
    return {"problem": problem.id, "status": status, **data}


def cmd_record(problems: Sequence[audit.Problem], out: Path, workers: int) -> int:
    log = audit.Log(out / "record.jsonl", lambda row: row["problem"])
    skipped = 0

    def job(problem: audit.Problem, params: list[str]) -> Callable[[], None]:
        def run() -> None:
            log.write(record_problem(problem, params))

        return run

    jobs = []
    for problem in problems:
        params = entry_params(problem)
        if params is None:
            skipped += 1
        elif problem.id not in log.done:
            jobs.append(job(problem, params))
    print(f"record: {len(jobs)} to run, {skipped} skipped (SQL or class entry)", flush=True)
    audit._run_all(jobs, workers)
    rows = audit.read_jsonl(log.path)
    bad = [row for row in rows if row["status"] != "passed"]
    print(f"record: {len(rows) - len(bad)}/{len(rows)} recorded with every test passing")
    for row in bad:
        print(f"  {row['problem']}: {row['status']}")
    return 1 if bad else 0


def _records(out: Path) -> dict[str, dict[str, Any]]:
    return {row["problem"]: row for row in audit.read_jsonl(out / "record.jsonl")}


def _types_seen(row: dict[str, Any], key: str) -> set[str]:
    return {feat[5:] for feat in row.get("ok", {}).get(key, {}) if feat.startswith("type_")}


# --- scribble and returns -----------------------------------------------------------------------


def scribble_plan(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    """(argument, type name, probe) for every in-place change that can apply to this problem."""
    plan: list[tuple[str, str, str]] = []
    for name in row.get("params", []):
        types = _types_seen(row, name)
        if "Parameter" in types:
            types = (types - {"Parameter"}) | {"Tensor"}
        array_like = bool(types & set(_ARRAY_TYPES))
        for (type_name, kind), _body in _SCRIBBLES.items():
            if type_name not in types:
                continue
            if type_name in ("list", "dict", "set") and array_like:
                continue
            plan.append((name, type_name, kind))
    return plan


def returns_plan(row: dict[str, Any]) -> list[str]:
    """The laxer return forms that would change something the reference returned."""
    seen = row.get("ok", {}).get("return", {})
    top = _types_seen(row, "return")
    leaves = {feat[5:] for feat in seen if feat.startswith("leaf_")}
    members = set(top)
    for key in row.get("ok", {}):
        if key.startswith("return."):
            members |= _types_seen(row, key)
    plan: list[str] = []
    if "ndarray" in members:
        plan.append("array_as_list")
    if "ndarray" in members and "ndarray_f" in leaves:
        plan.append("float64_as_float32")
    if leaves & {"int", "ndarray_i", "ndarray_u"}:
        plan.append("int_as_float")
    if members & {"float", "float64"}:
        plan.append("float_as_int")
        plan.append("float_as_one_element_array")
    if "tensor_float32" in leaves or "tensor_float32_grad" in leaves:
        plan.append("tensor_float32_as_float64")
    if any(leaf.startswith("tensor_") and leaf.endswith("_grad") for leaf in leaves):
        plan.append("tensor_detached")
    if "DataFrame" in members:
        plan.extend(name for name in _RETURNS if name.startswith("frame_"))
    if "tuple" in top:
        plan.append("tuple_as_list")
    if "list" in top:
        plan.extend(["list_as_tuple", "list_reversed"])
    return plan


def _killers(report: Any) -> list[str]:
    return [t.name for t in report.tests if t.outcome in ("fail", "error")]


def _grade_probes(
    problems: Sequence[audit.Problem],
    out: Path,
    workers: int,
    family: str,
    sources: Callable[[audit.Problem, dict[str, Any]], list[tuple[str, str]]],
) -> int:
    """Grade every (probe name, source) of every recorded problem on the learner's real path."""
    settings = Settings()
    records = _records(out)
    log = audit.Log(out / f"{family}.jsonl", lambda row: f"{row['problem']}:{row['probe']}")

    def job(problem: audit.Problem, probe: str, source: str) -> Callable[[], None]:
        def run() -> None:
            if not audit._passes_static(problem, source):
                log.write({"problem": problem.id, "probe": probe, "verdict": "rejected"})
                return
            report = audit.grade_source(problem, source, settings)
            log.write(
                {
                    "problem": problem.id,
                    "probe": probe,
                    **audit.record(report),
                    "killers": _killers(report)[:8],
                }
            )

        return run

    jobs = []
    for problem in problems:
        row = records.get(problem.id)
        if row is None or row["status"] != "passed":
            continue
        for probe, source in sources(problem, row):
            if f"{problem.id}:{probe}" not in log.done:
                jobs.append(job(problem, probe, source))
    print(f"{family}: {len(jobs)} probes to grade", flush=True)
    audit._run_all(jobs, workers)
    rows = audit.read_jsonl(log.path)
    survived = [row for row in rows if row["verdict"] == "survived"]
    print(f"{family}: {len(rows)} probes, {len(survived)} survived")
    return 0


def scribble_names(row: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """(probe name, argument, type name, probe kind). The name is ``ARG:KIND``.

    An argument the tests give in two containers (a list and a set) has one probe per container;
    the second one is named ``ARG:KIND@TYPE``.
    """
    taken: set[str] = set()
    named: list[tuple[str, str, str, str]] = []
    for name, type_name, kind in scribble_plan(row):
        probe = f"{name}:{kind}"
        if probe in taken:
            probe = f"{probe}@{type_name}"
        taken.add(probe)
        named.append((probe, name, type_name, kind))
    return named


def _scribble_sources(problem: audit.Problem, row: dict[str, Any]) -> list[tuple[str, str]]:
    params = row["params"]
    return [
        (probe, scribble_source(problem, params, name, kind, type_name))
        for probe, name, type_name, kind in scribble_names(row)
    ]


def _returns_sources(problem: audit.Problem, row: dict[str, Any]) -> list[tuple[str, str]]:
    return [(kind, returns_source(problem, kind)) for kind in returns_plan(row)]


def cmd_scribble(problems: Sequence[audit.Problem], out: Path, workers: int) -> int:
    return _grade_probes(problems, out, workers, "scribble", _scribble_sources)


def cmd_returns(problems: Sequence[audit.Problem], out: Path, workers: int) -> int:
    return _grade_probes(problems, out, workers, "returns", _returns_sources)


# --- report and write ---------------------------------------------------------------------------


def _clauses(problem: audit.Problem) -> list[tuple[str, str]]:
    import exercise_gate as gate  # noqa: PLC0415 - the gate imports the audit; keep this lazy

    return gate.contract_clauses((problem.path / "problem.md").read_text(encoding="utf-8"))


def mutation_clause(problem: audit.Problem) -> tuple[str, str] | None:
    """The first Contract bullet that speaks about leaving the inputs alone, if any."""
    for label, text in _clauses(problem):
        if label.startswith("C") and _MUTATION_WORDS.search(text):
            return label, text
    return None


# Input classes worth a look when no passing call was given one, by the kind of argument. A
# class that only raising calls were given is reported apart: the tests reach it, as an error.
_NUMERIC_CLASSES = ("empty", "one_row", "negative", "zero", "ties", "unsorted")
_FRAME_CLASSES = ("empty", "one_row", "nan", "duplicate_rows", "index_not_range")
_ONE = {"one_row": ("one_row", "one_item"), "unsorted": ("unsorted", "descending")}


def _gaps(ok: dict[str, int], raised: dict[str, int]) -> tuple[list[str], list[str]]:
    """(classes no call was given, classes only a raising call was given) for one argument."""
    types = {feat[5:] for feat in ok if feat.startswith("type_")}
    arrays = bool(types & {"ndarray", "Tensor", "Parameter"})
    numbers = arrays or any(f in ok for f in ("ints_only", "negative", "zero", "ascending"))
    one_dimensional = "ndim1" in ok or (not arrays and ("ties" in ok or "distinct" in ok))
    if "DataFrame" in types:
        wanted: tuple[str, ...] = _FRAME_CLASSES
    elif numbers and one_dimensional:
        wanted = _NUMERIC_CLASSES
    elif numbers:
        wanted = ("empty", "one_row", "negative", "zero")
    elif types & {"list", "tuple"} and one_dimensional:
        wanted = ("empty", "one_row", "ties")
    elif types & {"list", "tuple", "dict", "set"}:
        wanted = ("empty", "one_row")
    else:
        return [], []
    never: list[str] = []
    only_raised: list[str] = []
    for name in wanted:
        names = _ONE.get(name, (name,))
        if not any(n in ok for n in names):
            (only_raised if any(n in raised for n in names) else never).append(name)
    if "ndarray" in types and not types & {"list", "tuple"}:
        never.append("as_list")
    if "ndarray" in types and "dtype_i" not in ok and "ints_only" not in ok:
        never.append("integers")
    if types & {"Tensor", "Parameter"} and "noncontiguous" not in ok:
        never.append("noncontiguous")
    return never, only_raised


def visible_calls(row: dict[str, Any], probe: str) -> int:
    """In how many recorded calls the in-place change of ``probe`` would have shown."""
    name, _, kind = probe.partition(":")
    count = row.get("visible", {}).get(name, {}).get(kind.split("@")[0], 0)
    return int(count)


def survivors(out: Path, family: str, ids: set[str]) -> tuple[int, dict[str, list[str]]]:
    """(probes graded, surviving probe names by problem) for one family."""
    rows = [r for r in audit.read_jsonl(out / f"{family}.jsonl") if r["problem"] in ids]
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["verdict"] == "survived":
            grouped[row["problem"]].append(row["probe"])
    return len(rows), grouped


def _report_inputs(records: dict[str, dict[str, Any]], ids: set[str]) -> None:
    print("## inputs: classes no passing call was given (after the bar: only calls that raise)")
    for problem_id in sorted(records):
        row = records[problem_id]
        if problem_id not in ids or row["status"] != "passed":
            continue
        parts = []
        for name in row["params"]:
            never, only_raised = _gaps(row["ok"].get(name, {}), row["raised"].get(name, {}))
            if never or only_raised:
                parts.append(f"{name}: {' '.join(never)} | {' '.join(only_raised)}")
        if parts:
            print(f"- {problem_id}")
            for part in parts:
                print(f"    {part}")
    print()


def cmd_report(problems: Sequence[audit.Problem], out: Path) -> int:
    """Print what the recordings and the graded probes show. A survivor is a candidate."""
    by_id = {problem.id: problem for problem in problems}
    ids = set(by_id)
    records = _records(out)
    _report_inputs(records, ids)

    graded, grouped = survivors(out, "scribble", ids)
    shown = {
        problem_id: [p for p in probes if visible_calls(records.get(problem_id, {}), p)]
        for problem_id, probes in grouped.items()
    }
    shown = {problem_id: probes for problem_id, probes in shown.items() if probes}
    total = sum(len(probes) for probes in grouped.values())
    seen = sum(len(probes) for probes in shown.values())
    print(f"## scribble: {graded} probes, {total} survived in {len(grouped)} problems")
    print(f"## scribble, change shows on a tested input: {seen} in {len(shown)} problems")
    for problem_id in sorted(shown):
        clause = mutation_clause(by_id[problem_id])
        label = clause[0] if clause else "no clause"
        print(f"- {problem_id} ({label}): {', '.join(sorted(shown[problem_id]))}")
    print(f"## scribble, change never shows on a tested input: {total - seen}")
    for problem_id in sorted(grouped):
        rest = sorted(set(grouped[problem_id]) - set(shown.get(problem_id, [])))
        if rest:
            print(f"- {problem_id}: {', '.join(rest)}")
    print()

    graded, grouped = survivors(out, "returns", ids)
    total = sum(len(probes) for probes in grouped.values())
    print(f"## returns: {graded} probes, {total} survived in {len(grouped)} problems")
    for problem_id in sorted(grouped):
        print(f"- {problem_id}: {', '.join(sorted(grouped[problem_id]))}")
    print()
    return 0


def probe_source(problem: audit.Problem, out: Path, probe: str) -> str:
    """The source of one probe: ``ARG:KIND`` for a scribble, a bare name for a return form."""
    if ":" not in probe:
        if probe not in _RETURNS and probe not in _TOP_ONLY:
            raise SystemExit(f"unknown return probe {probe}")
        return returns_source(problem, probe)
    row = _records(out).get(problem.id)
    if row is None:
        raise SystemExit(f"no record for {problem.id}: run `record` first")
    for named, name, type_name, kind in scribble_names(row):
        if named == probe:
            return scribble_source(problem, row["params"], name, kind, type_name)
    argument, _, form = probe.partition(":")
    if form in _FORMS and argument in row["params"]:
        return form_source(problem, row["params"], argument, form)
    raise SystemExit(f"{probe} does not apply to {problem.id}")


def cmd_write(
    problems: Sequence[audit.Problem], out: Path, problem_id: str, probe: str, target: Path
) -> int:
    matches = [problem for problem in problems if problem.id == problem_id]
    if not matches:
        raise SystemExit(f"unknown problem {problem_id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(probe_source(matches[0], out, probe), encoding="utf-8")
    print(f"wrote {target}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["record", "scribble", "returns", "report", "write"])
    parser.add_argument("args", nargs="*", help="write: PROBLEM_ID PROBE FILE")
    parser.add_argument("--out", type=Path, required=True, help="directory for results")
    parser.add_argument("--only", action="append", default=[], help="problem id prefix")
    parser.add_argument("--workers", type=int, default=10)
    ns = parser.parse_args(argv)
    problems = audit.discover(audit.REPO, ns.only)
    ns.out.mkdir(parents=True, exist_ok=True)
    if ns.command == "record":
        return cmd_record(problems, ns.out, ns.workers)
    if ns.command == "scribble":
        return cmd_scribble(problems, ns.out, ns.workers)
    if ns.command == "returns":
        return cmd_returns(problems, ns.out, ns.workers)
    if ns.command == "report":
        return cmd_report(problems, ns.out)
    if len(ns.args) != len(("id", "probe", "file")):
        parser.error("write takes PROBLEM_ID PROBE FILE")
    problem_id, probe, target = ns.args
    return cmd_write(audit.discover(audit.REPO, []), ns.out, problem_id, probe, Path(target))


if __name__ == "__main__":
    raise SystemExit(main())
