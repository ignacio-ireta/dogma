"""Tests for scripts/audit_privacy.py. Uses tmp_path only; never reads the real denylist.

Test strings avoid literal email addresses and phone numbers (they are assembled from parts), so
that this file itself stays clean under `make audit`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "audit_privacy.py"


def load_audit() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_privacy", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


audit_mod = load_audit()


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def denylist(tmp_path: Path, text: str) -> Path:
    return write(tmp_path / "private", "denylist.txt", text)


def labels_for(text: str, rules: list[Any]) -> list[str]:
    return [hit.label for hit in audit_mod.scan_text("f.md", text, rules)]


def test_denylist_matches_case_insensitively_by_default(tmp_path: Path) -> None:
    rules = audit_mod.load_denylist(denylist(tmp_path, "zqxcorp\n"))
    assert labels_for("Works at ZQXCORP today", rules) == ["private rule 1"]


def test_case_sensitive_prefix_is_honoured(tmp_path: Path) -> None:
    rules = audit_mod.load_denylist(denylist(tmp_path, "(?-i)\\bQzv\\b\n"))
    assert labels_for("Qzv is flagged", rules) == ["private rule 1"]
    assert labels_for("qzv is not", rules) == []


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    rules = audit_mod.load_denylist(denylist(tmp_path, "# a comment\n\n   \nzqxcorp\n"))
    assert [rule.label for rule in rules] == ["private rule 4"]


def test_invalid_regex_exits_with_code_two(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        audit_mod.load_denylist(denylist(tmp_path, "(unclosed\n"))
    assert excinfo.value.code == 2


def test_builtin_email_flags_personal_domains_but_not_examples() -> None:
    rules = audit_mod.builtin_rules()
    personal = "reach me at " + "person" + "@" + "gmail.com"
    assert "builtin: email" in labels_for(personal, rules)
    assert labels_for("contact user@example.com or a@b.example.org", rules) == []
    assert labels_for("reserved: x@host.test and y@svc.invalid", rules) == []


def test_builtin_phone_requires_leading_plus() -> None:
    rules = audit_mod.builtin_rules()
    assert "builtin: phone" in labels_for("call " + "+" + "44 20 7946 0958", rules)
    assert labels_for("the sum is x + 1 = 2 and 2024-02-14", rules) == []


def test_history_free_discovery_covers_all_public_folders(tmp_path: Path) -> None:
    write(tmp_path, "sources/a.md", "x")
    write(tmp_path, "docs/b.md", "x")
    write(tmp_path, "random/c.md", "x")
    write(tmp_path, "tools/vendor/VENDORED.md", "x")
    write(tmp_path, "tools/vendor/README.md", "x")
    write(tmp_path, "README.md", "x")
    write(tmp_path, "private/d.md", "x")
    names = {p.relative_to(tmp_path).as_posix() for p in audit_mod.scan_targets(tmp_path)}
    assert names == {
        "sources/a.md",
        "docs/b.md",
        "tools/vendor/VENDORED.md",
        "README.md",
        "random/c.md",
        "tools/vendor/README.md",
    }


def test_git_publication_files_outside_the_standard_folders_are_checked(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "extra/note.md", "ZQXCORP\n")
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert any(hit.path == "extra/note.md" for hit in hits)


def test_history_free_exports_include_vendored_and_unconventional_files(tmp_path: Path) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    for name in ["extra/note.md", "tools/vendor/metadata.toml"]:
        write(tmp_path, name, "zqxcorp\n")
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert {hit.path for hit in hits} == {"extra/note.md", "tools/vendor/metadata.toml"}


@pytest.mark.parametrize("directory", ["private", "var", "workspace"])
def test_state_named_folders_nested_in_public_sources_are_checked(
    tmp_path: Path, directory: str
) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    name = f"extra/{directory}/note.md"
    write(tmp_path, name, "zqxcorp\n")
    assert any(hit.path == name for hit in audit_mod.audit(tmp_path, deny)[0])


def test_json_slash_escaping_cannot_hide_an_account_path(tmp_path: Path) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    path = "/" + "/".join(["Users", "sample-account", "Documents"])
    escaped = path.replace("/", chr(92) + "/")
    write(tmp_path, "web/public/account.json", '{"path": "' + escaped + '"}')
    assert any(hit.label == "builtin: account path" for hit in audit_mod.audit(tmp_path, deny)[0])


def test_publication_symlinks_are_rejected_without_reading_the_target(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    target = write(tmp_path, "private/hidden.md", "zqxcorp\n")
    (tmp_path / "link.md").symlink_to(target)
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert [(hit.path, hit.label) for hit in hits] == [("link.md", "builtin: publication symlink")]


@pytest.mark.parametrize(
    ("relative", "directory", "dangling"),
    [
        ("web/dist", True, False),
        ("web/dist/assets", True, False),
        ("web/dist/assets/leak.js", False, False),
        ("web/dist/assets/missing.js", False, True),
    ],
)
def test_ignored_build_symlinks_block_certification_without_reading_targets(
    tmp_path: Path, relative: str, directory: bool, dangling: bool
) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n/web/dist\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    target = write(tmp_path, "private/hidden.js", "zqxcorp\n")
    if directory:
        target = target.parent
    elif dangling:
        target = target.with_name("missing.js")
    link = tmp_path / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=directory)

    hits, scanned = audit_mod.audit(tmp_path, deny)

    assert [(hit.path, hit.label) for hit in hits] == [(relative, "builtin: publication symlink")]
    assert scanned == 1, "only .gitignore is text; symlink targets must never be read"


def test_staged_audit_checks_index_bytes_when_the_worktree_is_clean(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    path = write(tmp_path, "docs/note.md", "zqxcorp\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "docs/note.md"], check=True)
    path.write_text("Generic text.\n")
    hits, _ = audit_mod.audit(tmp_path, deny, staged=True)
    assert any(hit.path == "docs/note.md" and hit.label == "private rule 1" for hit in hits)


def test_staged_private_paths_are_rejected_even_when_deleted_from_disk(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    path = write(tmp_path, "private/hidden.md", "zqxcorp\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "--force", "private/hidden.md"], check=True)
    path.unlink()
    hits, _ = audit_mod.audit(tmp_path, deny, staged=True)
    assert [(hit.path, hit.label) for hit in hits] == [
        ("private/hidden.md", "builtin: forbidden publication")
    ]


@pytest.mark.parametrize("encoding", ["unicode-escape", "html", "utf-16"])
def test_serialized_private_text_is_checked(tmp_path: Path, encoding: str) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    if encoding == "unicode-escape":
        value = "".join(chr(92) + f"u{ord(char):04x}" for char in "zqxcorp")
        write(tmp_path, "web/public/metadata.json", '{"owner": "' + value + '"}')
    elif encoding == "html":
        write(tmp_path, "web/public/note.html", "".join(f"&#{ord(char)};" for char in "zqxcorp"))
    else:
        write(tmp_path, "docs/note.txt", "").write_bytes("zqxcorp".encode("utf-16"))
    assert audit_mod.audit(tmp_path, deny)[0]


def test_force_tracked_private_files_block_publication_without_reading_them(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path, ".gitignore", "/private/\n")
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "private/snapshot.txt", "zqxcorp\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--force", "private/snapshot.txt"], check=True
    )
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert [(hit.path, hit.label) for hit in hits] == [
        ("private/snapshot.txt", "builtin: forbidden publication")
    ]


def test_binary_bodies_are_not_text_and_oversized_files_block_certification(tmp_path: Path) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    binary = tmp_path / "sources" / "blob.bin"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x00zqxcorp")
    big = tmp_path / "sources" / "big.md"
    big.write_bytes(b"zqxcorp" + b"a" * (audit_mod.MAX_BYTES + 1))
    hits, scanned = audit_mod.audit(tmp_path, deny)
    assert [(hit.path, hit.label) for hit in hits] == [
        ("sources/big.md", "builtin: file size limit")
    ]
    assert scanned == 0


def test_main_reports_hits_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "sources/note.md", "line one\nline two mentions ZQXCORP\n")
    code = audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)])
    out = capsys.readouterr().out
    assert code == 1
    assert "sources/note.md:2: private rule 1" in out

    (tmp_path / "sources" / "note.md").write_text("clean\n", encoding="utf-8")
    assert audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)]) == 0


def test_private_is_never_scanned(tmp_path: Path) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "private/notes.md", "zqxcorp")
    assert audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)]) == 0


@pytest.mark.parametrize("target", ["web/public/note.md", "corpus/README.md"])
def test_symlinked_parent_cannot_expose_private_files(tmp_path: Path, target: str) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    first, remainder = target.split("/", 1)
    write(tmp_path, "private/" + remainder, "zqxcorp\n")
    (tmp_path / first).symlink_to(tmp_path / "private", target_is_directory=True)
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert [(hit.path, hit.label) for hit in hits] == (
        [("web", "builtin: publication symlink")] if first == "web" else []
    )


def test_missing_denylist_refuses_to_certify_the_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "docs/x.md", "reach " + "person" + "@" + "gmail.com")
    code = audit_mod.main(["--root", str(tmp_path), "--denylist", str(tmp_path / "missing.txt")])
    captured = capsys.readouterr()
    assert code == 2
    assert "privacy rules are required" in captured.err
    assert "audit: clean" not in captured.out


def test_empty_denylist_refuses_to_certify_the_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deny = denylist(tmp_path, "# no configured terms\n")
    assert audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)]) == 2
    assert "privacy rules are required" in capsys.readouterr().err


def test_oversized_text_cannot_silently_bypass_the_audit(tmp_path: Path) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "sources/large.md", "x" * (audit_mod.MAX_BYTES + 1))
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert any(hit.label == "builtin: file size limit" for hit in hits)


@pytest.mark.parametrize("location", ["public", "dist"])
def test_modified_license_cannot_hide_a_new_contact_address(tmp_path: Path, location: str) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(
        tmp_path,
        f"web/{location}/fonts/opendyslexic/OFL.txt",
        "Changed license with " + "person" + "@" + "gmail.com\n",
    )
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert any(hit.label == "builtin: email" for hit in hits)


def test_verified_parser_data_does_not_exempt_modified_data_or_surrounding_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    literal = json.dumps("compressed:data:zqxcorp:table")
    monkeypatch.setattr(
        audit_mod,
        "VERIFIED_PARSER_LITERALS",
        frozenset({hashlib.sha256(literal.encode()).hexdigest()}),
        raising=False,
    )
    path = write(tmp_path, "web/dist/assets/app.js", f"const table = {{stateData: {literal}}};\n")
    assert audit_mod.audit(tmp_path, deny)[0] == []
    path.write_text(f"const table = {{stateData: {literal}}}; const owner = 'zqxcorp';\n")
    assert audit_mod.audit(tmp_path, deny)[0]
    path.write_text(f"const table = {{stateData: {json.dumps('modified:data:zqxcorp:table')}}};\n")
    assert audit_mod.audit(tmp_path, deny)[0]


def test_private_terms_in_filenames_are_rejected_without_echoing_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "sources/zqxcorp.md", "Generic subject material.\n")
    assert audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)]) == 1
    output = capsys.readouterr().out
    assert "zqxcorp" not in output
    assert "sources/[redacted].md:0: private rule 1" in output


@pytest.mark.parametrize(
    "relative",
    [
        "web/public/data.json",
        "web/dist/assets/app.js",
        "corpus/README.md",
        "pyproject.toml",
        ".env.example",
        "Makefile",
        ".pre-commit-config.yaml",
    ],
)
def test_publishable_assets_and_root_metadata_are_checked(tmp_path: Path, relative: str) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, relative, "Contains ZQXCORP.\n")
    hits, _ = audit_mod.audit(tmp_path, deny)
    assert any(hit.path == relative for hit in hits)


@pytest.mark.parametrize("platform", ["unix", "windows"])
def test_builtin_rules_reject_local_account_paths(platform: str) -> None:
    path = (
        "/" + "/".join(["Users", "sample-account", "Documents", "note.md"])
        if platform == "unix"
        else "C:" + chr(92) + chr(92).join(["Users", "sample-account", "Documents", "note.md"])
    )
    assert "builtin: account path" in labels_for(path, audit_mod.builtin_rules())


def test_private_rule_diagnostics_never_print_the_term(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    deny = denylist(tmp_path, "zqxcorp\n")
    write(tmp_path, "docs/note.md", "ZQXCORP\n")
    assert audit_mod.main(["--root", str(tmp_path), "--denylist", str(deny)]) == 1
    output = capsys.readouterr().out
    assert "private rule 1" in output
    assert "zqxcorp" not in output.lower()


def test_invalid_private_regex_diagnostics_hide_pattern_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = denylist(tmp_path, "(?P<zqxcorp>)(?P<zqxcorp>)\n")
    with pytest.raises(SystemExit) as caught:
        audit_mod.load_denylist(path)
    assert caught.value.code == 2
    assert "zqxcorp" not in capsys.readouterr().err
