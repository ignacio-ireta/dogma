"""Diff classifier: added/updated/unchanged/moved/deleted resolution and move ambiguity."""

from __future__ import annotations

from docmd.tracking.diff import HashedFile, classify_changes
from docmd.tracking.snapshot import SnapshotEntry


def make_previous(files: dict[str, str]) -> dict[str, SnapshotEntry]:
    return {
        path: SnapshotEntry(sha256=sha, size=len(sha), output=path + ".md", last_run_id="r0")
        for path, sha in files.items()
    }


def make_current(files: dict[str, str]) -> dict[str, HashedFile]:
    return {
        path: HashedFile(relative_path=path, sha256=sha, size=len(sha))
        for path, sha in files.items()
    }


def test_classify_first_sync_all_added() -> None:
    plan = classify_changes({}, make_current({"b.docx": "h2", "a/x.pdf": "h1", "c.md": "h3"}))
    assert plan.added == ["a/x.pdf", "b.docx", "c.md"]
    assert plan.updated == []
    assert plan.unchanged == []
    assert plan.moved == []
    assert plan.deleted == []


def test_classify_both_trees_empty_all_empty() -> None:
    plan = classify_changes({}, {})
    assert plan.added == plan.updated == plan.unchanged == plan.deleted == []
    assert plan.moved == []


def test_classify_new_path_new_content_added() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "h1"}),
        make_current({"a.docx": "h1", "b.docx": "h2"}),
    )
    assert plan.added == ["b.docx"]
    assert plan.unchanged == ["a.docx"]
    assert plan.updated == [] and plan.moved == [] and plan.deleted == []


def test_classify_same_path_new_hash_updated() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "h1"}),
        make_current({"a.docx": "h2"}),
    )
    assert plan.updated == ["a.docx"]
    assert plan.added == [] and plan.unchanged == [] and plan.moved == [] and plan.deleted == []


def test_classify_same_path_same_hash_unchanged() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "h1"}),
        make_current({"a.docx": "h1"}),
    )
    assert plan.unchanged == ["a.docx"]
    assert plan.added == [] and plan.updated == [] and plan.moved == [] and plan.deleted == []


def test_classify_missing_path_deleted() -> None:
    plan = classify_changes(make_previous({"a.docx": "h1"}), {})
    assert plan.deleted == ["a.docx"]
    assert plan.added == [] and plan.updated == [] and plan.unchanged == [] and plan.moved == []


def test_classify_unique_hash_pair_moved_not_added_or_deleted() -> None:
    plan = classify_changes(
        make_previous({"old/a.docx": "h1"}),
        make_current({"new/b.docx": "h1"}),
    )
    assert plan.moved == [("old/a.docx", "new/b.docx")]
    assert plan.added == []
    assert plan.deleted == []
    assert plan.updated == [] and plan.unchanged == []


def test_classify_two_deleted_one_added_same_hash_falls_back_to_delete_add() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "dup", "b.docx": "dup"}),
        make_current({"c.docx": "dup"}),
    )
    assert plan.moved == []
    assert plan.deleted == ["a.docx", "b.docx"]
    assert plan.added == ["c.docx"]


def test_classify_one_deleted_two_added_same_hash_falls_back_to_delete_add() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "dup"}),
        make_current({"b.docx": "dup", "c.docx": "dup"}),
    )
    assert plan.moved == []
    assert plan.deleted == ["a.docx"]
    assert plan.added == ["b.docx", "c.docx"]


def test_classify_content_swap_between_existing_paths_both_updated_never_moved() -> None:
    plan = classify_changes(
        make_previous({"a.docx": "h1", "b.docx": "h2"}),
        make_current({"a.docx": "h2", "b.docx": "h1"}),
    )
    assert plan.updated == ["a.docx", "b.docx"]
    assert plan.moved == []
    assert plan.added == [] and plan.unchanged == [] and plan.deleted == []


def test_classify_update_plus_move_onto_old_path_hash_not_move_candidate() -> None:
    # A's old content (h1) reappears at B, but A still exists: h1 must not pair as a move.
    plan = classify_changes(
        make_previous({"a.docx": "h1"}),
        make_current({"a.docx": "h2", "b.docx": "h1"}),
    )
    assert plan.updated == ["a.docx"]
    assert plan.added == ["b.docx"]
    assert plan.moved == []
    assert plan.unchanged == [] and plan.deleted == []


def test_classify_deleted_content_lands_on_surviving_path_updated_plus_deleted() -> None:
    # B's content (h2) now lives at still-existing A: A is an update, B a plain delete.
    plan = classify_changes(
        make_previous({"a.docx": "h1", "b.docx": "h2"}),
        make_current({"a.docx": "h2"}),
    )
    assert plan.updated == ["a.docx"]
    assert plan.deleted == ["b.docx"]
    assert plan.moved == []
    assert plan.added == [] and plan.unchanged == []


def test_classify_output_lists_sorted_regardless_of_insertion_order() -> None:
    # Insert every mapping in reverse-sorted order to prove ordering is imposed, not inherited.
    previous = make_previous(
        {
            "z_same.md": "same-z",
            "z_moved_old.md": "mv-z",
            "z_gone.md": "gone-z",
            "y_upd.md": "old-y",
            "a_same.md": "same-a",
            "a_moved_old.md": "mv-a",
            "a_gone.md": "gone-a",
            "b_upd.md": "old-b",
        }
    )
    current = make_current(
        {
            "z_same.md": "same-z",
            "z_new.md": "new-z",
            "y_upd.md": "new-y",
            "b_upd.md": "new-b",
            "b_moved_new.md": "mv-z",
            "a_same.md": "same-a",
            "a_new.md": "new-a",
            "a_moved_new.md": "mv-a",
        }
    )
    plan = classify_changes(previous, current)
    assert plan.added == ["a_new.md", "z_new.md"]
    assert plan.updated == ["b_upd.md", "y_upd.md"]
    assert plan.unchanged == ["a_same.md", "z_same.md"]
    assert plan.deleted == ["a_gone.md", "z_gone.md"]
    # moved sorts by (old, new): a_moved_old precedes z_moved_old even though
    # z_moved_old's destination (b_moved_new) sorts before a_moved_old's.
    assert plan.moved == [
        ("a_moved_old.md", "a_moved_new.md"),
        ("z_moved_old.md", "b_moved_new.md"),
    ]


def test_classify_mixed_all_five_kinds_at_once() -> None:
    previous = make_previous(
        {
            "keep.md": "h-keep",
            "edit.md": "h-old",
            "moved/from.md": "h-move",
            "gone.md": "h-gone",
            "dup1.md": "h-dup",
            "dup2.md": "h-dup",
        }
    )
    current = make_current(
        {
            "keep.md": "h-keep",
            "edit.md": "h-new",
            "moved/to.md": "h-move",
            "fresh.md": "h-fresh",
            "dup3.md": "h-dup",
        }
    )
    plan = classify_changes(previous, current)
    assert plan.unchanged == ["keep.md"]
    assert plan.updated == ["edit.md"]
    assert plan.moved == [("moved/from.md", "moved/to.md")]
    # ambiguous h-dup pair falls back to delete+add alongside the plain ones
    assert plan.deleted == ["dup1.md", "dup2.md", "gone.md"]
    assert plan.added == ["dup3.md", "fresh.md"]
