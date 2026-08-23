from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOY_CORPUS = ROOT / "data/toy/corpus.jsonl"


def _toy_doc_ids() -> set[str]:
    ids = set()

    with TOY_CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            # The starter corpus format uses doc_id.  Keep this tolerant
            # of the common "_id" spelling too.
            doc_id = obj.get("doc_id", obj.get("_id"))

            if doc_id is not None:
                ids.add(str(doc_id))

    return ids


def test_submission_imports():
    import submission.retrieve  # noqa: F401


def test_retrieve_contract_after_harness():
    """
    Exercise the public retrieve() interface through the course harness.

    We deliberately don't inspect private index/scorer internals.
    """
    from submission import retrieve

    index_dir = ROOT / "dev" / "_test_index"

    try:
        index_dir.mkdir(parents=True, exist_ok=True)

        retrieve.build_index(
            str(TOY_CORPUS),
            str(index_dir),
        )
        retrieve.load_index(str(index_dir))

        results = retrieve.retrieve("test query", k=10)

        assert isinstance(results, list)
        assert len(results) <= 10

        doc_ids = [row[0] for row in results]
        scores = [row[1] for row in results]

        assert len(doc_ids) == len(set(doc_ids))
        assert all(isinstance(doc_id, str) for doc_id in doc_ids)
        assert all(isinstance(score, (int, float)) for score in scores)

        # Scores should be returned best-first.
        assert scores == sorted(scores, reverse=True)

        known_ids = _toy_doc_ids()

        if known_ids:
            assert set(doc_ids) <= known_ids

    finally:
        import shutil

        shutil.rmtree(index_dir, ignore_errors=True)