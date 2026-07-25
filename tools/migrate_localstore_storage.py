#!/usr/bin/env python3
"""Merge legacy nyaturingtest Chroma stores into workspace-local storage.

The tool reads stored embeddings directly, so it never calls an embedding API.
It always builds a new staging tree and refuses to overwrite an existing path.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable

import chromadb


COLLECTION_NAME = "nyabot_memory"
VECTOR_DIR_PATTERN = re.compile(r"^vector_index_(.+)$")
DEFAULT_BATCH_SIZE = 250


def discover_vector_dirs(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    result: dict[str, Path] = {}
    for path in root.iterdir():
        match = VECTOR_DIR_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            result[match.group(1)] = path
    return result


def _source_collection(path: Path):
    client = chromadb.PersistentClient(path=str(path))
    return client, client.get_collection(COLLECTION_NAME)


def _batched_collection_records(collection: Any, batch_size: int) -> Iterable[dict]:
    count = collection.count()
    for offset in range(0, count, batch_size):
        yield collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "embeddings", "metadatas"],
        )


def _embedding_dimension(collection: Any) -> int | None:
    if collection.count() == 0:
        return None
    result = collection.get(limit=1, include=["embeddings"])
    embeddings = result.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    return len(embeddings[0])


def describe_store(path: Path) -> dict:
    client, collection = _source_collection(path)
    try:
        return {
            "path": str(path),
            "count": collection.count(),
            "dimension": _embedding_dimension(collection),
            "metadata": collection.metadata or {},
        }
    finally:
        del collection
        del client


def _copy_collection(
    source_path: Path,
    target: Any,
    seen_ids: set[str],
    *,
    batch_size: int,
    skip_presets: bool,
) -> dict:
    client, source = _source_collection(source_path)
    stats = {
        "path": str(source_path),
        "source_count": source.count(),
        "imported": 0,
        "skipped_preset": 0,
        "skipped_id_collision": 0,
        "dimension": _embedding_dimension(source),
        "metadata": source.metadata or {},
    }
    try:
        for records in _batched_collection_records(source, batch_size):
            ids = records.get("ids") or []
            documents = records.get("documents") or []
            embeddings = records.get("embeddings")
            metadatas = records.get("metadatas") or []
            if embeddings is None:
                raise RuntimeError(f"{source_path} contains no stored embeddings")

            selected: list[int] = []
            for index, raw_id in enumerate(ids):
                record_id = str(raw_id)
                metadata = metadatas[index] or {}
                if skip_presets and str(metadata.get("source") or "") == "preset":
                    stats["skipped_preset"] += 1
                    continue
                if record_id in seen_ids:
                    stats["skipped_id_collision"] += 1
                    continue
                seen_ids.add(record_id)
                selected.append(index)

            if not selected:
                continue
            target.add(
                ids=[str(ids[index]) for index in selected],
                documents=[str(documents[index] or "") for index in selected],
                embeddings=[embeddings[index] for index in selected],
                metadatas=[dict(metadatas[index] or {}) for index in selected],
            )
            stats["imported"] += len(selected)
    finally:
        del source
        del client
    return stats


def merge_stores(
    current_root: Path,
    legacy_root: Path,
    stage_root: Path,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    current_root = current_root.resolve()
    legacy_root = legacy_root.resolve()
    stage_root = stage_root.resolve()
    if stage_root.exists():
        raise FileExistsError(f"staging path already exists: {stage_root}")
    if stage_root in {current_root, legacy_root}:
        raise ValueError("staging path must differ from both source roots")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    current_dirs = discover_vector_dirs(current_root)
    legacy_dirs = discover_vector_dirs(legacy_root)
    session_ids = sorted(set(current_dirs) | set(legacy_dirs))
    if not session_ids:
        raise RuntimeError("no vector_index_* directories found")

    stage_root.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "collection": COLLECTION_NAME,
        "target_space": "cosine",
        "current_root": str(current_root),
        "legacy_root": str(legacy_root),
        "stage_root": str(stage_root),
        "batch_size": batch_size,
        "sessions": {},
    }

    total = 0
    for session_id in session_ids:
        target_path = stage_root / f"vector_index_{session_id}"
        target_client = chromadb.PersistentClient(path=str(target_path))
        target = target_client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        seen_ids: set[str] = set()
        session_manifest = {
            "target_path": str(target_path),
            "sources": [],
        }

        current_path = current_dirs.get(session_id)
        legacy_path = legacy_dirs.get(session_id)
        if current_path is not None:
            session_manifest["sources"].append(
                {
                    "role": "current",
                    **_copy_collection(
                        current_path,
                        target,
                        seen_ids,
                        batch_size=batch_size,
                        skip_presets=False,
                    ),
                }
            )
        if legacy_path is not None:
            session_manifest["sources"].append(
                {
                    "role": "legacy",
                    **_copy_collection(
                        legacy_path,
                        target,
                        seen_ids,
                        batch_size=batch_size,
                        # If a current store exists, its preset state is
                        # authoritative.  Do not resurrect stale legacy roles.
                        skip_presets=current_path is not None,
                    ),
                }
            )

        expected = sum(source["imported"] for source in session_manifest["sources"])
        actual = target.count()
        if actual != expected:
            raise RuntimeError(
                f"count mismatch for {session_id}: expected {expected}, got {actual}"
            )
        session_manifest.update(
            {
                "count": actual,
                "dimension": _embedding_dimension(target),
                "metadata": target.metadata or {},
            }
        )
        manifest["sessions"][session_id] = session_manifest
        total += actual
        del target
        del target_client

    manifest["total_count"] = total
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = merge_stores(
        args.current_root,
        args.legacy_root,
        args.stage_root,
        batch_size=args.batch_size,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
