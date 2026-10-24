"""Validation, atomic artifact writes, and a process lock for local KB batches."""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

ASPECTS = ("description", "quality", "packaging", "delivery", "service", "price")
SENTIMENTS = {"negative", "neutral", "positive"}


def read_csv(path):
    return pd.read_csv(path, dtype={"product_id": str}, keep_default_na=False)


def normalize_ids(frame):
    frame = frame.copy()
    if "product_id" not in frame:
        raise ValueError("Missing product_id")
    values = frame["product_id"].astype(str).str.strip()
    if not values.str.fullmatch(r"[0-9]+").all():
        raise ValueError("product_id must contain positive integers")
    frame["product_id"] = values.map(int)
    if (frame["product_id"] <= 0).any():
        raise ValueError("product_id must be positive")
    return frame


def validate_labels(frame):
    if "label" not in frame:
        raise ValueError("Missing label column")
    for index, value in frame["label"].items():
        try:
            labels = ast.literal_eval(value) if isinstance(value, str) else value
            if not isinstance(labels, list) or len(labels) != len(ASPECTS):
                raise ValueError()
            mapping = {item["aspect"]: item["sentiment"] for item in labels}
            if set(mapping) != set(ASPECTS) or not set(mapping.values()) <= SENTIMENTS:
                raise ValueError()
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid six-aspect labels at row {index}") from exc


def fingerprint(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(frame, path):
    atomic_write(path, lambda temporary: frame.to_csv(temporary, index=False))


def write_json(value, path):
    atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(docs, path):
    def writer(temporary):
        with temporary.open("w", encoding="utf-8") as stream:
            for doc in docs:
                stream.write(json.dumps(doc, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_write(path, writer)


@contextmanager
def batch_lock(path):
    """OS lock releases automatically on process exit; do not delete lock files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise RuntimeError("Another incremental KB update is running") from exc
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
