from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass
class Registry:
    """Append-only JSON-Lines store of versioned entries (filters, features, ...).

    Each entry is keyed by ``(kind, name)`` and carries a ``version`` (int,
    starting at 1) and a ``fingerprint``. Registering a payload whose
    fingerprint already exists under that ``(kind, name)`` returns the
    existing entry unchanged; a new fingerprint appends a new line with
    ``version`` incremented. Existing lines are never rewritten.
    """

    path: Path
    entries: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.entries = _read_jsonl(self.path)

    def versions(self, kind: str, name: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == kind and e["name"] == name]

    def register(
        self,
        kind: str,
        name: str,
        fingerprint: str,
        payload: dict[str, Any],
        run: str | None = None,
    ) -> dict[str, Any]:
        existing = self.versions(kind, name)
        for entry in existing:
            if entry["fingerprint"] == fingerprint:
                return entry
        version = max((e["version"] for e in existing), default=0) + 1
        entry = {
            "kind": kind,
            "name": name,
            "version": version,
            "fingerprint": fingerprint,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
            "run": run,
            **payload,
        }
        self.entries.append(entry)
        _append_jsonl(self.path, entry)
        return entry

    def resolve(self, kind: str, name: str, version: int) -> dict[str, Any]:
        for entry in self.versions(kind, name):
            if entry["version"] == version:
                return entry
        raise KeyError(f"registry entry not found: {kind}/{name} v{version}")


def register_expr(
    registry: Registry,
    kind: str,
    name: str,
    expr: pl.Expr,
    run: str | None = None,
) -> dict[str, Any]:
    fingerprint, payload = expr_registry_payload(expr)
    entry = registry.register(kind, name, fingerprint, payload, run=run)
    return registry_ref(entry)


def register_pickle_feature(
    registry: Registry,
    name: str,
    feature: Any,
    run: str | None = None,
) -> dict[str, Any]:
    payload, fingerprint = pickle_feature_payload(feature)
    entry = registry.register("feature", name, fingerprint, payload, run=run)
    return registry_ref(entry)


def registry_ref(entry: dict[str, Any]) -> dict[str, Any]:
    ref = {
        "name": entry["name"],
        "version": entry["version"],
        "fingerprint": entry["fingerprint"],
    }
    storage = entry.get("storage")
    if storage is not None:
        ref["storage"] = storage
    return ref


def versioned_name(ref: dict[str, Any]) -> str:
    return f"{ref['name']}@v{ref['version']}"


def expr_registry_payload(expr: pl.Expr) -> tuple[str, dict[str, Any]]:
    blob = canonical_json_blob(expr.meta.serialize(format="json"))
    fingerprint = hashlib.sha256(blob.encode()).hexdigest()
    return fingerprint, {
        "storage": "expr",
        "expr": blob,
        "repr": str(expr),
        "roots": expr.meta.root_names(),
        "format": "json",
        "polars_version": pl.__version__,
    }


def pickle_feature_payload(feature: Any) -> tuple[dict[str, Any], str]:
    pickle_blob = pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL)
    config = normalize_registry_value(feature_config(feature))
    if config:
        fingerprint_blob = canonical_json(config).encode()
    else:
        fingerprint_blob = pickle_blob
    fingerprint = hashlib.sha256(fingerprint_blob).hexdigest()
    payload = {
        "storage": "pickle",
        "pickle": base64.b64encode(pickle_blob).decode("ascii"),
        "pickle_encoding": "base64",
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "pickle_sha256": hashlib.sha256(pickle_blob).hexdigest(),
        "repr": repr(feature),
        "type": type(feature).__name__,
    }
    if config:
        payload["config"] = config
    return payload, fingerprint


def feature_config(feature: Any) -> dict[str, Any]:
    if hasattr(feature, "to_config"):
        return feature.to_config()
    name = getattr(feature, "name", None)
    config = {"kind": "stateful_feature", "type": type(feature).__name__}
    if name is not None:
        config["name"] = name
    return config


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_json_blob(blob: str) -> str:
    try:
        return canonical_json(json.loads(blob))
    except json.JSONDecodeError:
        return blob


def normalize_registry_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {str(k): normalize_registry_value(v) for k, v in value.items()}
        if out.get("format") == "json" and isinstance(out.get("expr"), str):
            out["expr"] = canonical_json_blob(out["expr"])
            out["fingerprint"] = hashlib.sha256(out["expr"].encode()).hexdigest()
        return out
    if isinstance(value, (list, tuple)):
        return [normalize_registry_value(v) for v in value]
    return _json_ready(value)


def file_set_manifest(
    paths: list[Path] | tuple[Path, ...],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    entries = []
    combined = hashlib.sha256()
    unique_paths = sorted(_unique_paths(paths), key=lambda item: str(item))
    sizes = {path: path.stat().st_size for path in unique_paths}
    total_size = sum(sizes.values())
    completed_size = 0
    for i, path in enumerate(unique_paths, 1):
        size = sizes[path]
        if progress is not None:
            progress(
                {
                    "event": "file-start",
                    "index": i,
                    "total": len(unique_paths),
                    "path": path,
                    "size": size,
                    "completed_size": completed_size,
                    "total_size": total_size,
                }
            )

        def file_progress(file_bytes: int) -> None:
            if progress is None:
                return
            progress(
                {
                    "event": "file-progress",
                    "index": i,
                    "total": len(unique_paths),
                    "path": path,
                    "size": size,
                    "file_bytes": file_bytes,
                    "completed_size": completed_size + file_bytes,
                    "total_size": total_size,
                }
            )

        digest = file_sha256(path, progress=file_progress if progress else None)
        entry = {
            "path": str(path),
            "size": size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": digest,
        }
        entries.append(entry)
        combined.update(str(path).encode())
        combined.update(b"\0")
        combined.update(digest.encode())
        combined.update(b"\0")
        completed_size += size
        if progress is not None:
            progress(
                {
                    "event": "file-done",
                    "index": i,
                    "total": len(unique_paths),
                    "path": path,
                    "size": size,
                    "sha256": digest,
                    "completed_size": completed_size,
                    "total_size": total_size,
                }
            )
    return {
        "algorithm": "sha256",
        "hash": combined.hexdigest() if entries else None,
        "files": entries,
    }


def file_sha256(
    path: Path, progress: Callable[[int], None] | None = None
) -> str:
    h = hashlib.sha256()
    bytes_read = 0
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            bytes_read += len(chunk)
            if progress is not None:
                progress(bytes_read)
    return h.hexdigest()


def append_jsonl_record(path: Path, entry: dict[str, Any]) -> None:
    _append_jsonl(Path(path), entry)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(_json_ready(entry), sort_keys=True, separators=(",", ":")))
        f.write("\n")


def _unique_paths(paths: list[Path] | tuple[Path, ...]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        item = Path(path)
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
