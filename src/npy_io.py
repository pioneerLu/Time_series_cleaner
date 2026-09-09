import json
import os
from collections import defaultdict
from typing import Iterator, List, Optional, Tuple

import numpy as np

DTYPE = np.float32
ITEMSIZE = np.dtype(DTYPE).itemsize
LAYOUTS = ("auto", "NT", "TC")
CHANNEL_MODES = ("flatten", "first")
BLOCK_BYTES = 64 << 20


def list_npy_files(path: str, recursive: bool = True) -> List[str]:
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if not path.endswith(".npy"):
            raise ValueError(f"input is not a .npy file: {path}")
        return [path]
    if os.path.isdir(path):
        files = []
        if recursive:
            for dirpath, _, names in os.walk(path):
                for name in names:
                    if name.endswith(".npy"):
                        files.append(os.path.join(dirpath, name))
        else:
            files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".npy")
            ]
        files.sort()
        if not files:
            raise FileNotFoundError(f"no .npy files in {path}")
        return files
    raise FileNotFoundError(f"input not found: {path}")


def input_root(input_path: str) -> str:
    path = os.path.abspath(input_path)
    if os.path.isdir(path):
        return path
    return os.path.dirname(path) or os.getcwd()


def relkey(path: str, root: str) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")


def fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}


def _as_float(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=DTYPE).reshape(-1))


def load_npy(npy_path: str):
    try:
        return np.load(npy_path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError):
        return np.load(npy_path, allow_pickle=True)


def resolve_2d_layout(shape, layout: str, npy_path: str) -> str:
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}, choose from {LAYOUTS}")
    if layout != "auto":
        return layout
    rows, cols = shape
    if cols == 1:
        return "TC"
    if rows == 1:
        return "NT"
    raise ValueError(
        f"ambiguous 2D shape {tuple(shape)} in {npy_path}: "
        "pass --layout NT if rows are separate series, "
        "or --layout TC if columns are channels of one series"
    )


def iter_series(
    npy_path: str,
    layout: str = "auto",
    channel_mode: str = "flatten",
) -> Iterator[Tuple[int, np.ndarray]]:
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"unknown channel_mode {channel_mode!r}, choose from {CHANNEL_MODES}")
    arr = load_npy(npy_path)
    if getattr(arr, "dtype", None) == np.dtype("O"):
        for i, item in enumerate(np.asarray(arr, dtype=object).ravel()):
            yield i, _as_float(item)
        return

    if arr.ndim == 1:
        yield 0, _as_float(arr)
        return

    if arr.ndim == 2:
        resolved = resolve_2d_layout(arr.shape, layout, npy_path)
        if resolved == "TC":
            for j in range(arr.shape[1]):
                yield j, _as_float(arr[:, j])
        else:
            for i in range(arr.shape[0]):
                yield i, _as_float(arr[i])
        return

    if arr.ndim == 3:
        n, _, c = arr.shape
        if channel_mode == "first":
            c = min(1, c)
        idx = 0
        for i in range(n):
            for j in range(c):
                yield idx, _as_float(arr[i, :, j])
                idx += 1
        return

    raise ValueError(f"unsupported npy shape {tuple(arr.shape)} in {npy_path}")


def safe_stem(npy_path: str, root: Optional[str] = None) -> str:
    if root:
        rel = os.path.splitext(relkey(npy_path, root))[0]
    else:
        rel = os.path.splitext(os.path.basename(npy_path))[0]
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in rel)


class ShardWriter:
    def __init__(self, shard_dir: str, source_path: str, root: Optional[str] = None):
        self.shard_dir = shard_dir
        self.prefix = safe_stem(source_path, root)
        self.counts = defaultdict(int)
        self._handles = {}
        self._meta = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def append(self, series: np.ndarray, record: dict) -> None:
        data = np.ascontiguousarray(series, dtype=DTYPE)
        length = int(data.size)
        handle = self._handles.get(length)
        if handle is None:
            os.makedirs(self.shard_dir, exist_ok=True)
            raw_path = os.path.join(self.shard_dir, f"{self.prefix}__T{length}.raw")
            meta_path = os.path.join(self.shard_dir, f"{self.prefix}__T{length}.jsonl")
            handle = open(raw_path, "ab")
            meta = open(meta_path, "a", encoding="utf-8")
            self._handles[length] = handle
            self._meta[length] = meta
        handle.write(data.tobytes())
        record = dict(record)
        record["length"] = length
        self._meta[length].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[length] += 1

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        for handle in self._meta.values():
            handle.close()
        self._handles.clear()
        self._meta.clear()


def reset_shards(shard_dir: str, source_path: str, root: Optional[str] = None) -> None:
    if not os.path.isdir(shard_dir):
        return
    prefix = f"{safe_stem(source_path, root)}__T"
    for name in os.listdir(shard_dir):
        if name.startswith(prefix) and (name.endswith(".raw") or name.endswith(".jsonl")):
            os.remove(os.path.join(shard_dir, name))


def has_shards(shard_dir: str, source_path: str, root: Optional[str] = None) -> bool:
    if not os.path.isdir(shard_dir):
        return False
    prefix = f"{safe_stem(source_path, root)}__T"
    return any(name.startswith(prefix) and name.endswith(".raw") for name in os.listdir(shard_dir))


def _raw_shards_by_length(shard_dir: str) -> dict:
    grouped = defaultdict(list)
    if not os.path.isdir(shard_dir):
        return grouped
    for name in sorted(os.listdir(shard_dir)):
        if not name.endswith(".raw") or "__T" not in name:
            continue
        try:
            length = int(name.rsplit("__T", 1)[1][: -len(".raw")])
        except ValueError:
            continue
        grouped[length].append(os.path.join(shard_dir, name))
    return grouped


def _existing_lengths(output_dir: str) -> set:
    lengths = set()
    if not os.path.isdir(output_dir):
        return lengths
    for name in os.listdir(output_dir):
        if name.startswith("data_") and name.endswith(".npy"):
            try:
                lengths.add(int(name[len("data_"): -len(".npy")]))
            except ValueError:
                pass
    return lengths


def merge_shards(
    shard_dir: str,
    output_dir: str,
    append_existing: bool = False,
    keep_raw: bool = False,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    raw_by_length = _raw_shards_by_length(shard_dir)
    lengths = set(raw_by_length)
    if append_existing:
        lengths |= _existing_lengths(output_dir)

    outputs = {}
    index_path = os.path.join(output_dir, "index.jsonl")
    index_mode = "a" if append_existing and os.path.exists(index_path) else "w"
    with open(index_path, index_mode, encoding="utf-8") as index_file:
        for length in sorted(lengths):
            raws = raw_by_length.get(length, [])
            out_path = os.path.join(output_dir, f"data_{length}.npy")
            base = out_path if append_existing and os.path.exists(out_path) else None
            if not raws:
                if base is not None:
                    rows = int(np.load(base, mmap_mode="r").shape[0])
                    outputs[str(length)] = {"path": out_path, "shape": [rows, length, 1]}
                continue

            total = sum(os.path.getsize(p) // (length * ITEMSIZE) for p in raws)
            offset = 0
            if base is not None:
                offset = int(np.load(base, mmap_mode="r").shape[0])
                total += offset
            if total == 0:
                continue

            tmp_path = out_path + ".tmp.npy"
            written = _write_merged(tmp_path, base, raws, length, total)
            os.replace(tmp_path, out_path)
            _append_index(index_file, raws, out_path, length, offset)
            if not keep_raw:
                for path in raws:
                    os.remove(path)
                    meta = path[: -len(".raw")] + ".jsonl"
                    if os.path.exists(meta):
                        os.remove(meta)
            outputs[str(length)] = {"path": out_path, "shape": [written, length, 1]}
    return outputs


def _append_index(index_file, raws: List[str], out_path: str, length: int, offset: int) -> None:
    cursor = offset
    out_name = os.path.basename(out_path)
    for raw in raws:
        meta = raw[: -len(".raw")] + ".jsonl"
        if not os.path.exists(meta):
            n = os.path.getsize(raw) // (length * ITEMSIZE)
            for _ in range(n):
                index_file.write(json.dumps({
                    "output_file": out_name,
                    "output_index": cursor,
                    "length": length,
                }, ensure_ascii=False) + "\n")
                cursor += 1
            continue
        with open(meta, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["output_file"] = out_name
                record["output_index"] = cursor
                index_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                cursor += 1


def _write_merged(tmp_path: str, base, raws: List[str], length: int, total: int) -> int:
    step = max(1, BLOCK_BYTES // (length * ITEMSIZE))
    out = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=DTYPE, shape=(total, length, 1)
    )
    pos = 0
    try:
        if base is not None:
            src = np.load(base, mmap_mode="r")
            for start in range(0, src.shape[0], step):
                chunk = np.asarray(src[start: start + step], dtype=DTYPE)
                count = chunk.shape[0]
                out[pos: pos + count] = chunk.reshape(count, length, 1)
                pos += count
            del src
        for path in raws:
            with open(path, "rb") as handle:
                while True:
                    buf = handle.read(step * length * ITEMSIZE)
                    count = len(buf) // (length * ITEMSIZE)
                    if count == 0:
                        break
                    block = np.frombuffer(buf, dtype=DTYPE, count=count * length)
                    out[pos: pos + count] = block.reshape(count, length, 1)
                    pos += count
    finally:
        out.flush()
        del out
    return pos
