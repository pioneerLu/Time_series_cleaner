import json
import os
from collections import defaultdict

import numpy as np

from src.filters import drop_reason, expand_nan, fix_anomalies, series_hash
from src.npy_io import (
    ShardWriter,
    fingerprint,
    has_shards,
    input_root,
    iter_series,
    list_npy_files,
    merge_shards,
    relkey,
    reset_shards,
)


class PretrainCleaner:
    def __init__(self, config):
        self.config = config

    def run(self, input_path: str) -> str:
        cfg = self.config
        cfg.ensure_pretrain_dirs()
        root = input_root(input_path)
        files = list_npy_files(input_path, recursive=cfg.recursive)
        progress = {} if cfg.overwrite else self._load_progress()
        had_manifest = os.path.exists(cfg.manifest_path)

        if cfg.overwrite or self._stale_keys(files, progress, root):
            if not cfg.overwrite:
                print("source files changed, rebuilding output")
            self._clear_shards()
            self._clear_outputs()
            had_manifest = False
            progress = {}

        pending_shards = any(has_shards(cfg.shard_dir, path, root) for path in files)
        done = all(self._record(progress, path, root) for path in files)
        if had_manifest and not pending_shards and done and all(
            self._fingerprint_ok(progress, path, root) for path in files
        ):
            print(f"already complete: {cfg.manifest_path}")
            return cfg.output_dir

        seen = self._load_hashes() if cfg.dedup else None
        for path in files:
            if self._should_skip(path, progress, root):
                print(f"skip (done): {relkey(path, root)}")
                continue
            reset_shards(cfg.shard_dir, path, root)
            record = self._process_file(path, root, seen)
            key = relkey(path, root)
            record.update(fingerprint(path))
            progress[key] = record
            self._save_progress(progress)
            print(f"{key} kept={record['kept']} dropped={record['dropped']}")

        outputs = merge_shards(
            cfg.shard_dir,
            cfg.output_dir,
            append_existing=had_manifest,
            keep_raw=cfg.keep_temp_files,
        )
        if not cfg.keep_temp_files:
            self._clear_shards()

        manifest = {
            "mode": "pretrain",
            "input": os.path.abspath(input_path),
            "layout": cfg.layout,
            "channel_mode": cfg.channel_mode,
            "nan_policy": cfg.nan_policy,
            "recursive": cfg.recursive,
            "dedup": cfg.dedup,
            "files": progress,
            "outputs": outputs,
            "kept": sum(item.get("kept", 0) for item in progress.values()),
            "dropped": _sum_dropped(progress),
        }
        with open(cfg.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"manifest: {cfg.manifest_path}")
        return cfg.output_dir

    def _process_file(self, path: str, root: str, seen) -> dict:
        cfg = self.config
        dropped = defaultdict(int)
        kept = 0
        source = relkey(path, root)
        with ShardWriter(cfg.shard_dir, path, root) as writer:
            for src_index, ts in iter_series(path, layout=cfg.layout, channel_mode=cfg.channel_mode):
                pieces, nan_reason = expand_nan(ts, cfg.min_length, cfg.nan_policy)
                if nan_reason:
                    dropped[nan_reason] += 1
                    continue
                for piece_i, piece in enumerate(pieces):
                    reason = drop_reason(
                        piece,
                        min_length=cfg.min_length,
                        enable_zero_check=cfg.enable_zero_check,
                        zero_check_len=cfg.zero_check_len,
                        zero_ratio_threshold=cfg.zero_ratio_threshold,
                        zero_streak_threshold=cfg.zero_streak_threshold,
                        near_zero_threshold=cfg.near_zero_threshold,
                        zero_std_threshold=cfg.zero_std_threshold,
                    )
                    if reason:
                        dropped[reason] += 1
                        continue
                    if seen is not None:
                        digest = series_hash(piece)
                        if digest in seen:
                            dropped["duplicate"] += 1
                            continue
                        seen.add(digest)
                    cleaned = fix_anomalies(piece) if cfg.fix_anomalies else piece
                    writer.append(cleaned, {
                        "source": source,
                        "source_index": src_index,
                        "piece": piece_i,
                    })
                    kept += 1
            lengths = {str(k): v for k, v in writer.counts.items()}
        return {
            "status": "done",
            "kept": kept,
            "dropped": dict(dropped),
            "lengths": lengths,
        }

    def _record(self, progress: dict, path: str, root: str):
        rec = progress.get(relkey(path, root)) or progress.get(os.path.abspath(path))
        return rec if rec and rec.get("status") == "done" else None

    def _fingerprint_ok(self, progress: dict, path: str, root: str) -> bool:
        rec = self._record(progress, path, root)
        if not rec:
            return False
        fp = fingerprint(path)
        return rec.get("size") == fp["size"] and rec.get("mtime_ns") == fp["mtime_ns"]

    def _stale_keys(self, files, progress: dict, root: str) -> bool:
        for path in files:
            rec = self._record(progress, path, root)
            if not rec:
                continue
            if not self._fingerprint_ok(progress, path, root):
                return True
        return False

    def _should_skip(self, path: str, progress: dict, root: str) -> bool:
        if not self._fingerprint_ok(progress, path, root):
            return False
        return has_shards(self.config.shard_dir, path, root) or os.path.exists(self.config.manifest_path)

    def _load_hashes(self) -> set:
        seen = set()
        output_dir = self.config.output_dir
        if not os.path.isdir(output_dir):
            return seen
        for name in os.listdir(output_dir):
            if not (name.startswith("data_") and name.endswith(".npy")):
                continue
            arr = np.load(os.path.join(output_dir, name), mmap_mode="r")
            for i in range(arr.shape[0]):
                seen.add(series_hash(arr[i]))
        return seen

    def _load_progress(self) -> dict:
        if not os.path.exists(self.config.progress_path):
            return {}
        with open(self.config.progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "files" in data:
            return data["files"]
        return data if isinstance(data, dict) else {}

    def _save_progress(self, progress: dict) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(self.config.progress_path, "w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def _clear_shards(self) -> None:
        _clear_dir_files(self.config.shard_dir)

    def _clear_outputs(self) -> None:
        output_dir = self.config.output_dir
        if not os.path.isdir(output_dir):
            return
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                continue
            if name in ("progress.json", "manifest.json", "index.jsonl") or (
                name.startswith("data_") and name.endswith(".npy")
            ):
                os.remove(path)


def _sum_dropped(progress: dict) -> dict:
    total = defaultdict(int)
    for item in progress.values():
        for key, value in item.get("dropped", {}).items():
            total[key] += int(value)
    return dict(total)


def _clear_dir_files(directory: str) -> None:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            os.remove(path)
    try:
        os.rmdir(directory)
    except OSError:
        pass
