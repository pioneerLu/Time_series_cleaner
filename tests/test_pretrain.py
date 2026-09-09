import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.config import Config
from src.filters import drop_reason, expand_nan, is_zero_or_constant
from src.npy_io import iter_series, list_npy_files
from src.pretrain import PretrainCleaner


def _sine(length: int, phase: float = 0.0) -> np.ndarray:
    return np.sin(np.linspace(0, 12, length) + phase).astype(np.float32)


def _args():
    return dict(
        min_length=32,
        enable_zero_check=True,
        zero_check_len=100,
        zero_ratio_threshold=0.9,
        zero_streak_threshold=50,
        near_zero_threshold=0.005,
        zero_std_threshold=0.0,
    )


class FilterTests(unittest.TestCase):
    def test_too_short(self):
        self.assertEqual(drop_reason(np.ones(8), **{**_args(), "enable_zero_check": False}), "too_short")

    def test_non_finite(self):
        x = np.linspace(0, 1, 64)
        x[3] = np.nan
        self.assertEqual(drop_reason(x, **{**_args(), "enable_zero_check": False}), "non_finite")

    def test_constant(self):
        self.assertTrue(is_zero_or_constant(np.zeros(200)))
        self.assertTrue(is_zero_or_constant(np.full(200, 5.0)))

    def test_keep_sine(self):
        rng = np.random.default_rng(0)
        x = np.sin(np.linspace(0, 20, 128)) + 0.05 * rng.normal(size=128)
        self.assertIsNone(drop_reason(x, **_args()))

    def test_nan_interp(self):
        x = np.linspace(0, 1, 64)
        expected = x.copy()
        x[10] = np.nan
        pieces, reason = expand_nan(x, min_length=32, nan_policy="interp")
        self.assertIsNone(reason)
        self.assertEqual(len(pieces), 1)
        np.testing.assert_allclose(pieces[0][10], expected[10], atol=1e-6)

    def test_nan_split(self):
        left = np.linspace(1, 2, 40)
        right = np.linspace(3, 4, 40)
        x = np.concatenate([left, [np.nan, np.nan], right])
        pieces, reason = expand_nan(x, min_length=32, nan_policy="split")
        self.assertIsNone(reason)
        self.assertEqual(len(pieces), 2)
        np.testing.assert_allclose(pieces[0], left)
        np.testing.assert_allclose(pieces[1], right)


class IOTests(unittest.TestCase):
    def test_layout_unambiguous(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        a = os.path.join(tmp, "a.npy")
        b = os.path.join(tmp, "b.npy")
        c = os.path.join(tmp, "c.npy")
        d = os.path.join(tmp, "d.npy")
        np.save(a, np.ones((3, 40, 1), dtype=np.float32))
        np.save(b, np.ones((40,), dtype=np.float32))
        np.save(c, np.ones((40, 1), dtype=np.float32))
        np.save(d, np.ones((1, 40), dtype=np.float32))
        self.assertEqual(len(list(iter_series(a))), 3)
        self.assertEqual(len(list(iter_series(b))), 1)
        self.assertEqual(len(list(iter_series(c))), 1)
        self.assertEqual(len(list(iter_series(d))), 1)

    def test_layout_ambiguous_raises(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "amb.npy")
        np.save(path, np.ones((5, 40), dtype=np.float32))
        with self.assertRaises(ValueError):
            list(iter_series(path, layout="auto"))

    def test_layout_explicit(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "m.npy")
        np.save(path, np.ones((5, 40), dtype=np.float32))
        nt = list(iter_series(path, layout="NT"))
        tc = list(iter_series(path, layout="TC"))
        self.assertEqual(len(nt), 5)
        self.assertEqual(nt[0][1].size, 40)
        self.assertEqual(len(tc), 40)
        self.assertEqual(tc[0][1].size, 5)

    def test_station_layout(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "wind.npy")
        np.save(path, np.ones((500, 4), dtype=np.float32))
        series = list(iter_series(path, layout="TC"))
        self.assertEqual(len(series), 4)
        self.assertEqual(series[0][1].size, 500)

    def test_channel_mode_first(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "m.npy")
        np.save(path, np.ones((2, 50, 3), dtype=np.float32))
        self.assertEqual(len(list(iter_series(path, channel_mode="flatten"))), 6)
        self.assertEqual(len(list(iter_series(path, channel_mode="first"))), 2)

    def test_recursive_list(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        nested = os.path.join(tmp, "sub", "deep")
        os.makedirs(nested)
        np.save(os.path.join(nested, "x.npy"), np.ones((40,), dtype=np.float32))
        files = list_npy_files(tmp, recursive=True)
        self.assertEqual(len(files), 1)
        self.assertRaises(FileNotFoundError, list_npy_files, tmp, False)


class PretrainPipelineTests(unittest.TestCase):
    def test_group_by_length_and_drop(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        good_a = _sine(64)[None, :, None]
        good_b = _sine(96, phase=1.0)[None, :, None]
        bad = np.zeros((2, 64, 1), dtype=np.float32)
        nan = np.ones((1, 64, 1), dtype=np.float32)
        nan[0, 2, 0] = np.nan
        np.save(os.path.join(data_dir, "a.npy"), np.concatenate([good_a, bad, nan], axis=0))
        np.save(os.path.join(data_dir, "b.npy"), good_b)

        config = Config(output_dir=out_dir, min_length=32, enable_zero_check=True)
        PretrainCleaner(config).run(data_dir)

        out_64 = np.load(os.path.join(out_dir, "data_64.npy"))
        out_96 = np.load(os.path.join(out_dir, "data_96.npy"))
        self.assertEqual(out_64.shape, (1, 64, 1))
        self.assertEqual(out_96.shape, (1, 96, 1))
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["kept"], 2)
        self.assertGreater(manifest["dropped"].get("zero_or_constant", 0), 0)
        self.assertGreater(manifest["dropped"].get("non_finite", 0), 0)

    def test_values_roundtrip(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        expected = _sine(80)
        np.save(os.path.join(data_dir, "sine.npy"), expected.reshape(1, 80, 1))

        PretrainCleaner(Config(output_dir=out_dir, min_length=32)).run(data_dir)
        got = np.load(os.path.join(out_dir, "data_80.npy"))
        self.assertEqual(got.shape, (1, 80, 1))
        np.testing.assert_allclose(got[0, :, 0], expected, rtol=0, atol=0)

    def test_incremental_append(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        a = _sine(72)
        b = _sine(72, phase=2.0)
        np.save(os.path.join(data_dir, "a.npy"), a.reshape(1, 72, 1))

        PretrainCleaner(Config(output_dir=out_dir, min_length=32)).run(data_dir)
        first = np.load(os.path.join(out_dir, "data_72.npy"))
        self.assertEqual(first.shape, (1, 72, 1))

        np.save(os.path.join(data_dir, "b.npy"), b.reshape(1, 72, 1))
        PretrainCleaner(Config(output_dir=out_dir, min_length=32)).run(data_dir)
        second = np.load(os.path.join(out_dir, "data_72.npy"))
        self.assertEqual(second.shape, (2, 72, 1))
        np.testing.assert_allclose(second[0, :, 0], a, rtol=0, atol=0)
        np.testing.assert_allclose(second[1, :, 0], b, rtol=0, atol=0)

        PretrainCleaner(Config(output_dir=out_dir, min_length=32)).run(data_dir)
        third = np.load(os.path.join(out_dir, "data_72.npy"))
        self.assertEqual(third.shape, (2, 72, 1))

    def test_overwrite_resets(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        np.save(os.path.join(data_dir, "a.npy"), _sine(60).reshape(1, 60, 1))
        np.save(os.path.join(data_dir, "b.npy"), _sine(60, phase=1.5).reshape(1, 60, 1))

        PretrainCleaner(Config(output_dir=out_dir, min_length=32)).run(data_dir)
        self.assertEqual(np.load(os.path.join(out_dir, "data_60.npy")).shape, (2, 60, 1))

        PretrainCleaner(Config(output_dir=out_dir, min_length=32, overwrite=True)).run(data_dir)
        self.assertEqual(np.load(os.path.join(out_dir, "data_60.npy")).shape, (2, 60, 1))

    def test_dedup_and_index(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        x = _sine(80)
        np.save(os.path.join(data_dir, "a.npy"), x.reshape(1, 80, 1))
        np.save(os.path.join(data_dir, "b.npy"), x.reshape(1, 80, 1))
        PretrainCleaner(Config(output_dir=out_dir, min_length=32, dedup=True)).run(data_dir)
        self.assertEqual(np.load(os.path.join(out_dir, "data_80.npy")).shape, (1, 80, 1))
        with open(os.path.join(out_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["dropped"].get("duplicate", 0), 1)
        with open(os.path.join(out_dir, "index.jsonl"), encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "a.npy")
        self.assertEqual(rows[0]["output_file"], "data_80.npy")

    def test_recursive_pipeline(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        nested = os.path.join(tmp, "in", "nested")
        os.makedirs(nested)
        np.save(os.path.join(nested, "c.npy"), _sine(88).reshape(1, 88, 1))
        out_dir = os.path.join(tmp, "out")
        PretrainCleaner(Config(output_dir=out_dir, min_length=32, recursive=True)).run(os.path.join(tmp, "in"))
        self.assertEqual(np.load(os.path.join(out_dir, "data_88.npy")).shape, (1, 88, 1))

    def test_nan_split_pipeline(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        data_dir = os.path.join(tmp, "in")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(data_dir)
        left = _sine(40)
        right = _sine(40, phase=2.0)
        series = np.concatenate([left, [np.nan], right])
        np.save(os.path.join(data_dir, "gap.npy"), series.reshape(1, -1, 1))
        PretrainCleaner(Config(output_dir=out_dir, min_length=32, nan_policy="split")).run(data_dir)
        got = np.load(os.path.join(out_dir, "data_40.npy"))
        self.assertEqual(got.shape, (2, 40, 1))


if __name__ == "__main__":
    unittest.main()
