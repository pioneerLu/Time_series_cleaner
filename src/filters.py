import hashlib
from typing import List, Optional, Tuple

import numpy as np

NAN_POLICIES = ("drop", "interp", "split")


def expand_nan(ts: np.ndarray, min_length: int, nan_policy: str) -> Tuple[List[np.ndarray], Optional[str]]:
    x = np.asarray(ts, dtype=np.float64).reshape(-1)
    if nan_policy not in NAN_POLICIES:
        raise ValueError(f"unknown nan_policy {nan_policy!r}, choose from {NAN_POLICIES}")
    if nan_policy == "drop":
        if not np.isfinite(x).all():
            return [], "non_finite"
        if x.size < min_length:
            return [], "too_short"
        return [x.astype(np.float32, copy=False)], None
    if nan_policy == "interp":
        filled = _interp_nonfinite(x)
        if filled is None:
            return [], "non_finite"
        if filled.size < min_length:
            return [], "too_short"
        return [filled.astype(np.float32, copy=False)], None
    pieces = _split_finite(x, min_length)
    if not pieces:
        return [], "non_finite"
    return pieces, None


def _interp_nonfinite(x: np.ndarray) -> Optional[np.ndarray]:
    finite = np.isfinite(x)
    if finite.all():
        return x
    if finite.sum() < 2:
        return None
    out = x.copy()
    idx = np.arange(x.size)
    out[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
    return out


def _split_finite(x: np.ndarray, min_length: int) -> List[np.ndarray]:
    finite = np.isfinite(x)
    pieces = []
    start = None
    for i, ok in enumerate(finite):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_length:
                pieces.append(x[start:i].astype(np.float32, copy=False))
            start = None
    if start is not None and x.size - start >= min_length:
        pieces.append(x[start:].astype(np.float32, copy=False))
    return pieces


def drop_reason(
    ts: np.ndarray,
    min_length: int,
    enable_zero_check: bool,
    zero_check_len: int,
    zero_ratio_threshold: float,
    zero_streak_threshold: int,
    near_zero_threshold: float,
    zero_std_threshold: float = 0.0,
) -> Optional[str]:
    x = np.asarray(ts, dtype=np.float64).reshape(-1)
    if x.size < min_length:
        return "too_short"
    if not np.isfinite(x).all():
        return "non_finite"
    if enable_zero_check and is_zero_or_constant(
        x,
        zero_check_len=zero_check_len,
        zero_ratio_threshold=zero_ratio_threshold,
        zero_streak_threshold=zero_streak_threshold,
        near_zero_threshold=near_zero_threshold,
        zero_std_threshold=zero_std_threshold,
    ):
        return "zero_or_constant"
    return None


def is_zero_or_constant(
    ts: np.ndarray,
    zero_check_len: int = 100,
    zero_ratio_threshold: float = 0.9,
    zero_streak_threshold: int = 50,
    near_zero_threshold: float = 0.005,
    zero_std_threshold: float = 0.0,
) -> bool:
    x = np.asarray(ts, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return True
    n = min(zero_check_len, x.size)
    initial = x[:n]
    near = np.abs(initial) <= near_zero_threshold
    zero_ratio = float(np.mean(near))
    streak = 0
    for val in x:
        if abs(val) <= near_zero_threshold:
            streak += 1
        else:
            break
    too_many_zeros = zero_ratio > zero_ratio_threshold or streak > zero_streak_threshold
    nearly_constant = float(np.ptp(x)) <= near_zero_threshold
    if zero_std_threshold > 0:
        nearly_constant = nearly_constant or float(np.std(x)) <= zero_std_threshold
    return bool(too_many_zeros or nearly_constant)


def series_hash(ts: np.ndarray) -> bytes:
    data = np.ascontiguousarray(ts, dtype=np.float32).reshape(-1).tobytes()
    return hashlib.md5(data).digest()


def fix_anomalies(series_data: np.ndarray, threshold: float = 7,
                  window_size: int = 24) -> np.ndarray:
    import pandas as pd

    series = pd.Series(np.asarray(series_data, dtype=np.float64).reshape(-1))
    if series.size == 0:
        return series.to_numpy()
    window = max(3, min(int(window_size), int(series.size)))
    if window % 2 == 0:
        window = max(3, window - 1)

    rolling_mean = series.rolling(window=window, center=True).mean().bfill().ffill()
    rolling_std = series.rolling(window=window, center=True).std().bfill().ffill()
    global_std = float(series.std()) if series.size else 0.0
    rolling_std = rolling_std.replace(0, global_std if global_std > 0 else 1.0)

    threshold_anomalies = np.abs(series - rolling_mean) > threshold * rolling_std
    median_value = float(series.median())
    if median_value > 0 and float(series.min()) >= 0:
        extreme_low = series < median_value * 0.2
    else:
        extreme_low = pd.Series(False, index=series.index)

    anomalies = (threshold_anomalies | extreme_low).to_numpy()
    anomaly_indices = np.flatnonzero(anomalies)
    if anomaly_indices.size == 0:
        return series.to_numpy()

    fixed = series.to_numpy().copy()
    values = series.to_numpy()
    n = values.size
    for idx in anomaly_indices:
        left_idx = idx - 1
        while left_idx >= 0 and anomalies[left_idx]:
            left_idx -= 1
        right_idx = idx + 1
        while right_idx < n and anomalies[right_idx]:
            right_idx += 1
        if left_idx >= 0 and right_idx < n:
            fixed[idx] = np.interp(idx, [left_idx, right_idx], [values[left_idx], values[right_idx]])
        elif left_idx >= 0:
            fixed[idx] = values[left_idx]
        elif right_idx < n:
            fixed[idx] = values[right_idx]
    return fixed
