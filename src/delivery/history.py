"""Time-aware group statistics: for each order, statistics over earlier events only.

For a query (time q, key k) this returns the number of events and the sums of some values over events with the same key and
event time in [q - window, q). The upper bound is strict, so an order can never see itself or anything that happens later.
"""
import numpy as np
import pandas as pd


def group_stats(q_ts: np.ndarray, q_key: np.ndarray, ev_ts: np.ndarray, ev_key: np.ndarray, ev_vals: dict, window_ns: int | None = None):
    """q_ts, ev_ts: int64 nanoseconds. Returns (counts, {name: sums}) aligned with the queries."""
    order = np.lexsort((ev_ts, ev_key))
    ts, key = ev_ts[order], ev_key[order]
    cum = {n: np.concatenate([[0.0], np.cumsum(np.asarray(v, dtype=float)[order])]) for n, v in ev_vals.items()}
    uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
    span = {k: (a, a + c) for k, a, c in zip(uniq, start, cnt)}

    n_out = np.zeros(len(q_ts))
    sums = {n: np.zeros(len(q_ts)) for n in ev_vals}
    for k, idx in pd.Series(np.arange(len(q_ts))).groupby(q_key).indices.items():
        if k not in span:
            continue
        a, b = span[k]
        slice_ts, qs = ts[a:b], q_ts[idx]
        hi = a + np.searchsorted(slice_ts, qs, side="left")                       # events strictly before q
        lo = a + (np.searchsorted(slice_ts, qs - window_ns, side="left") if window_ns else 0)
        n_out[idx] = hi - lo
        for n in ev_vals:
            sums[n][idx] = cum[n][hi] - cum[n][lo]
    return n_out, sums


def smoothed_rate(late_sum, n, prior, strength):
    """Late rate shrunk toward a prior; with no history it equals the prior."""
    return (late_sum + strength * prior) / (n + strength)
