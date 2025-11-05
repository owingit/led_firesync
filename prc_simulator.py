import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Tuple, Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed


def _nanargmin_unravel(A):
    """Like nanargmin but returns None if all-NaN; otherwise unravel indices."""
    if not np.isfinite(A).any():
        return None
    return np.unravel_index(np.nanargmin(A), A.shape)


def nearest_all_res_key(all_res, p_ms, kap, phic, ymax, y0, tol=None):
    """
    Return the key in all_res for this LED with params closest (L2) to (kap, phic, ymax, y0).
    If tol is set (float), return None if the nearest is farther than tol.
    """
    # collect candidates with same LED
    candidates = [k for k in all_res.keys() if int(k[0]) == int(p_ms)]
    if not candidates:
        return None

    target = np.array([float(kap), float(phic), float(ymax), float(y0)], dtype=float)
    arr = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in candidates], dtype=float)

    diffs = arr - target
    d2 = np.einsum('ij,ij->i', diffs, diffs)  # squared L2 distance
    idx = int(np.argmin(d2))
    if tol is not None and float(np.sqrt(d2[idx])) > float(tol):
        return None
    return candidates[idx]


# =========================
# Helpers
# =========================
def _wrap_pm(x, L):
    x = np.asarray(x, float)
    return ((x + L) % (2.0 * L)) - L


def _wrap_pm_half(phi):
    # keep your original name working everywhere
    return _wrap_pm(phi, 0.5)


def wrap1(u):
    """wrap_1(u) = u - 2*floor((u+1)/2)  ->  (-1, 1]"""
    u = np.asarray(u, float)
    return u - 2.0 * np.floor((u + 1.0) / 2.0)


def wrap_half(phi):
    """wrap to (-1/2, 1/2]"""
    phi = np.asarray(phi, float)
    return ((phi + 0.5) % 1.0) - 0.5


def _alpha(kappa, phi_c, eps=1e-15):
    t1 = np.tanh(kappa * (0.5 - phi_c))
    t2 = np.tanh(kappa * (0.5 + phi_c))
    denom = t1 + t2
    tiny = np.where(np.abs(denom) < eps,
                    np.sign(denom) * eps + (denom == 0.0) * eps, denom)
    return 2.0 / tiny


def R_tanh_step(phi, Ymax, kappa, phi_c, y0, sgn=+1.0):
    """
    Measured ratio model:
        R_tanh(φ_eval) = 1 + Ymax * wrap1( sgn * α(κ, φ_c) * [ y0 + tanh(κ(φ_eval − φ_c)) ] ),
    where φ_eval = wrap_half(φ) ∈ (-1/2, 1/2].
    """
    phi_eval = wrap_half(phi)
    a = _alpha(kappa, phi_c)
    core = sgn * a * (y0 + np.tanh(kappa * (phi_eval - phi_c)))
    return 1.0 + Ymax * wrap1(core)


def Z_from_R(phi, R_func):
    """
    Impulse PRC from your construction:
        Z(φ_eval) = 1 - R(φ_eval).
    """
    return 1.0 - R_func(phi)


def forward_R_from_Z(phi, Z_func, r_max=2.0):
    """
    With the chosen convention Z = 1 - R, we simply invert:
        R(φ_eval) = 1 - Z(φ_eval).
    """
    Z = np.asarray(Z_func(phi), float)
    R = 1.0 - Z
    return np.clip(R, 0.0, r_max)


def make_tanh_prc(Ymax, kappa, phi_c, y0, sgn=+1.0, r_max=2.0):
    """
    Factory returning (R, Z) callables consistent with Z = 1 - R.
    """
    def R(phi):
        return R_tanh_step(phi, Ymax=Ymax, kappa=kappa, phi_c=phi_c, y0=y0, sgn=sgn)
    def Z(phi):
        return Z_from_R(phi, R)
    return R, Z


def _nearest_key(keys, target):
    keys = np.array(list(keys), dtype=float)
    return float(keys[np.argmin(np.abs(keys - float(target)))])


def load_empirical_before_seconds(path, lo=0.3, hi=2.0):
    """Load once; return a 1D float array of periods in seconds."""
    with open(path, "rb") as h:
        obj = pickle.load(h)
    if isinstance(obj, dict):
        vals = []
        for v in obj.values():
            a = np.asarray(v, float).ravel()
            a = a[np.isfinite(a)]
            vals.append(a)
        arr = np.concatenate(vals) if vals else np.array([], float)
    else:
        arr = np.asarray(obj, float).ravel()
        arr = arr[np.isfinite(arr)]
    # if looks like ms, convert to s
    if arr.size and np.nanmedian(arr) > 5.0:
        arr = arr * 1e-3
    # keep a broad range in seconds
    arr = arr[(arr > lo) & (arr < hi)]
    if arr.size == 0:
        raise ValueError("Empirical 'befores' array empty after filtering.")
    return np.ascontiguousarray(arr, dtype=float)


def load_experimental_samples_ms(path, led_periods_ms, lo_s=0.2, hi_s=2.0):
    """
    Return {p_ms: 1D float array of ISIs in *ms*} for each LED period key.
    Filters to [lo_s, hi_s] in seconds, converts to ms.
    """
    with open(path, "rb") as h:
        obj = pickle.load(h)

    out = {}
    for p_ms in led_periods_ms:
        vals = []
        if isinstance(obj, dict):
            mk = None
            try:
                w = float(p_ms)
                for k in obj.keys():
                    try:
                        if abs(float(k) - w) < 1e-9:
                            mk = k
                            break
                    except:
                        pass
            except:
                pass
            if mk is None:
                for k in obj.keys():
                    if str(k).strip().lower() == str(p_ms).strip().lower():
                        mk = k
                        break
            iters = obj.values() if mk is None else [obj[mk]]
            for v in iters:
                a = np.asarray(v, float).ravel()
                a = a[np.isfinite(a)]
                vals.append(a)
        else:
            a = np.asarray(obj, float).ravel()
            a = a[np.isfinite(a)]
            vals.append(a)

        arr = np.concatenate(vals) if vals else np.array([], float)
        if arr.size and np.nanmedian(arr) > 5.0:  # looks like ms → s
            arr = arr * 1e-3
        # seconds window
        arr = arr[(arr > float(lo_s)) & (arr < float(hi_s))]
        out[p_ms] = 1000.0 * arr  # ms
    return out


def _range_filter_ms(arr_ms, bins_ms):
    lo = float(bins_ms[0])
    hi = float(bins_ms[-1])
    a = np.asarray(arr_ms, float)
    a = a[np.isfinite(a)]
    return a[(a >= lo) & (a <= hi)]


def sample_from_loaded_empirical(rng, emp_arr, lo=None, hi=None):
    """Fast sampler from preloaded EMP_ARR with optional (lo, hi) filter in seconds."""
    arr = emp_arr
    if lo is None and hi is None:
        return float(arr[rng.integers(arr.size)])
    mask = np.ones(arr.shape, dtype=bool)
    if lo is not None: mask &= (arr > float(lo))
    if hi is not None: mask &= (arr < float(hi))
    pool = arr[mask] if np.any(mask) else arr
    return float(pool[rng.integers(pool.size)])


def precompute_led_onsets(led_periods_ms, sim_duration_s, led_start_s):
    onsets = {}
    for p_ms in led_periods_ms:
        p_s = float(p_ms) / 1000.0
        if led_start_s >= sim_duration_s:
            arr = np.array([], float)
        else:
            n_led = int(np.floor((sim_duration_s - led_start_s) / p_s)) + 1
            arr = led_start_s + p_s * np.arange(n_led, dtype=float)
        onsets[p_ms] = np.ascontiguousarray(arr, dtype=float)
    return onsets


def build_R_for_led_from_dir(save_dir, led_ms, fill="edge"):
    csv_path = Path(save_dir) / f"prc_ratio_{int(led_ms)}ms.csv"
    phases, ratios = load_prc_ratio_from_csv(csv_path)
    return make_R_from_table(phases, ratios, fill=fill)


def load_prc_ratio_from_csv(csv_path, seed=None):
    """
    Load a PRC CSV with columns: phase,ratio,sem,n -> (phases[], ratios[])
    Applies SEM logic:
      1) if ratio > 1 and ratio - sem < 1  -> ratio = 1.0
      2) if ratio < 1 and ratio + sem > 1  -> ratio = 1.0
      3) otherwise ratio ~ Uniform(ratio - sem, ratio + sem)
    Returns strictly-monotone (phases, ratios) sorted by phase.
    """

    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    x = np.asarray(arr["phase"], float)
    y = np.asarray(arr["ratio"], float)
    s = np.asarray(arr["sem"], float)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    x, y, s = x[m], y[m], s[m]
    if x.size == 0: return x, y

    rng = np.random.default_rng(seed)
    low = y - s
    high = y + s

    hit_one = ((y > 1.0) & (low < 1.0)) | ((y < 1.0) & (high > 1.0))
    y_adj = y.copy()
    y_adj[hit_one] = 1.0
    other = ~hit_one
    if np.any(other):
        y_adj[other] = rng.uniform(low[other], high[other])

    order = np.argsort(x)
    x, y_adj = x[order], y_adj[order]
    uniq_x = [x[0]]
    agg_y = [y_adj[0]]
    for xi, yi in zip(x[1:], y_adj[1:]):
        if np.isclose(xi, uniq_x[-1], atol=1e-12):
            agg_y[-1] = 0.5 * (agg_y[-1] + yi)
        else:
            uniq_x.append(xi)
            agg_y.append(yi)
    phases = np.asarray(uniq_x, float)
    ratios = np.asarray(agg_y, float)
    return phases, ratios


def make_R_from_table(phases, ratios, fill="edge", periodic=True):
    """
    Build R(phi) from samples on phases in [-0.5, 0.5).
    Caller is free to pass any phi; we wrap internally.

    fill: 'edge' -> clamp to table endpoints within the chosen interpolation window
          'one'  -> return 1.0 when extrapolating (non-periodic mode only)
    periodic: if True, interpolate on the circle (recommended for PRCs)
    """
    phases = np.asarray(phases, float)
    ratios = np.asarray(ratios, float)
    assert phases.ndim == ratios.ndim == 1 and phases.size == ratios.size
    assert phases.size >= 2, "Need at least 2 samples"
    # Require strictly increasing samples inside [-0.5, 0.5)
    assert np.all(np.diff(phases) > 0), "phases must be strictly increasing"
    assert np.all((-0.5 <= phases) & (phases < 0.5)), "phases must lie in [-0.5, 0.5)"

    if periodic:
        # Tile samples at ±1 to avoid the seam
        P = np.concatenate([phases - 1.0, phases, phases + 1.0])
        R = np.tile(ratios, 3)

        def Rfun(phi):
            phi = _wrap_pm_half(phi)
            # For each phi, choose the contiguous window [phi-0.5, phi+0.5) to interpolate in,
            # which guarantees we don't cross the seam.
            lo = phi - 0.5
            hi = phi + 0.5
            # Build indices to slice P,R within [lo,hi] for vectorized phi.
            # We do this by interpolating once per phi in its own window.
            phi = np.atleast_1d(phi)
            out = np.empty_like(phi, dtype=float)
            for i, x in enumerate(phi):
                mask = (P >= x - 0.5) & (P < x + 0.5)
                Pi = P[mask]; Ri = R[mask]
                # Safety: need at least two points to interpolate
                if Pi.size < 2:
                    # Fallback: nearest
                    j = np.argmin(np.abs(P - x))
                    out[i] = R[j]
                else:
                    # Interpolate within this local contiguous arc
                    out[i] = np.interp(x, Pi, Ri, left=Ri[0], right=Ri[-1])
            return float(out[0]) if out.size == 1 else out

        return Rfun
    else:
        # Non-periodic: behave like your original version
        left_val  = ratios[0]  if fill == "edge" else 1.0
        right_val = ratios[-1] if fill == "edge" else 1.0

        def Rfun(phi):
            d = _wrap_pm_half(phi)
            y = np.interp(d, phases, ratios, left=left_val, right=right_val)
            return float(y) if np.ndim(y) == 0 else y

        return Rfun


# =========================
# Distance functions
# =========================
def wasserstein_from_hist(p_density, q_density, bin_edges):
    """
    1D Wasserstein (W1) between two histogram densities on common bin_edges.
    p_density, q_density are 'density=True' hist outputs (sum p*width = 1).
    """
    p_density = np.asarray(p_density, float)
    q_density = np.asarray(q_density, float)
    w = np.diff(bin_edges)
    # CDFs at right edges of each bin
    cdf_p = np.cumsum(p_density * w)
    cdf_q = np.cumsum(q_density * w)
    # Integral of |F_p - F_q| over support ≈ sum |ΔCDF| * bin_width
    return float(np.sum(np.abs(cdf_p - cdf_q) * w))


def _hist_mode_center(hist: np.ndarray, bins: np.ndarray) -> float:
    """Return the center of the highest-density bin."""
    if hist.size == 0 or bins.size < 2 or np.all(~np.isfinite(hist)):
        return np.nan
    idx = int(np.nanargmax(hist))
    centers = 0.5 * (bins[:-1] + bins[1:])
    return float(centers[idx])


def _mode_distance(sim_hist: np.ndarray, exp_hist: np.ndarray, bins: np.ndarray) -> float:
    """Absolute distance between primary modes (bin centers)."""
    s = _hist_mode_center(sim_hist, bins)
    e = _hist_mode_center(exp_hist, bins)
    return np.inf if (not np.isfinite(s) or not np.isfinite(e)) else abs(s - e)


def dphi_from_R_phasejump(phi, Rphi):
    """
    Given current phase phi in [0,1) and PRC ratio R(phi),
    return the phase jump Δφ so that the post-LED IFI equals R(phi)*T.
    """
    # target next-interval fraction = R(phi)
    # we need 1 - frac(phi + Δφ) = R(phi)
    raw = 1.0 - phi - Rphi           # unwrapped Δφ
    # wrap Δφ so that new phase is in [0,1)
    # (this also handles the case raw < -phi or raw >= 1-phi)
    dphi = ((raw + 1.0) % 1.0) - 1.0 if raw < -phi else (raw % 1.0) if raw >= (1.0 - phi) else raw
    return dphi


def _rank_top_n(values: Dict[Any, float], n: int = 3, ascending: bool = True) -> Tuple[list, Dict[Any, int]]:
    """
    Return the top-n keys by value and a rank map (1=best).
    Skips NaNs/inf. Ties are broken by stable sort on (value, key).
    """
    items = [(k, v) for k, v in values.items() if np.isfinite(v)]
    if not ascending:
        items.sort(key=lambda kv: (kv[1], kv[0]))  # but we'll never use descending here
    else:
        items.sort(key=lambda kv: (kv[1], kv[0]))
    top = [k for k, _ in items[:n]]
    rank = {k: i+1 for i, (k, _) in enumerate(items)}  # 1-based
    return top, rank


# =========================
# Fast simulator (kept as utility; parametric version below)
# =========================
def simulate_firefly_with_led_from_prc_fast(
        led_period_ms,
        R_fn,                # callable R(phi) built from CSV (kept for compatibility)
        emp_arr,             # baseline period sampler
        w,                   # phase offset
        sim_duration_s,
        led_start_s,
        seed=None,
        precomputed_led_onsets=None,  # <- pass array to reuse across sims
        idle_timeout_init=(15.0, 25.0)  # <- randomized range each reset
    ):
    """
    Faster but behavior-identical version:
      - Vectorized bulk emission of flashes between events
      - Optional precomputed LED onsets reuse
      - Inlined helpers to avoid Python call overhead in hot paths
    """
    rng = np.random.default_rng(seed)

    # ----- helpers (inlined math for speed) -----
    def sample_period():
        idx = rng.integers(0, emp_arr.shape[0])
        return float(emp_arr[idx])

    def wrap_pm_half(x):
        y = (x + 0.5) % 1.0
        return y - 0.5

    def _dphi_from_R(phi01, Rval):
        return 1.0 - Rval * (1.0 - phi01)

    # ----- LED onsets -----
    if precomputed_led_onsets is not None:
        led_onsets = precomputed_led_onsets
    else:
        if led_start_s >= sim_duration_s:
            led_onsets = np.empty(0, dtype=float)
        else:
            led_period_s = float(led_period_ms) / 1000.0
            n_led = int(np.floor((sim_duration_s - led_start_s) / led_period_s)) + 1
            led_onsets = led_start_s + led_period_s * np.arange(n_led, dtype=float)

    # ----- initial state -----
    idle_timeout_s = 30.0
    last_boost_time = 0.0
    next_quiet_time = last_boost_time + idle_timeout_s if idle_timeout_s is not None else np.inf

    t = 0.0
    phi = 0.0
    base_period_s = sample_period()

    flash_times = [0.0]  # start at t=0 like your original

    # ----- bulk advance: emit flashes up to target_t (no LED or quiescence events inside) -----
    def advance_to(target_t):
        nonlocal t, phi, base_period_s, flash_times

        if target_t <= t:
            return

        time_to_next = (1.0 - phi) * base_period_s
        if t + time_to_next > target_t:
            dt = (target_t - t)
            phi = (phi + dt / base_period_s)
            if phi >= 1.0:
                phi = phi % 1.0
            t = target_t
            return

        first_flash_t = t + time_to_next
        n_flashes = int(np.floor((target_t - first_flash_t) / base_period_s)) + 1

        new_flashes = first_flash_t + base_period_s * np.arange(n_flashes, dtype=float)
        flash_times.extend(new_flashes.tolist())

        t = target_t
        last_flash_t = new_flashes[-1]
        dt_after_last = t - last_flash_t
        phi = (dt_after_last / base_period_s)

    def randomize_base_period(now_t):
        nonlocal base_period_s, last_boost_time, next_quiet_time, idle_timeout_s
        base_period_s = sample_period()
        last_boost_time = now_t
        lo, hi = idle_timeout_init
        idle_timeout_s = float(np.random.uniform(lo, hi))
        next_quiet_time = last_boost_time + idle_timeout_s if idle_timeout_s is not None else np.inf

    for onset in led_onsets:
        while next_quiet_time <= onset:
            advance_to(next_quiet_time)
            randomize_base_period(now_t=next_quiet_time)

        advance_to(onset)

        phi_eval = wrap_pm_half(phi - w)
        Rval = float(R_fn(phi_eval))
        phi01 = phi if (0.0 <= phi < 1.0) else (phi % 1.0)
        dphi = _dphi_from_R(phi01, Rval)
        phi = (phi01 + dphi) % 1.0

    while next_quiet_time <= sim_duration_s:
        advance_to(next_quiet_time)
        randomize_base_period(now_t=next_quiet_time)

    advance_to(sim_duration_s)

    return np.asarray(flash_times, dtype=float), np.asarray(led_onsets, dtype=float)


# =========================
# Simulator (phase in (-0.5,0.5)) with optional precomputed LED onsets (for workers)
# =========================
def simulate_firefly_with_led_parametric(led_period_ms, z_fn, r_fn, emp_arr, w,
                                         sim_duration_s, led_start_s, seed=None,
                                         flash_len_s=0.0, refrac_kick_scale=0.0,
                                         precomputed_led_onsets: np.ndarray = None):
    rng = np.random.default_rng(seed)
    led_period_s = float(led_period_ms) / 1000.0
    base_period_s = sample_from_loaded_empirical(rng, emp_arr)

    # LED onsets
    if precomputed_led_onsets is not None:
        led_onsets = precomputed_led_onsets
    else:
        if led_start_s >= sim_duration_s:
            led_onsets = np.array([], float)
        else:
            n_led = int(np.floor((sim_duration_s - led_start_s) / led_period_s)) + 1
            led_onsets = led_start_s + led_period_s * np.arange(n_led)

    # optional "quiescence" timer (unchanged)
    idle_timeout_s = 20.0
    last_boost_time = 0.0
    next_quiet_time = last_boost_time + idle_timeout_s if idle_timeout_s is not None else np.inf

    # state
    t = 0.0
    phi = 0.0
    t_refrac_end = -np.inf  # refractory end time
    flash_times = [0.0]

    onset_phases, onset_time_delays, onset_dphis, onset_next_intervals = [], [], [], []

    def advance_to(target_t):
        nonlocal t, phi, t_refrac_end
        while True:
            if t < t_refrac_end:
                t = min(target_t, t_refrac_end)
                phi = 0.0
                if t < target_t:
                    continue
                else:
                    break

            time_to_next_flash = (1.0 - phi) * base_period_s
            t_next = t + time_to_next_flash

            if t_next <= target_t:
                t = t_next
                flash_times.append(t)
                phi = 0.0
                t_refrac_end = t + float(flash_len_s)
            else:
                dt = target_t - t
                phi = (phi + dt / base_period_s) % 1.0
                t = target_t
                break

    def randomize_base_period(now_t, ea):
        nonlocal base_period_s, last_boost_time, next_quiet_time, idle_timeout_s
        base_period_s = sample_from_loaded_empirical(np.random.default_rng(), ea)
        last_boost_time = now_t
        idle_timeout_s = float(np.random.uniform(25.0, 35.0))
        next_quiet_time = last_boost_time + idle_timeout_s

    for onset in led_onsets:
        while next_quiet_time <= onset:
            advance_to(next_quiet_time)
            randomize_base_period(now_t=next_quiet_time, ea=emp_arr)

        advance_to(onset)

        if t < t_refrac_end:
            continue

        phi_eval = _wrap_pm_half(phi - w)
        onset_phases.append(float(phi_eval))
        onset_time_delays.append(float(phi_eval) * float(base_period_s))

        dphi = float(z_fn(phi_eval))
        onset_dphis.append(dphi)

        phi_post = phi + dphi
        if phi_post >= 1.0:
            flash_times.append(t)
            phi = 0.0
            t_refrac_end = t + float(flash_len_s)
            onset_next_intervals.append(0.0)
        else:
            phi = phi_post % 1.0
            R_here = float(r_fn(phi_eval))
            onset_next_intervals.append(R_here * base_period_s)

        if abs(dphi) >= 0.05:
            last_boost_time = t
            next_quiet_time = last_boost_time + idle_timeout_s if idle_timeout_s is not None else np.inf

    while next_quiet_time <= sim_duration_s:
        advance_to(next_quiet_time)
        randomize_base_period(now_t=next_quiet_time, ea=emp_arr)

    advance_to(sim_duration_s)

    return np.array(flash_times), np.array(led_onsets)


# =========================
# Visualization util (unchanged)
# =========================
def visualize_best_by_beta(
        best_by_beta,
        BINS_MS,
        exp_samples_by_led,  # optional: dict[led_ms] -> 1D array of samples (ms)
        save_dir,
        show,
        max_cols,
        suptitle_prefix="LED"
):
    bin_edges = np.asarray(BINS_MS, dtype=float)
    bin_lefts = bin_edges[:-1]
    bin_widths = np.diff(bin_edges)
    n_bins = len(bin_edges) - 1

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for led_ms, recs in best_by_beta.items():
        if not recs:
            continue

        parsed = []
        for r in recs:
            beta = r.get('beta', r.get('beta_scale', None))
            hist = r.get('hist', None)
            if beta is None or hist is None:
                continue
            hist = np.asarray(hist, dtype=float)
            if hist.shape[0] != n_bins:
                continue
            parsed.append({
                'beta': float(beta),
                'k1': r.get('k1', None),
                'k2': r.get('k2', None),
                'dist': r.get('dist', np.nan),
                'hist': hist
            })
        if not parsed:
            continue

        parsed.sort(key=lambda d: d['beta'])
        betas = np.array([d['beta'] for d in parsed], dtype=float)
        dists = np.array([d['dist'] for d in parsed], dtype=float)

        exp_hist = None
        if exp_samples_by_led is not None and led_ms in exp_samples_by_led:
            samples = np.asarray(exp_samples_by_led[led_ms], dtype=float)
            mask = (samples >= bin_edges[0]) & (samples <= bin_edges[-1])
            samples = samples[mask]
            if samples.size > 0:
                exp_hist, _ = np.histogram(samples, bins=bin_edges, density=True)

        n = len(parsed)
        cols = min(max_cols, n)
        rows = math.ceil(n / cols)

        fig_h, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
        fig_h.suptitle(f"{suptitle_prefix} {led_ms} ms — histograms by β", y=0.995)

        for i, rec in enumerate(parsed):
            ax = axes[i // cols, i % cols]
            ax.bar(bin_lefts, rec['hist'], width=bin_widths, align='edge', alpha=0.65, label='sim')

            if exp_hist is not None:
                ax.step(bin_lefts, exp_hist, where='post', linewidth=1.0, label='exp')

            k1 = rec['k1']
            k2 = rec['k2']
            dist = rec['dist']
            ttl = [fr"$\\beta$={rec['beta']:.3g}"]
            if k1 is not None and k2 is not None:
                ttl.append(fr"$k_1$={float(k1):.3g}, $k_2$={float(k2):.3g}")
            if np.isfinite(dist):
                ttl.append(fr"dist={dist:.3g}")
            ax.set_title(" | ".join(ttl), fontsize=9)

            ax.set_xlabel("Interval (ms)")
            ax.set_ylabel("Density")
            ax.tick_params(labelsize=8)

            if i == 0:
                ax.legend(fontsize=8, frameon=False)

        for j in range(n, rows * cols):
            axes[j // cols, j % cols].axis('off')

        fig_h.tight_layout(rect=[0, 0, 1, 0.97])

        if save_dir is not None:
            out_path = os.path.join(save_dir, f"led_{int(led_ms)}ms_hist_by_beta.png")
            fig_h.savefig(out_path, dpi=200)

        plt.close(fig_h)

        fig_d, axd = plt.subplots(figsize=(5, 3.2))
        axd.plot(betas, dists, marker='o')
        axd.set_title(f"{suptitle_prefix} {led_ms} ms — distance vs β")
        axd.set_xlabel(r"$\\beta$")
        axd.set_ylabel("distance")
        axd.grid(True, alpha=0.3)

        if np.isfinite(dists).any():
            j_best = np.nanargmin(dists)
            axd.scatter([betas[j_best]], [dists[j_best]], s=60, zorder=3)
            axd.annotate(f"best β={betas[j_best]:.3g}\n{dists[j_best]:.3g}",
                         (betas[j_best], dists[j_best]),
                         textcoords="offset points", xytext=(6, 8), fontsize=8)

        fig_d.tight_layout()
        if save_dir is not None:
            out_path = os.path.join(save_dir, f"led_{int(led_ms)}ms_dist_vs_beta.png")
            fig_d.savefig(out_path, dpi=200)
        plt.close(fig_d)


# =========================
# Parallel worker infra
# =========================
_G = {}


def _worker_init(emp_arr: np.ndarray, bins_ms: np.ndarray, led_onsets_by_p: Dict[int, np.ndarray],
                 led_start_by_p: Dict[int, float], trials_per_combo: int, exp_hist_by_led: Dict[int, np.ndarray]):
    _G["emp_arr"] = np.asarray(emp_arr, dtype=float)
    _G["bins_ms"] = np.asarray(bins_ms, dtype=float)
    _G["led_onsets_by_p"] = {int(k): np.asarray(v, dtype=float) for k, v in led_onsets_by_p.items()}
    _G["led_start_by_p"] = {int(k): float(v) for k, v in led_start_by_p.items()}
    _G["trials"] = int(trials_per_combo)
    _G["exp_hist_by_led"] = {int(k): np.asarray(v, dtype=float) for k, v in exp_hist_by_led.items()}


def _simulate_param_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    One task = aggregate over TRIALS_PER_COMBO trials for a single (p_ms, kappa, phi_c, Ymax, y0).
    Returns dict with keys: led_ms, kappa, phi_c, Ymax, y0, dist, hist
    """
    p_ms = int(task["led_ms"])
    kappa = float(task["kappa"])
    phi_c = round(float(task["phi_c"]), 3)
    Ymax = round(float(task["Ymax"]), 3)
    y0 = round(float(task["y0"]), 3)

    # Build parametric PRC functions (fast pure-numpy callables)
    def r_fn(phi):
        return R_tanh_step(phi, Ymax=Ymax, kappa=kappa, phi_c=phi_c, y0=y0)
    def z_fn(phi):
        return Z_from_R(phi, r_fn)

    emp_arr = _G["emp_arr"]
    led_onsets = _G["led_onsets_by_p"][p_ms]
    led_start = _G["led_start_by_p"][p_ms]
    bins_ms = _G["bins_ms"]
    trials = _G["trials"]

    pieces: List[np.ndarray] = []
    for tt in range(trials):
        flashes, onsets = simulate_firefly_with_led_parametric(
            p_ms, z_fn, r_fn, emp_arr, 0.0,
            sim_duration_s=float(task["SIM_DURATION_S"]),
            led_start_s=led_start,
            seed=None,
            flash_len_s=0.03,
            refrac_kick_scale=0.0,
            precomputed_led_onsets=led_onsets,
        )
        ff_post = flashes[flashes >= led_start]
        if ff_post.size >= 2:
            pieces.append(np.diff(ff_post))

    if len(pieces) == 0:
        return {"led_ms": p_ms, "kappa": kappa, "phi_c": phi_c, "Ymax": Ymax, "y0": y0,
                "dist": float("nan"), "hist": None}

    sim_all = np.concatenate(pieces)
    sim_ms = 1000.0 * sim_all[np.isfinite(sim_all)]

    sim_ms_win = _range_filter_ms(sim_ms, bins_ms)
    if sim_ms_win.size == 0:
        return {"led_ms": p_ms, "kappa": kappa, "phi_c": phi_c, "Ymax": Ymax, "y0": y0,
                "dist": float("nan"), "hist": None}

    exp_hist = _G["exp_hist_by_led"][p_ms]
    sim_hist, _ = np.histogram(sim_ms_win, bins=bins_ms, density=True)

    dist = wasserstein_from_hist(sim_hist, exp_hist, bins_ms)

    print(
        f"{datetime.now()} — task completed. "
        f"{trials} trials of {p_ms} ms, "
        f"{kappa} kappa, {phi_c} phi_c, "
        f"{Ymax} Ymax, {y0} y0."
    )

    return {
        "led_ms": p_ms,
        "kappa": kappa,
        "phi_c": phi_c,
        "Ymax": Ymax,
        "y0": y0,
        "dist": float(dist),
        "hist": sim_hist.astype(float),
    }


# =========================
# Main driver with parallelization
# =========================
def simulate_from_prc(args):
    # ------------- unchanged preamble (paths, bins, data loads) -------------
    EMPIRICAL_PATH = args.before_path
    EXPER_PATH = args.after_path
    LED_PERIODS_MS = args.led_periods_ms

    SIM_DURATION_S = 300.0
    LED_START_S = 60.0
    TRIALS_PER_COMBO = args.simulation_trials

    # ---- parameter grids ----
    KAPPA_VALUES = np.array(getattr(args, "kappa_values", [6.0, 10.0, 14.0]), float)
    PHIC_VALUES  = np.array(getattr(args, "phi_c_values", [-0.05, -0.02, 0.0, 0.02, 0.05]), float)
    YMAX_VALUES  = np.array(getattr(args, "ymax_values", [0.30, 0.50, 0.80]), float)
    Y0_VALUES    = np.array(getattr(args, "y0_values", [0.96, 0.98, 1.00]), float)

    print(f'Running with κ={KAPPA_VALUES}, φ_c={PHIC_VALUES}, Ymax={YMAX_VALUES}, y0={Y0_VALUES}; '
          f'{TRIALS_PER_COMBO} trials per combo')

    # Histogram bins for ISIs (ms)
    BINS_MS = np.linspace(200, 1400, 35)
    BIN_CENTERS_MS = 0.5 * (BINS_MS[:-1] + BINS_MS[1:])

    emp_arr = load_empirical_before_seconds(EMPIRICAL_PATH, lo=0.3, hi=2.0)
    exp_samples_by_led = load_experimental_samples_ms(EXPER_PATH, LED_PERIODS_MS, lo_s=0.2, hi_s=2.0)

    # Precompute LED onsets & start times per LED period (consistent per-LED)
    LED_ONSETS_BY_P = {}
    LED_START_BY_P = {}
    for p_ms in LED_PERIODS_MS:
        led_start_s = float(np.random.uniform(low=LED_START_S - 2.0, high=LED_START_S))
        LED_START_BY_P[p_ms] = led_start_s
        LED_ONSETS_BY_P[p_ms] = precompute_led_onsets([p_ms], SIM_DURATION_S, led_start_s)[p_ms]

    # Precompute experimental histogram once per LED (density=True, matching previous logic)
    exp_hist_by_led = {}
    for p_ms in LED_PERIODS_MS:
        exp_ms_win = _range_filter_ms(exp_samples_by_led[p_ms], BINS_MS)
        if exp_ms_win.size == 0:
            exp_hist_by_led[p_ms] = np.zeros(len(BINS_MS) - 1, dtype=float)
        else:
            exp_hist_by_led[p_ms], _ = np.histogram(exp_ms_win, bins=BINS_MS, density=True)

    # ---- outputs (note: heatmaps now per-y0) ----
    best_params = {}        # per LED: overall best across all 4 dims
    heatmaps = {}           # per LED: dict[y0] -> 3D cube [len(κ), len(φ_c), len(Ymax)]
    best_by_y0_ymax = {}    # per LED: dict[(y0, Ymax)] -> best (over κ, φ_c)
    all_res = {}

    # Build all tasks across LEDs and param combos (aggregate per combo over trials inside worker)
    tasks: List[Dict[str, Any]] = []
    for p_ms in LED_PERIODS_MS:
        for y0 in Y0_VALUES:
            for kap in KAPPA_VALUES:
                for phic in PHIC_VALUES:
                    for Ymax in YMAX_VALUES:
                        tasks.append({
                            "led_ms": int(p_ms),
                            "kappa": float(kap),
                            "phi_c": float(phic),
                            "Ymax": float(Ymax),
                            "y0": float(y0),
                            "SIM_DURATION_S": float(SIM_DURATION_S),
                        })

    # Run in parallel
    max_workers = min(os.cpu_count() // 2 or 4, getattr(args, "max_workers", os.cpu_count() or 4))
    print(f"Launching {len(tasks)} param-combo tasks across {max_workers} workers ...")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
        initargs=(emp_arr, BINS_MS, LED_ONSETS_BY_P, LED_START_BY_P, TRIALS_PER_COMBO, exp_hist_by_led),
    ) as ex:
        futures = [ex.submit(_simulate_param_task, t) for t in tasks]
        results = []
        for fut in as_completed(futures):
            results.append(fut.result())

    # Assemble cubes and bests from results
    # Prepare index maps for fast placement
    idx_kappa = {round(float(v),3): i for i, v in enumerate(KAPPA_VALUES)}
    idx_phic  = {round(float(v),3): j for j, v in enumerate(PHIC_VALUES)}
    idx_ymax  = {round(float(v),3): k for k, v in enumerate(YMAX_VALUES)}

    for p_ms in LED_PERIODS_MS:
        cubes_for_y0 = {}
        per_y0_ymax_bests = {}
        overall_best = {"dist": np.inf, "kappa": None, "phi_c": None, "Ymax": None, "y0": None, "hist": None}

        for y0 in Y0_VALUES:
            cube = np.full((len(KAPPA_VALUES), len(PHIC_VALUES), len(YMAX_VALUES)), np.nan, float)
            cubes_for_y0[round(float(y0),3)] = cube
            # initialize slice bests for this y0
            for k_idx, Ymax in enumerate(YMAX_VALUES):
                per_y0_ymax_bests[(round(float(y0), 3), round(float(Ymax), 3))] = {
                    "dist": np.inf, "kappa": None, "phi_c": None,
                    "Ymax": round(float(Ymax), 3), "y0": round(float(y0), 3), "hist": None
                }

        # place results for this LED into cubes
        for rec in results:
            if int(rec["led_ms"]) != int(p_ms):
                continue
            k_i = idx_kappa.get(float(rec["kappa"]))
            j_i = idx_phic.get(float(rec["phi_c"]))
            y_i = idx_ymax.get(round(float(rec["Ymax"]),3))
            y0_val = round(float(rec["y0"]),3)
            cube = cubes_for_y0[y0_val]
            cube[k_i, j_i, y_i] = rec["dist"]

            key = (p_ms, float(rec["kappa"]), float(rec["phi_c"]), round(float(rec["Ymax"]),3), y0_val)
            all_res[key] = {
                "led_ms": p_ms,
                "dist": float(rec["dist"]),
                "kappa": float(rec["kappa"]),
                "phi_c": float(rec["phi_c"]),
                "Ymax": round(float(rec["Ymax"]),3),
                "y0": float(y0_val),
                "hist": None if rec["hist"] is None else np.asarray(rec["hist"], dtype=float),
            }

            # update slice bests for this (y0, Ymax)
            s_key = (y0_val, round(float(rec["Ymax"]),3))
            if np.isfinite(rec["dist"]) and rec["dist"] < per_y0_ymax_bests[s_key]["dist"]:
                per_y0_ymax_bests[s_key].update({
                    "dist": float(rec["dist"]),
                    "kappa": float(rec["kappa"]),
                    "phi_c": float(rec["phi_c"]),
                    "hist": None if rec["hist"] is None else np.asarray(rec["hist"], dtype=float),
                })

            # update overall best
            if np.isfinite(rec["dist"]) and rec["dist"] < overall_best["dist"]:
                overall_best.update({
                    "dist": float(rec["dist"]),
                    "kappa": float(rec["kappa"]),
                    "phi_c": float(rec["phi_c"]),
                    "Ymax": round(float(rec["Ymax"]),3),
                    "y0": float(y0_val),
                    "hist": None if rec["hist"] is None else np.asarray(rec["hist"], dtype=float),
                })

        heatmaps[p_ms] = cubes_for_y0
        best_params[p_ms] = overall_best
        best_by_y0_ymax[p_ms] = per_y0_ymax_bests

    # ----------------- (optional) plotting -----------------
    SHOW_HEATMAPS = args.show_heatmaps
    SHOW_OVERLAY = args.show_overlay
    SAVEFIGS = args.save_sim_figs
    SAVEDATA = args.save_sim_data

    if SHOW_HEATMAPS:
        for p_ms in LED_PERIODS_MS:
            cubes_for_y0 = heatmaps[p_ms]  # dict[y0] -> cube shape [len(κ), len(φ_c), len(Ymax)]

            Y0_list = list(sorted(cubes_for_y0.keys()))
            Y0_idx = {y0: i for i, y0 in enumerate(Y0_list)}
            big = np.full((len(Y0_list), len(KAPPA_VALUES), len(PHIC_VALUES), len(YMAX_VALUES)),
                          np.nan, float)
            for y0_val, cube in cubes_for_y0.items():
                big[Y0_idx[y0_val], :, :, :] = cube

            best4_idx = _nanargmin_unravel(big)
            if best4_idx is None:
                print(f"[warn] LED {p_ms}: all NaNs in 4D tensor; skipping heatmaps")
                continue
            glob_i_y0, glob_i_kap, glob_i_pc, glob_i_ymax = best4_idx

            collapsed_phi = np.nanmin(big, axis=2)  # (y0, κ, Ymax)

            kap_star_idx = int(glob_i_kap)
            kap_star = float(KAPPA_VALUES[kap_star_idx])

            H_y0_Ymax = collapsed_phi[:, kap_star_idx, :]

            fig1, ax1 = plt.subplots(figsize=(6.8, 4.8))
            im1 = ax1.imshow(
                H_y0_Ymax,
                origin="lower",
                aspect="auto",
                extent=[YMAX_VALUES[0], YMAX_VALUES[-1], Y0_list[0], Y0_list[-1]],
                cmap="viridis"
            )
            im1.set_rasterized(True)
            cb1 = fig1.colorbar(im1, ax=ax1, shrink=0.9, label="Wasserstein distance")

            # Plot the star using the true global best indices.
            ax1.plot(
                YMAX_VALUES[glob_i_ymax],
                Y0_list[glob_i_y0],
                marker="*",
                ms=14, mfc="white", mec="black", mew=1.1
            )

            ax1.set_xlabel(r"$Y_{max}$")
            ax1.set_ylabel(r"$y_0$")
            ax1.set_title(
                f"LED {p_ms} ms — min D over $\phi_c$ with best $\kappa$={kap_star:.2f}\n"
                f"(map: $y_0 x Y_{{max}}$)"
            )
            ax1.set_xticks(YMAX_VALUES)
            ax1.set_yticks(Y0_list)
            ax1.grid(color="k", alpha=0.10, linestyle=":", linewidth=0.8)
            plt.tight_layout()
            if SAVEFIGS:
                Path(args.simulation_save_path).mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{args.simulation_save_path}/heatmap_y0_by_ymax_bestKappa_led{int(p_ms)}.png', dpi=200)
                plt.close(fig1)
            else:
                plt.show()

            y0_star_idx = int(glob_i_y0)
            y0_star = float(Y0_list[y0_star_idx])
            H_kap_Ymax = collapsed_phi[y0_star_idx, :, :]

            fig2, ax2 = plt.subplots(figsize=(6.8, 4.8))
            im2 = ax2.imshow(
                H_kap_Ymax,
                origin="lower",
                aspect="auto",
                extent=[YMAX_VALUES[0], YMAX_VALUES[-1], KAPPA_VALUES[0], KAPPA_VALUES[-1]],
                cmap="viridis"
            )
            im2.set_rasterized(True)
            cb2 = fig2.colorbar(im2, ax=ax2, shrink=0.9, label="Wasserstein distance")

            # Plot the star using the true global best indices.
            ax2.plot(
                YMAX_VALUES[glob_i_ymax],
                KAPPA_VALUES[glob_i_kap],
                marker="*",
                ms=14, mfc="white", mec="black", mew=1.1
            )

            ax2.set_xlabel(r"$Y_{max}$")
            ax2.set_ylabel(r"$\kappa$")
            ax2.set_title(
                f"LED {p_ms} ms — min D over $\phi_c$ with best $y_0$={y0_star:.3f}\n"
                f"(map: $\kappa x Y_{{max}}$)"
            )
            ax2.set_xticks(YMAX_VALUES)
            ax2.set_yticks(KAPPA_VALUES)
            ax2.grid(color="k", alpha=0.10, linestyle=":", linewidth=0.8)
            plt.tight_layout()
            if SAVEFIGS:
                Path(args.simulation_save_path).mkdir(parents=True, exist_ok=True)
                plt.savefig(f'{args.simulation_save_path}/heatmap_kappa_by_ymax_bestY0_led{int(p_ms)}.png', dpi=200)
                plt.close(fig2)
            else:
                plt.show()

    def _panel_heat(ax, H, xvals, yvals, xlabel, ylabel, title, star_x, star_y, cmap="viridis"):
        im = ax.imshow(
            H, origin="lower", aspect="auto",
            extent=[xvals[0], xvals[-1], yvals[0], yvals[-1]],
            cmap=cmap
        )
        im.set_rasterized(True)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Wasserstein D")
        ax.plot(star_x, star_y, marker="*", ms=12, mfc="white", mec="black", mew=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(xvals)
        ax.set_yticks(yvals)
        ax.grid(color="k", alpha=0.10, linestyle=":", linewidth=0.8)

    if SHOW_OVERLAY:
        for p_ms in LED_PERIODS_MS:
            cubes_for_y0 = heatmaps[p_ms]
            Y0_list = list(sorted(cubes_for_y0.keys()))
            Y0_idx = {y0: i for i, y0 in enumerate(Y0_list)}

            big = np.full((len(Y0_list), len(KAPPA_VALUES), len(PHIC_VALUES), len(YMAX_VALUES)), np.nan, float)
            for y0_val, cube in cubes_for_y0.items():
                big[Y0_idx[y0_val], :, :, :] = cube

            # This part correctly finds the global minimum.
            best4_idx = _nanargmin_unravel(big)
            if best4_idx is None:
                print(f"[warn] LED {p_ms}: all NaNs in 4D tensor; skipping FULL dashboard")
                continue
            gi_y0, gi_kap, gi_pc, gi_ymax = best4_idx
            y0_star, kap_star = round(float(Y0_list[gi_y0]), 3), float(KAPPA_VALUES[gi_kap])
            phic_star, ymax_star = float(PHIC_VALUES[gi_pc]), round(float(YMAX_VALUES[gi_ymax]), 3)
            D_best = float(big[gi_y0, gi_kap, gi_pc, gi_ymax])

            best_key = nearest_all_res_key(all_res, p_ms, kap_star, phic_star, ymax_star, y0_star)
            sim_hist = all_res.get(best_key, {}).get("hist", None)
            exp_ms_win = _range_filter_ms(exp_samples_by_led[p_ms], BINS_MS)
            exp_hist, _ = np.histogram(exp_ms_win, bins=BINS_MS, density=True)
            BIN_CENTERS_MS = 0.5 * (BINS_MS[:-1] + BINS_MS[1:])

            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(12.5, 12.5))
            gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[1.2, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.30)
            axTop = fig.add_subplot(gs[0, :])
            if sim_hist is not None and len(sim_hist) == len(BINS_MS) - 1:
                axTop.plot(BIN_CENTERS_MS, sim_hist, lw=2.2, label="Sim (best 4D cell)")
            else:
                axTop.text(0.5, 0.50, "No sim hist for best cell", transform=axTop.transAxes,
                           ha="center", va="center")
            axTop.plot(BIN_CENTERS_MS, exp_hist, lw=1.8, alpha=0.85, label="Exp")
            axTop.set_ylabel("Density")
            axTop.set_title(
                (r"LED {} ms — Global best: $y_0={:.3f},\kappa={:.2f}, \phi_c={:+.3f}, "
                 r"Y_{{max}}={:.2f}$ (D={:.3g})").format(p_ms, y0_star, kap_star, phic_star, ymax_star, D_best)
            )
            axTop.legend(frameon=False)
            axTop.grid(alpha=0.25)
            heat_axes = [
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]),
            ]
            H_pairs = []
            # D(κ,φc). Slice is (κ,φc), but labels are (κ,φc)
            H_pairs.append((
                big[gi_y0, :, :, gi_ymax].T, PHIC_VALUES, KAPPA_VALUES,
                r"$\phi_c$", r"$\kappa$",
                rf"(fix $y_0={y0_star:.3f}$, $Y_{{max}}={ymax_star:.2f}$)", phic_star, kap_star
            ))
            # D(κ,Ymax). Slice is (κ,Ymax), labels are (κ,Ymax).
            H_pairs.append((
                big[gi_y0, :, gi_pc, :].T, YMAX_VALUES, KAPPA_VALUES,
                r"$Y_{{max}}$", r"$\kappa$",
                rf"(fix $y_0={y0_star:.3f}$, $\phi_c={phic_star:+.3f}$)", ymax_star, kap_star
            ))
            # D(y0,κ). Slice is (y0,κ). Transposed slice is (κ,y0), labels are (y0,κ)
            H_pairs.append((
                big[:, :, gi_pc, gi_ymax].T, Y0_list, KAPPA_VALUES,
                r"$y_0$", r"$\kappa$",
                rf"(fix $\phi_c={phic_star:+.3f}$, $Y_{{max}}={ymax_star:.2f}$)", y0_star, kap_star
            ))
            # D(y0,φc). Slice is (y0,φc), labels are (y0,φc)
            H_pairs.append((
                big[:, gi_kap, :, gi_ymax].T, PHIC_VALUES, Y0_list,
                r"$\phi_c$", r"$y_0$",
                rf"(fix $\kappa={kap_star:.2f}$, $Y_{{max}}={ymax_star:.2f}$)", phic_star, y0_star
            ))
            # D(y0,Ymax). Slice is (y0,Ymax), labels are (y0,Ymax)
            H_pairs.append((
                big[:, gi_kap, gi_pc, :].T, YMAX_VALUES, Y0_list,
                r"$Y_{{max}}$", r"$y_0$",
                rf"(fix $\kappa={kap_star:.2f}$, $\phi_c={phic_star:+.3f}$)", ymax_star, y0_star
            ))
            # D(φc,Ymax). Slice is (φc,Ymax), labels are (φc,Ymax)
            H_pairs.append((
                big[gi_y0, gi_kap, :, :].T, YMAX_VALUES, PHIC_VALUES,
                r"$Y_{{max}}$", r"$\phi_c$",
                rf"(fix $\kappa={kap_star:.2f}$, $y_0={y0_star:.3f}$)", ymax_star, phic_star
            ))

            titles = [
                r"$D(\kappa,\phi_c)$ ",
                rf"$D(\kappa,Y_{{max}})$ ",
                r"$D(y_0, \kappa)$ ",
                r"$D(y_0,\phi_c)$ ",
                rf"$D(y_0,Y_{{max}})$ ",
                rf"$D(\phi_c,Y_{{max}})$ ",
            ]
            for ax, (H, xvals, yvals, xlabel, ylabel, fixlbl, star_x, star_y), ttl in zip(heat_axes, H_pairs, titles):
                _panel_heat(
                    ax, H, np.asarray(xvals, float), np.asarray(yvals, float),
                    xlabel, ylabel, ttl + fixlbl,
                    star_x, star_y
                )

            fig.suptitle(f"LED {p_ms} ms — Full pairwise landscapes at global best, plus best histogram", y=0.995,
                         fontsize=13)
            if SAVEFIGS:
                Path(args.simulation_save_path).mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{args.simulation_save_path}/FULL_dashboard_led{int(p_ms)}.png", dpi=200)
                plt.close(fig)
            else:
                plt.show()

        exp_hist_by_led_plot = {}
        for p_ms in LED_PERIODS_MS:
            exp_ms_win = _range_filter_ms(exp_samples_by_led[p_ms], BINS_MS)
            exp_hist_by_led_plot[p_ms], _ = np.histogram(exp_ms_win, bins=BINS_MS, density=True)
        figO, axesO = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
        axesO = axesO.ravel()
        for ax, p_ms in zip(axesO, LED_PERIODS_MS):
            bp = best_params[p_ms]
            if bp["hist"] is None:
                ax.text(0.5, 0.5, "No sim", transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"LED {p_ms} ms")
                continue
            ax.plot(BIN_CENTERS_MS, bp["hist"], label="Sim (best)", lw=2)
            ax.plot(BIN_CENTERS_MS, exp_hist_by_led_plot[p_ms], label="Exp", lw=2, alpha=0.8)
            ax.set_title(
                f"LED {p_ms} ms\nbest: κ={bp['kappa']:.2f}, φ_c={bp['phi_c']:+.3f}, "
                f"Ymax={bp['Ymax']:.2f}, y0={bp['y0']:.3f}, dist={bp['dist']:.3f}")
            ax.set_xlabel("ISI (ms)")
            ax.set_ylabel("Density")
        axesO[0].legend()
        figO.suptitle("Best-fit simulated vs experimental ISI distributions (tanh PRC)", y=1.02, fontsize=12)
        plt.tight_layout()
        if SAVEFIGS:
            Path(args.simulation_save_path).mkdir(parents=True, exist_ok=True)
            plt.savefig(f'{args.simulation_save_path}/simulation_histogram_BESTFITS_tanh_prc.png')
            plt.close()
        else:
            plt.show()

    if SAVEDATA:
        out_dir = Path(args.simulation_save_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(f"{out_dir}/all_res_tanh.pkl", "wb") as f:
            pickle.dump(all_res, f)

        for p_ms in LED_PERIODS_MS:
            for y0, cube in heatmaps[p_ms].items():
                np.savez_compressed(out_dir / f"heatmap_tanh_led{int(p_ms)}_y0_{y0:.3f}.npz",
                                    cube=np.asarray(cube, dtype=np.float32),
                                    kappa=np.asarray(KAPPA_VALUES, dtype=np.float32),
                                    phi_c=np.asarray(PHIC_VALUES, dtype=np.float32),
                                    Ymax=np.asarray(YMAX_VALUES, dtype=np.float32),
                                    y0=float(y0))

        def _to_py(obj):
            if isinstance(obj, (np.floating, np.integer)): return obj.item()
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj

        best_params_clean = {
            str(int(p_ms)): {k: _to_py(v) for k, v in bp.items() if k != "hist"}
            for p_ms, bp in best_params.items()
        }
        with open(out_dir / "best_params_tanh_prc.json", "w") as f:
            json.dump(best_params_clean, f, indent=2)

        np.savez_compressed(
            out_dir / "bestfit_hists_tanh_prc.npz",
            **{f"led_{int(p_ms)}": (np.asarray(best_params[p_ms]["hist"], dtype=np.float32)
                                    if best_params[p_ms]["hist"] is not None else np.array([], np.float32))
               for p_ms in LED_PERIODS_MS}
        )

        manifest = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "led_periods_ms": [int(x) for x in LED_PERIODS_MS],
            "kappa_values": [float(x) for x in KAPPA_VALUES],
            "phi_c_values": [float(x) for x in PHIC_VALUES],
            "ymax_values": [float(x) for x in YMAX_VALUES],
            "y0_values": [float(x) for x in Y0_VALUES],
            "bins_ms": [float(x) for x in BINS_MS],
            "distance_metric": "wasserstein_distance(W1, histogram-based, ms)",
            "sim_controls": {"SIM_DURATION_S": SIM_DURATION_S,
                             "LED_START_S": LED_START_S,
                             "TRIALS_PER_COMBO": TRIALS_PER_COMBO},
            "files": {
                "best_params_json": "best_params_tanh_prc.json",
                "bestfit_hists_npz": "bestfit_hists_tanh_prc.npz",
                "all_res_pkl": "all_res_tanh.pkl",
                "heatmaps_prefix": "heatmap_tanh_led<LED>_y0_<y0>.npz"
            }
        }
        with open(out_dir / "manifest_tanh_prc.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return str(out_dir / "manifest_tanh_prc.json")
    else:
        return 'simulation data not saved'


def load_sim_outputs(save_path='sim_data'):
    out_dir = Path(save_path)
    manifest = json.load(open(out_dir / "manifest_triangle_prc.json"))
    heatmaps = np.load(out_dir / manifest["files"]["heatmaps_npz"])
    with open(out_dir / manifest["files"]["best_params_json"]) as f:
        best_params = json.load(f)
    bestfit_hists = np.load(out_dir / manifest["files"]["bestfit_hists_npz"])
    return heatmaps, best_params, bestfit_hists, manifest
