import os
import glob
import json
import traceback
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter, peak_widths
from pyuvdata import UVData


# =========================
# 설정
# =========================
DATA_DIR = "./hera_uvh5"   # uvh5 파일 폴더
GLOB_PATTERN = "zen.LST.baseline.*.sum.uvh5"
OUT_CSV = "comb_results.csv"
OUT_JSON = "comb_results.json"

# comb spacing search range (MHz)
DF_MIN_MHZ = 1.0
DF_MAX_MHZ = 5.0

# peak finding
PEAK_SIGMA = 2.5
TOOTH_MATCH_TOL_FRAC = 0.20   # spacing의 20% 안이면 같은 tooth로 간주

# smoothing
SG_WINDOW = 31   # 홀수여야 함
SG_POLY = 3

# drift 계산용: 시간축별 offset 추정
USE_TIME_DRIFT = False


# =========================
# 결과 구조
# =========================
@dataclass
class CombResult:
    file: str
    pol: str
    n_times: int
    n_freqs: int
    fmin_mhz: float
    fmax_mhz: float
    delta_f_hat_mhz: float
    conf: float
    grid_score: float
    C: float
    O: float
    dwell_med: float
    jitter_mhz: float
    drift_mhz_per_timebin: float
    linewidth_mhz: float
    A_pol: float
    band_class: str
    morph_class: str
    pol_class: str
    sat_stage: str
    sat_model: str
    peak_freq: list


# =========================
# 유틸
# =========================
def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def safe_savgol(y: np.ndarray, window: int = 31, poly: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) < 7:
        return np.full_like(y, np.nanmedian(y))
    w = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if w < poly + 2:
        return np.full_like(y, np.nanmedian(y))
    if w % 2 == 0:
        w -= 1
    return savgol_filter(y, window_length=w, polyorder=poly, mode="interp")


def detrend_spectrum(spec: np.ndarray) -> np.ndarray:
    trend = safe_savgol(spec, SG_WINDOW, SG_POLY)
    resid = spec - trend
    return resid


def autocorr_nonneg(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0)
    x = x - np.mean(x)
    ac = np.correlate(x, x, mode="full")
    ac = ac[len(ac)//2:]
    if ac[0] != 0:
        ac = ac / ac[0]
    return ac


def estimate_spacing_from_autocorr(freq_mhz: np.ndarray, resid: np.ndarray):
    """
    주파수 residual의 자기상관에서 spacing 추정
    """
    dnu = np.median(np.diff(freq_mhz))
    ac = autocorr_nonneg(resid)

    lag_axis = np.arange(len(ac)) * dnu
    valid = (lag_axis >= DF_MIN_MHZ) & (lag_axis <= DF_MAX_MHZ)
    if np.sum(valid) < 5:
        return np.nan, 0.0, lag_axis, ac

    ac_valid = ac[valid]
    lag_valid = lag_axis[valid]

    peak_idx = np.argmax(ac_valid)
    delta_f_hat = lag_valid[peak_idx]

    background = np.nanmedian(ac_valid)
    spread = np.nanstd(ac_valid) + 1e-12
    conf = (ac_valid[peak_idx] - background) / spread
    return float(delta_f_hat), float(conf), lag_axis, ac


def build_comb_grid(freq_mhz: np.ndarray, delta_f_mhz: float, phase_mhz: float):
    """
    f = phase + k * delta_f 형태의 comb tooth 좌표 생성
    """
    if not np.isfinite(delta_f_mhz) or delta_f_mhz <= 0:
        return np.array([], dtype=float)

    f0 = freq_mhz[0]
    f1 = freq_mhz[-1]

    kmin = int(np.floor((f0 - phase_mhz) / delta_f_mhz)) - 1
    kmax = int(np.ceil((f1 - phase_mhz) / delta_f_mhz)) + 1

    teeth = phase_mhz + np.arange(kmin, kmax + 1) * delta_f_mhz
    teeth = teeth[(teeth >= f0) & (teeth <= f1)]
    return teeth


def nearest_offsets(peaks_freq: np.ndarray, teeth_freq: np.ndarray):
    """
    각 peak가 가장 가까운 tooth에서 얼마나 벗어났는지 계산
    """
    if len(peaks_freq) == 0 or len(teeth_freq) == 0:
        return np.array([]), np.array([], dtype=int)

    idx = np.searchsorted(teeth_freq, peaks_freq)
    idx = np.clip(idx, 1, len(teeth_freq) - 1)

    left = teeth_freq[idx - 1]
    right = teeth_freq[idx]

    use_right = np.abs(peaks_freq - right) < np.abs(peaks_freq - left)
    nearest = np.where(use_right, right, left)
    nearest_idx = np.where(use_right, idx, idx - 1)

    offsets = peaks_freq - nearest
    return offsets, nearest_idx


def estimate_comb_phase(freq_mhz: np.ndarray, peaks_freq: np.ndarray, delta_f_mhz: float):
    """
    peak들을 가장 잘 설명하는 comb phase를 coarse search로 추정
    """
    if len(peaks_freq) == 0 or not np.isfinite(delta_f_mhz):
        return np.nan

    phases = np.linspace(freq_mhz[0], freq_mhz[0] + delta_f_mhz, 200)
    best_phase = np.nan
    best_score = -np.inf

    tol = TOOTH_MATCH_TOL_FRAC * delta_f_mhz

    for phase in phases:
        teeth = build_comb_grid(freq_mhz, delta_f_mhz, phase)
        if len(teeth) == 0:
            continue
        offsets, _ = nearest_offsets(peaks_freq, teeth)
        score = np.sum(np.abs(offsets) < tol)
        if score > best_score:
            best_score = score
            best_phase = phase

    return float(best_phase)


def find_spectral_peaks(freq_mhz: np.ndarray, resid: np.ndarray):
    z = robust_zscore(resid)
    peaks, props = find_peaks(z, height=PEAK_SIGMA)
    return peaks, props, z


def linewidth_from_peaks(freq_mhz: np.ndarray, z: np.ndarray, peaks: np.ndarray):
    if len(peaks) == 0:
        return np.nan
    widths, _, _, _ = peak_widths(z, peaks, rel_height=0.5)
    dnu = np.median(np.diff(freq_mhz))
    widths_mhz = widths * dnu
    return float(np.nanmedian(widths_mhz))


def comb_metrics_for_spectrum(freq_mhz: np.ndarray, spec: np.ndarray):
    """
    1개 spectrum에서 comb 관련 metric 계산
    """
    resid = detrend_spectrum(spec)
    peaks, props, z = find_spectral_peaks(freq_mhz, resid)
    peak_freq = freq_mhz[peaks]

    delta_f_hat, conf, lag_axis, ac = estimate_spacing_from_autocorr(freq_mhz, resid)

    if not np.isfinite(delta_f_hat) or len(peak_freq) == 0:
        return {
            "delta_f_hat_mhz": np.nan,
            "conf": 0.0,
            "grid_score": 0.0,
            "C": 0.0,
            "O": 0.0,
            "jitter_mhz": np.nan,
            "linewidth_mhz": linewidth_from_peaks(freq_mhz, z, peaks),
            "phase_mhz": np.nan,
            "n_peaks": len(peaks),
            "peak_freq": peak_freq,
        }

    phase_mhz = estimate_comb_phase(freq_mhz, peak_freq, delta_f_hat)
    teeth = build_comb_grid(freq_mhz, delta_f_hat, phase_mhz)

    if len(teeth) == 0:
        return {
            "delta_f_hat_mhz": delta_f_hat,
            "conf": conf,
            "grid_score": 0.0,
            "C": 0.0,
            "O": 0.0,
            "jitter_mhz": np.nan,
            "linewidth_mhz": linewidth_from_peaks(freq_mhz, z, peaks),
            "phase_mhz": phase_mhz,
            "n_peaks": len(peaks),
            "peak_freq": peak_freq,
        }

    offsets, nearest_idx = nearest_offsets(peak_freq, teeth)
    tol = TOOTH_MATCH_TOL_FRAC * delta_f_hat
    matched = np.abs(offsets) < tol

    # grid_score: peak 중 comb tooth에 잘 맞는 비율
    grid_score = float(np.mean(matched)) if len(matched) > 0 else 0.0

    # C: matched peak offset의 coherence를 spacing으로 정규화한 값
    if np.any(matched):
        jitter_mhz = float(np.nanstd(offsets[matched]))
        C = float(np.clip(1.0 - jitter_mhz / (delta_f_hat / 2 + 1e-12), 0, 1))
    else:
        jitter_mhz = np.nan
        C = 0.0

    # O: occupancy = tooth 중 실제 peak가 붙은 비율
    occupied_tooth_ids = np.unique(nearest_idx[matched]) if np.any(matched) else np.array([], dtype=int)
    O = float(len(occupied_tooth_ids) / len(teeth)) if len(teeth) > 0 else 0.0

    return {
        "delta_f_hat_mhz": float(delta_f_hat),
        "conf": float(conf),
        "grid_score": grid_score,
        "C": C,
        "O": O,
        "jitter_mhz": jitter_mhz,
        "linewidth_mhz": linewidth_from_peaks(freq_mhz, z, peaks),
        "phase_mhz": float(phase_mhz),
        "n_peaks": len(peaks),
        "peak_freq": peak_freq,
    }


def estimate_drift(freq_mhz: np.ndarray, waterfall: np.ndarray):
    """
    시간축별 comb phase 변화를 선형회귀로 추정
    drift 단위 = MHz / time-bin
    """
    if waterfall.ndim != 2 or waterfall.shape[0] < 3:
        return np.nan, np.nan

    phases = []
    valid_t = []

    for t in range(waterfall.shape[0]):
        spec = waterfall[t]
        metrics = comb_metrics_for_spectrum(freq_mhz, spec)
        phase = metrics["phase_mhz"]
        if np.isfinite(phase) and np.isfinite(metrics["delta_f_hat_mhz"]):
            phases.append(phase)
            valid_t.append(t)

    if len(phases) < 3:
        return np.nan, np.nan

    phases = np.array(phases)
    valid_t = np.array(valid_t)

    # 위상 wrapping 보정: delta_f 범위 안에서 unwrap 흉내
    for i in range(1, len(phases)):
        step = phases[i] - phases[i - 1]
        # phase jump가 너무 크면 주기 보정
        if np.abs(step) > np.nanmedian(np.abs(np.diff(phases))) * 3 and np.isfinite(step):
            pass

    coef = np.polyfit(valid_t, phases, 1)
    slope = coef[0]

    # dwell_med: 시간이 지나도 comb가 유지된 median run length
    is_valid = np.zeros(waterfall.shape[0], dtype=bool)
    is_valid[valid_t] = True
    runs = []
    run = 0
    for v in is_valid:
        if v:
            run += 1
        elif run > 0:
            runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    dwell_med = float(np.median(runs)) if len(runs) > 0 else 0.0

    return float(slope), dwell_med


def provisional_label(delta_f_hat, jitter, O, A_pol):
    if not np.isfinite(delta_f_hat):
        return "no_comb"

    if O > 0.25 and (np.isfinite(jitter) and jitter < 0.15 * delta_f_hat):
        if np.isfinite(A_pol) and A_pol > 0.30:
            return "stable_polarized_comb"
        return "stable_comb"

    if O > 0.10:
        return "sparse_comb"

    if np.isfinite(A_pol) and A_pol > 0.50:
        return "polarized_irregular"

    return "weak_or_irregular"


def read_uvh5_to_waterfalls(path: str):
    """
    uvh5 -> polarization별 waterfall 반환
    반환:
        freq_mhz, {"ee": 2D [time, freq], ...}

    shape 가 파일마다 다를 수 있어서 robust 하게 처리한다.
    """
    uv = UVData()
    uv.read(path)

    print(f"\n[DEBUG] file = {os.path.basename(path)}")
    print(f"[DEBUG] data_array.shape = {uv.data_array.shape}")
    print(f"[DEBUG] Nblts = {uv.Nblts}, Nfreqs = {uv.Nfreqs}, Npols = {uv.Npols}, Ntimes = {uv.Ntimes}")
    print(f"[DEBUG] polarization_array = {uv.polarization_array}")

    freq_mhz = np.asarray(uv.freq_array).squeeze() / 1e6
    if freq_mhz.ndim != 1:
        raise ValueError(f"freq_array squeeze 후 ndim={freq_mhz.ndim}, expected 1")

    data = uv.data_array
    dnu = np.median(np.diff(freq_mhz))
    print(f"[INFO] {os.path.basename(path)} channel spacing = {dnu:.9f} MHz")

    # pyuvdata 버전에 따라 shape가 다를 수 있음:
    # 1) (Nblts, Nfreqs, Npols)
    # 2) (Nblts, 1, Nfreqs, Npols)
    # 3) 기타 예외적 케이스
    if data.ndim == 3:
        # (Nblts, Nfreqs, Npols)
        arr = data
    elif data.ndim == 4:
        # (Nblts, Nspws, Nfreqs, Npols) 가정
        if data.shape[1] != 1:
            raise ValueError(f"Unexpected Nspws={data.shape[1]} in data_array.shape={data.shape}")
        arr = data[:, 0, :, :]
    else:
        raise ValueError(f"Unsupported data_array ndim={data.ndim}, shape={data.shape}")

    # arr shape = (Nblts, Nfreqs, Npols)
    if arr.shape[1] != len(freq_mhz):
        raise ValueError(
            f"Frequency axis mismatch: arr.shape[1]={arr.shape[1]} vs len(freq_mhz)={len(freq_mhz)}"
        )

    pol_nums = list(np.asarray(uv.polarization_array).astype(int))
    pol_map = {}
    for i, p in enumerate(pol_nums):
        if p == -5:
            pol_map["xx"] = i
        elif p == -6:
            pol_map["yy"] = i
        elif p == -7:
            pol_map["xy"] = i
        elif p == -8:
            pol_map["yx"] = i
        elif p == -24:
            pol_map["ee"] = i
        elif p == -25:
            pol_map["nn"] = i

    out = {}

    # amplitude waterfall 생성
    for label in ["ee", "nn", "xx", "yy"]:
        if label in pol_map:
            pol_idx = pol_map[label]
            wf = np.abs(arr[:, :, pol_idx])   # (Nblts, Nfreqs)
            out[label] = wf

    if len(out) == 0:
        raise ValueError(
            f"No supported pol found. polarization_array={uv.polarization_array}"
        )

    return freq_mhz, out


def analyze_file(path: str):
    freq_mhz, wf_dict = read_uvh5_to_waterfalls(path)

    # A_pol 계산을 위해 가능한 편파 쌍 선택
    pol_pair = None
    if "ee" in wf_dict and "nn" in wf_dict:
        pol_pair = ("ee", "nn")
    elif "xx" in wf_dict and "yy" in wf_dict:
        pol_pair = ("xx", "yy")

    pol_strength = {}
    for pol, wf in wf_dict.items():
        pol_strength[pol] = float(np.nanmedian(wf))

    A_pol_global = np.nan
    if pol_pair is not None:
        a = pol_strength[pol_pair[0]]
        b = pol_strength[pol_pair[1]]
        A_pol_global = float(np.abs(a - b) / (a + b + 1e-12))

    results = []

    # Band split ranges
    band_ranges = {
        "low_band": (46.9, 87.5),
        "high_band": (108.0, 234.3)
    }

    for pol, waterfall in wf_dict.items():
        mean_spec = np.nanmedian(waterfall, axis=0)
        for band_class, (band_min, band_max) in band_ranges.items():
            band_mask = (freq_mhz >= band_min) & (freq_mhz <= band_max)
            if np.sum(band_mask) < 10:
                continue
            band_freq = freq_mhz[band_mask]
            band_spec = mean_spec[band_mask]
            metrics = comb_metrics_for_spectrum(band_freq, band_spec)

            if USE_TIME_DRIFT:
                drift, dwell_med = estimate_drift(band_freq, waterfall[:, band_mask])
            else:
                drift, dwell_med = np.nan, np.nan

            # Morphology class
            morph_class = "stable_comb" if metrics["grid_score"] > 0.45 else "sparse_comb"
            # Polarization class
            pol_class = "polarized" if np.isfinite(A_pol_global) and A_pol_global > 0.1 else "unpolarized"
            # Satellite stage/model (default unknown)
            sat_stage = "unknown"
            sat_model = "unknown"

            result = CombResult(
                file=os.path.basename(path),
                pol=pol,
                n_times=int(waterfall.shape[0]),
                n_freqs=int(np.sum(band_mask)),
                fmin_mhz=float(band_freq.min()),
                fmax_mhz=float(band_freq.max()),
                delta_f_hat_mhz=float(metrics["delta_f_hat_mhz"]) if np.isfinite(metrics["delta_f_hat_mhz"]) else np.nan,
                conf=float(metrics["conf"]),
                grid_score=float(metrics["grid_score"]),
                C=float(metrics["C"]),
                O=float(metrics["O"]),
                dwell_med=float(dwell_med) if np.isfinite(dwell_med) else np.nan,
                jitter_mhz=float(metrics["jitter_mhz"]) if np.isfinite(metrics["jitter_mhz"]) else np.nan,
                drift_mhz_per_timebin=float(drift) if np.isfinite(drift) else np.nan,
                linewidth_mhz=float(metrics["linewidth_mhz"]) if np.isfinite(metrics["linewidth_mhz"]) else np.nan,
                A_pol=float(A_pol_global) if np.isfinite(A_pol_global) else np.nan,
                band_class=band_class,
                morph_class=morph_class,
                pol_class=pol_class,
                sat_stage=sat_stage,
                sat_model=sat_model,
                peak_freq=list(metrics["peak_freq"]) if "peak_freq" in metrics else []
            )
            results.append(result)

    return results


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, GLOB_PATTERN)))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {os.path.join(DATA_DIR, GLOB_PATTERN)}")

    all_rows = []

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] analyzing {os.path.basename(path)}")
        try:
            results = analyze_file(path)
            for r in results:
                all_rows.append(asdict(r))
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_JSON}")

    if len(df) > 0:
        print("\nTop candidates by conf:")
        print(df.sort_values(["conf", "grid_score", "O"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()