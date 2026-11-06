"""Поиск отрезка с минимальным спектральным разрывом на стыке.

Границы ставим на обнаруженные доли. Перебираем 8, 16, 32 и 64 доли
и оставляем кандидатов длиной не меньше 30 % записи по умолчанию.
Решение о пригодности найденного отрезка принимает criteria_v2.
"""
from __future__ import annotations

import librosa
import numpy as np

from ..metrics import game_criteria

BEATS_PER_BAR = 4
N_BARS_CHOICES = [2, 4, 8, 16]
SEAM_WINDOW_MS_PRIMARY = 50.0


def find_best_beat_aligned_loop(y: np.ndarray, sr: int, min_loop_fraction: float = 0.3) -> dict | None:
    """Возвращает границы лучшего кандидата или None, если подходящих границ нет."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, units="time", trim=False)
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 8:
        return None

    duration = len(y) / sr
    min_len = min_loop_fraction * duration

    w = game_criteria._window_samples(sr, SEAM_WINDOW_MS_PRIMARY)
    n_fft = game_criteria._n_fft_for(w)
    base = game_criteria._internal_transition_stats(y, w, n_fft)
    med = base["median"] if np.isfinite(base["median"]) else 1.0

    n_beats_total = len(beats)
    candidates = []
    for n_bars in N_BARS_CHOICES:
        span_beats = n_bars * BEATS_PER_BAR
        for bi in range(0, n_beats_total - span_beats):
            start_s = float(beats[bi])
            end_s = float(beats[bi + span_beats])
            span_s = end_s - start_s
            if span_s < min_len or end_s > duration - 0.02:
                continue
            candidates.append((start_s, end_s, n_bars))

    if not candidates:
        return None

    results = []
    for start_s, end_s, n_bars in candidates:
        start_i, end_i = int(round(start_s * sr)), int(round(end_s * sr))
        if end_i - start_i < 4 * w:
            continue
        seg = y[start_i:end_i]
        end_frame, start_frame = seg[-w:], seg[:w]
        spec = game_criteria._frame_log_spectra(np.stack([end_frame, start_frame]), n_fft)
        dist = float(game_criteria._spectral_distance(spec[0], spec[1]))
        results.append({"start_s": start_s, "end_s": end_s, "n_bars": n_bars, "dist": dist})

    if not results:
        return None
    best = min(results, key=lambda r: r["dist"])
    best["seam_score"] = best["dist"] / (med + 1e-8)
    best["internal_p95"] = base["p95"] if np.isfinite(base["p95"]) else float("inf")
    best["n_candidates"] = len(results)
    return best
