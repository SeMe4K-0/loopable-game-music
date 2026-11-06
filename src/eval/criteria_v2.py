"""Проверка петли по протоколу статьи.

Сравниваем спектры на стыке (50 и 200 мс), скачок сигнала с атаками нот
и долевую сетку. Для сетки трекер видит хвост и начало соседних повторов,
чтобы не терять доли на границах. Порог щелчка калибруется отдельно.
"""
from __future__ import annotations

import librosa
import numpy as np

from ..metrics import game_criteria

TOL = 0.10                     # те же допуски, что в версии 1
CONTEXT_S = 15.0               # сколько петли подклеивается с каждой стороны
CLICK_Q_GRID = (0.25, 0.5, 0.75, 0.9, 1.0)
ONSET_HALF_MS = 12.0           # окно поиска скачка вокруг атаки (полкадра onset-огибающей)
MIN_ONSETS = 4


def with_context(seg: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Петля в контексте своего зацикливания: [хвост | петля | начало].
    Возвращает сигнал и индекс начала петли в нём."""
    c = min(len(seg), int(CONTEXT_S * sr))
    return np.concatenate([seg[-c:], seg, seg[:c]]), c


def grid_v2(seg: np.ndarray, sr: int) -> dict:
    ctx, c = with_context(seg, sr)
    start_s, end_s = c / sr, (c + len(seg)) / sr
    g = game_criteria.beat_grid_metrics(ctx, sr, loop_start_s=start_s, loop_end_s=end_s)
    if g.get("degenerate"):
        return {"degenerate": True, "ok": False, "ok_bars44": False}
    ok = bool(g["beat_fraction_error"] <= TOL and g["seam_gap_beat_error"] <= TOL)
    ok44 = bool(g["bar_fraction_error"] <= TOL and g["seam_gap_beat_error"] <= TOL)
    return {
        "degenerate": False, "ok": ok, "ok_bars44": ok44,
        "beat_period_s": g["beat_period_s"], "loop_duration_s": g["loop_duration_s"],
        "beat_fraction_error": g["beat_fraction_error"], "bar_fraction_error": g["bar_fraction_error"],
        "seam_gap_beat_error": g["seam_gap_beat_error"],
        # для перестановочной нулевой линии: щель на стыке в секундах
        "gap_s": float((end_s - g["last_beat_s"]) + (g["first_beat_s"] - start_s)),
        "n_beats": g["n_beats"],
    }


def click_stats(seg: np.ndarray, sr: int) -> dict:
    """Скачок на стыке и квантили скачков на атаках трека."""
    r = max(1, int(sr * 0.001))
    seam = np.concatenate([seg[-r - 1:], seg[:r + 1]])
    boundary = float(np.max(np.abs(np.diff(seam))))
    d = np.abs(np.diff(seg))
    onsets = librosa.onset.onset_detect(y=seg, sr=sr, units="samples", backtrack=False)
    h = int(sr * ONSET_HALF_MS / 1000)
    jumps = [float(d[max(0, o - h):min(len(d), o + h)].max()) for o in onsets if 0 < o < len(d)]
    out = {"boundary": boundary, "n_onsets": len(jumps),
           "p99_all": float(np.percentile(d, 99))}               # эталон версии 1 — для сравнения
    for q in CLICK_Q_GRID:
        out[f"onset_q{q}"] = float(np.quantile(jumps, q)) if len(jumps) >= MIN_ONSETS else None
    return out


def click_ok_v2(cs: dict, q: float) -> bool:
    ref = cs.get(f"onset_q{q}")
    if ref is None:                       # атак почти нет — эталон версии 1
        ref = cs["p99_all"]
    return bool(cs["boundary"] <= ref + 1e-8)


def seam_spectral(seg: np.ndarray, sr: int) -> dict:
    s50 = game_criteria.seam_discontinuity(seg, sr, window_ms=50.0)
    s200 = game_criteria.seam_discontinuity(seg, sr, window_ms=200.0)
    return {"seam50_ok": bool(s50["seam_ok_p95"]), "seam200_ok": bool(s200["seam_ok_p95"]),
            "click_ok_v1": bool(s50["click_ok"]), "seam_score_50": float(s50["seam_score"])}


def evaluate_loop_v2(seg: np.ndarray, sr: int) -> dict:
    """Сырые величины для петли «как есть»; решение по q — в decide()."""
    return {"seam": seam_spectral(seg, sr), "click": click_stats(seg, sr), "grid": grid_v2(seg, sr)}


def decide(ev: dict, q: float, degenerate: bool = False) -> dict:
    s = ev["seam"]
    click_ok = click_ok_v2(ev["click"], q)
    seam_ok = bool(s["seam50_ok"] and s["seam200_ok"] and click_ok)
    grid_ok = bool(ev["grid"]["ok"])
    return {"seam_ok": seam_ok, "click_ok": click_ok, "grid_ok": grid_ok,
            "both_ok": bool(seam_ok and grid_ok and not degenerate)}
