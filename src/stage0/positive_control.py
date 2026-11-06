"""Найти контрольные петли NES по повтору звуковой формы.

Кандидаты берём по сходству мел-спектров, затем проверяем совпадение
волновой формы по обе стороны от границы. Так размечаем петлю независимо
от проверяемой долевой сетки. Синтетические петли дают отдельный контроль.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import librosa
import yaml

from .. import audio_utils
from ..metrics import degeneracy, game_criteria

ROOT = Path(__file__).resolve().parents[2]
AUDIO_ROOT = Path(os.environ.get("NES_VMDB_AUDIO", ROOT / "Audio"))
OUT_DIR = ROOT / "results" / "stage0_positive_control"
SR = 22050
PEAK = 0.9
RULE = yaml.safe_load((ROOT / "configs" / "durations.yaml").read_text(encoding="utf-8"))["decision_rule"]
THRESHOLDS = json.loads((ROOT / "configs" / "degeneracy_thresholds.json").read_text(encoding="utf-8"))

# поиск повторов
HOP = 512
MIN_LOOP_S, MAX_LOOP_S = 4.0, 60.0
MAX_TRACK_S = 150.0
FEAT_SIM_MIN = 0.97      # косинус мел-спектров для кандидата
RUN_MIN_S = 1.0          # повтор должен держаться не меньше секунды
NCC_HALF_S = 0.5         # окно волновой проверки: ±0.5 с вокруг стыка
NCC_MIN = 0.95


# --------------------------------------------------------------------------
# Оценка одной петли тем же правилом, что в статье
# --------------------------------------------------------------------------

def evaluate(loop: np.ndarray, sr: int = SR) -> dict:
    loop = audio_utils.peak_normalize(loop, PEAK)
    deg = degeneracy.report(loop, sr, thresholds=THRESHOLDS)
    s50 = game_criteria.seam_discontinuity(loop, sr, window_ms=50.0)
    s200 = game_criteria.seam_discontinuity(loop, sr, window_ms=200.0)
    seam_ok = bool(s50["seam_ok_p95"] and s200["seam_ok_p95"] and s50["click_ok"])
    g = game_criteria.beat_grid_metrics(loop, sr)
    grid_ok = bool(not g.get("degenerate")
                   and g["bar_fraction_error"] <= RULE["bar_fraction_error_max"]
                   and g["seam_gap_beat_error"] <= RULE["seam_gap_beat_error_max"])
    is_deg = bool(deg.get("degenerate"))
    return {
        "degenerate": is_deg,
        "seam_ok": seam_ok, "seam50_ok": bool(s50["seam_ok_p95"]), "seam200_ok": bool(s200["seam_ok_p95"]),
        "click_ok": bool(s50["click_ok"]), "seam_score_50": s50["seam_score"],
        "grid_ok": grid_ok,
        "bpm": g.get("bpm_from_beats", g.get("bpm")), "beat_period_s": g.get("beat_period_s"),
        "n_bars_estimate": g.get("n_bars_estimate"),
        "bar_fraction_error": g.get("bar_fraction_error"), "seam_gap_beat_error": g.get("seam_gap_beat_error"),
        "both_ok": bool(seam_ok and grid_ok and not is_deg),
    }


# --------------------------------------------------------------------------
# Уровень A: синтетические петли
# --------------------------------------------------------------------------

def synth_loop(bpm: float, n_bars: int, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    n = int(round(n_bars * 4 * beat * sr))
    y = np.zeros(n)

    def add(event: np.ndarray, t0: float):
        i0 = int(round(t0 * sr))
        idx = (np.arange(len(event)) + i0) % n       # хвосты заворачиваются: петля идеальна
        np.add.at(y, idx, event)

    tt = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    th = np.arange(int(0.05 * sr)) / sr
    hat = rng.standard_normal(len(th)) * np.exp(-th * 80) * 0.25
    bass_notes = [55.0, 55.0, 65.41, 49.0]           # A1 A1 C2 G1 — по такту
    for bar in range(n_bars):
        f = bass_notes[bar % len(bass_notes)]
        for b in range(4):
            t0 = (bar * 4 + b) * beat
            add(kick if b in (0, 2) else 0.6 * kick, t0)
            add(hat, t0 + beat / 2)
            tb = np.arange(int(beat * 0.9 * sr)) / sr
            add(0.3 * np.sign(np.sin(2 * np.pi * f * tb)) * np.exp(-tb * 3), t0)   # «чиптюн»-меандр
    return y


# --------------------------------------------------------------------------
# Уровень C': настоящие петли по дословному повтору
# --------------------------------------------------------------------------

def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def find_repeat_loop(y: np.ndarray, sr: int = SR) -> dict | None:
    """Самый длинный период L ∈ [MIN_LOOP_S, MAX_LOOP_S], на котором звук
    повторяется дословно в окрестности некоторой точки a. Возвращает
    {a, L, ncc} в сэмплах или None."""
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64)
    X = np.log1p(M)
    X = X - X.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True) + 1e-9
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    loud = rms > 0.05 * rms.max()                     # тишина «повторяется» тривиально
    n = X.shape[1]
    run_min = int(RUN_MIN_S * sr / HOP)
    w = int(NCC_HALF_S * sr)
    k_lo, k_hi = int(MIN_LOOP_S * sr / HOP), min(int(MAX_LOOP_S * sr / HOP), n - run_min - 1)
    for k in range(k_hi, k_lo - 1, -1):               # от длинных к коротким: берём полный период
        s = np.sum(X[:, :n - k] * X[:, k:], axis=0)
        ok = (s >= FEAT_SIM_MIN) & loud[:n - k] & loud[k:]
        if ok.sum() < run_min:
            continue
        # самый длинный непрерывный участок совпадения
        edges = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        j = int(np.argmax(ends - starts))
        best_len, best_mid = int(ends[j] - starts[j]), int((starts[j] + ends[j]) // 2)
        if best_len < run_min:
            continue
        a = best_mid * HOP
        if a - w < 0 or a + k * HOP + 2 * w >= len(y):
            continue
        # уточнение L до сэмпла по корреляции волны в окне ±0.5 с вокруг стыка
        ref = y[a - w:a + w]
        best = (-1.0, 0)
        for L in range(k * HOP - HOP, k * HOP + HOP + 1):
            c = _ncc(ref, y[a + L - w:a + L + w])
            if c > best[0]:
                best = (c, L)
        if best[0] >= NCC_MIN:
            return {"a": int(a), "L": int(best[1]), "ncc": best[0]}
    return None


# --------------------------------------------------------------------------
# Дефекты (§4.2)
# --------------------------------------------------------------------------

def defect_click(loop, amp):
    out = loop.copy(); out[-1] += amp; out[0] -= amp      # импульс ровно на стыке
    return out


def defect_gap(loop, ms):
    return np.concatenate([loop, np.zeros(int(round(ms / 1000 * SR)))])


def defect_trim(loop, beats, beat_period_s):
    k = int(round(beats * beat_period_s * SR))
    return loop[:-k] if 0 < k < len(loop) // 2 else None


def defect_gain(loop, db):
    # плавная рампа громкости по всей петле: внутри переходы гладкие, на стыке скачок db
    g = 10 ** (-np.linspace(0, db, len(loop)) / 20)
    return loop * g


DEFECTS = {
    "click_amp":   ([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0], defect_click),
    "gap_ms":      ([1, 2, 5, 10, 20, 50, 100, 200], defect_gap),
    "trim_beats":  ([0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0], defect_trim),
    "gain_db":     ([0.5, 1, 2, 3, 4.5, 6, 9, 12], defect_gain),
}
ROTATIONS = [0.125, 0.25, 0.375, 0.5]


# --------------------------------------------------------------------------

def prop_ci(k: int, n: int, n_boot: int = 1000, seed: int = 0) -> dict:
    if n == 0:
        return {"p": None, "ci_lo": None, "ci_hi": None, "k": 0, "n": 0}
    rng = np.random.default_rng(seed)
    x = np.zeros(n); x[:k] = 1
    boots = rng.choice(x, size=(n_boot, n), replace=True).mean(1)
    return {"p": k / n, "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
            "k": int(k), "n": int(n)}


def run_controls(loops: list[dict], label: str, log) -> dict:
    """loops: [{id, y, ...}] — для каждой: как есть, повороты, дефекты."""
    rows = []
    for i, L in enumerate(loops):
        y = L["y"]
        base = evaluate(y)
        row = {"id": L["id"], **{k: v for k, v in L.items() if k not in ("y",)}, "base": base, "rot": {}, "defects": {}}
        for r in ROTATIONS:
            row["rot"][str(r)] = evaluate(np.roll(y, -int(round(r * len(y)))))
        # стык точно на доле (так режут петли на практике) и между долями
        onb, offb = _beat_aligned_shifts(y)
        if onb is not None:
            row["rot"]["on_beat"] = evaluate(np.roll(y, -onb))
            row["rot"]["off_beat"] = evaluate(np.roll(y, -offb))
        # сетка той же петли в контексте: трекер видит её трижды подряд,
        # оценивается средняя копия — краевых эффектов трекера нет
        Ls = len(y) / SR
        g = game_criteria.beat_grid_metrics(audio_utils.peak_normalize(np.tile(y, 3), PEAK), SR, Ls, 2 * Ls)
        row["grid_ctx"] = {"bar_fraction_error": g.get("bar_fraction_error"),
                           "seam_gap_beat_error": g.get("seam_gap_beat_error"),
                           "ok": bool(not g.get("degenerate")
                                      and g["bar_fraction_error"] <= RULE["bar_fraction_error_max"]
                                      and g["seam_gap_beat_error"] <= RULE["seam_gap_beat_error_max"])}
        if base["both_ok"]:
            P = base["beat_period_s"]
            for name, (levels, fn) in DEFECTS.items():
                res = []
                for lv in levels:
                    yd = fn(y, lv, P) if name == "trim_beats" else fn(y, lv)
                    res.append(None if yd is None else evaluate(yd))
                row["defects"][name] = res
        rows.append(row)
        if (i + 1) % 10 == 0:
            log(f"  [{label}] {i + 1}/{len(loops)}")
    return summarize(rows, label), rows


def _beat_aligned_shifts(y: np.ndarray) -> tuple[int | None, int | None]:
    """Сдвиги (в сэмплах), ставящие стык на обнаруженную долю около середины
    петли и на 0.37 доли позже неё."""
    onset_env = librosa.onset.onset_strength(y=y, sr=SR)
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, units="samples", trim=False)
    beats = np.asarray(beats)
    if len(beats) < 4:
        return None, None
    b = int(beats[np.argmin(np.abs(beats - len(y) / 2))])
    period = float(np.median(np.diff(beats)))
    return b, int(b + 0.37 * period) % len(y)


def summarize(rows: list[dict], label: str) -> dict:
    n = len(rows)
    out = {"label": label, "n_loops": n}
    for key in ("degenerate", "seam_ok", "grid_ok", "both_ok"):
        out[f"base_{key}"] = prop_ci(sum(r["base"][key] for r in rows), n)
    out["false_rejection_both"] = prop_ci(sum(not r["base"]["both_ok"] for r in rows), n)
    out["grid_in_context"] = prop_ci(sum(r["grid_ctx"]["ok"] for r in rows), n)
    out["rotation"] = {}
    for r in [str(x) for x in ROTATIONS] + ["on_beat", "off_beat"]:
        sub = [x for x in rows if r in x["rot"]]
        out["rotation"][r] = {k: prop_ci(sum(x["rot"][r][k] for x in sub), len(sub))
                              for k in ("seam_ok", "click_ok", "grid_ok", "both_ok")}
    acc = [r for r in rows if r["base"]["both_ok"]]
    out["defects"] = {"n_accepted_loops": len(acc)}
    for name, (levels, _) in DEFECTS.items():
        curve = []
        for j, lv in enumerate(levels):
            ev = [r["defects"][name][j] for r in acc if r["defects"][name][j] is not None]
            m = len(ev)
            curve.append({
                "level": lv, "n": m,
                "detected_any": prop_ci(sum(not e["both_ok"] for e in ev), m),
                "detected_seam": prop_ci(sum(not e["seam_ok"] for e in ev), m),
                "detected_grid": prop_ci(sum(not e["grid_ok"] for e in ev), m),
            })
        out["defects"][name] = curve
    # отказы по причинам — что именно отвергает настоящую петлю
    rej = [r["base"] for r in rows if not r["base"]["both_ok"]]
    out["rejection_reasons"] = {
        "degenerate": sum(e["degenerate"] for e in rej),
        "seam50": sum(not e["seam50_ok"] for e in rej), "seam200": sum(not e["seam200_ok"] for e in rej),
        "click": sum(not e["click_ok"] for e in rej),
        "bar_fraction": sum((e["bar_fraction_error"] is None) or not (e["bar_fraction_error"] <= 0.1) for e in rej),
        "seam_gap": sum((e["seam_gap_beat_error"] is None) or not (e["seam_gap_beat_error"] <= 0.1) for e in rej),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-loops", type=int, default=150)
    ap.add_argument("--max-scan", type=int, default=1500, help="сколько пьес просмотреть в поиске повторов")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-synth", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logf = open(OUT_DIR / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"rule: bar≤{RULE['bar_fraction_error_max']} gap≤{RULE['seam_gap_beat_error_max']}; NCC_MIN={NCC_MIN}")
    result = {"config": {"sr": SR, "ncc_min": NCC_MIN, "feat_sim_min": FEAT_SIM_MIN, "min_loop_s": MIN_LOOP_S,
                         "max_loop_s": MAX_LOOP_S, "rotations": ROTATIONS,
                         "defects": {k: v[0] for k, v in DEFECTS.items()}, "seed": args.seed}}

    # --- A
    if not args.skip_synth:
        synth = []
        for bpm in range(60, 181, 10):
            for nb in (2, 4, 8, 16):
                y = synth_loop(bpm, nb, seed=bpm * 100 + nb)
                if len(y) / SR > 90:
                    continue
                synth.append({"id": f"synth_bpm{bpm}_bars{nb}", "y": y, "bpm_true": bpm, "n_bars_true": nb,
                              "duration_s": len(y) / SR})
        log(f"A: {len(synth)} синтетических петель")
        sumA, rowsA = run_controls(synth, "A", log)
        result["A_synthetic"] = sumA
        _dump(OUT_DIR / "A_synthetic_rows.json", rowsA)
        log(f"A: оба={sumA['base_both_ok']['p']:.3f} стык={sumA['base_seam_ok']['p']:.3f} сетка={sumA['base_grid_ok']['p']:.3f}")

    # --- C'
    files = sorted(AUDIO_ROOT.rglob("*.mp3"))
    random.Random(args.seed).shuffle(files)
    real, seen_games, scanned = [], set(), 0
    t0 = time.time()
    for f in files[:args.max_scan]:
        if len(real) >= args.max_loops:
            break
        game = f.parent.name
        if game in seen_games:                 # не больше одной петли на игру — разнообразие
            continue
        scanned += 1
        try:
            y, sr = audio_utils.load_for_analysis(f, SR, PEAK)
        except Exception as e:
            log(f"  не прочитан {f.name}: {e!r}")
            continue
        if len(y) / SR > MAX_TRACK_S:
            y = y[:int(MAX_TRACK_S * SR)]
        hit = find_repeat_loop(y)
        if hit is None:
            continue
        seg = y[hit["a"]:hit["a"] + hit["L"]]
        real.append({"id": f.stem, "y": seg, "file": str(f.relative_to(AUDIO_ROOT)), "a_s": hit["a"] / SR,
                     "duration_s": hit["L"] / SR, "ncc": hit["ncc"]})
        seen_games.add(game)
        if len(real) % 10 == 0:
            log(f"  C': найдено {len(real)} петель из {scanned} пьес ({time.time() - t0:.0f} с)")
    log(f"C': {len(real)} петель из {scanned} просмотренных пьес")
    result["C_real_repeats"] = {"n_scanned": scanned}
    sumC, rowsC = run_controls(real, "C'", log)
    result["C_real_repeats"].update(sumC)
    _dump(OUT_DIR / "C_real_rows.json", rowsC)
    log(f"C': оба={sumC['base_both_ok']['p']:.3f} стык={sumC['base_seam_ok']['p']:.3f} "
        f"сетка={sumC['base_grid_ok']['p']:.3f} брак={sumC['base_degenerate']['p']:.3f}")

    _dump(OUT_DIR / "summary.json", result)
    log(f"готово: {OUT_DIR / 'summary.json'}")


def _dump(path: Path, obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=conv), encoding="utf-8")


if __name__ == "__main__":
    main()
