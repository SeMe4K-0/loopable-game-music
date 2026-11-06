"""Подобрать порог щелчка на одной половине петель NES и оценить его на другой.

Выбираем самый строгий квантиль, при котором проверка щелчка отклоняет
не более 5 % калибровочных петель. Проверочные петли не участвуют в подборе.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..eval import criteria_v2 as v2
from . import positive_control as pc

ROOT = Path(__file__).resolve().parents[2]
PC_DIR = ROOT / "results" / "stage0_positive_control"
OUT_DIR = ROOT / "results" / "stage0_v2_calibration"

SR = pc.SR
CLICK_FR_MAX_CALIB = 0.05


def _variants(y: np.ndarray, P: float | None):
    """Все варианты одной петли: как есть, повороты, дефекты."""
    yield "base", y
    for r in pc.ROTATIONS:
        yield f"rot{r}", np.roll(y, -int(round(r * len(y))))
    onb, offb = pc._beat_aligned_shifts(y)
    if onb is not None:
        yield "rot_on_beat", np.roll(y, -onb)
        yield "rot_off_beat", np.roll(y, -offb)
    for name, (levels, fn) in pc.DEFECTS.items():
        for lv in levels:
            if name == "trim_beats":
                yd = fn(y, lv, P) if P else None
            else:
                yd = fn(y, lv)
            if yd is not None:
                yield f"{name}={lv}", yd


def process(item: dict) -> dict:
    y = pc.audio_utils.peak_normalize(item["y"], pc.PEAK)
    g0 = v2.grid_v2(y, SR)
    P = g0.get("beat_period_s")
    out = {"id": item["id"], "set": item["set"], "duration_s": len(y) / SR, "beat_period_s": P, "ev": {}}
    for name, yv in _variants(y, P):
        yv = pc.audio_utils.peak_normalize(yv, pc.PEAK)
        deg = bool(pc.degeneracy.report(yv, SR, thresholds=pc.THRESHOLDS).get("degenerate"))
        ev = v2.evaluate_loop_v2(yv, SR)
        ev["degenerate"] = deg
        out["ev"][name] = ev
    return out


def load_items(seed: int, loop_index: Path) -> list[dict]:
    rows = json.loads(loop_index.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rows))
    half = set(perm[: len(rows) // 2].tolist())
    items = []
    for i, r in enumerate(rows):
        y, _ = pc.audio_utils.load_for_analysis(pc.AUDIO_ROOT / r["file"].replace("\\", "/"), SR, pc.PEAK)
        a = int(round(r["a_s"] * SR)); L = int(round(r["duration_s"] * SR))
        items.append({"id": r["id"], "set": "calib" if i in half else "test", "y": y[a:a + L]})
    for bpm in range(60, 181, 10):
        for nb in (2, 4, 8, 16):
            y = pc.synth_loop(bpm, nb, seed=bpm * 100 + nb)
            if len(y) / SR <= 90:
                items.append({"id": f"synth_bpm{bpm}_bars{nb}", "set": "synth", "y": y})
    return items


def rates(recs: list[dict], key: str, q: float) -> dict:
    sub = [r for r in recs if key in r["ev"]]
    dec = [v2.decide(r["ev"][key], q, r["ev"][key]["degenerate"]) for r in sub]
    n = len(dec)
    res = {k: pc.prop_ci(sum(d[k] for d in dec), n) for k in ("seam_ok", "click_ok", "grid_ok", "both_ok")}
    res["degenerate"] = pc.prop_ci(sum(r["ev"][key]["degenerate"] for r in sub), n)
    res["grid_ok_bars44"] = pc.prop_ci(sum(r["ev"][key]["grid"]["ok_bars44"] for r in sub), n)
    res["both_ok_excl_degeneracy_filter"] = pc.prop_ci(sum(d["seam_ok"] and d["grid_ok"] for d in dec), n)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loop-index", type=Path, default=PC_DIR / "C_real_rows.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = load_items(args.seed, args.loop_index)
    print(f"петель: {len(items)} (calib/test/synth)", flush=True)
    recs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(process, items, chunksize=1)):
            recs.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(items)}  {time.time() - t0:.0f} с", flush=True)
    by = {s: [r for r in recs if r["set"] == s] for s in ("calib", "test", "synth")}

    # --- выбор q на калибровочной половине
    calib_click = {}
    for q in v2.CLICK_Q_GRID:
        fr = sum(not v2.click_ok_v2(r["ev"]["base"]["click"], q) for r in by["calib"]) / len(by["calib"])
        calib_click[str(q)] = fr
    q_star = next((q for q in v2.CLICK_Q_GRID if calib_click[str(q)] <= CLICK_FR_MAX_CALIB), v2.CLICK_Q_GRID[-1])
    print(f"ложное отвержение щелчком на калибровке по q: {calib_click} -> q* = {q_star}", flush=True)

    summary = {"rule": {"split_seed": args.seed, "click_fr_max_calib": CLICK_FR_MAX_CALIB,
                        "q_grid": list(v2.CLICK_Q_GRID), "context_s": v2.CONTEXT_S, "tol": v2.TOL},
               "calib_click_false_rejection_by_q": calib_click, "q_star": q_star, "sets": {}}
    for s, rs in by.items():
        S = {"n": len(rs), "base": rates(rs, "base", q_star), "rotations": {}, "defects": {}}
        for key in [f"rot{r}" for r in pc.ROTATIONS] + ["rot_on_beat", "rot_off_beat"]:
            S["rotations"][key] = rates(rs, key, q_star)
        acc = [r for r in rs if v2.decide(r["ev"]["base"], q_star, r["ev"]["base"]["degenerate"])["both_ok"]]
        S["defects"]["n_accepted"] = len(acc)
        for name, (levels, _) in pc.DEFECTS.items():
            curve = []
            for lv in levels:
                key = f"{name}={lv}"
                sub = [r for r in acc if key in r["ev"]]
                dec = [v2.decide(r["ev"][key], q_star, r["ev"][key]["degenerate"]) for r in sub]
                curve.append({"level": lv, "n": len(dec),
                              "detected_any": pc.prop_ci(sum(not d["both_ok"] for d in dec), len(dec)),
                              "detected_seam": pc.prop_ci(sum(not d["seam_ok"] for d in dec), len(dec)),
                              "detected_grid": pc.prop_ci(sum(not d["grid_ok"] for d in dec), len(dec))})
            S["defects"][name] = curve
        summary["sets"][s] = S
        b = S["base"]
        print(f"[{s}] n={len(rs)} оба={b['both_ok']['p']:.3f} [{b['both_ok']['ci_lo']:.3f}; {b['both_ok']['ci_hi']:.3f}] "
              f"стык={b['seam_ok']['p']:.3f} сетка={b['grid_ok']['p']:.3f} брак={b['degenerate']['p']:.3f}", flush=True)

    pc._dump(OUT_DIR / "summary.json", summary)
    pc._dump(OUT_DIR / "rows.json", recs)
    print(f"готово за {time.time() - t0:.0f} с: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
