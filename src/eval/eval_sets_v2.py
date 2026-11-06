"""Проверить целые записи и обрезки из опыта с заданным темпом.

Для каждой записи сохраняем признаки стыка и долевой сетки; для порога
щелчка используются те же данные, что в основном опыте.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .. import audio_utils
from ..metrics import degeneracy
from . import beat_aligned_cut as bac, criteria_v2 as v2

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SETS = {
    "tempo_sa3": "stage0_5_tempo_follow_stable_audio_3_small_music",
    "tempo_ace": "stage0_5_tempo_follow_ace_step_v1_3_5b",
}
OUT = RES / "v2_sets_metrics.jsonl"
THRESHOLDS = json.loads((ROOT / "configs" / "degeneracy_thresholds.json").read_text(encoding="utf-8"))

def process(job: tuple[str, str]) -> dict:
    set_name, path = job
    y, sr = audio_utils.load_for_analysis(Path(path), 22050, 0.9)
    name = Path(path).name
    bpm_req = int(name.split("_bpm")[1].split("_")[0]) if "_bpm" in name else None
    rec = {"set": set_name, "file": name, "bpm_requested": bpm_req,
           "degenerate": bool(degeneracy.report(y, sr, thresholds=THRESHOLDS).get("degenerate"))}
    try:
        rec["naive"] = v2.evaluate_loop_v2(y, sr)
        best = v2.game_criteria.find_best_loop(y, sr, window_ms=50.0, min_loop_fraction=0.5)
        seg = y[int(round(best["loop_start_s"] * sr)):int(round(best["loop_end_s"] * sr))]
        rec["best"] = {"best_seam_ok_p95": bool(best["best_seam_ok_p95"]), "grid": v2.grid_v2(seg, sr)}
        ba = bac.find_best_beat_aligned_loop(y, sr)
        if ba is not None:
            seg = y[int(round(ba["start_s"] * sr)):int(round(ba["end_s"] * sr))]
            rec["beat_aligned"] = {"n_bars": ba["n_bars"], **v2.evaluate_loop_v2(seg, sr)}
    except Exception as e:
        rec["error"] = repr(e)
    return rec


def _conv(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    jobs = [(s, str(p)) for s, d in SETS.items() for p in sorted((RES / d / "audio").glob("*.flac"))]
    if not jobs:
        ap.error("в results нет FLAC из опыта с заданным темпом")
    print(f"треков: {len(jobs)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex, open(OUT, "w", encoding="utf-8") as f:
        for i, r in enumerate(ex.map(process, jobs, chunksize=2)):
            f.write(json.dumps(r, ensure_ascii=False, default=_conv) + "\n")
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"  {i + 1}/{len(jobs)} {time.time() - t0:.0f} с", flush=True)
    print(f"готово за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
