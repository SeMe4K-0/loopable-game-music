"""Пересчитать долевую сетку и щелчок по сохранённому аудио.

Спектральный стык и фильтр берём из результатов compute_track_metrics.
Генерацию не повторяем. Отсутствующее аудио отмечаем в выходном JSONL.
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
from . import criteria_v2 as v2

ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = ROOT
OUT = ROOT / "results" / "v2_per_track_metrics.jsonl"
KEY = ("model_id", "duration_s", "prompt_id", "set_idx", "track_idx")

def load_v1() -> list[dict]:
    latest = {}
    for line in open(EVAL_ROOT / "results" / "per_track_metrics.jsonl", encoding="utf-8"):
        r = json.loads(line)
        latest[tuple(r[k] for k in KEY)] = r          # последняя запись по ключу, как в summarize.py
    return list(latest.values())


def process(r: dict) -> dict:
    out = {k: r[k] for k in KEY + ("prompt_category", "loop_eval", "seed")}
    wav = EVAL_ROOT / r["wav_path"]
    if not wav.exists():
        return {**out, "status": "no_audio"}
    try:
        y, sr = audio_utils.load_for_analysis(wav, r.get("analysis_sr", 22050), 0.9)
        g = r["game"]
        best = g["best_loop"]
        a, b = int(round(best["loop_start_s"] * sr)), int(round(best["loop_end_s"] * sr))
        seg = y[a:b]
        return {**out, "status": "done",
                "v1": {"degenerate": bool(r.get("degeneracy", {}).get("degenerate", False)),
                       "seam50_ok": bool(g["seam_50ms"]["seam_ok_p95"]),
                       "seam200_ok": bool(g["seam_200ms"]["seam_ok_p95"]),
                       "click_ok_v1": bool(g["seam_50ms"]["click_ok"]),
                       "seam_naive_ok_v1": bool(g["seam_naive_ok"]),
                       "best_seam_ok_p95": bool(best["best_seam_ok_p95"]),
                       "best_loop_start_s": best["loop_start_s"], "best_loop_end_s": best["loop_end_s"]},
                "click_naive": v2.click_stats(y, sr),
                "grid_naive": v2.grid_v2(y, sr),
                "grid_best": v2.grid_v2(seg, sr)}
    except Exception as e:                       # фиксируем, не роняем прогон
        return {**out, "status": "error", "error": repr(e)}


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
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exclude-models", nargs="*", default=["audioldm2_music"],
                    help="AudioLDM2 исключена из версии 2: аудио большей части треков не сохранилось")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [r for r in load_v1() if r.get("loop_eval", True) and "game" in r
            and r["model_id"] not in set(args.exclude_models)]
    done = set()
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            d = json.loads(line)
            done.add(tuple(d[k] for k in KEY))
    todo = [r for r in rows if tuple(r[k] for k in KEY) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"петлевых треков {len(rows)}, уже посчитано {len(done)}, к расчёту {len(todo)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex, open(OUT, "a", encoding="utf-8") as f:
        for i, res in enumerate(ex.map(process, todo, chunksize=4)):
            f.write(json.dumps(res, ensure_ascii=False, default=_conv) + "\n")
            if (i + 1) % 200 == 0:
                f.flush()
                el = time.time() - t0
                print(f"  {i + 1}/{len(todo)}  {el / 60:.1f} мин, осталось ~{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} мин",
                      flush=True)
    print(f"готово за {(time.time() - t0) / 60:.1f} мин", flush=True)


if __name__ == "__main__":
    main()
