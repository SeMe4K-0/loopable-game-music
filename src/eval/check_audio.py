"""Проверить пригодность аудиофайла для зацикливания."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..audio_utils import load_for_analysis
from ..metrics import degeneracy
from .beat_aligned_cut import find_best_beat_aligned_loop
from .criteria_v2 import decide, evaluate_loop_v2

THRESHOLDS = Path(__file__).resolve().parents[2] / "configs" / "degeneracy_thresholds.json"
CLICK_QUANTILE = 0.9


def inspect(y, sr, is_filtered):
    evidence = evaluate_loop_v2(y, sr)
    return {**evidence, "decision": decide(evidence, CLICK_QUANTILE, is_filtered)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cut", action="store_true", help="также проверить лучшую обрезку по долям")
    parser.add_argument("--output", type=Path, help="сохранить обрезку, если она прошла проверку")
    args = parser.parse_args()
    if args.output and not args.cut:
        parser.error("для --output нужен --cut")
    if args.output and args.output.exists():
        parser.error("выходной файл уже существует")

    y, sr = load_for_analysis(args.audio)
    if not np.isfinite(y).all() or len(y) < sr:
        parser.error("нужна запись длительностью от одной секунды с конечными значениями отсчётов")

    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    preliminary = degeneracy.report(y, sr, thresholds=thresholds)
    # Как в опыте с темпом: фильтр применяется к целой записи и учитывается для её обрезки.
    is_filtered = bool(preliminary["degenerate"])
    report = {"duration_s": len(y) / sr, "filter": preliminary, "whole": inspect(y, sr, is_filtered)}
    if args.cut:
        report["cut"] = None
        best = find_best_beat_aligned_loop(y, sr)
        if best is not None:
            first = round(best["start_s"] * sr)
            last = round(best["end_s"] * sr)
            segment = y[first:last]
            result = inspect(segment, sr, is_filtered)
            report["cut"] = {
                "start_s": best["start_s"], "end_s": best["end_s"], "beats": 4 * best["n_bars"], **result,
            }
            if args.output and result["decision"]["both_ok"]:
                sf.write(args.output, segment, sr)
                report["cut"]["output"] = str(args.output)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
