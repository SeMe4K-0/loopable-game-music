"""Сгенерировать восьмитактовые записи с указанием BPM и размера 4/4 в запросе.

Для SA3 использовано 24 записи на BPM,
для ACE-Step — 12 записей на BPM. Список сцен и порядок запросов фиксированы.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..audio_utils import prepare_for_analysis, write_wav
from ..generate import _BACKENDS, load_models_config
from ..metrics.game_criteria import beat_grid_metrics
from ..stats import bootstrap_ci, proportion_ci

ROOT = Path(__file__).resolve().parents[2]
SA3 = "stable_audio_3_small_music"
ACE = "ace_step_v1_3_5b"

BPM_GRID = [80, 90, 100, 110, 120, 130, 140, 160]
N_BARS = 8
BEATS_PER_BAR = 4
SEED_BASE = 2026092000
SAVE_AUDIO = True

# Фанфара звучит один раз и не входит в опыт с зацикливанием.
PROMPTS = [
    ("boss_battle", "intense chiptune boss battle theme, driving bassline, urgent melody, 8-bit", "combat"),
    ("regular_battle", "upbeat 8-bit action battle theme, fast arpeggios, retro video game combat music", "combat"),
    ("final_boss_hybrid", "epic final boss theme, orchestral chiptune hybrid, driving retro video game music", "combat"),
    ("ambient_exploration", "calm ambient exploration theme for a video game overworld, chiptune, sparse melody", "exploration"),
    ("peaceful_overworld", "peaceful overworld theme, bright chiptune melody, looping video game music", "exploration"),
    ("village_town", "cheerful village town theme, cozy chiptune music, friendly video game atmosphere", "location"),
    ("dungeon_danger", "dark dungeon danger theme, tense low chiptune drone, video game suspense music", "location"),
]

# Опорные числа PLAN.md §1.3 (перестановочный нуль исходного эксперимента,
# наивная петля, безусловная генерация — n = 8722).
CHANCE_BAR_FRACTION = 0.257
CHANCE_SEAM_GAP = 0.340
CHANCE_BOTH = 0.088
DECISION_THRESHOLD = 0.5


def octave_corrected_error(bpm_detected: float, bpm_requested: float) -> tuple[float, float, float]:
    """beat_grid_metrics использует beat-трекер librosa: известна октавная
    неоднозначность (x2/÷2), задокументированная в самой функции. Возвращает
    (относительная ошибка без коррекции, относительная ошибка с коррекцией)
    и не подменяет одно другим — обе величины сохраняются в записи трека."""
    raw = abs(bpm_detected - bpm_requested) / bpm_requested
    cands = {
        1.0: abs(bpm_detected - bpm_requested),
        2.0: abs(bpm_detected - 2 * bpm_requested),
        0.5: abs(bpm_detected - 0.5 * bpm_requested),
    }
    best_mult = min(cands, key=cands.get)
    corrected = cands[best_mult] / bpm_requested
    return raw, corrected, best_mult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(SA3, ACE), default=SA3)
    parser.add_argument("--n-per-bpm", type=int, choices=range(1, 25), metavar="N")
    args = parser.parse_args()
    tracks_per_bpm = args.n_per_bpm or (24 if args.model == SA3 else 12)
    out_dir = ROOT / "results" / f"stage0_5_tempo_follow_{args.model}"
    cfg_list = load_models_config(ROOT / "configs" / "models.yaml")
    cfg = next(c for c in cfg_list if c["id"] == args.model)
    print(f"Загрузка {cfg['id']} ({cfg['backend']})...", flush=True)
    backend = _BACKENDS[cfg["backend"]](cfg)
    print("Модель загружена.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    tracks_path = out_dir / "tracks.jsonl"

    records = []
    t_start = time.time()
    track_idx = 0
    total = len(BPM_GRID) * tracks_per_bpm

    with open(tracks_path, "w", encoding="utf-8") as f:
        for bpm in BPM_GRID:
            duration_s = N_BARS * BEATS_PER_BAR * 60.0 / bpm
            for i in range(tracks_per_bpm):
                pid, base_text, category = PROMPTS[i % len(PROMPTS)]
                prompt = f"{base_text}, {bpm} BPM, 4/4 time signature"
                seed = SEED_BASE + track_idx

                t0 = time.time()
                audio, sr = backend.generate(prompt, duration_s, seed)
                gen_time = time.time() - t0
                actual_duration_s = audio.shape[-1] / sr

                y, asr = prepare_for_analysis(audio, sr)
                grid = beat_grid_metrics(y, asr, loop_start_s=0.0, loop_end_s=None,
                                          beats_per_bar=BEATS_PER_BAR)

                degenerate = bool(grid.get("degenerate", True))
                bpm_detected = grid.get("bpm_from_beats", float("nan"))
                if degenerate or not np.isfinite(bpm_detected):
                    bpm_err_raw = bpm_err_oct = float("nan")
                    oct_mult = None
                else:
                    bpm_err_raw, bpm_err_oct, oct_mult = octave_corrected_error(bpm_detected, bpm)

                grid_pass = bool(
                    not degenerate
                    and grid["bar_fraction_error"] <= 0.10
                    and grid["seam_gap_beat_error"] <= 0.10
                )

                rec = {
                    "track_idx": track_idx,
                    "bpm_requested": bpm,
                    "prompt_id": pid,
                    "category": category,
                    "duration_s_requested": duration_s,
                    "duration_s_actual": actual_duration_s,
                    "seed": seed,
                    "gen_time_s": gen_time,
                    "bpm_detected": bpm_detected,
                    "bpm_error_rel": bpm_err_raw,
                    "bpm_error_rel_oct_corrected": bpm_err_oct,
                    "octave_mult": oct_mult,
                    "bar_fraction_error": grid.get("bar_fraction_error"),
                    "beat_fraction_error": grid.get("beat_fraction_error"),
                    "seam_gap_beat_error": grid.get("seam_gap_beat_error"),
                    "n_beats": grid.get("n_beats"),
                    "degenerate": degenerate,
                    "grid_pass": grid_pass,
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                if SAVE_AUDIO:
                    wav_path = out_dir / "audio" / f"{track_idx:04d}_bpm{bpm}_{pid}.flac"
                    write_wav(wav_path, audio, sr)

                track_idx += 1
                if track_idx % 20 == 0 or track_idx == total:
                    elapsed = time.time() - t_start
                    print(f"[{track_idx}/{total}] elapsed={elapsed:.0f}s", flush=True)

    print(f"Готово: {total} треков за {time.time() - t_start:.0f}s. Считаю сводку...", flush=True)
    summarize(records, args.model, tracks_per_bpm, out_dir)


def summarize(records: list[dict], model_id: str, tracks_per_bpm: int, out_dir: Path):
    n = len(records)
    grid_pass_vals = [1.0 if r["grid_pass"] else 0.0 for r in records]
    overall_ci = bootstrap_ci(grid_pass_vals, n_resamples=1000, seed=0)
    overall_wilson = proportion_ci(sum(int(v) for v in grid_pass_vals), n)

    valid_bpm_err = [r["bpm_error_rel"] for r in records if np.isfinite(r["bpm_error_rel"])]
    valid_bpm_err_oct = [r["bpm_error_rel_oct_corrected"] for r in records if np.isfinite(r["bpm_error_rel_oct_corrected"])]
    n_degenerate = sum(1 for r in records if r["degenerate"])

    by_bpm = {}
    for bpm in BPM_GRID:
        cell = [r for r in records if r["bpm_requested"] == bpm]
        cell_pass = [1.0 if r["grid_pass"] else 0.0 for r in cell]
        cell_ci = bootstrap_ci(cell_pass, n_resamples=1000, seed=0)
        cell_bpm_err = [r["bpm_error_rel"] for r in cell if np.isfinite(r["bpm_error_rel"])]
        by_bpm[bpm] = {
            "n": len(cell),
            "grid_pass_fraction": cell_ci["point"],
            "grid_pass_ci_lo": cell_ci["ci_lo"],
            "grid_pass_ci_hi": cell_ci["ci_hi"],
            "n_degenerate": sum(1 for r in cell if r["degenerate"]),
            "mean_bpm_error_rel": float(np.mean(cell_bpm_err)) if cell_bpm_err else None,
            "median_bar_fraction_error": float(np.nanmedian([r["bar_fraction_error"] for r in cell])),
            "median_seam_gap_beat_error": float(np.nanmedian([r["seam_gap_beat_error"] for r in cell])),
        }

    summary = {
        "n_tracks": n,
        "n_degenerate": n_degenerate,
        "grid_pass_fraction": overall_ci["point"],
        "grid_pass_ci_lo_bootstrap": overall_ci["ci_lo"],
        "grid_pass_ci_hi_bootstrap": overall_ci["ci_hi"],
        "grid_pass_ci_lo_wilson": overall_wilson["ci_lo"],
        "grid_pass_ci_hi_wilson": overall_wilson["ci_hi"],
        "decision_threshold": DECISION_THRESHOLD,
        "chance_bar_fraction": CHANCE_BAR_FRACTION,
        "chance_seam_gap": CHANCE_SEAM_GAP,
        "chance_both": CHANCE_BOTH,
        "mean_bpm_error_rel": float(np.mean(valid_bpm_err)) if valid_bpm_err else None,
        "mean_bpm_error_rel_oct_corrected": float(np.mean(valid_bpm_err_oct)) if valid_bpm_err_oct else None,
        "median_bpm_error_rel": float(np.median(valid_bpm_err)) if valid_bpm_err else None,
        "by_bpm": by_bpm,
        "above_chance_both": bool(overall_ci["ci_lo"] > CHANCE_BOTH),
        "above_decision_threshold": bool(overall_ci["ci_lo"] > DECISION_THRESHOLD),
        "config": {
            "n_bars": N_BARS, "beats_per_bar": BEATS_PER_BAR,
            "bpm_grid": BPM_GRID, "n_tracks_per_cell": tracks_per_bpm,
            "seed_base": SEED_BASE, "model": model_id,
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== §5: следование темпу, {model_id} ===")
    print(f"N треков: {n} (degenerate: {n_degenerate})")
    print(f"Средняя ошибка BPM (без октавной коррекции): {summary['mean_bpm_error_rel']}")
    print(f"Средняя ошибка BPM (с октавной коррекцией): {summary['mean_bpm_error_rel_oct_corrected']}")
    print(f"Доля прохождения сетки (оба подкритерия): {overall_ci['point']:.3f} "
          f"[{overall_ci['ci_lo']:.3f}, {overall_ci['ci_hi']:.3f}] (bootstrap 1000)")
    print(f"  Порог решающего правила: {DECISION_THRESHOLD}")
    print(f"  Случайность (оба сразу, из родительского эксперимента): {CHANCE_BOTH}")
    print(f"  Выше случайности (нижняя граница CI > {CHANCE_BOTH}): {summary['above_chance_both']}")
    print(f"  Выше порога 0.5 (нижняя граница CI > 0.5): {summary['above_decision_threshold']}")
    print(f"\nСводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()
