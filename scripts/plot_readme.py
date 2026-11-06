"""Render the README figures from the published observations."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from src.eval.criteria_v2 import decide


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

MODELS = {
    "stable_audio_open_1_0": ("Stable Audio Open", "#53dfba", "o"),
    "musicgen_small": ("MusicGen", "#7dbcff", "D"),
    "stable_audio_3_small_music": ("Stable Audio 3 Small", "#c3a5fd", "s"),
    "ace_step_v1_3_5b": ("ACE-Step", "#f5bb72", "^"),
    "riffusion_v1": ("Riffusion", "#f78fa6", "o"),
}

BG = "#101922"
PANEL = "#172430"
INK = "#f2f6fb"
MUTED = "#b7c7d8"
GRID = "#354655"
MINT = "#53dfba"
BLUE = "#7dbcff"


def canvas(title, subtitle, height):
    fig = plt.figure(figsize=(10, height))
    fig.add_artist(Rectangle((0.048, 0.92), 0.028, 0.008,
                             transform=fig.transFigure, facecolor=MINT))
    fig.text(0.048, 0.875, title, fontsize=24, fontweight="bold", va="top")
    fig.text(0.049, 0.753, subtitle, fontsize=11, color=MUTED, va="top")
    return fig


def style(axis, grid):
    axis.set_axisbelow(True)
    axis.grid(axis=grid, color=GRID, linewidth=0.85)
    axis.tick_params(length=0, pad=7)
    axis.spines[:].set_visible(False)


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=160)
    plt.close(fig)


def sound_distance():
    summary = json.loads((DATA / "eval_summary_30s.json").read_text(encoding="utf-8"))
    labels, colors, _ = zip(*MODELS.values())
    fig = canvas("How close is the music to NES?",
                 "30-second clips compared with NES-VMDB. Lower scores indicate greater similarity.", 4.3)
    for metric, title, multiplier, digits, position in (
        ("fad", "Fréchet Audio Distance", 1, 2, [0.22, 0.145, 0.345, 0.48]),
        ("kad", "Kernel Audio Distance × 100", 100, 1, [0.613, 0.145, 0.338, 0.48]),
    ):
        axis = fig.add_axes(position)
        values = [summary[model]["dist"][metric]["mean"] * multiplier for model in MODELS]
        bars = axis.barh(range(len(values)), values, color=colors, height=0.43, zorder=3)
        axis.set_xlim(0, max(values) * 1.22)
        axis.set_yticks(range(len(values)), labels if metric == "fad" else [""] * len(labels))
        axis.invert_yaxis()
        axis.set_title(title, loc="left", pad=11, fontweight="bold", fontsize=11)
        axis.bar_label(bars, [f"{value:.{digits}f}" for value in values], padding=5, fontsize=10)
        style(axis, "x")
    save(fig, "nes-audio-distance-en.png")


def raw_pass_rates():
    counts = defaultdict(lambda: [0, 0])
    with (DATA / "v2_per_track_metrics.jsonl").open(encoding="utf-8") as rows:
        for line in rows:
            record = json.loads(line)
            checks = record["v1"]
            evidence = {"seam": checks, "click": record["click_naive"], "grid": record["grid_naive"]}
            count = counts[(record["model_id"], record["duration_s"])]
            count[0] += decide(evidence, q=0.9, degenerate=checks["degenerate"])["both_ok"]
            count[1] += 1

    fig = canvas("Which clips loop as generated?",
                 "Share of clips that passed the loop check, by target duration.", 4.9)
    riffusion = fig.add_axes([0.09, 0.273, 0.35, 0.358])
    others = fig.add_axes([0.56, 0.273, 0.39, 0.358])

    for model, (label, color, marker) in MODELS.items():
        points = sorted((duration, 100 * passed / total)
                        for (model_id, duration), (passed, total) in counts.items()
                        if model_id == model)
        durations, rates = zip(*points)
        axis = riffusion if model == "riffusion_v1" else others
        axis.plot(durations, rates, marker=marker, color=color,
                  label=label, linewidth=2.1, markersize=5.3)
        if axis is riffusion:
            axis.fill_between(durations, rates, color=color, alpha=0.13)
            for duration, rate in points:
                offset = (-7, 7) if duration == durations[0] else (0, 7)
                align = "right" if offset[0] else "center"
                axis.annotate(f"{rate:.1f}", (duration, rate), xytext=offset,
                              textcoords="offset points", ha=align, color=color, fontsize=9)

    for axis, title, maximum, step in (
        (riffusion, "Riffusion  /  0–80%", 80, 20),
        (others, "Other models  /  0–6%", 6, 2),
    ):
        axis.set_title(title, loc="left", pad=10, fontweight="bold", fontsize=11)
        axis.set(xlim=(8, 127), ylim=(0, maximum), xlabel="Target duration (s)")
        axis.set_xticks([15, 30, 60, 90, 120])
        axis.set_yticks(range(0, maximum + 1, step))
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        style(axis, "y")

    others.legend(loc="upper center", bbox_to_anchor=(0.52, -0.33), ncol=2,
                  frameon=False, fontsize=9, columnspacing=1.2)
    save(fig, "loop-pass-rate-by-duration-en.png")


def beat_cut():
    summary = json.loads((DATA / "review_extension_summary.json").read_text(encoding="utf-8"))
    fig = canvas("What changes after a beat-aligned cut?",
                 "Requested-tempo experiment. Bars show the share of clips that passed.", 4.5)
    axis = fig.add_axes([0.20, 0.21, 0.75, 0.42])
    categories = [("Untrimmed", "#879ab1"), ("Random position", BLUE), ("Seam selected", MINT)]
    model_sets = [summary["sets"][key] for key in ("tempo_sa3", "tempo_ace")]
    for series, (label, color) in enumerate(categories):
        counts = [
            (
                group["paired_naive_optimized"]["before"],
                group["paired_random_optimized"]["before"],
                group["paired_naive_optimized"]["after"],
            )[series]
            for group in model_sets
        ]
        values = [100 * count / group["n"] for count, group in zip(counts, model_sets)]
        bars = axis.barh([i + (series - 1) * 0.20 for i in range(len(model_sets))],
                         values, height=0.15, color=color, label=label)
        axis.bar_label(bars, [f"{value:.1f}%" for value in values], padding=5, fontsize=10)

    axis.set_xlim(0, 29)
    axis.set_yticks([0, 1], ["Stable Audio 3 Small\n192 clips", "ACE-Step\n96 clips"])
    axis.invert_yaxis()
    axis.set_xticks(range(0, 26, 5))
    axis.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.legend(loc="upper center", bbox_to_anchor=(0.49, -0.24), ncol=3,
                frameon=False, fontsize=10, columnspacing=1.4)
    style(axis, "x")
    save(fig, "beat-aligned-trimming-en.png")


def loop_seam():
    fig = canvas("Anatomy of a loop seam",
                 "At playback, the end of the clip is followed by its beginning.", 4.9)
    axis = fig.add_axes([0.049, 0.38, 0.902, 0.252])
    axis.set(xlim=(-1.65, 1.65), ylim=(-1.05, 1.23))
    axis.axvspan(-0.32, 0, color=BLUE, alpha=0.13)
    axis.axvspan(0, 0.32, color=MINT, alpha=0.13)
    axis.axhline(0, color=GRID, linewidth=1)

    for lower, upper, color in ((-1.65, 0, BLUE), (0, 1.65, MINT)):
        time = np.linspace(lower, upper, 1200)
        phase = 2 * np.pi * time
        wave = np.sin(18 * phase) + 0.45 * np.sin(30 * phase)
        envelope = 0.14 + 0.42 * (0.5 + 0.5 * np.cos(2 * phase)) ** 3
        axis.plot(time, envelope * wave, color=color, linewidth=1.15)

    for location in (-1.25, -0.75, -0.25, 0.25, 0.75, 1.25):
        axis.vlines(location, -0.92, -0.55, color=MUTED, linewidth=1.2)
    axis.axvline(0, color=INK, linewidth=1.4, linestyle=(0, (4, 4)))
    axis.text(-1.54, 0.89, "End of clip", color=BLUE, fontsize=10)
    axis.text(1.54, 0.89, "Start of clip", color=MINT, fontsize=10, ha="right")
    axis.text(0, 1.01, "SEAM", ha="center", color=INK, fontsize=9,
              bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 2})
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines[:].set_visible(False)

    details = [
        (0.05, "Spectrum", "Compare the audio in short\nwindows on either side."),
        (0.365, "Click", "Check the sample jump\nacross the join."),
        (0.68, "Beat grid", "Check beat spacing across\nthe join and the full clip."),
    ]
    for left, label, detail in details:
        fig.text(left, 0.25, label, fontweight="bold", fontsize=12, va="top")
        fig.text(left, 0.181, detail, color=MUTED, fontsize=10.5, va="top", linespacing=1.4)
    save(fig, "loop-seam.png")


def main():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": MUTED,
    })
    FIGURES.mkdir(exist_ok=True)
    sound_distance()
    raw_pass_rates()
    beat_cut()
    loop_seam()


if __name__ == "__main__":
    main()
