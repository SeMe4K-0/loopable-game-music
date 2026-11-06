"""Построить графики README по опубликованным данным."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.eval.criteria_v2 import decide


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

MODELS = {
    "stable_audio_open_1_0": ("Stable Audio Open", "#087866", "o"),
    "musicgen_small": ("MusicGen", "#325dc5", "D"),
    "stable_audio_3_small_music": ("Stable Audio 3 Small", "#8843aa", "s"),
    "ace_step_v1_3_5b": ("ACE-Step", "#a35810", "^"),
    "riffusion_v1": ("Riffusion", "#a43d5d", "o"),
}

BG = "#f2f5fa"
INK = "#182434"
MUTED = "#46586f"
GRID = "#dce4ee"


def setup(title, subtitle, height):
    fig = plt.figure(figsize=(10, height))
    fig.text(0.05, 0.94, title, fontsize=22, fontweight="bold", va="top")
    fig.text(0.05, 0.845, subtitle, fontsize=11.5, color=MUTED, va="top")
    return fig


def style(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.85)
    ax.tick_params(length=0, pad=8)
    ax.spines[:].set_visible(False)


def local_number(value, decimals=1):
    return f"{value:.{decimals}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def sound_distance():
    summary = json.loads((DATA / "eval_summary_30s.json").read_text(encoding="utf-8"))
    labels, colors, _ = zip(*MODELS.values())
    fig = setup("Сходство с музыкой NES", "30-секундные записи · меньшая величина означает большее сходство", 4.25)
    for metric, title, multiplier, digits, position in (
        ("fad", "FAD", 1, 2, [0.235, 0.15, 0.345, 0.58]),
        ("kad", "KAD × 100", 100, 1, [0.63, 0.15, 0.32, 0.58]),
    ):
        axis = fig.add_axes(position)
        values = [summary[model]["dist"][metric]["mean"] * multiplier for model in MODELS]
        bars = axis.barh(range(len(values)), values, color=colors, height=0.53)
        axis.set_xlim(0, max(values) * 1.24)
        axis.set_yticks(range(len(values)), labels if metric == "fad" else [""] * len(labels))
        axis.invert_yaxis()
        axis.set_title(title, loc="left", pad=10, color=INK, fontweight="bold")
        axis.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(f"{x:g}").replace(".", ",")))
        axis.bar_label(bars, [local_number(value, digits) for value in values], padding=5)
        style(axis, "x")
    save(fig, "nes-distance.png")


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

    fig = setup("Зацикливание без обрезки", "Доля записей, прошедших проверку, по заданной длительности", 4.8)
    riffusion = fig.add_axes([0.095, 0.305, 0.35, 0.39])
    other_models = fig.add_axes([0.572, 0.305, 0.36, 0.39])
    for model, (label, color, marker) in MODELS.items():
        points = sorted((duration, 100 * passed / total)
                        for (model_id, duration), (passed, total) in counts.items()
                        if model_id == model)
        durations, rates = zip(*points)
        axis = riffusion if model == "riffusion_v1" else other_models
        axis.plot(durations, rates, marker=marker, color=color, label=label, linewidth=2, markersize=5)
        if axis is riffusion:
            for duration, rate in zip(durations, rates):
                offset = (-7, 8) if duration == durations[0] else (0, 8)
                alignment = "right" if duration == durations[0] else "center"
                axis.annotate(local_number(rate), (duration, rate), xytext=offset,
                              textcoords="offset points", ha=alignment, color=color, fontsize=9)

    for axis, title, maximum in (
        (riffusion, "Riffusion · шкала 0–80 %", 80),
        (other_models, "Другие модели · шкала 0–6 %", 6),
    ):
        axis.set_title(title, loc="left", pad=11, color=INK, fontweight="bold")
        axis.set(xlim=(8, 127), ylim=(0, maximum), xlabel="Длительность, с")
        axis.set_xticks([15, 30, 60, 90, 120])
        axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        style(axis, "y")

    other_models.legend(loc="upper center", bbox_to_anchor=(0.5, -0.33), ncol=2,
                        frameon=False, fontsize=9, columnspacing=1.3)
    save(fig, "raw-loop-rates.png")


def beat_cut():
    summary = json.loads((DATA / "review_extension_summary.json").read_text(encoding="utf-8"))
    fig = setup("Что меняет обрезка по долям", "Отдельный опыт с заданным темпом · доля записей, прошедших проверку", 4.15)
    axis = fig.add_axes([0.09, 0.22, 0.86, 0.49])
    centers = [0, 1]
    colors = ["#73839c", "#325dc5", "#087866"]
    categories = ["Без обрезки", "Случайная позиция", "Подбор стыка"]
    model_sets = [summary["sets"][key] for key in ("tempo_sa3", "tempo_ace")]
    for series, (label, color) in enumerate(zip(categories, colors)):
        counts = [
            (
                group["paired_naive_optimized"]["before"],
                group["paired_random_optimized"]["before"],
                group["paired_naive_optimized"]["after"],
            )[series]
            for group in model_sets
        ]
        values = [100 * count / group["n"] for count, group in zip(counts, model_sets)]
        bars = axis.bar([x + (series - 1) * 0.23 for x in centers], values,
                        width=0.195, color=color, label=label)
        axis.bar_label(bars, [f"{local_number(value)} %" for value in values], padding=4)

    axis.set(xlim=(-0.52, 1.52), ylim=(0, 30))
    axis.set_xticks(centers, ["Stable Audio 3 Small · 192 записи", "ACE-Step · 96 записей"])
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)
    style(axis, "y")
    save(fig, "beat-cut.png")


def main():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": "#ffffff",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": MUTED,
    })
    FIGURES.mkdir(exist_ok=True)
    sound_distance()
    raw_pass_rates()
    beat_cut()


if __name__ == "__main__":
    main()
