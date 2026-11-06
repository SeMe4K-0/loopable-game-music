"""Сверить числа статьи с сохранёнными наблюдениями. Зависимостей нет."""
import csv
import io
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def read_data(name):
    return (DATA / name).read_text(encoding="utf-8-sig")


def jl(name):
    return [json.loads(line) for line in read_data(name).splitlines() if line.strip()]


def js(name):
    return json.loads(read_data(name))


def ru(x, nd=1):
    """Округление «половина вверх», как принято в тексте (31,25 -> 31,3), с запятой."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)).replace(".", ",")


# ---------- правила приёмки (как в исходных скриптах проекта) ----------
def click_ok(d):
    ref = d.get("onset_q0.9")
    if ref is None:
        ref = d["p99_all"]
    return d["boundary"] <= ref + 1e-8


def main_ok(r):
    """Готовая запись основного опыта: фильтр + стык 50 и 200 мс + щелчок + длина."""
    s = r["v1"]
    return (not s["degenerate"] and s["seam50_ok"] and s["seam200_ok"]
            and click_ok(r["click_naive"]) and r["grid_naive"]["ok"])


def set_ok(r, mode):
    """Запись опыта с темпом: mode = naive (как есть) или beat_aligned (обрезка по долям)."""
    d = r.get(mode)
    if not d or r["degenerate"]:
        return False
    s = d["seam"]
    return bool(s["seam50_ok"] and s["seam200_ok"] and click_ok(d["click"]) and d["grid"]["ok"])


def wilson(k, n, z=1.959963984540054):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def fisher_exact(a, b, c, d):
    """Двусторонний точный критерий Фишера для таблицы [[a, b], [c, d]]."""
    r1, c1, n = a + b, a + c, a + b + c + d
    def pmf(x):
        return math.comb(c1, x) * math.comb(n - c1, r1 - x) / math.comb(n, r1)
    p0 = pmf(a)
    lo, hi = max(0, r1 + c1 - n), min(r1, c1)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-7)))


# ---------- сверка ----------
rows_out, bad = [], 0


def check(what, printed, computed, fragment=None):
    """При расхождении fragment подсказывает, где найти число в тексте."""
    global bad
    ok = printed == computed
    bad += not ok
    note = f"  [фрагмент в статье: {fragment}]" if not ok and fragment else ""
    rows_out.append(f"{'OK ' if ok else 'НЕТ'} | {what}: в статье {printed}, по данным {computed}{note}")


# ===== Материалы и методы =====
tracks = jl("v2_per_track_metrics.jsonl")
cells = defaultdict(list)
for r in tracks:
    cells[(r["model_id"], r["duration_s"])].append(r)
check("Проверено записей", "8736", str(len(tracks)), "Проверено 8736 записей пяти моделей")
check("Моделей в проверке зацикливания", "5", str(len({m for m, _ in cells})))
check("Записей на модель и длительность", "252 или 420",
      " или ".join(str(x) for x in sorted({len(v) for v in cells.values()})), "252 или 420 записей")
check("Длительности Stable Audio Open", "15, 30",
      ", ".join(str(int(d)) for m, d in sorted(cells) if m == "stable_audio_open_1_0"),
      "Stable Audio Open 1.0 длительностью 15 и 30 с")
check("Запросов в проверке повтора (без фанфары)", "7",
      str(len({r["prompt_id"] for r in tracks})), "в проверке повтора не участвовала")

man = [m for m in jl("generation_manifest.jsonl") if m.get("status") == "done"]
gpus = {m["env"]["gpu"] for m in man}
check("Видеокарта генерации", "RTX 3060", "RTX 3060" if gpus == {"NVIDIA GeForce RTX 3060"} else str(gpus),
      "видеокарте RTX 3060 с 12 ГБ памяти")

thr = js("degeneracy_thresholds.json")
check("Порог тихих кадров, %", "6,7", ru(thr["silent_frac_max"] * 100), "более 6,7 %")
check("Порог плоскостности", "0,042", ru(thr["spectral_flatness_max"], 3), "выше 0,042")
check("Порог спектрального потока", "1,51", ru(thr["spectral_flux_min"], 2), "ниже 1,51")
check("Фрагментов NES для порогов", "1373", str(thr["_calibration"]["n_reference"]), "по 1373 фрагментам")

ref = js("fad_kad_reference_meta.json")["reference"]
check("Образец для FAD/KAD", "1000 фрагментов, 279 игр",
      ("1000 фрагментов, 279 игр" if "1000 segments" in ref and "279 games" in ref else ref),
      "1000 фрагментов по 30 с из 279 игр")
s30 = js("eval_summary_30s.json")
check("Наборов × записей для FAD/KAD", "5 × 96",
      f"{s30['riffusion_v1']['dist']['fad']['k']} × {s30['riffusion_v1']['tracks']['n_tracks'] // s30['riffusion_v1']['dist']['fad']['k']}",
      "пять наборов по 96 записей")

# ===== Проверка на настоящих петлях NES =====
v2 = js("v2_summary.json")
ct = v2["control_test_half"]["both_ok"]
lo, hi = wilson(ct["k"], ct["n"])
check("Настоящие петли: принято", "47 из 75", f"{ct['k']} из {ct['n']}", "из остальных 75 проверка приняла 47")
check("Доля принятых петель, %", "62,7", ru(100 * ct["k"] / ct["n"]), "62,7 %")
check("Интервал Уилсона, %", "51,4–72,7", f"{ru(100 * lo)}–{ru(100 * hi)}", "51,4–72,7 %")
dfx = {(r["defect"], r["level"]): r for r in csv.DictReader(io.StringIO(read_data("control_defects.csv")))}
check("Щелчок 0,4: найден", "47 из 47", f"{dfx[('click_amp', '0.4')]['rejected']} из 47", "щелчок амплитудой 0,4")
check("Пауза 50 мс: найдена", "47 из 47", f"{dfx[('gap_ms', '50')]['rejected']} из 47", "паузу 50 мс всегда")
check("Укорочение 0,15 доли: найдено", "42 из 47", f"{dfx[('trim_beats', '0.15')]['rejected']} из 47", "в 42 случаях из 47")
check("Щелчок 0,1: пропущен", "38 из 47", f"{dfx[('click_amp', '0.1')]['accepted']} из 47", "щелчок амплитудой 0,1 в 38 случаях из 47")
gaps = {dfx[("gap_ms", g)]["accepted"] for g in ("2", "5", "10", "20")}
check("Паузы 2–20 мс: пропущены", "24", "/".join(sorted(gaps)), "паузы 2–20 мс в 24")

# ===== Таблица 1 =====
names = {"stable_audio_open_1_0": "Stable Audio Open 1.0", "musicgen_small": "MusicGen-small",
         "stable_audio_3_small_music": "Stable Audio 3 Small", "ace_step_v1_3_5b": "ACE-Step v1 3.5B",
         "riffusion_v1": "Riffusion v1"}
table1 = {  # как напечатано в табл. 1: FAD, KAD, время, память, приняты сразу
    "stable_audio_open_1_0": ("0,49", "29,1", "26", "5,6", "2,6"),
    "musicgen_small": ("0,67", "33,5", "52", "1,7", "4,3"),
    "stable_audio_3_small_music": ("0,67", "37,3", "0,7", "1,7", "5,2"),
    "ace_step_v1_3_5b": ("0,78", "41,4", "6,4", "8,0", "3,3"),
    "riffusion_v1": ("1,03", "62,8", "37", "3,1", "72,6"),
}
acc = {k: 100 * sum(map(main_ok, v)) / len(v) for k, v in cells.items()}
speed = {}
for m, (fad, kad, t, mem, best) in table1.items():
    mm = [x for x in man if x["model_id"] == m and x["duration_s"] == 30.0]
    tt = st.median(x["wall_time_s"] for x in mm)
    speed[m] = tt
    vram = max(x["peak_vram_gb"] for x in mm)
    row = f"| {names[m]} | {fad} | {kad} | {t} | {mem} | {best} |"
    check(f"Табл. 1, {names[m]}, FAD", fad, ru(s30[m]["dist"]["fad"]["mean"], 2), row)
    check(f"Табл. 1, {names[m]}, KAD×100", kad, ru(100 * s30[m]["dist"]["kad"]["mean"]))
    check(f"Табл. 1, {names[m]}, время 30 с (медиана), с", t, ru(tt, 1) if tt < 10 else str(round(tt)))
    check(f"Табл. 1, {names[m]}, память (пик), ГБ", mem, ru(vram))
    check(f"Табл. 1, {names[m]}, приняты сразу при лучшей длительности, %", best,
          ru(max(v for (mm_, d), v in acc.items() if mm_ == m)))
check("ACE-Step медленнее SA3, раз", "почти 10",
      "почти " + str(round(speed["ace_step_v1_3_5b"] / speed["stable_audio_3_small_music"])), "медленнее почти в 10 раз")
check("Остальные модели на 30 с, с", "от 26 до 52",
      f"от {round(min(speed[m] for m in ('stable_audio_open_1_0', 'musicgen_small', 'riffusion_v1')))} до "
      f"{round(max(speed[m] for m in ('stable_audio_open_1_0', 'musicgen_small', 'riffusion_v1')))}",
      "от 26 до 52 с")

# ===== Готовые записи, Riffusion =====
four = max(v for (m, d), v in acc.items() if m != "riffusion_v1")
check("Четыре модели: максимум принятых при любой длительности, %", "5,2", ru(four), "не более 5,2 %")
riff = {int(d): v for (m, d), v in acc.items() if m == "riffusion_v1"}
check("Riffusion при 30/60/90 с, %", "72,6, 56,7, 44,5", ", ".join(ru(riff[d]) for d in (30, 60, 90)),
      "72,6, 56,7 и 44,5 %")
check("Riffusion при 15/120 с, %", "9,5, 2,4", ", ".join(ru(riff[d]) for d in (15, 120)), "только 9,5 и 2,4 %")
bpm = [60 / r["grid_naive"]["beat_period_s"] for r in tracks
       if r["model_id"] == "riffusion_v1" and r["grid_naive"].get("beat_period_s")]
med = st.median(bpm)
share = 100 * sum(abs(b - med) <= 0.5 for b in bpm) / len(bpm)
check("Riffusion: медианный темп, уд/мин", "93,9", ru(med), "темп равен 93,9")
check("Riffusion: доля записей с темпом 93,9 ± 0,5, %", "87", str(round(share)), "у 87 % её записей")
check("Доли при медианном темпе для 15/30/60/120 с", "23,5, 47, 94, 187,9",
      ", ".join(ru(L * med / 60) if L in (15, 120) else str(round(L * med / 60)) for L in (15, 30, 60, 120)),
      "дают 23,5 и 187,9 доли")

# ===== Опыт с темпом и обрезкой =====
tf_sa3 = js("tempo_follow_sa3_summary.json")
tf_ace = js("tempo_follow_ace_summary.json")
check("Медианная ошибка темпа SA3, %", "0,18", ru(100 * tf_sa3["median_bpm_error_rel"], 2), "ошибка 0,18 %")
check("Медианная ошибка темпа ACE-Step, %", "17,6", ru(100 * tf_ace["median_bpm_error_rel"]), "ACE-Step нет (17,6 %)")

sets = defaultdict(list)
for r in jl("v2_sets_metrics.jsonl"):
    sets[r["set"]].append(r)
ext = [r for r in jl("review_extension_tracks.jsonl")]
rnd = {(r["set"], r["file"]): r["random_ok"] for r in ext}
exts = js("review_extension_summary.json")["sets"]
printed = {"tempo_sa3": ("192", "14,6", "25,0", "32", "12", "0,004", "20,8", "0,22", "107"),
           "tempo_ace": ("96", "1,0", "20,8", "19", "0", "< 0,001", "15,6", "0,42", "17")}
for s, (n_, before_, after_, imp_, wor_, p_, rand_, prand_, filt_) in printed.items():
    rr = sets[s]
    n = len(rr)
    nv = [set_ok(r, "naive") for r in rr]
    ba = [set_ok(r, "beat_aligned") for r in rr]
    rd = [rnd[(s, r["file"])] for r in rr]
    imp = sum((not a) and b for a, b in zip(nv, ba))
    wor = sum(a and (not b) for a, b in zip(nv, ba))
    p = mcnemar_exact(imp, wor)
    imp2 = sum((not a) and b for a, b in zip(rd, ba))
    wor2 = sum(a and (not b) for a, b in zip(rd, ba))
    lab = "SA3" if s == "tempo_sa3" else "ACE-Step"
    check(f"{lab}: записей в опыте с темпом", n_, str(n))
    check(f"{lab}: принято без обработки, %", before_, ru(100 * sum(nv) / n))
    check(f"{lab}: принято после обрезки, %", after_, ru(100 * sum(ba) / n))
    check(f"{lab}: стали приниматься / перестали", f"{imp_} / {wor_}", f"{imp} / {wor}")
    check(f"{lab}: p (Мак-Немар, до/после)", p_, "< 0,001" if p < 0.001 else ru(p, 3))
    check(f"{lab}: случайная обрезка той же длины, %", rand_, ru(100 * sum(rd) / n))
    check(f"{lab}: p (Мак-Немар, подбор/случайно)", prand_, ru(mcnemar_exact(imp2, wor2), 2))
    check(f"{lab}: исключено фильтром", filt_, str(sum(r["degenerate"] for r in rr)))
check("SA3: исключено из-за тихих мест", "86", str(exts["tempo_sa3"]["filter_reasons"]["mostly_silence"]),
      "(86 из-за тихих мест)")
check("Остаётся доля записи (медиана, SA3)", "примерно половина",
      "примерно половина" if abs(exts["tempo_sa3"]["retained_fraction_accepted"]["median"] - 0.5) < 0.05 else "иначе",
      "оставалась примерно половина записи")
check("Длительность фрагмента (медиана, SA3), с", "около 8",
      f"около {round(exts['tempo_sa3']['optimized_seconds_all_candidates']['median'])}", "около 8 с")

# ===== Таблица 2 и сцены =====
SCENE = {"boss_battle": "Бой", "regular_battle": "Бой", "final_boss_hybrid": "Бой",
         "ambient_exploration": "Исследование мира", "peaceful_overworld": "Исследование мира",
         "village_town": "Деревня и подземелье", "dungeon_danger": "Деревня и подземелье"}
agg = defaultdict(lambda: [0, 0, 0, 0])  # n, как есть, после обрезки, исключено
for s in ("tempo_sa3", "tempo_ace"):
    for r in sets[s]:
        sc = SCENE[re.sub(r"^\d+_bpm\d+_|\.flac$", "", r["file"])]
        a = agg[(s, sc)]
        a[0] += 1
        a[1] += set_ok(r, "naive")
        a[2] += set_ok(r, "beat_aligned")
        a[3] += r["degenerate"]
table2 = (
    ("Бой", 96, 48, ("22,9", "29,2", "2,1", "14,6")),
    ("Исследование мира", 48, 32, ("12,5", "33,3", "0", "25,0")),
    ("Деревня и подземелье", 48, 16, ("0", "8,3", "0", "31,3")),
)
for sc, n_sa3, n_ace, vals in table2:
    a, b = agg[("tempo_sa3", sc)], agg[("tempo_ace", sc)]
    comp = [ru(100 * a[1] / a[0]), ru(100 * a[2] / a[0]), ru(100 * b[1] / b[0]), ru(100 * b[2] / b[0])]
    comp = ["0" if c == "0,0" else c for c in comp]
    check(f"Табл. 2, {sc}: записей SA3 / ACE-Step", f"{n_sa3} / {n_ace}", f"{a[0]} / {b[0]}")
    check(f"Табл. 2, {sc}: SA3, SA3+обрезка, ACE, ACE+обрезка, %", " | ".join(vals), " | ".join(comp))
scene_p = {}
for sc, pr, frag in (("Бой", "0,06", "(p = 0,06)"), ("Деревня и подземелье", "0,04", "(p = 0,04)"),
                     ("Исследование мира", "0,47", "(p = 0,47)")):
    a, b = agg[("tempo_sa3", sc)], agg[("tempo_ace", sc)]
    p = fisher_exact(a[2], a[0] - a[2], b[2], b[0] - b[2])
    scene_p[sc] = p
    check(f"Фишер, {sc}, SA3 против ACE-Step после обрезки", pr, ru(p, 2), frag)
a, b = agg[("tempo_sa3", "Бой")], agg[("tempo_ace", "Бой")]
check("Бой: SA3 вдвое больше, чем ACE-Step", "2", str(round((a[2] / a[0]) / (b[2] / b[0]))), "вдвое больше")
a = agg[("tempo_sa3", "Деревня и подземелье")]
b = agg[("tempo_ace", "Деревня и подземелье")]
check("Деревня и подземелье: исключено у SA3", "44 из 48", f"{a[3]} из {a[0]}", "исключил 44 из 48 записей SA3")
check("Деревня и подземелье: принято у ACE-Step", "5 из 16", f"{b[2]} из {b[0]}", "прошли 5 из 16")

# Поправка Холма применяется к трём разведочным сравнениям по сценам.
holm = {}
running = 0.0
for rank, (scene, p) in enumerate(sorted(scene_p.items(), key=lambda item: item[1])):
    running = max(running, min(1.0, (len(scene_p) - rank) * p))
    holm[scene] = running
check("Поправка Холма для деревни и подземелья", "0,11", ru(holm["Деревня и подземелье"], 2))
for group, model, expected in (("tempo_sa3", "SA3", "8,7"), ("tempo_ace", "ACE-Step", "7,8")):
    lengths = [row["optimized_duration_s"] for row in ext if row["set"] == group and row["optimized_ok"]]
    check(f"{model}: медиана длительности принятых обрезок, с", expected, ru(st.median(lengths)))
raw_30 = [pct for (model, length), pct in acc.items() if model != "riffusion_v1" and length == 30]
check("Четыре модели при 30 с, диапазон принятых, %", "1,2–4,3", f"{ru(min(raw_30))}–{ru(max(raw_30))}")
controls = js("nes_control_measurements.json")
held_out = [row["ev"]["base"] for row in controls if row["set"] == "test"]
accepted = sum(
    not row["degenerate"] and row["seam"]["seam50_ok"] and row["seam"]["seam200_ok"]
    and click_ok(row["click"]) and row["grid"]["ok"]
    for row in held_out
)
check("Контроль NES по признакам каждой петли", "47 из 75", f"{accepted} из {len(held_out)}")
for model, duration in (("stable_audio_open_1_0", 30), ("musicgen_small", 30),
                        ("stable_audio_3_small_music", 120), ("ace_step_v1_3_5b", 120), ("riffusion_v1", 30)):
    best_duration = max((d for m, d in acc if m == model), key=lambda d: acc[model, d])
    check(f"{names[model]}: лучшая длительность готовой петли, с", str(duration), str(int(best_duration)))

print("\n".join(rows_out))
print(f"\nПроверено утверждений: {len(rows_out)}; расхождений: {bad}")
sys.exit(1 if bad else 0)
