"""
Стадия 3: FAD/KAD на уровне набора (модель × длительность × set) — набор
объединяет все промпты (см. configs/durations.yaml). Плюс полы:
split-half и codec round-trip (по подготовленным папкам).

Вход:  results/generation_manifest.jsonl, results/references/<D>s/*.wav
       (+ results/references/<D>s_half_A|B, results/references/<D>s_rt_<codec>)
Выход: results/distributional_metrics.jsonl, results/floors.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from .metrics import distributional as dm
from .metrics.codec_floor import split_half, split_half_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("distributional")

# --------------------------------------------------------------------------
# Параллельный бутстрап по наборам.
#
# FAD и KAD на один набор уже считаются вместе, за один проход по общим
# эмбеддингам (см. docstring distributional.py) — раздельного счёта нет и
# нечего параллелить внутри пары. Настоящее узкое место — 1000 бутстрап-
# пересэмплирований на КАЖДЫЙ набор (модель × D × set), а наборов сотни;
# это чистый CPU/numpy (эмбеддинги уже извлечены и в памяти), без GPU и
# без общего состояния между наборами, поэтому распараллеливается тривиально.
#
# Эмбеддинги всё ещё снимаются СЕРИАЛЬНО в основном процессе — это GPU-шаг
# через единственный загруженный экземпляр fadtk, и держать по копии модели
# в каждом воркере было бы дорого и не нужно (эмбеддинг — не узкое место,
# судя по логам: тысячи файлов в секунду).
#
# Контекст референса передаётся воркерам ОДИН РАЗ при старте пула
# (initializer), а не с каждой задачей — иначе сотни задач сериализовали бы
# одни и те же ~десятки МБ (gauss.sigma, ref_chunks) заново на каждый набор.
# ВАЖНО (найдено 2026-09-14, после первого прогона параллельного кода):
# полы (split_half + codec round-trip) используют ТОТ ЖЕ дорогой вызов
# evaluate_set — 1000 бутстрапов, где каждый frechet_distance это
# scipy.linalg.sqrtm матрицы 512x512, ~0.4-0.5 с сам по себе (проверено
# замером; альтернатива через конгруэнцию sqrt(Σa)@Σb@sqrt(Σa) не быстрее
# на РАЗНЫХ ковариациях — быстрее только на вырожденном тесте Σ@Σ). Полов
# до 15 (5 длительностей x до 3 типов), и первая версия параллельного кода
# считала их СЕРИАЛЬНО в основном процессе, как и раньше — заняло 13+ часов
# между эмбеддингом референсов D=30 и D=60 на 8 из 15 полов. Теперь полы и
# наборы моделей идут в ОДНУ очередь на пул.
# Найдено 2026-09-18: eval-сторона floor-задач (весь референсный корпус,
# сотни файлов) на порядки больше eval-стороны набора модели (96 файлов);
# см. подробности у места использования ниже. Подрезаем до того же
# масштаба -- решение пользователя, не тихая подгонка чисел.
FLOOR_EVAL_FILES = 96


def _floor_seed(duration_s: float, floor_name: str) -> int:
    """Детерминированный сид подвыборки: одна и та же длительность+имя пола
    всегда даёт одну и ту же подвыборку файлов при перезапуске."""
    return hash((round(duration_s), floor_name)) & 0xFFFFFFFF


def _subsample_indices(idx: list[int], n: int, seed: int) -> list[int]:
    if len(idx) <= n:
        return idx
    import numpy as np
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(idx, n, replace=False))


def _subsample_files(files: list[Path], n: int, seed: int) -> list[Path]:
    if len(files) <= n:
        return files
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), n, replace=False)
    return [files[i] for i in sorted(idx)]


_worker_ref: dm.ReferenceContext | None = None


def _init_worker(ref: dm.ReferenceContext) -> None:
    global _worker_ref
    _worker_ref = ref
    # Страховка поверх переменных среды из .bat-обёртки: threadpoolctl
    # управляет УЖЕ ЗАГРУЖЕННОЙ BLAS напрямую, а не через переменную,
    # которую нативная библиотека читает лишь при первом импорте — на
    # Windows-spawn это надёжнее, чем полагаться на наследование окружения.
    # Найдено 2026-09-15: OpenBLAS брала 12 потоков НА КАЖДЫЙ воркер —
    # 6-11 параллельных процессов давали 72-132 потока на 12 ядер, и набор
    # на D=30 (не длиннее D=15 по объёму работы в sqrtm) занимал втрое
    # дольше просто из-за конкуренции потоков за процессор.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except ImportError:
        pass


def _bootstrap_task(args):
    """kind: 'set' | 'floor'. ref_override — свой референс (нужен только
    split_half: его "референс" — это половина A, а не общий ref_ctx[D],
    который используют остальные задачи этого пула через initializer)."""
    kind, key, ref_override, eval_embs, n_resamples, seed = args
    ref = ref_override if ref_override is not None else _worker_ref
    res = dm.evaluate_set(ref, eval_embs, n_resamples, seed=seed)
    return kind, key, res


def _sets(manifest: Path) -> dict:
    sets = defaultdict(list)
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "done":
                sets[(r["model_id"], float(r["duration_s"]), int(r["set_idx"]))].append(Path(r["wav_path"]))
    return sets


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["model_id"], r["duration_s"], r["set_idx"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def _done_floors(path: Path) -> set:
    """Как _done, но для floors.jsonl: без этой проверки пол пересчитывался
    и дописывался заново при КАЖДОМ перезапуске (в т.ч. после обрыва
    фонового процесса вместе с сессией) — файл накопил дубли одних и тех же
    (D, floor) при нескольких перезапусках расчёта."""
    if not path.exists():
        return set()
    return {(r["duration_s"], r["floor"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def run(out_dir: Path, config_dir: Path, embedder: str, n_resamples: int, crosscheck: bool) -> None:
    dcfg = yaml.safe_load(open(config_dir / "durations.yaml", encoding="utf-8"))
    ref_root = out_dir / "references"
    out_path, floors_path = out_dir / "distributional_metrics.jsonl", out_dir / "floors.jsonl"
    sets, done = _sets(out_dir / "generation_manifest.jsonl"), _done(out_path)
    done_floors = _done_floors(floors_path)
    # Найдено на практике (2026-09-14): при 11 воркерах процесс падал с
    # numpy MemoryError (~250 МБ на один временный массив n x n float64 в
    # kad_from_embeddings, на длинных D их несколько на итерацию бутстрапа) и
    # с OSError WinError 1450 "недостаточно системных ресурсов" — это именно
    # нехватка ресурсов IPC Windows (каналы ProcessPoolExecutor), а не общей
    # RAM: 11 воркеров синхронно попадают в пик временного выделения на
    # длинных D одновременно. Понижено вдвое с явным запасом; переменная
    # окружения позволяет подстроить без правки кода, если снова не хватит.
    max_workers = max(1, int(os.environ.get("GME_FAD_WORKERS", "0")) or
                      max(1, (os.cpu_count() or 4) // 2))

    # --- Фаза 1: ВСЕ эмбеддинги серийно (GPU, единственный экземпляр fadtk
    # в основном процессе — не узкое место, тысячи файлов в секунду). Сюда
    # входят референс, обе половины split_half (уже в кэше референса, без
    # доп. эмбеддинга), round-trip файлы полов и файлы наборов моделей.
    # Никакого evaluate_set в этой фазе — она только готовит данные для фазы 2.
    ref_ctx: dict[float, dm.ReferenceContext] = {}
    floor_tasks: list[tuple] = []   # (kind='floor', key=(D,name), ref_override, eval_embs, seed)
    for D in sorted({d for (_, d, _) in sets}):
        ref_dir = ref_root / f"{int(D)}s"
        files = sorted(ref_dir.glob("*.wav"))
        if not files:
            log.error("нет референса %s — сначала python -m src.corpus_prep", ref_dir)
            continue
        log.info("референс D=%.0f: %d файлов, эмбеддинг...", D, len(files))
        ref_embs = dm.embed_files(files, embedder)
        ref_ctx[D] = dm.ReferenceContext(ref_embs)

        manifest = json.load(open(ref_dir / "manifest.json", encoding="utf-8"))
        piece_of = {m["file"]: m["piece"] for m in manifest}
        if (D, "split_half") not in done_floors:
            # Половины — ПОДМНОЖЕСТВА уже посчитанных эмбеддингов референса,
            # без повторного эмбеддинга (см. коммит про двойную работу).
            idx_a, idx_b = split_half_indices(files, piece_of)
            # Найдено 2026-09-18: eval-сторона пола (idx_b, ~половина всего
            # референсного корпуса -- сотни файлов) на порядки больше eval-
            # стороны обычного набора модели (96 файлов), а KAD квадратичен
            # по числу чанков -- пол выходил в ~100 раз дороже набора и не
            # укладывался в разумное время (round-trip D=60 не досчитался
            # даже за 7+ часов). По решению пользователя eval-сторона полов
            # подрезается до FLOOR_EVAL_FILES файлов (тот же масштаб, что и
            # у наборов моделей) -- честное сравнение "пол vs модель" на
            # одинаковом размере выборки, а не искусственно точный пол на
            # недостижимом для наборов объёме. n_resamples (1000) не трогаем.
            idx_b = _subsample_indices(idx_b, FLOOR_EVAL_FILES, seed=_floor_seed(D, "split_half"))
            log.info("пол split_half D=%.0f: половины %d (ref) и %d (eval, подрезано с учётом "
                    "FLOOR_EVAL_FILES=%d) окон", D, len(idx_a), len(idx_b), FLOOR_EVAL_FILES)
            ctx_a = dm.ReferenceContext([ref_embs[i] for i in idx_a])
            floor_tasks.append(("floor", (D, "split_half"), ctx_a,
                                [ref_embs[i] for i in idx_b], n_resamples, 0))
        for rt in sorted(ref_root.glob(f"{int(D)}s_rt_*")):
            floor_name = rt.name.split("_rt_")[1]
            if (D, floor_name) in done_floors:
                continue
            rt_files = sorted(rt.glob("*.wav"))
            if not rt_files:
                continue
            rt_files = _subsample_files(rt_files, FLOOR_EVAL_FILES, seed=_floor_seed(D, floor_name))
            log.info("эмбеддинг round-trip %s D=%.0f: %d файлов (подрезано с учётом "
                    "FLOOR_EVAL_FILES=%d)", floor_name, D, len(rt_files), FLOOR_EVAL_FILES)
            floor_tasks.append(("floor", (D, floor_name), None,
                                dm.embed_files(rt_files, embedder), n_resamples, 0))

    pending = [(mid, D, k, files) for (mid, D, k), files in sorted(sets.items())
              if (mid, D, k) not in done and D in ref_ctx]
    set_embs: dict[tuple, list] = {}
    usable_by_key: dict[tuple, list] = {}
    for mid, D, k, files in pending:
        # Файлы без аудио И без кэшированного эмбеддинга исключаются из набора
        # явно, с логом — не тихо. Найдено на практике (2026-09-14): один
        # трек Kaggle (audioldm2_music D=15 boss_battle set0 track0) упал при
        # снятии эмбеддинга с "PytorchStreamReader failed reading zip archive"
        # (транзиентный сбой площадки), CLAP при этом посчитался. Аудио на
        # площадке уже удалено — эмбеддинг для этого трека не восстановить
        # никогда (review_log.md F.31). Без этой проверки одна испорченная
        # запись роняла расчёт FAD/KAD целиком. res.n_files по набору честно
        # отразит 95 вместо 96 — числа не подгоняются, пропуск виден в логе
        # и в итоговом файле.
        usable, missing = [], []
        for f in sorted(files):
            if f.exists() or dm.embedding_cache_path(f, out_dir, embedder).exists():
                usable.append(f)
            else:
                missing.append(f)
        if missing:
            log.warning("набор %s D=%.0f set=%d: %d из %d файлов без аудио и без "
                       "эмбеддинга, исключены: %s", mid, D, k, len(missing), len(files),
                       ", ".join(m.name for m in missing))
        log.info("эмбеддинг набора %s D=%.0f set=%d: %d файлов", mid, D, k, len(usable))
        set_embs[(mid, D, k)] = dm.embed_files_cached(usable, out_dir, embedder)
        usable_by_key[(mid, D, k)] = usable
    files_by_key = {(mid, D, k): files for mid, D, k, files in pending}

    # --- Фаза 2: бутстрап — параллельно, ПОЛЫ И НАБОРЫ В ОДНОЙ ОЧЕРЕДИ на
    # каждую D. Полы используют тот же evaluate_set (1000 x sqrtm 512x512,
    # ~0.4-0.5 с/вызов) и раньше считались серийно в фазе 1 — 13+ часов на
    # 8 из ~15 полов при первом прогоне параллельного кода по наборам,
    # потому что полы в параллелизацию не попадали вовсе.
    floors_by_D: dict[float, list] = defaultdict(list)
    for t in floor_tasks:
        floors_by_D[t[1][0]].append(t)
    sets_by_D: dict[float, list] = defaultdict(list)
    for mid, D, k, _ in pending:
        sets_by_D[D].append((mid, D, k))

    # Найдено 2026-09-16: полы (особенно round-trip) считаются на ВСЁМ
    # референсном корпусе (для D=60 это 975 файлов = ~58500 чанков) против
    # ~96 файлов (~5760 чанков) у обычного набора модели -- дороже набора в
    # ~100 раз (KAD квадратичен по числу чанков), а полная (58500, 58500)
    # float64-матрица расстояний вообще падает MemoryError (~25.5 ГБ разом;
    # исправлено блочным KAD в distributional.py). Раньше полы были в ОДНОМ
    # пуле с наборами того же D: сначала занимали все воркеры на десятки
    # часов (полы первыми в очереди), а после переноса набор+пол в один
    # список — необработанное исключение в floor-задаче убивало ВЕСЬ прогон
    # через as_completed/fut.result(), даже не дав перейти к следующей D.
    # Теперь два независимых прохода: сначала ВСЕ наборы по всем D (то, что
    # реально нужно для таблиц статьи) в своих пулах, потом отдельно ВСЕ
    # полы -- каждый со своим try/except, чтобы падение одного пола не
    # трогало ни наборы, ни остальные полы.
    for D in sorted(sets_by_D):
        tasks = [("set", (mid, D, k), None, set_embs[(mid, D, k)], n_resamples, k)
                 for mid, D, k in sets_by_D[D]]
        log.info("параллельный бутстрап (наборы) D=%.0f: %d наборов на %d ядрах",
                 D, len(tasks), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                                 initargs=(ref_ctx[D],)) as pool:
            futs = [pool.submit(_bootstrap_task, t) for t in tasks]
            for fut in as_completed(futs):
                kind, key, res = fut.result()
                mid, D_, k = key
                rec = {"model_id": mid, "duration_s": D_, "set_idx": k, "embedder": embedder,
                       "fad": res.fad, "kad": res.kad, "fad_boot": res.fad_boot, "kad_boot": res.kad_boot,
                       "n_files": res.n_files, "n_chunks": res.n_chunks, "bandwidth": res.bandwidth}
                if crosscheck and k == 0:
                    # Только файлы с реальным аудио на диске: toolkit_fad и
                    # kadtk читают wav напрямую, а не кэш эмбеддингов. Трек
                    # может быть в usable (эмбеддинг есть) и не иметь
                    # локального аудио — площадка вроде Kaggle его удаляет.
                    files = [f for f in usable_by_key[(mid, D_, k)] if f.exists()]
                    eval_dir = out_dir / "sets" / mid / f"{int(D_)}s" / f"set{k}"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    for f in files:
                        dst = eval_dir / f"{f.parent.parent.name}_{f.name}"
                        if not dst.exists():
                            dst.symlink_to(f.resolve())
                    try:
                        rec["toolkit_fad"] = dm.toolkit_fad(ref_root / f"{int(D_)}s", eval_dir, embedder)
                        rec["toolkit_fad_inf"] = dm.toolkit_fad_inf(
                            ref_root / f"{int(D_)}s", sorted(eval_dir.glob("*.wav")), embedder)
                        rec["toolkit_kad"] = dm.toolkit_kad_cli(ref_root / f"{int(D_)}s", eval_dir, embedder)
                    except Exception as e:
                        rec["toolkit_error"] = repr(e)
                # append сразу по готовности задачи — не ждём весь пул: при
                # обрыве процесса уже посчитанные наборы этого D не теряются.
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for D in sorted(floors_by_D):
        tasks = list(floors_by_D[D])
        log.info("параллельный бутстрап (полы) D=%.0f: %d полов на %d ядрах",
                 D, len(tasks), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                                 initargs=(ref_ctx[D],)) as pool:
            futs = {pool.submit(_bootstrap_task, t): t for t in tasks}
            for fut in as_completed(futs):
                submitted_key = futs[fut][1]   # (D, floor_name), известен ещё до result()
                try:
                    _, key, res = fut.result()
                except Exception as e:
                    log.error("пол %s D=%.0f упал, пропускаю: %r",
                             submitted_key[1], submitted_key[0], e)
                    continue
                D_, floor_name = key
                rec = {"duration_s": D_, "floor": floor_name, "fad": res.fad, "kad": res.kad,
                       "fad_boot": res.fad_boot, "kad_boot": res.kad_boot}
                with open(floors_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--config-dir", type=Path, default=Path("configs"))
    ap.add_argument("--embedder", default=dm.DEFAULT_EMBEDDER)
    ap.add_argument("--n-resamples", type=int, default=1000)
    ap.add_argument("--crosscheck", action="store_true", help="сверить точечные FAD/KAD с fadtk/kadtk на set0")
    a = ap.parse_args()
    run(a.out_dir, a.config_dir, a.embedder, a.n_resamples, a.crosscheck)


if __name__ == "__main__":
    main()
