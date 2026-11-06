# Воспроизведение эксперимента

## Числа статьи

`python verify.py` сверяет статистики, таблицы и контрольные числа с сохранёнными наблюдениями. Для FAD и KAD используются сохранённые оценки пяти наборов на модель. Для нового расчёта расстояний нужны исходные аудиофайлы и CLAP.

## Стык и долевая сетка

Общий формат сигнала для проверки — моно, 22 050 Гц, пик 0,9. Запись проходит предварительный фильтр тишины, плоскостности спектра и спектрального потока. Спектры на стыке сравниваются с собственными переходами записи на окнах 50 и 200 мс. Скачок на стыке сравнивается с 90-м перцентилем скачков на атаках нот. Если трекер обнаружил меньше четырёх атак, берётся 99-й перцентиль всех соседних отсчётов. Сетка оценивается на записи с подклеенными хвостом и началом. И длина фрагмента, и интервал через стык должны попадать в 0,1 доли от ближайшего целого числа долей.

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

Для калибровки скачка возьмите файл nes_loop_index.json из папки data и положите MP3 из [NES-VMDB](https://github.com/rubensolv/NES-VMDB) в `Audio/` либо задайте `NES_VMDB_AUDIO` как путь к папке корпуса. В опубликованном опыте индекс пополам разделён с сидом 0.

```bash
python -m src.stage0.calibrate_v2 --loop-index data/nes_loop_index.json
```

## Прогон моделей

Запускайте команды из корня репозитория. Выберите отдельную среду для нужного бэкенда и установите подходящую сборку PyTorch с CUDA. Для MusicGen используются Transformers, для Stable Audio Open — Diffusers и `torchsde`, для Riffusion — [`riffusion-hobby`](https://github.com/riffusion/riffusion-hobby) и FFmpeg, для ACE-Step — пакет [`acestep` версии v1](https://github.com/ace-step/ACE-Step), для Stable Audio 3 — [`stable_audio_3`](https://github.com/Stability-AI/stable-audio-3). Для Riffusion возьмите `og_beat.png` из `riffusion-hobby` и положите его по пути, указанному в `configs/models.yaml`. Для локальных весов ACE-Step задайте `ACE_STEP_CHECKPOINT_DIR`. Проверьте доступ к весам в выбранной среде.

```bash
python -m src.run_experiment --dry-run
python -m src.run_experiment --models stable_audio_3_small_music
python -m src.compute_track_metrics
python -m src.eval.recompute_v2
```

Повторите генерацию для остальных моделей в их средах, сохраняя выход в общем `results/`. `compute_track_metrics` применяет пороги из `configs/degeneracy_thresholds.json`, если в `results/` нет порогов своего прогона. `recompute_v2` работает с полными файлами из манифеста. В план входит фанфара для сравнения звучания, но она исключена из расчёта пригодности к повтору.

Для пересчёта FAD и KAD подготовьте образец и поставьте `fadtk` в среду с бэкендом CLAP. Сравниваются окна по 10 с. Опция `--crosscheck` дополнительно требует `kadtk`.

```bash
python -m src.corpus_prep --audio-root Audio --durations 30 --n-segments 1000
python -m src.compute_distributional
python -m src.summarize
```

## Заданный темп и обрезка

Опыт использует восемь значений BPM и восьмитактовые записи (192 для SA3, 96 для ACE-Step). Базовая линия берёт случайную позицию с тем же числом обнаруженных долей, что у спектрально лучшей обрезки.

```bash
python -m src.eval.tempo_follow --model stable_audio_3_small_music
python -m src.eval.tempo_follow --model ace_step_v1_3_5b
python -m src.eval.eval_sets_v2
python -m scripts.compare_beat_cuts
```
