"""Сравнить спектральный подбор стыка и случайную обрезку с тем же числом долей.

Случайная позиция выбирается среди допустимых границ, а длина в долях
берётся у подобранного отрезка. Это сравнение позиции среза при заданной длине.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

for env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[env] = '1'
import numpy as np
import librosa

from src import audio_utils as audio
from src.eval import beat_aligned_cut as bac, criteria_v2 as v2
from src.metrics import degeneracy as deg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 20261016
SETS = {
    "tempo_sa3": "stage0_5_tempo_follow_stable_audio_3_small_music",
    "tempo_ace": "stage0_5_tempo_follow_ace_step_v1_3_5b",
}
THRESHOLDS = json.loads((ROOT / "configs" / "degeneracy_thresholds.json").read_text(encoding="utf-8"))

def load_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def ok(row, mode):
    ev = row.get(mode)
    return bool(ev and v2.decide(ev,.9,row['degenerate'])['both_ok'])

def mcnemar(before, after):
    improved = sum(not a and b for a,b in zip(before,after))
    worsened = sum(a and not b for a,b in zip(before,after))
    n = improved+worsened
    p = min(1.,2*sum(math.comb(n,k) for k in range(min(improved,worsened)+1))/2**n) if n else 1.
    return dict(n=len(before),before=sum(before),after=sum(after),improved=improved,worsened=worsened,p_exact=p)

def process(job):
    old, path = job
    started = time.monotonic()
    y,sr = audio.load_for_analysis(Path(path),22050,.9)
    features = deg.report(y,sr,thresholds=THRESHOLDS)
    rec = {'set':old['set'],'file':old['file'],'bpm':old['bpm_requested'],
           'audio_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
           'duration_s':len(y)/sr,'degenerate':old['degenerate'],
           'filter_recomputed':features,'naive_ok':ok(old,'naive'),
           'optimized_saved_ok':ok(old,'beat_aligned')}
    assert features['degenerate'] == old['degenerate'], ('filter mismatch',path)
    best = bac.find_best_beat_aligned_loop(y,sr)
    rec['optimized_selection'] = best
    if best is None:
        rec.update(optimized_ok=False,random_ok=False,random_selection=None)
        assert not old.get('beat_aligned')
        return rec
    assert best['n_bars'] == old['beat_aligned']['n_bars'], ('bar mismatch',path)
    a,b = [int(round(best[k]*sr)) for k in ('start_s','end_s')]
    rec['optimized_duration_s'] = (b-a)/sr
    rec['retained_fraction'] = (b-a)/len(y)
    # Используем те же параметры трекера, чтобы набор допустимых границ совпал.
    env = librosa.onset.onset_strength(y=y,sr=sr)
    _, beats = librosa.beat.beat_track(onset_envelope=env,sr=sr,units='time',trim=False)
    beats = np.asarray(beats,dtype=float)
    span = best['n_bars']*4
    w = v2.game_criteria._window_samples(sr,50.)
    candidates = []
    for i in range(len(beats)-span):
        start,end = float(beats[i]),float(beats[i+span])
        aa,bb = int(round(start*sr)),int(round(end*sr))
        if end-start < .3*len(y)/sr or end > len(y)/sr-.02 or bb-aa < 4*w:
            continue
        candidates.append((start,end,aa,bb))
    assert candidates
    seed = int.from_bytes(hashlib.sha256(f'{SEED}|{old["set"]}|{old["file"]}'.encode()).digest()[:8],'big')
    idx = int(np.random.default_rng(seed).integers(len(candidates)))
    start,end,aa,bb = candidates[idx]
    rec['random_selection'] = dict(start_s=start,end_s=end,n_bars=best['n_bars'],
                                   n_candidates=len(candidates),index=idx,seed=seed)
    rec['random_duration_s'] = (bb-aa)/sr
    rec['random_retained_fraction'] = (bb-aa)/len(y)
    rec['length_difference_fraction'] = ((bb-aa)-(b-a))/(b-a)
    # Сохраняем метрики и для записей, отклонённых фильтром.
    for mode,segment in [('optimized',y[a:b]),('random',y[aa:bb])]:
        ev = v2.evaluate_loop_v2(segment,sr)
        decision = v2.decide(ev,.9,old['degenerate'])
        rec[mode] = ev
        rec[f'{mode}_decision'] = decision
        rec[f'{mode}_ok'] = decision['both_ok']
    assert rec['optimized_ok'] == rec['optimized_saved_ok'], ('decision mismatch',path)
    # Сверяем обе ошибки сетки с результатом исходного прогона.
    for key in ('beat_fraction_error','seam_gap_beat_error'):
        oldvalue = old['beat_aligned']['grid'].get(key)
        newvalue = rec['optimized']['grid'].get(key)
        if oldvalue is not None:
            assert math.isclose(oldvalue,newvalue,abs_tol=1e-9), ('grid mismatch',path,key,oldvalue,newvalue)
    rec['elapsed_s'] = time.monotonic()-started
    return rec

def quantiles(values):
    a = np.asarray(values,dtype=float)
    return dict(n=len(a),q25=float(np.quantile(a,.25)),median=float(np.median(a)),
                q75=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max())) if len(a) else None

def summarize(rr):
    from collections import Counter
    output = {'design':{'random_seed':SEED,'random_draws_per_track':1,
                       'baseline':'uniform position with same detected beat span as optimized segment',
                       'all_attempts_denominator':True,'new_generation':False,
                       'post_review_exploratory':True},'sets':{}}
    for name in SETS:
        sub = [r for r in rr if r['set']==name]
        if not sub:
            continue
        candidates = [r for r in sub if r['optimized_selection'] is not None]
        clean = [r for r in candidates if not r['degenerate']]
        accepted = [r for r in candidates if r['optimized_ok']]
        before = [r['naive_ok'] for r in sub]
        optimized = [r['optimized_ok'] for r in sub]
        random = [r['random_ok'] for r in sub]
        # Stratified pair bootstrap: retain allocation to each prompt/tempo cell.
        cells = {}
        for r in sub:
            prompt = r['file'].split('_',2)[2].rsplit('.',1)[0]
            cells.setdefault((r['bpm'],prompt),[]).append(int(r['optimized_ok'])-int(r['naive_ok']))
        rng = np.random.default_rng(SEED)
        boot = np.zeros(10000)
        for values in cells.values():
            a = np.asarray(values)
            boot += rng.choice(a,size=(len(boot),len(a)),replace=True).sum(axis=1)
        boot /= len(sub)
        # More conservative sensitivity analysis: resample tempo blocks.
        blocks = [[int(r['optimized_ok'])-int(r['naive_ok']) for r in sub if r['bpm']==b]
                  for b in sorted({r['bpm'] for r in sub})]
        means = np.array([np.mean(b) for b in blocks])
        boot_blocks = rng.choice(means,size=(10000,len(means)),replace=True).mean(axis=1)
        output['sets'][name] = {
            'n':len(sub),'excluded':sum(r['degenerate'] for r in sub),
            'filter_reasons':dict(Counter(k for r in sub for k in r['filter_recomputed']['reasons'])),
            'n_with_candidate':len(candidates),'n_clean_with_candidate':len(clean),
            'paired_naive_optimized':mcnemar(before,optimized),
            'paired_random_optimized':mcnemar(random,optimized),
            'paired_naive_random':mcnemar(before,random),
            'retained_fraction_all_candidates':quantiles([r['retained_fraction'] for r in candidates]),
            'retained_fraction_clean_candidates':quantiles([r['retained_fraction'] for r in clean]),
            'retained_fraction_accepted':quantiles([r['retained_fraction'] for r in accepted]),
            'optimized_seconds_all_candidates':quantiles([r['optimized_duration_s'] for r in candidates]),
            'random_seconds_all_candidates':quantiles([r['random_duration_s'] for r in candidates]),
            'absolute_relative_length_difference':quantiles([abs(r['length_difference_fraction']) for r in candidates]),
            'delta_stratified_bootstrap_ci':list(np.quantile(boot,[.025,.975])),
            'delta_tempo_block_bootstrap_ci':list(np.quantile(boot_blocks,[.025,.975])),
            'tempo_block_deltas':{str(b):float(np.mean([int(r['optimized_ok'])-int(r['naive_ok']) for r in sub if r['bpm']==b]))
                                  for b in sorted({r['bpm'] for r in sub})},
        }
    (RESULTS/'review_extension_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    return output

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--limit',type=int,default=0)
    args = ap.parse_args()
    source = load_rows(RESULTS/'v2_sets_metrics.jsonl')
    jobs = [(r,str(RESULTS/SETS[r['set']]/'audio'/r['file'])) for r in source if r['set'] in SETS]
    if not jobs:
        ap.error("нет строк опыта с заданным темпом в results/v2_sets_metrics.jsonl")
    if args.limit:
        jobs = jobs[:args.limit]
    expected_keys = {(row["set"], row["file"]) for row, _ in jobs}
    out = RESULTS/('review_extension_probe.jsonl' if args.limit else 'review_extension_tracks.jsonl')
    existing = load_rows(out) if out.exists() else []
    keys = {(r['set'],r['file']) for r in existing}
    if not keys <= expected_keys:
        ap.error(f"{out} содержит строки другого прогона; переместите файл и повторите запуск")
    jobs = [j for j in jobs if (j[0]['set'],j[0]['file']) not in keys]
    started = time.monotonic()
    print('jobs',len(jobs),'cached',len(existing),'workers',args.workers,flush=True)
    with out.open('a',encoding='utf-8') as f, ProcessPoolExecutor(args.workers) as pool:
        for i,r in enumerate(pool.map(process,jobs,chunksize=1)):
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
            f.flush()
            existing.append(r)
            if (i+1)%12==0 or i+1==len(jobs):
                print('completed',i+1,'/',len(jobs),'seconds',round(time.monotonic()-started,1),flush=True)
    if not args.limit:
        assert len(existing) == len(expected_keys), "не все пары обработаны"
        result = summarize(existing)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    main()
