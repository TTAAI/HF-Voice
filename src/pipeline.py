"""Fixed-model workflow and command-line interface."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from data import ROOT, META, KEYS, config, paths, read_csv, read_json, write_json, load_data
from training import epoch_summary
from label3 import train_heads
from inference import export_weights, predict
from metrics import evaluate, validate_probabilities


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_expert_outputs(out, stage, cfg, frame):
    from data import INTERNAL, EXTERNAL
    expected = frame.loc[frame.EvalSet.eq(INTERNAL if stage == 'cv' else EXTERNAL)]
    for key, expert in cfg['experts'].items():
        d = read_csv(out/stage/f'{key}.csv')
        if d.duplicated(KEYS).any() or set(map(tuple, d[KEYS].to_numpy())) != set(map(tuple, expected[KEYS].to_numpy())):
            raise ValueError(f'Incomplete expert predictions: {key}')
        j = d.merge(expected[KEYS+['Fold']], on=KEYS, validate='one_to_one', suffixes=('', '_expected'))
        if not j.Fold.eq(j.Fold_expected).all():
            raise ValueError(f'Fold mismatch: {key}')
        validate_probabilities(d[[f'{expert["task"]}_p{i}' for i in range(expert['classes'])]].to_numpy(), expert['classes'])
        if stage == 'cv' and not d.EarlyStopFold.eq(d.Fold % 10+1).all():
            raise ValueError(f'Early-stop fold mismatch: {key}')


def run_experts(out, stage, cfg, path_file, locations, gpus, resume, frame):
    def worker(gpu, items):
        for key in items:
            target = out/stage/f'{key}.csv'
            if resume and target.exists():
                continue
            logfile = out/'logs'/f'{stage}_{key}.log'
            logfile.parent.mkdir(exist_ok=True)
            device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
            cmd = [sys.executable, str(ROOT/'src/training.py'), '--expert', key, '--stage', stage,
                   '--paths', str(Path(path_file).resolve()), '--output', str(out), '--device', device]
            print('[START]', stage, key, device, flush=True)
            with logfile.open('w', encoding='utf-8') as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                             OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            print('[DONE]', stage, key, flush=True)
    jobs = [[] for _ in gpus]
    for i, key in enumerate(cfg['experts']):
        jobs[(i//4) % len(gpus)].append(key)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, items) for gpu, items in zip(gpus, jobs)]
        for future in futures:
            future.result()
    validate_expert_outputs(out, stage, cfg, frame)


def export_stage1(out, cfg, frame):
    base = frame.copy()
    for key, expert in cfg['experts'].items():
        d = pd.concat([read_csv(out/'cv'/f'{key}.csv'), read_csv(out/'refit'/f'{key}.csv')], ignore_index=True)
        probability_cols = [f'{expert["task"]}_p{i}' for i in range(expert['classes'])]
        columns = probability_cols+['BestEpoch', 'EarlyStopFold', 'ConfigID', 'MappedLabel']
        d = d[KEYS+['Fold']+columns].rename(columns={c: f'{expert["task"]}_{c}' for c in columns if c not in probability_cols})
        base = base.merge(d, on=KEYS+['Fold'], how='left', validate='one_to_one')
    base.to_csv(out/'stage1_predictions.csv', index=False, encoding='utf-8-sig')


def train(args):
    cfg, locations = config(), paths(args.paths)
    frame = load_data(locations)
    out = Path(args.output).resolve()
    contract = {'model': sha(ROOT/'config/model.json'), 'locations': locations,
                'source': {p.name: sha(p) for p in (ROOT/'src').glob('*.py')},
                'data': {k: sha(locations[k]) for k in ['labels', 'folds']}}
    contract['features'] = {name: sha(Path(locations['features'])/f'{name}_hubert.csv') for name in cfg['speech_tasks'].values()}
    if out.exists() and any(out.iterdir()):
        if not args.resume or not (out/'contract.json').exists() or read_json(out/'contract.json') != contract:
            raise ValueError('Use a new output directory or --resume with unchanged code and data')
    write_json(out/'contract.json', contract)
    device = f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu'
    try:
        (out/'status.txt').write_text('RUNNING\n')
        for stage in ['cv', 'refit']:
            run_experts(out, stage, cfg, args.paths, locations, args.gpus, args.resume, frame)
        internal, external = train_heads(out, cfg, frame, device)
        for d in [internal, external]:
            d['ModelID'], d['HeadID'] = cfg['id'], cfg['label3']['name']
            d['Label3_pred'] = d[[f'Label3_p{i}' for i in range(4)]].to_numpy().argmax(axis=1)+1
        internal.to_csv(out/'internal_oof_predictions.csv', index=False, encoding='utf-8-sig')
        external.to_csv(out/'external_predictions.csv', index=False, encoding='utf-8-sig')
        pd.concat([internal, external], ignore_index=True).to_csv(out/'stage2_predictions.csv', index=False, encoding='utf-8-sig')
        result = [evaluate(internal, 'InternalOOF'), evaluate(external, 'ExternalOnce')]
        pd.DataFrame(result).to_csv(out/'metrics.csv', index=False, encoding='utf-8-sig')
        plan = []
        for key in cfg['experts']:
            epochs = read_csv(out/'cv'/f'{key}.csv').groupby('Fold').BestEpoch.first().astype(int).tolist()
            plan.append({'expert': key, 'fold_best_epochs': ';'.join(map(str, epochs)), 'refit_epochs': epoch_summary(epochs)})
        pd.DataFrame(plan).to_csv(out/'refit_epochs.csv', index=False)
        export_stage1(out, cfg, frame)
        export_weights(out)
        checks = [read_json(p) for p in (out/'models').rglob('check.json')]
        write_json(out/'verification.json', {'reload_checks': len(checks), 'all_pass': all(x['status']=='PASS' for x in checks),
            'Internal': result[0]['Label_4class_macro_ovr'], 'External': result[1]['Label_4class_macro_ovr']})
        print('[RESULT]', result[0]['Label_4class_macro_ovr'], result[1]['Label_4class_macro_ovr'], flush=True)
        (out/'status.txt').write_text('COMPLETE\n')
    except BaseException:
        (out/'status.txt').write_text('FAILED; see error.txt\n')
        (out/'error.txt').write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train')
    p.add_argument('--paths', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', nargs='+', help='GPU indices; default uses up to two available GPUs, or CPU')
    p.add_argument('--resume', action='store_true')
    p = sub.add_parser('predict')
    p.add_argument('--features', required=True)
    p.add_argument('--samples', required=True)
    p.add_argument('--weights', default=str(ROOT/'weights'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    p = sub.add_parser('evaluate')
    p.add_argument('--predictions', required=True)
    p.add_argument('--cohort', choices=['InternalOOF', 'ExternalOnce'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train':
        if args.gpus is None:
            args.gpus = [str(i) for i in range(min(torch.cuda.device_count(), 2))] or ['0']
        if any(not g.isdigit() for g in args.gpus) or len(set(args.gpus)) != len(args.gpus):
            parser.error('--gpus requires unique nonnegative integer indices')
        if torch.cuda.is_available() and any(int(g) >= torch.cuda.device_count() for g in args.gpus):
            parser.error('A requested GPU index does not exist; use --gpus 0 for a single GPU')
        train(args)
    elif args.command == 'predict':
        out = Path(args.output)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Use an empty prediction directory')
        first, second = predict(args.features, args.samples, args.weights, args.device)
        out.mkdir(parents=True, exist_ok=True)
        first.to_csv(out/'stage1_predictions.csv', index=False, encoding='utf-8-sig')
        second.to_csv(out/'stage2_predictions.csv', index=False, encoding='utf-8-sig')
        print('[PREDICTED]', len(second), 'records')
    else:
        import json
        print(json.dumps(evaluate(read_csv(args.predictions), args.cohort), ensure_ascii=False, indent=2))
