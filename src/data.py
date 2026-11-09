"""Record keys, patient folds and fold-fitted feature preprocessing."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['uid', 'index']
META = KEYS + ['hid', 'hospital', 'Fold', 'EvalSet']
INTERNAL = 'InternalCV'
EXTERNAL = 'External(CD+L)'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    try:
        frame = pd.read_csv(path, encoding='utf-8-sig', low_memory=False, dtype={'uid': str, 'index': str})
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding='gbk', low_memory=False, dtype={'uid': str, 'index': str})
    for key in KEYS:
        if key not in frame or frame[key].isna().any():
            raise ValueError(f'Missing record key {key}: {path}')
        frame[key] = frame[key].astype(str).str.strip()
        if frame[key].eq('').any():
            raise ValueError(f'Empty record key {key}: {path}')
    return frame


def keys(frame):
    return list(map(tuple, frame[KEYS].to_numpy()))


def config():
    return read_json(ROOT/'config/model.json')


def paths(path):
    result = read_json(path)
    for field in ['labels', 'folds', 'features']:
        p = Path(result[field]).expanduser()
        result[field] = str((p if p.is_absolute() else ROOT/p).resolve())
    return result


def mapped_label(values, classes):
    values = pd.to_numeric(values, errors='coerce')
    valid = values.isin([1, 2, 3, 4] if classes == 4 else [0, 1])
    return values.where(valid)-(1 if classes == 4 else 0)


def validate_folds(frame):
    if not set(META).issubset(frame.columns) or frame.duplicated(KEYS).any():
        raise ValueError('Fold table requires complete metadata and unique uid,index')
    if frame.hid.isna().any() or not frame.EvalSet.isin([INTERNAL, EXTERNAL]).all():
        raise ValueError('Invalid patient ID or cohort name')
    internal = frame.loc[frame.EvalSet.eq(INTERNAL)]
    external = frame.loc[frame.EvalSet.eq(EXTERNAL)]
    if set(internal.Fold) != set(range(1, 11)) or not external.Fold.eq(0).all():
        raise ValueError('Expected internal folds 1..10 and external fold 0')
    if internal.groupby('hid').Fold.nunique().gt(1).any():
        raise ValueError('Patient leakage across internal folds')
    if set(internal.hid.astype(str)) & set(external.hid.astype(str)):
        raise ValueError('Patient leakage across cohorts')


def load_data(locations):
    manifest = read_csv(locations['folds'])
    manifest.Fold = pd.to_numeric(manifest.Fold, errors='raise').astype(int)
    validate_folds(manifest)
    labels = read_csv(locations['labels'])
    if labels.duplicated(KEYS).any():
        raise ValueError('Duplicate label records')
    result = manifest[META].merge(labels, on=KEYS, how='left', validate='one_to_one',
                                  suffixes=('', '_label'), indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError('A manifest record is absent from the label table')
    return result.drop(columns='_merge')


def load_raw(directory, cfg, eligible=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Feature directory does not exist: {directory}')
    if not any((directory/f'{name}_hubert.csv').is_file() for name in cfg['speech_tasks'].values()):
        raise FileNotFoundError('No expected HuBERT speech feature files in the feature directory')
    stores = {}
    for voice, name in cfg['speech_tasks'].items():
        path = Path(directory)/f'{name}_hubert.csv'
        if not path.exists():
            stores[voice] = {}
            continue
        frame = read_csv(path)
        columns = [c for c in frame if c.startswith('mean_F') or (c.startswith('F') and c[1:].isdigit())]
        if len(columns) != cfg['input_dimension']:
            raise ValueError(f'Expected {cfg["input_dimension"]} ordered features: {path}')
        dimension = cfg['input_dimension']
        accepted = [[f'{prefix}{i}' for i in range(start, start+dimension)]
                    for prefix in ['F', 'mean_F'] for start in [0, 1]]
        if columns not in accepted:
            raise ValueError(f'Feature columns must be in numerical order, without mixed prefixes: {path}')
        if eligible is not None:
            frame = frame.loc[frame[KEYS].apply(tuple, axis=1).isin(eligible)]
        store = {tuple(map(str, k)): g[columns].to_numpy(dtype=np.float32, copy=True)
                 for k, g in frame.groupby(KEYS, sort=False)}
        if any(not np.isfinite(x).all() for x in store.values()):
            raise ValueError(f'Nonfinite features: {path}')
        stores[voice] = store
    return stores


def impute(raw, fit_keys, record_keys):
    fills, stores = {}, {}
    for voice, observed in raw.items():
        selected = sorted(set(observed).intersection(fit_keys))
        if not selected:
            raise ValueError(f'No observed training recording: {voice}')
        fills[voice] = np.stack([observed[k].mean(axis=0) for k in selected]).mean(axis=0).astype(np.float32)
        stores[voice] = {k: observed[k] if k in observed else fills[voice][None, :] for k in record_keys}
    return stores, fills


def scalers(frame, stores):
    result = {}
    for voice, store in stores.items():
        dimension = next(iter(store.values())).shape[1]
        total, total_sq, count = np.zeros(dimension, np.float64), np.zeros(dimension, np.float64), 0
        for key in keys(frame):
            values = store[key].astype(np.float64)
            total += values.sum(axis=0)
            total_sq += np.square(values).sum(axis=0)
            count += len(values)
        if count == 0:
            raise ValueError('No training segments')
        mean = total/count
        std = np.sqrt(np.maximum(total_sq/count-np.square(mean), 1e-12))
        std = np.where(std < 1e-6, 1.0, std)
        result[voice] = (mean.astype(np.float32), std.astype(np.float32))
    return result


def mean_embeddings(frame, stores, fitted_scalers):
    tasks = []
    for voice, store in stores.items():
        mean, std = fitted_scalers[voice]
        tasks.append(np.asarray([((store[k]-mean)/std).mean(axis=0) for k in keys(frame)], dtype=np.float32))
    return np.stack(tasks, axis=1).mean(axis=1)


class Segments(Dataset):
    def __init__(self, frame, stores, fitted_scalers, cfg, seed=0, train=False):
        self.keys = keys(frame)
        self.labels = frame.target.astype(int).to_numpy() if train else None
        self.stores, self.scalers, self.cfg, self.train = stores, fitted_scalers, cfg, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        values = {}
        for voice, store in self.stores.items():
            x = store[self.keys[index]]
            if self.train:
                count = len(x)
                if self.cfg['sampling'] == 'fixed_cap':
                    if self.cfg['max_segments'] > 0:
                        count = min(self.cfg['max_segments'], len(x))
                elif len(x) > 1 and self.rng.random() >= .5:
                    count = int(self.rng.integers(max(1, math.ceil(len(x)/2)), len(x)+1))
                if count < len(x):
                    x = x[self.rng.choice(len(x), size=count, replace=False)]
            mean, std = self.scalers[voice]
            values[voice] = ((x-mean)/std).astype(np.float32)
        item = {'inputs': values}
        if self.train:
            item['y'] = int(self.labels[index])
        return item


def collate(batch):
    inputs, masks = {}, {}
    for voice in batch[0]['inputs']:
        maximum = max(len(x['inputs'][voice]) for x in batch)
        dimension = batch[0]['inputs'][voice].shape[1]
        values = np.zeros((len(batch), maximum, dimension), np.float32)
        mask = np.zeros((len(batch), maximum), np.float32)
        for row, item in enumerate(batch):
            x = item['inputs'][voice]
            values[row, :len(x)], mask[row, :len(x)] = x, 1.
        inputs[voice], masks[voice] = torch.from_numpy(values), torch.from_numpy(mask)
    result = {'inputs': inputs, 'masks': masks}
    if 'y' in batch[0]:
        result['y'] = torch.tensor([x['y'] for x in batch], dtype=torch.long)
    return result
