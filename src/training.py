"""One trainer for the seven neural experts and the PCA ordinal expert."""
import argparse
from collections import deque
import math
from pathlib import Path
import random

import joblib
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import (META, INTERNAL, EXTERNAL, config, paths, load_data, load_raw, keys,
                  mapped_label, impute, scalers, mean_embeddings, Segments, collate, write_json)
from models import expert_model
from metrics import auc


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_summary(values):
    if len(values) != 10 or min(values) < 1:
        raise ValueError('Expected ten positive best epochs')
    return max(1, math.floor(float(np.median(sorted(set(values))))+.5))


def class_weights(labels, classes, mode):
    if mode == 'none':
        return None
    counts = np.maximum(np.bincount(labels, minlength=classes), 1)
    values = counts.sum()/(classes*counts)
    if mode == 'balanced_sqrt':
        values = np.sqrt(values)
        values = np.clip(values/values.mean(), .5, 2.)
    elif mode == 'balanced_clip3':
        values = np.clip(values, 1/3, 3)
    else:
        raise ValueError(mode)
    return torch.tensor(values, dtype=torch.float32)


def ordinal_loss(logits, labels):
    p = torch.softmax(logits, dim=1)
    cumulative = torch.stack([p[:, 1:].sum(dim=1), p[:, 2:].sum(dim=1), p[:, 3:].sum(dim=1)], dim=1).clamp(1e-6, 1-1e-6)
    targets = torch.stack([labels > 0, labels > 1, labels > 2], dim=1).float()
    return F.binary_cross_entropy(cumulative, targets)


def expert_loss(logits, labels, cfg, weight):
    loss = F.cross_entropy(logits, labels, weight=weight, label_smoothing=cfg['label_smoothing'])
    if cfg['classes'] == 4 and cfg['ordinal_weight'] > 0:
        loss = loss+cfg['ordinal_weight']*ordinal_loss(logits, labels)
    if cfg['brier_weight'] > 0:
        targets = F.one_hot(labels, num_classes=cfg['classes']).float()
        loss = loss+cfg['brier_weight']*torch.mean(torch.sum((torch.softmax(logits, dim=1)-targets)**2, dim=1))
    return loss


def predict_neural(model, frame, stores, prep, cfg, device):
    loader = DataLoader(Segments(frame, stores, prep['scalers'], cfg), batch_size=cfg['batch_size'],
                        shuffle=False, num_workers=0, collate_fn=collate)
    parts = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            masks = {k: v.to(device) for k, v in batch['masks'].items()}
            parts.append(torch.softmax(model(inputs, masks), dim=1).cpu().numpy())
    return np.concatenate(parts)


def predict_expert(model, frame, stores, prep, cfg, device):
    if cfg['architecture'] != 'ordinal':
        return predict_neural(model, frame, stores, prep, cfg, device)
    x = prep['pca'].transform(mean_embeddings(frame, stores, prep['scalers']))
    model.eval()
    with torch.no_grad():
        return model.probabilities(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()


def average_states(states):
    return {k: torch.stack([s[k] for s in states]).mean(dim=0) if torch.is_floating_point(v)
            else v.clone() for k, v in states[-1].items()}


def fit_neural(train, early, stores, prep, cfg, speech, seed, device, fixed_epochs=None):
    seed_all(seed)
    model = expert_model(cfg, speech).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    weights = class_weights(train.target.to_numpy(dtype=int), cfg['classes'], cfg['class_weight'])
    weight = None if weights is None else weights.to(device)
    loader = DataLoader(Segments(train, stores, prep['scalers'], cfg, seed, True),
                        batch_size=cfg['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed),
                        num_workers=0, collate_fn=collate)
    recent = deque(maxlen=cfg['average_window'])
    best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
    maximum = fixed_epochs if fixed_epochs is not None else cfg['max_epochs']
    for epoch in range(1, maximum+1):
        model.train()
        total, count = 0., 0
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch['inputs'].items()}
            masks = {k: v.to(device) for k, v in batch['masks'].items()}
            labels = batch['y'].to(device)
            optimizer.zero_grad()
            loss = expert_loss(model(inputs, masks), labels, cfg, weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
            total += float(loss.item())*len(labels)
            count += len(labels)
        recent.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if fixed_epochs is not None:
            best_epoch, best_state = epoch, average_states(recent)
            continue
        if epoch % 2 != 0 and epoch != maximum:
            continue
        score = auc(early.target.to_numpy(dtype=int), predict_neural(model, early, stores, prep, cfg, device), cfg['classes']) if len(early) else float('nan')
        if not math.isfinite(score):
            score = -total/max(count, 1)
        if score > best_score:
            best_score, best_epoch, best_state, stale = score, epoch, average_states(recent), 0
        else:
            stale += 2
            if stale >= cfg['patience']:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def fit_ordinal(train, early, stores, prep, cfg, seed, device, fixed_epochs=None):
    x = mean_embeddings(train, stores, prep['scalers'])
    dimension = min(cfg['pca_dimension'], len(x)-1, x.shape[1])
    prep['pca'] = PCA(n_components=dimension, whiten=True, random_state=seed).fit(x)
    x = prep['pca'].transform(x)
    early_x = prep['pca'].transform(mean_embeddings(early, stores, prep['scalers'])) if fixed_epochs is None else None
    seed_all(seed)
    model = expert_model(cfg, []).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    tensor_x = torch.tensor(x, dtype=torch.float32, device=device)
    tensor_y = torch.tensor(train.target.to_numpy(dtype=int), dtype=torch.long, device=device)
    tensor_early = torch.tensor(early_x, dtype=torch.float32, device=device) if early_x is not None else None
    best_score, best_epoch, best_state, stale = -math.inf, 0, None, 0
    maximum = fixed_epochs if fixed_epochs is not None else cfg['max_epochs']
    for epoch in range(1, maximum+1):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(torch.log(model.probabilities(tensor_x)), tensor_y)
        loss.backward()
        optimizer.step()
        if fixed_epochs is not None:
            best_epoch = epoch
            continue
        if epoch % 2 != 0 and epoch != maximum:
            continue
        model.eval()
        with torch.no_grad():
            p = model.probabilities(tensor_early).detach().cpu().numpy()
        score = auc(early.target.to_numpy(dtype=int), p, 4)
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 2
            if stale >= cfg['patience']:
                break
    if fixed_epochs is None:
        if best_state is None:
            raise ValueError('Ordinal early-stop fold has no valid AUC')
        model.load_state_dict(best_state)
    return model, best_epoch


def train_expert(expert_id, stage, locations, output, device):
    cfg = config()
    recipe = cfg['experts'][expert_id]
    data = load_data(locations)
    data['target'] = mapped_label(data[recipe['label']], recipe['classes'])
    internal = data.loc[data.EvalSet.eq(INTERNAL)]
    raw = load_raw(locations['features'], cfg, set(keys(data)))
    out = Path(output)
    frames, best_epochs = [], []
    if stage == 'refit':
        import pandas as pd
        oof = pd.read_csv(out/'cv'/f'{expert_id}.csv')
        best_epochs = oof.groupby('Fold').BestEpoch.first().astype(int).tolist()
        fixed = epoch_summary(best_epochs)
    else:
        fixed = None
    for fold in ([0] if stage == 'refit' else range(1, 11)):
        early_fold = fold % 10+1 if fold else 0
        available_train = internal if fold == 0 else internal.loc[internal.Fold.ne(fold) & internal.Fold.ne(early_fold)]
        train = available_train.loc[available_train.target.notna()].copy()
        early = internal.loc[internal.Fold.eq(early_fold) & internal.target.notna()].copy()
        test = data.loc[data.EvalSet.eq(EXTERNAL)] if fold == 0 else internal.loc[internal.Fold.eq(fold)]
        train.target = train.target.astype(int)
        early.target = early.target.astype(int)
        fit = train if fold == 0 and recipe['architecture'] == 'ordinal' else available_train
        stores, fills = impute(raw, set(keys(fit)), set(keys(data)))
        prep = {'scalers': scalers(train, stores), 'fills': fills}
        print(f'[TRAIN] {expert_id} fold={fold} train={len(train)} early={len(early)} test={len(test)}', flush=True)
        trainer = fit_ordinal if recipe['architecture'] == 'ordinal' else fit_neural
        if recipe['architecture'] == 'ordinal':
            model, epoch = trainer(train, early, stores, prep, recipe, cfg['seed']+fold, device, fixed)
        else:
            model, epoch = trainer(train, early, stores, prep, recipe, cfg['speech_tasks'], cfg['seed']+fold, device, fixed)
        p = predict_expert(model, test, stores, prep, recipe, device)
        directory = out/'models'/stage/expert_id/f'fold{fold:02d}'
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory/'weights.pt')
        joblib.dump(prep, directory/'preprocessing.joblib')
        # Loading can initialize model parameters; restore RNG so checks do not affect training.
        devices = [torch.device(device).index or 0] if device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            loaded = expert_model(recipe, cfg['speech_tasks']).to(device)
            loaded.load_state_dict(torch.load(directory/'weights.pt', map_location=device, weights_only=True))
            again = predict_expert(loaded, test, stores, joblib.load(directory/'preprocessing.joblib'), recipe, device)
        difference = float(np.max(np.abs(p-again)))
        if difference > 1e-6:
            raise ValueError(f'Saved expert changed probabilities: {expert_id}')
        write_json(directory/'check.json', {'best_epoch': epoch, 'early_stop_fold': early_fold,
            'reload_max_difference': difference, 'status': 'PASS'})
        result = test[META+[recipe['label'], 'target']].copy().rename(columns={'target': 'MappedLabel'})
        result['BestEpoch'], result['EarlyStopFold'], result['ConfigID'] = epoch, early_fold, recipe['name']
        for i in range(recipe['classes']):
            result[f'{recipe["task"]}_p{i}'] = p[:, i]
        frames.append(result)
        print(f'[DONE] {expert_id} fold={fold} epoch={epoch}', flush=True)
    import pandas as pd
    (out/stage).mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out/stage/f'{expert_id}.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert', required=True)
    parser.add_argument('--stage', choices=['cv', 'refit'], required=True)
    parser.add_argument('--paths', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    train_expert(args.expert, args.stage, paths(args.paths), args.output, args.device)
