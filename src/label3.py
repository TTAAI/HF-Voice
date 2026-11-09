"""Twenty-dimensional posterior input and the fixed Label3 MLP."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from data import KEYS, META, read_csv, mapped_label, write_json
from training import seed_all, class_weights


class LabelHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(20, cfg['hidden']), nn.ReLU(),
                                    nn.Dropout(cfg['dropout']), nn.Linear(cfg['hidden'], 4))

    def forward(self, values):
        return self.layers(values)


def posterior_columns(cfg):
    return [f'{c["task"]}_p{i}' for c in cfg['experts'].values() for i in
            (range(4) if c['classes'] == 4 else [1])]


def posterior_table(run, stage, cfg, labels):
    base = None
    for key, expert in cfg['experts'].items():
        frame = read_csv(Path(run)/stage/f'{key}.csv')
        if frame.duplicated(KEYS).any():
            raise ValueError(f'Duplicate expert output: {key}')
        if base is None:
            base = frame[META].merge(labels[KEYS+['Label3']], on=KEYS, validate='one_to_one')
        columns = [f'{expert["task"]}_p{i}' for i in (range(4) if expert['classes'] == 4 else [1])]
        base = base.merge(frame[KEYS+['Fold']+columns], on=KEYS+['Fold'], how='left', validate='one_to_one')
    columns = posterior_columns(cfg)
    if base[columns].isna().any().any():
        raise ValueError('Missing expert posterior')
    base['target'] = mapped_label(base.Label3, 4)
    return base


def fit_head(matrix, labels, cfg, seed, device):
    seed_all(seed)
    model = LabelHead(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    weight = class_weights(labels, 4, cfg['class_weight']).to(device)
    loader = DataLoader(TensorDataset(torch.tensor(matrix, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)),
                        batch_size=cfg['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(cfg['epochs']):
        model.train()
        for values, targets in loader:
            values, targets = values.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(values), targets, weight=weight, label_smoothing=cfg['label_smoothing'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.)
            optimizer.step()
    return model


def predict_head(model, matrix, device):
    parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), 512):
            parts.append(torch.softmax(model(torch.tensor(matrix[start:start+512], dtype=torch.float32, device=device)), dim=1).cpu().numpy())
    return np.concatenate(parts)


def save_head(model, cfg, matrix, probabilities, directory, device):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, directory/'weights.pt')
    devices = [torch.device(device).index or 0] if device.startswith('cuda') else []
    with torch.random.fork_rng(devices=devices):
        loaded = LabelHead(cfg).to(device)
        loaded.load_state_dict(torch.load(directory/'weights.pt', map_location=device, weights_only=True))
        difference = float(np.max(np.abs(probabilities-predict_head(loaded, matrix, device))))
    if difference > 1e-6:
        raise ValueError('Label3 reload probability mismatch')
    write_json(directory/'check.json', {'status': 'PASS', 'reload_max_difference': difference})


def train_heads(run, cfg, labels, device):
    import pandas as pd
    run = Path(run)
    internal, external = posterior_table(run, 'cv', cfg, labels), posterior_table(run, 'refit', cfg, labels)
    columns, parts = posterior_columns(cfg), []
    for fold in range(1, 11):
        train = internal.loc[internal.Fold.ne(fold) & internal.target.notna()]
        test = internal.loc[internal.Fold.eq(fold)]
        model = fit_head(train[columns].to_numpy(dtype=np.float32), train.target.to_numpy(dtype=int), cfg['label3'], cfg['seed']+fold, device)
        mask = test.target.notna().to_numpy()
        matrix = test[columns].to_numpy(dtype=np.float32)
        probabilities = predict_head(model, matrix, device)[mask]
        result = test.loc[mask, META+['Label3']].copy()
        for k in range(4):
            result[f'Label3_p{k}'] = probabilities[:, k]
        parts.append(result)
        save_head(model, cfg['label3'], matrix[mask], probabilities, run/'models/label3'/f'fold{fold:02d}', device)
    train = internal.loc[internal.target.notna()]
    final = fit_head(train[columns].to_numpy(dtype=np.float32), train.target.to_numpy(dtype=int), cfg['label3'], cfg['seed'], device)
    matrix = external[columns].to_numpy(dtype=np.float32)
    p = predict_head(final, matrix, device)
    result = external[META+['Label3']].copy()
    for k in range(4):
        result[f'Label3_p{k}'] = p[:, k]
    save_head(final, cfg['label3'], matrix, p, run/'models/label3/final', device)
    return pd.concat(parts, ignore_index=True), result
