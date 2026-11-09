"""Save the complete final model and predict unlabelled feature records."""
from pathlib import Path
import joblib
import numpy as np
import torch

from data import KEYS, config, read_csv, keys, load_raw
from models import expert_model
from training import predict_expert
from label3 import LabelHead, predict_head
from metrics import validate_probabilities


def export_weights(run):
    run, cfg = Path(run), config()
    weights, preprocessing = {}, {}
    for key in cfg['experts']:
        directory = run/'models/refit'/key/'fold00'
        weights[key] = torch.load(directory/'weights.pt', map_location='cpu', weights_only=True)
        preprocessing[key] = joblib.load(directory/'preprocessing.joblib')
    weights['label3'] = torch.load(run/'models/label3/final/weights.pt', map_location='cpu', weights_only=True)
    directory = run/'weights'
    directory.mkdir(exist_ok=True)
    torch.save(weights, directory/'weights.pt')
    joblib.dump(preprocessing, directory/'preprocessing.joblib')


def predict(features, samples, weights_directory, device='cpu'):
    cfg = config()
    frame = read_csv(samples)
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError('Sample table requires nonempty unique uid,index')
    raw = load_raw(features, cfg, set(keys(frame)))
    observed = set().union(*(set(store) for store in raw.values()))
    missing = set(keys(frame))-observed
    if missing:
        raise ValueError(f'{len(missing)} sample records have no matching speech; check uid,index and feature files')
    weights_directory = Path(weights_directory)
    weights = torch.load(weights_directory/'weights.pt', map_location='cpu', weights_only=True)
    prep = joblib.load(weights_directory/'preprocessing.joblib')
    stage1, head_values = frame.copy(), []
    for key, expert in cfg['experts'].items():
        stores = {s: {k: observed[k] if k in observed else prep[key]['fills'][s][None, :] for k in keys(frame)}
                  for s, observed in raw.items()}
        model = expert_model(expert, cfg['speech_tasks']).to(device)
        model.load_state_dict(weights[key])
        probabilities = predict_expert(model, frame, stores, prep[key], expert, device)
        validate_probabilities(probabilities, expert['classes'])
        for k in range(expert['classes']):
            stage1[f'{expert["task"]}_p{k}'] = probabilities[:, k]
        stage1[f'{expert["task"]}_pred_label'] = probabilities.argmax(axis=1)+(1 if expert['classes'] == 4 else 0)
        stage1[f'{expert["task"]}_ConfigID'] = expert['name']
        for k in (range(4) if expert['classes'] == 4 else [1]):
            head_values.append(probabilities[:, k])
    head = LabelHead(cfg['label3']).to(device)
    head.load_state_dict(weights['label3'])
    probabilities = predict_head(head, np.stack(head_values, axis=1).astype(np.float32), device)
    validate_probabilities(probabilities, 4)
    stage2 = frame.copy()
    for k in range(4):
        stage2[f'Label3_p{k}'] = probabilities[:, k]
    stage2['Label3_pred'] = probabilities.argmax(axis=1)+1
    return stage1, stage2
