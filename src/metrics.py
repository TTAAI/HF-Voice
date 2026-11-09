"""Record-level discrimination and classification metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score


def auc(labels, probabilities, classes):
    if len(labels) == 0:
        return float('nan')
    if classes == 2:
        return float(roc_auc_score(labels, probabilities[:, 1])) if len(np.unique(labels)) == 2 else float('nan')
    values = [roc_auc_score(labels == k, probabilities[:, k]) for k in range(classes)
              if len(np.unique(labels == k)) == 2]
    return float(np.mean(values)) if values else float('nan')


def validate_probabilities(probabilities, classes=None):
    p = np.asarray(probabilities)
    if (p.ndim != 2 or (classes is not None and p.shape[1] != classes)
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid probability matrix')


def evaluate(frame, cohort):
    from data import KEYS, INTERNAL, EXTERNAL
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate records')
    if 'Fold' in frame:
        expected = set(range(1, 11)) if cohort == 'InternalOOF' else {0}
        if set(frame.Fold) != expected:
            raise ValueError('Unexpected evaluation folds')
    if 'EvalSet' in frame and set(frame.EvalSet) != {INTERNAL if cohort == 'InternalOOF' else EXTERNAL}:
        raise ValueError('Mixed or incorrect evaluation cohort')
    if cohort == 'InternalOOF' and frame.groupby('hid').Fold.nunique().gt(1).any():
        raise ValueError('Patient leakage across test folds')
    p = frame[[f'Label3_p{i}' for i in range(4)]].to_numpy(dtype=float)
    validate_probabilities(p, 4)
    labels = pd.to_numeric(frame.Label3, errors='coerce')-1
    mask = labels.isin(range(4)).to_numpy()
    y, p = labels[mask].to_numpy(dtype=int), p[mask]
    if not len(y):
        raise ValueError('No valid Label3 labels')
    out = {'EvalSet': cohort, 'N': len(y)}
    ovr = []
    for k in range(4):
        value = float(roc_auc_score(y == k, p[:, k])) if len(np.unique(y == k)) == 2 else float('nan')
        out[f'Label_{k+1}_vs_rest'] = value
        ovr.append(value)
    out['Label_4class_macro_ovr'] = float(np.nanmean(ovr))
    for lower, upper in [(0, 1), (1, 2), (2, 3)]:
        pair = np.isin(y, [lower, upper])
        out[f'Label{lower+1}_vs_Label{upper+1}'] = float(roc_auc_score(y[pair] == upper, p[pair, upper])) if len(np.unique(y[pair])) == 2 else float('nan')
    pred = p.argmax(axis=1)
    out.update(Accuracy=float(accuracy_score(y, pred)), BalancedAccuracy=float(balanced_accuracy_score(y, pred)),
               MacroF1=float(f1_score(y, pred, labels=[0, 1, 2, 3], average='macro', zero_division=0)))
    return out
