"""Checks for the fixed configuration and evaluation protocol."""
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from data import config, validate_folds, INTERNAL, EXTERNAL, load_raw
from models import expert_model
from training import epoch_summary, expert_loss, class_weights
from label3 import posterior_columns
from metrics import evaluate, validate_probabilities
from inference import predict


class ModelTests(unittest.TestCase):
    def test_epoch_rule(self):
        self.assertEqual(epoch_summary([30,36,28,18,4,2,26,2,2,8]), 22)
        self.assertEqual(epoch_summary([6,10,24,6,4,16,48,2,16,18]), 13)
        with self.assertRaises(ValueError):
            epoch_summary([2,4])

    def test_patient_grouping(self):
        d = pd.DataFrame([{'uid':str(i), 'index':'1', 'hid':str(i), 'hospital':'example',
                           'Fold':i, 'EvalSet':INTERNAL} for i in range(1,11)] +
                         [{'uid':'external', 'index':'1', 'hid':'external', 'hospital':'example',
                           'Fold':0, 'EvalSet':EXTERNAL}])
        validate_folds(d)
        bad = d.copy()
        bad.loc[1,'hid'] = bad.loc[0,'hid']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_folds(bad)
        bad = d.copy()
        bad.loc[10,'hid'] = bad.loc[0,'hid']
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_folds(bad)

    def test_pooled_record_auc(self):
        rng = np.random.default_rng(2024)
        y = np.tile(np.arange(4), 10)
        p = rng.dirichlet(np.ones(4), len(y))
        d = pd.DataFrame({'uid':[f'v{i}' for i in range(40)], 'index':'1',
                          'hid':[f'p{i}' for i in range(40)], 'Fold':np.repeat(np.arange(1,11),4),
                          'EvalSet':INTERNAL, 'Label3':y+1})
        d.loc[1,'hid'] = d.loc[0,'hid']
        for k in range(4):
            d[f'Label3_p{k}'] = p[:,k]
        result = evaluate(d, 'InternalOOF')
        expected = np.mean([roc_auc_score(y==k, p[:,k]) for k in range(4)])
        fold_mean = np.mean([np.mean([roc_auc_score(y[s:s+4]==k, p[s:s+4,k]) for k in range(4)]) for s in range(0,40,4)])
        self.assertAlmostEqual(result['Label_4class_macro_ovr'], expected)
        self.assertNotAlmostEqual(expected, fold_mean)
        self.assertEqual(result['N'], 40)
        self.assertEqual(d.hid.nunique(), 39)

    def test_configuration_and_posterior_order(self):
        cfg = config()
        self.assertEqual(list(cfg['speech_tasks']), ['yesterday','count1','i3'])
        columns = posterior_columns(cfg)
        self.assertEqual(len(columns), 20)
        self.assertEqual(columns[-4:], ['肺水肿_2_p1','外周水肿_p1','运动耐量下降_p1','LVFP_p1'])
        expected_counts = [117988,18900,28,375524,41173,34050,375266,38002]
        for c, count in zip(cfg['experts'].values(), expected_counts):
            network = expert_model(c, cfg['speech_tasks'])
            self.assertEqual(sum(p.numel() for p in network.parameters()), count)

    def test_weighted_loss_is_finite(self):
        c = config()['experts']['dyspnea']
        y = torch.tensor([0,1,2,3,0,1], dtype=torch.long)
        logits = torch.randn(6,4,requires_grad=True)
        loss = expert_loss(logits,y,c,class_weights(y.numpy(),4,c['class_weight']))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_probability(self):
        for p in [np.array([[.1,.1,.1,.1]]), np.array([[np.nan,0,0,1]]), np.array([[-.1,.1,.4,.6]])]:
            with self.assertRaises(ValueError):
                validate_probabilities(p,4)

    def test_invalid_feature_path_and_order(self):
        cfg = {'input_dimension': 2, 'speech_tasks': {'test': 'test'}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                load_raw(root/'missing', cfg)
            with self.assertRaises(FileNotFoundError):
                load_raw(root, cfg)
            pd.DataFrame({'uid':['sample'], 'index':['1'], 'F1':[1.], 'F0':[0.]}).to_csv(root/'test_hubert.csv',index=False)
            with self.assertRaisesRegex(ValueError, 'numerical order'):
                load_raw(root, cfg)
            pd.DataFrame({'uid':['sample'], 'index':['1'], 'F0':[0.], 'F1':[1.]}).to_csv(root/'test_hubert.csv',index=False)
            self.assertEqual(load_raw(root,cfg)['test'][('sample','1')].tolist(), [[0.,1.]])

    def test_no_matching_speech_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pd.DataFrame({'uid':['wanted'], 'index':['1']}).to_csv(root/'samples.csv', index=False)
            features = pd.DataFrame(np.zeros((1,1024)), columns=[f'F{i}' for i in range(1024)])
            features.insert(0,'index','1')
            features.insert(0,'uid','different')
            features.to_csv(root/'昨日活动_hubert.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'no matching speech'):
                predict(root, root/'samples.csv', root/'unused_weights')


if __name__ == '__main__':
    unittest.main()
