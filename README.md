# Two-stage model for speech-based classification of heart failure worsening grade

This repository provides the core training, inference, and evaluation code for a **two-stage model** that predicts heart failure (HF) worsening grade from speech. The model uses **predicted clinical-subtask probabilities as intermediate representations**, followed by a classifier for the final four-class outcome.

This release contains the fixed R72 configuration, pretrained weights, fitted preprocessing parameters, patient-grouped cross-validation, and full internal-cohort refitting.

## Model overview

The input comprises 1,024-dimensional, segment-level HuBERT features from three speech tasks: describing the previous day's activities, repeated counting of “1”, and three repetitions of the vowel /i/.

```text
Three speech tasks → segment-level HuBERT features
                              ↓
Stage 1: eight independently trained clinical-subtask models
                              ↓
         20-dimensional vector of predicted probabilities
                              ↓
Stage 2: multilayer perceptron (20 → 32 → 4)
                              ↓
                 Four-class HF worsening grade
```

| Stage 1 clinical subtask | Output | Input to stage 2 |
| --- | --- | --- |
| New York Heart Association (NYHA) functional class | Four-class probabilities | All 4 probabilities |
| Dyspnea | Four-class probabilities | All 4 probabilities |
| Rales | Four-class probabilities | All 4 probabilities |
| B-type natriuretic peptide (BNP) category | Four-class probabilities | All 4 probabilities |
| Pulmonary edema | Binary probabilities | Positive-class probability |
| Peripheral edema | Binary probabilities | Positive-class probability |
| Reduced exercise tolerance | Binary probabilities | Positive-class probability |
| Left ventricular filling pressure (LVFP) | Binary probabilities | Positive-class probability |

The probabilities are concatenated in the order shown above. Stage 2 maps this 20-dimensional intermediate representation to four HF worsening grades, stored as `Label3` with values 1–4.

Clinical-subtask labels supervise stage 1 during training. **Inference requires speech features only**, without measured clinical-subtask values or outcome labels. Raw-audio preprocessing and HuBERT feature extraction are outside this release.

## Reference performance

The fixed configuration reproduced the following record-level macro-average one-versus-rest AUCs:

| Evaluation | Records with valid outcome labels | Macro-average AUC |
| --- | --- | --- |
| Internal validation, pooled out-of-fold predictions | 386 | 0.700 |
| External validation | 71 | 0.652 |

The external prediction file contains 105 records; only the 71 records with valid outcome labels contribute to the reported AUC. Internal records represent 288 patients, with repeated visits retained as separate records.

The bundled [reproduction report](docs/reproduction.json) records exact agreement with reference probabilities after full retraining in the reference environment and 99 successful model-reloading checks. These are recorded verification results, rather than a guarantee of identical outputs across environments.

### Evaluation scope

Internal evaluation uses **two-stage out-of-fold (OOF) stacking** with ten patient-grouped folds. For each stage 1 model, eight folds train the model, one selects the stopping epoch, and one provides test predictions. Stage 2 trains on the other nine folds of stage 1 OOF probabilities and predicts the current fold.

This procedure is not fully nested cross-validation of the complete pipeline. Upstream models producing stage 2 training features may have used clinical-subtask labels from the current stage 2 test fold.

For external prediction, stage 1 models are refitted on the internal cohort. The final stage 2 classifier is trained on internal OOF probabilities. External labels do not update parameters or determine early stopping in this fixed implementation. External results were examined during historical model development, so the external cohort should not be described as an untouched, blinded test set.

See [model and evaluation details](docs/methods.md) for preprocessing, refitting rules, and additional evaluation limitations. That document currently retains the original Chinese technical notes.

## Repository contents

| Path | Description |
| --- | --- |
| `run.py` | Command-line entry point |
| `config/model.json` | Fixed architectures, hyperparameters, and speech-task mappings |
| `config/paths.example.json` | Example paths for labels, fold assignments, and features |
| `src/data.py` | Data loading, patient grouping, imputation, scaling, and segment handling |
| `src/models.py` | Stage 1 neural and ordinal model architectures |
| `src/training.py` | Stage 1 training, losses, early stopping, and weight averaging |
| `src/label3.py` | Probability concatenation and stage 2 training |
| `src/inference.py` | Final-model export and prediction on new samples |
| `src/metrics.py` | AUC and classification metrics |
| `src/pipeline.py` | Training workflow and command-line arguments |
| `weights/` | Pretrained weights and fitted preprocessing parameters |
| `tests/test_model.py` | Patient separation, model structure, probability mapping, epoch, and AUC checks |
| `docs/` | Technical methods and recorded reproduction results |

## Installation

The reference environment used Linux, Python 3.10.12, PyTorch 2.5.1 with CUDA 12.1, and two NVIDIA RTX 4090 GPUs. Other dependencies are pinned in `requirements.txt`.

Run all commands from the repository directory containing `run.py`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU inference on Linux or Windows, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

On Windows PowerShell, activate the virtual environment with `.venv\Scripts\Activate.ps1`. Hardware and library changes may produce small numerical differences.

## Input data

Patient-level data and speech features are not included. Reproducing the reference results requires the corresponding cohort data, labels, fold assignments, and feature-extraction protocol.

### Speech features

Feature filenames match `config/model.json`:

| Filename | Speech task |
| --- | --- |
| `昨日活动_hubert.csv` | Previous day's activities |
| `连续数1_hubert.csv` | Repeated counting of “1” |
| `三次i音_hubert.csv` | Three repetitions of /i/ |

Each row represents one segment. Multiple rows may share the same record key, consisting of the string columns `uid` and `index`.

Each file must contain 1,024 feature columns in numerical order, with a consistent `F` or `mean_F` prefix. Accepted ranges are `F0`–`F1023`, `F1`–`F1024`, `mean_F0`–`mean_F1023`, or `mean_F1`–`mean_F1024`. Feature extraction, segmentation, and audio preprocessing must match the original experiment.

### Labels and fold assignments

Training requires two CSV tables:

- **Fold manifest:** `uid,index,hid,hospital,Fold,EvalSet`. Internal records use `EvalSet=InternalCV` and folds 1–10; external records use `EvalSet=External(CD+L)` and fold 0.
- **Label table:** `uid,index` and the columns `NYHA_4`, `呼吸困难_4`, `啰音_4`, `BNP_4`, `肺水肿_2`, `外周水肿`, `运动耐量下降`, `LVFP`, and `Label3`.

Four-class labels use 1–4, binary labels use 0/1, and missing labels use −1. All internal records from the same patient (`hid`) must remain in one fold. Internal and external cohorts must not share patients.

## Train from scratch

Set data locations in `config/paths.example.json`. Relative paths are resolved from the repository root.

```bash
python run.py train --paths config/paths.example.json --output outputs/reproduction --gpus 0 1
```

Use `--gpus 0` for a single GPU. If omitted, the program selects up to two available GPUs, or uses CPU when CUDA is unavailable.

To resume an interrupted run, repeat the command with `--resume`. Code and data fingerprints are checked, completed stage 1 models are skipped, and stage 2 is recomputed. Use a new output directory for a new experiment.

Training saves the following outputs:

- `cv/` and `refit/`: stage 1 probabilities, labels, fold metadata, stopping epochs, and configuration identifiers.
- `stage1_predictions.csv`: combined clinical-subtask predictions.
- `internal_oof_predictions.csv`, `external_predictions.csv`, and `stage2_predictions.csv`: final outcome probabilities and labels.
- `models/`: 80 CV stage 1 models, 8 refitted stage 1 models, and 11 stage 2 classifiers.
- `weights/`: exported final weights and preprocessing parameters for independent inference.
- `metrics.csv`, `refit_epochs.csv`, and `verification.json`: evaluation, refitting, and checkpoint checks.

Run status and training logs are stored under the selected output directory.

## Predict new samples

Provide a sample CSV containing unique `uid,index` keys and a directory of speech features. Labels and access to the original training cohort are not required.

```bash
python run.py predict --samples new_data/samples.csv --features new_data/features --weights weights --output outputs/predictions --device cpu
```

Missing speech tasks are imputed using saved training-set values. Each sample must match at least one observed speech task; samples without any matching speech are rejected.

Prediction produces:

- `stage1_predictions.csv`: clinical-subtask probabilities and predicted labels.
- `stage2_predictions.csv`: four HF worsening-grade probabilities and the predicted grade (`Label3_pred`).

**Probability indexing:** `Label3_p0`–`Label3_p3` correspond to outcome labels **1–4**, respectively. Stage 1 four-class probabilities follow the same indexing; binary `p0` and `p1` correspond to labels 0 and 1.

## Evaluate predictions and run checks

```bash
python run.py evaluate --predictions outputs/reproduction/internal_oof_predictions.csv --cohort InternalOOF
python run.py evaluate --predictions outputs/reproduction/external_predictions.csv --cohort ExternalOnce
python -m unittest discover -s tests -v
```

Supply a prediction file containing only the relevant cohort. Records without valid outcome labels are excluded from scoring. Internal AUC pools OOF predictions across ten folds and averages the four class-specific one-versus-rest AUCs equally.

The tests check the implementation; reproducing study performance additionally requires the original data and reference environment. This release contains the fixed model pipeline, rather than the historical candidate-model search.

## Data, citation, and license

This is research code for the two-stage model described above. Patient-level data, recordings, and generated outputs containing patient records are not distributed here.

Publication metadata, citation instructions, and the license for source code and model weights will be added once finalized by the study team.
