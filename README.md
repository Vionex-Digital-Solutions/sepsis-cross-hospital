# Cross-Hospital Early Sepsis Prediction

Reproducibility repository for a bidirectional cross-hospital study of early sepsis prediction using the public PhysioNet/Computing in Cardiology Challenge 2019 dataset.

The study evaluates direct transfer between the two public hospital systems:

- **A -> B:** train, tune, and select using System A only; evaluate unchanged on System B.

- **B -> A:** train, tune, and select using System B only; evaluate unchanged on System A.

Target-hospital data are not used for preprocessing statistics, feature selection, hyperparameter selection, calibration, threshold selection, or model selection.

## Revision analyses

Additional RF, ExtraTrees and MLP reproduction, reference outputs and interpretation are documented in [docs/REVISION.md](docs/REVISION.md).

## Dataset

The study uses:

**Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019**

- Version: `1.0.0`

- DOI: `10.13026/v64v-d857`

- System A: 20,336 patients

- System B: 20,000 patients

Patient-level source data are not redistributed in this repository.

See [`docs/DATA.md`](docs/DATA.md) for download instructions, expected directory structure, cohort inventory, and frozen dataset hashes.

## Models

The frozen study includes:

- Logistic regression using V0 features

- XGBoost M0

- XGBoost V0

- XGBoost V1

- XGBoost V2

- Causal GRU

- Process-Attenuated Ensemble (PAE)

Feature representations are:

- **M0:** context + current observation masks + time-since-last measurement + six-hour measurement counts

- **V0:** physiological values + context

- **V1:** V0 + current observation masks

- **V2:** V0 + current masks + time-since-last measurement + six-hour counts

The complete frozen protocol is recorded in:

```text

configs/frozen_protocol.yaml

```

## Reproduction environment

The validated reference environment uses:

- Python `3.11.15`

- PyTorch `2.13.0+cu126`

- XGBoost `3.2.0`

- pyarrow `22.0.0`

The exact package versions are pinned in:

```text

environment.yml

```

A CUDA-capable NVIDIA GPU is required for the configured XGBoost and GRU runs.

The repository was validated with an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB of GPU memory.

### 1. Clone the repository

```bash

git clone https://github.com/Vionex-Digital-Solutions/sepsis-cross-hospital.git

cd sepsis-cross-hospital

```

### 2. Create the locked environment

```bash

conda env create -f environment.yml

conda activate sepsis-cross-hospital

```

Install the repository itself in editable mode without changing the locked dependency set:

```bash

python -m pip install -e . --no-deps --no-build-isolation

```

### 3. Download the public PhysioNet data

Follow:

```text

docs/DATA.md

```

Then copy:

```text

configs/paths.example.yaml

```

to:

```text

configs/paths.local.yaml

```

and set:

```yaml

training_set_a: "C:/path/to/training_setA"

training_set_b: "C:/path/to/training_setB"

```

`configs/paths.local.yaml` is ignored by Git.

### 4. Check the software and CUDA environment

```bash

python scripts/check_environment.py

```

A successful check ends with:

```text

Environment check: PASS

```

### 5. Run the complete frozen reproduction

From a fresh repository state with no previous frozen run directories:

```bash

python scripts/reproduce.py

```

or, when using a different local configuration path:

```bash

python scripts/reproduce.py --config path/to/paths.local.yaml

```

The driver performs, in order:

1\. locked environment verification;

2\. automated tests;

3\. raw PhysioNet dataset verification;

4\. deterministic patient-level splitting;

5\. causal preprocessing for both transfer directions;

6\. processed-data hash validation;

7\. all source-side Logistic, XGBoost, GRU, and PAE training runs;

8\. frozen external-evaluation preflight;

9\. atomic external evaluation for both directions;

10\. comparison against the frozen expected outputs.

A successful complete reproduction ends with:

```text

FULL REPRODUCTION: PASS

```

Model training and the 1,000-replicate paired patient bootstrap can take substantial time.

## Manual pipeline

The individual stages can also be run separately.

### Verify the raw public dataset

```bash

python scripts/verify_data.py --config configs/paths.local.yaml

```

### Reproduce the frozen patient split

```bash

python scripts/create_splits.py --config configs/paths.local.yaml --seed 1729

```

### Reproduce preprocessing

```bash

python scripts/preprocess.py --config configs/paths.local.yaml --direction all --batch-patients 250 --overwrite

```

### Validate processed artifacts

```bash

python scripts/validate_processed.py

```

### Train Logistic baselines

```bash

python scripts/train_logistic.py --direction A_to_B --run-id A_to_B__logistic_v0

python scripts/train_logistic.py --direction B_to_A --run-id B_to_A__logistic_v0

```

### Train XGBoost models

For A -> B:

```bash

python scripts/train_xgboost.py --direction A_to_B --representation M0 --run-id A_to_B__xgb_m0

python scripts/train_xgboost.py --direction A_to_B --representation V0 --run-id A_to_B__xgb_v0

python scripts/train_xgboost.py --direction A_to_B --representation V1 --run-id A_to_B__xgb_v1

python scripts/train_xgboost.py --direction A_to_B --representation V2 --run-id A_to_B__xgb_v2

```

For B -> A:

```bash

python scripts/train_xgboost.py --direction B_to_A --representation M0 --run-id B_to_A__xgb_m0

python scripts/train_xgboost.py --direction B_to_A --representation V0 --run-id B_to_A__xgb_v0

python scripts/train_xgboost.py --direction B_to_A --representation V1 --run-id B_to_A__xgb_v1

python scripts/train_xgboost.py --direction B_to_A --representation V2 --run-id B_to_A__xgb_v2

```

### Train causal GRU models

```bash

python scripts/train_gru.py --direction A_to_B --run-id A_to_B__gru

python scripts/train_gru.py --direction B_to_A --run-id B_to_A__gru

```

### Train PAE models

```bash

python scripts/train_pae.py --direction A_to_B --run-id A_to_B__pae

python scripts/train_pae.py --direction B_to_A --run-id B_to_A__pae

```

### Verify source artifacts before external evaluation

```bash

python scripts/evaluate_external.py --preflight

```

This preflight verifies the frozen external Parquet hashes, all 14 source-run contracts, and required model artifacts without calculating external performance.

### Run atomic external evaluation

```bash

python scripts/evaluate_external.py --run-id reproduction__external

```

### Verify the reproduced results

```bash

python scripts/verify_results.py

```

A successful result check ends with:

```text

ALL FROZEN RESULTS: PASS

```

## Frozen reference outputs

Machine-readable expected outputs are stored in:

```text

expected_outputs/internal_metrics.csv

expected_outputs/external_metrics.csv

expected_outputs/bootstrap_summary.csv

expected_outputs/frozen_data_hashes.json

```

These references cover:

- source-internal model metrics;

- external Challenge utility;

- AUROC;

- AUPRC;

- Brier score;

- calibration intercept and slope;

- 1,000-replicate patient-bootstrap utility confidence intervals;

- paired utility contrasts;

- raw-data hashes;

- split hash;

- processed-data hashes.

## Evaluation

Thresholds are selected using source-validation Challenge utility only and are transferred unchanged to the external hospital.

The threshold grid is:

```text

0.001, 0.002, ..., 0.999

```

with higher threshold used as the tie-break.

External evaluation reports:

- PhysioNet/CinC Challenge normalized utility

- AUROC

- AUPRC

- Brier score

- sensitivity

- specificity

- PPV

- NPV

- calibration intercept

- calibration slope

Calibration statistics are evaluation-only. Predictions are not recalibrated using target-hospital data.

Uncertainty is estimated with 1,000 paired patient-level bootstrap replicates using seed `1729`.

The frozen paired utility contrasts are:

- PAE - XGB V2

- XGB V1 - XGB V0

- XGB V2 - XGB V0

No BCa correction or multiplicity adjustment is applied.

## Testing

Run:

```bash

python -m pytest -q

```

The automated tests cover splitting, labels, leakage safeguards, preprocessing, Challenge utility, thresholding, GRU behavior, PAE behavior, and external evaluation.

## Reproducibility safeguards

The public pipeline includes explicit checks for:

- exact public dataset inventory;

- raw cohort SHA-256 tree hashes;

- deterministic patient-level split SHA-256;

- source/target patient separation;

- causal forward filling;

- source-training-only preprocessing statistics;

- source-unavailable measurement channels;

- exact processed Parquet hashes;

- frozen model configurations and seeds;

- source-only model and threshold selection;

- locked external cohort hashes;

- paired patient bootstrap behavior;

- frozen numerical result reproduction.

The supplied `SepsisLabel` already represents the Challenge's early prediction target and is therefore not shifted a second time.

## Repository structure

```text

configs/

    frozen_protocol.yaml

    paths.example.yaml

docs/

    DATA.md

expected_outputs/

    bootstrap_summary.csv

    external_metrics.csv

    frozen_data_hashes.json

    internal_metrics.csv

scripts/

    check_environment.py

    create_splits.py

    evaluate_external.py

    preprocess.py

    reproduce.py

    train_gru.py

    train_logistic.py

    train_pae.py

    train_xgboost.py

    validate_processed.py

    verify_data.py

    verify_results.py

src/sepsis_cross_hospital/

    data/

    evaluation/

    models/

tests/

```

Generated data, model checkpoints, run artifacts, local path configuration, and raw patient-level files are intentionally excluded from version control.

## External-transfer findings

The original XGBoost results show direction-dependent utility effects.
The additional RF, ExtraTrees and MLP analyses show that effects also depend
on the classifier. The direct timing/count comparison is V2 minus V1; those
follow-up intervals and their exploratory status are documented in
[docs/REVISION.md](docs/REVISION.md).

PAE is an exploratory mitigation strategy and does not consistently outperform V1.

## Data and privacy

Only the official public PhysioNet Challenge 2019 data are used.

No patient-level data are included in this repository.
