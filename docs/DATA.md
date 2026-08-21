# Data Acquisition

This repository does not redistribute the patient-level data from the
PhysioNet/Computing in Cardiology Challenge 2019.

The study uses version 1.0.0 of:

**Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019**

PhysioNet DOI: `10.13026/v64v-d857`

## Required cohorts

This study uses the two publicly available hospital systems:

- `training_setA`: 20,336 patients
- `training_setB`: 20,000 patients

Each patient is represented by one pipe-delimited `.psv` file containing
hourly clinical measurements and the supplied `SepsisLabel`.

The Challenge labels already encode the six-hour early-prediction target.
For septic patients, `SepsisLabel` becomes 1 from six hours before the
derived sepsis onset. This repository therefore does not shift the labels
again.

## Download

Download version `1.0.0` directly from PhysioNet.

The dataset is publicly accessible from the PhysioNet resource identified
by DOI:

`10.13026/v64v-d857`

PhysioNet also provides anonymous AWS S3 access. If the AWS CLI is
installed, the complete version 1.0.0 resource can be downloaded with:

```bash
aws s3 sync --no-sign-request \
  s3://physionet-open/challenge-2019/1.0.0/ \
  /path/to/physionet-challenge-2019/
```

Only the public training cohorts are required for this study.

## Expected directory layout

After downloading or extracting the training data, the required layout is:

```text
physionet-challenge-2019/
+-- training/
    +-- training_setA/
    |   +-- *.psv
    +-- training_setB/
        +-- *.psv
```

The dataset does not need to be placed inside this Git repository.

## Configure local paths

Copy:

```text
configs/paths.example.yaml
```

to:

```text
configs/paths.local.yaml
```

Then edit `configs/paths.local.yaml` to point to the downloaded cohorts.

Example:

```yaml
training_set_a: "C:/datasets/physionet-challenge-2019/training/training_setA"
training_set_b: "C:/datasets/physionet-challenge-2019/training/training_setB"
```

`configs/paths.local.yaml` is excluded from version control.

## Frozen dataset checks

Before preprocessing or model training, the reproduction pipeline verifies
the input dataset structure, schema, and cohort inventory.

Expected patient counts:

| Cohort | Patients |
| --- | ---: |
| System A | 20,336 |
| System B | 20,000 |

The dataset snapshot used for the reported experiments had the following
aggregate counts:

| Cohort | Patient-hours | Sepsis-positive patients |
| --- | ---: | ---: |
| System A | 790,215 | 1,790 |
| System B | 761,995 | 1,142 |

These aggregate values are included as reproducibility checks for the
dataset snapshot used in the study.

The complete cohort trees are also verified using deterministic SHA-256
digests over the sorted `.psv` files:

| Cohort | Frozen tree SHA-256 |
| --- | --- |
| System A | `ad23f06c4580a873871f08f635a8f02a52e538089b1dc352f576be41d949fc70` |
| System B | `6ae3634d90dd3003e5c3c761d24b75afe11d3741e7803ffd1ebfe6b886869da1` |

These hashes identify the exact public dataset snapshot used for the
reported experiments.

The pipeline also verifies the expected input schema before generating
patient-level splits or processed model inputs.

## Data-use note

Patient-level source data are intentionally excluded from this repository.

Users must obtain the official PhysioNet release independently and comply
with the terms associated with that resource.

## Dataset citation

When using the dataset, cite:

Matthew Reyna, Chris Josef, Russell Jeter, Supreeth Shashikumar,
Benjamin Moody, M. Brandon Westover, Ashish Sharma, Shamim Nemati,
and Gari D. Clifford.

*Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing
in Cardiology Challenge 2019.*

PhysioNet, Version 1.0.0, 2019.

DOI: `10.13026/v64v-d857`