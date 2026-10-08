# Revision robustness analyses

These are additional exploratory revision analyses of the original two
PhysioNet Challenge 2019 hospital systems. They add classifiers, not hospitals.
They were not part of an originally preregistered protocol.

## Reproduce

Complete the original pipeline in the main README first. It prepares the
processed data and runs/reproduction__external.

Then, from the repository root in the locked, activated Conda environment:

    python scripts/reproduce_revision.py --data-root . --original-evaluation runs/reproduction__external

The revision driver trains all 90 RF, ExtraTrees and MLP models; evaluates
their five-seed probability ensembles; evaluates individual RF seeds; builds
the combined tables; and verifies the numerical references. Use --resume to
verify and reuse matching model artifacts. It does not overwrite models.

The recorded reference environment includes the RTX 3050 Laptop GPU.
Numerical verification uses absolute tolerance 1e-10 and zero relative tolerance.
A successful fresh reproduction ends with REVISION REPRODUCTION: PASS.

## Fixed modeling and evaluation choices

- Both transfer directions; V0=40, V1=74 and V2=142 features.
- Seeds: 1729, 2718, 31415, 57721 and 65537.
- RF/ExtraTrees: 200 trees per seed, depth 12, minimum leaf size 20,
  square-root feature sampling, no class weighting; bootstrap sampling for
  RF, without bootstrap sampling for ExtraTrees.
- MLP: two ReLU hidden layers of 64 and 32 units. Source-training-only
  RobustScaler with quartiles 25/75 and clipping to [-10,10].
  Unweighted BCE, Adam learning rate 0.001, batches of 2048 hours, maximum
  60 epochs and patience 8. Retain minimum source-validation BCE, earliest
  epoch on ties. The complete settings are in the revision JSON configurations.
- Average five seed probabilities equally before selecting the ensemble threshold.
- Each evaluator freezes source-validation thresholds for both directions
  before reading its held-out partitions. Thresholds transfer unchanged.
- Calibration intercept/slope are diagnostics; they never modify predictions.
- Original early SepsisLabel values are not shifted again.
- Patient bootstrap: 1000 paired replicates, seed 1729, percentile intervals,
  without multiplicity adjustment or BCa correction.
- Bootstrap intervals condition on the trained models. RF seed summaries
  describe training/threshold variability; their sample SD is not a confidence interval.

## Interpretation

V2 improves A-to-B utility relative to V0 for XGBoost, RF, ExtraTrees and MLP.
Observation-mask effects depend on classifier and direction.

The direct V2-minus-V1 contrasts are exploratory follow-ups calculated after
reviewing the revision results, using the existing paired bootstrap draws.
For B-to-A this contrast is negative for XGBoost, RF and MLP, and positive for
ExtraTrees; all four intervals exclude zero. For A-to-B all four point
estimates are positive, but the XGBoost interval includes zero.

This supports classifier- and direction-dependent utility effects.
It does not establish universal superiority of one feature representation.
The individual RF seed results also include mixed signs.

## Reference outputs and provenance

expected_outputs/revision contains aggregate metrics, feature contrasts,
RF seed summaries, frozen evaluation protocols and a checksum manifest.
external_comparison.csv contains 32 external configurations across both directions.
feature_contrasts.csv contains 24 contrasts, including the eight explicitly
marked exploratory V2-minus-V1 follow-ups.

The portable RF evaluator retains compatibility with the original pilot's
missing training-hash field only when that exact artifact matches the recorded
reference fingerprint. Fresh public RF training records the training hash for
every seed.

Generated data, model artifacts and patient-level predictions stay in ignored
local directories. No patient-level data are redistributed.

Verification of existing completed outputs:

    python scripts/verify_revision_results.py --rf-evaluation PATH --extra-mlp-evaluation PATH --rf-seed-evaluation PATH --summary PATH

Independent hospital-system and prospective validation remain future work.
