from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from sepsis_cross_hospital.data.splitting import (
    DEFAULT_SEED,
    SPLIT_VERSION,
    assign_stratified_splits,
    canonical_manifest_bytes,
    manifest_sha256,
)

FROZEN_SPLIT_SHA256 = (
    "1883b58134c36ecba43c6228aa4712d3"
    "cdc94b47ca3ad86507834c826d1a4c08"
)

FROZEN_SPLIT_COUNTS = {
    "A": {
        "train": 14235,
        "validation": 3049,
        "internal_test": 3052,
    },
    "B": {
        "train": 13999,
        "validation": 2999,
        "internal_test": 3002,
    },
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic patient-level "
            "70/15/15 source-hospital splits."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/paths.local.yaml"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def load_paths(
    config_path: Path,
) -> dict[str, Path]:
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle) or {}

    paths = {
        "A": Path(
            config["training_set_a"]
        ).expanduser().resolve(),
        "B": Path(
            config["training_set_b"]
        ).expanduser().resolve(),
    }

    for cohort, path in paths.items():
        if not path.is_dir():
            raise FileNotFoundError(
                f"Cohort {cohort}: {path}"
            )

    return paths


def read_patient_labels(
    cohort: str,
    directory: Path,
) -> pd.DataFrame:
    records: list[dict] = []

    files = sorted(
        directory.glob("*.psv")
    )

    print()
    print(
        f"Cohort {cohort}: "
        f"reading {len(files):,} patient labels"
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        labels = pd.read_csv(
            path,
            sep="|",
            usecols=["SepsisLabel"],
        )["SepsisLabel"]

        if labels.isna().any():
            raise ValueError(
                f"{path.name}: missing SepsisLabel"
            )

        if not labels.isin([0, 1]).all():
            raise ValueError(
                f"{path.name}: invalid SepsisLabel"
            )

        records.append(
            {
                "patient_id": path.stem,
                "septic": int(
                    (labels == 1).any()
                ),
            }
        )

        if (
            index == 1
            or index % 2000 == 0
            or index == len(files)
        ):
            print(
                f"  processed "
                f"{index:,}/{len(files):,}"
            )

    return pd.DataFrame(records)


def build_summary(
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cohort in ("A", "B"):
        cohort_frame = manifest.loc[
            manifest["cohort"] == cohort
        ]

        for split in (
            "train",
            "validation",
            "internal_test",
        ):
            group = cohort_frame.loc[
                cohort_frame["split"] == split
            ]

            n_patients = len(group)
            septic = int(
                group["septic"].sum()
            )

            prevalence = (
                septic / n_patients
                if n_patients
                else float("nan")
            )

            rows.append(
                {
                    "cohort": cohort,
                    "split": split,
                    "patients": n_patients,
                    "septic_patients": septic,
                    "nonseptic_patients": (
                        n_patients - septic
                    ),
                    "sepsis_prevalence": prevalence,
                }
            )

    return pd.DataFrame(rows)


def load_dataset_hashes() -> dict[str, str]:
    manifest_path = Path(
        "data/metadata/dataset_manifest.csv"
    )

    if not manifest_path.exists():
        return {}

    dataset_manifest = pd.read_csv(
        manifest_path
    )

    hashes = {}

    for _, row in dataset_manifest.iterrows():
        hashes[str(row["cohort"])] = str(
            row["dataset_tree_sha256"]
        )

    return hashes

def verify_frozen_split(
    manifest: pd.DataFrame,
    split_hash: str,
    seed: int,
) -> list[str]:
    """
    Verify that the generated patient-level split
    is exactly the frozen study split.
    """
    failures: list[str] = []

    print()
    print("=" * 72)
    print("FROZEN SPLIT VERIFICATION")
    print("=" * 72)

    seed_ok = (
        seed == DEFAULT_SEED
    )

    print(
        f"Seed                  : "
        f"{'PASS' if seed_ok else 'FAIL'}"
    )

    if not seed_ok:
        failures.append(
            f"Split seed: expected "
            f"{DEFAULT_SEED}, found {seed}."
        )

    hash_ok = (
        split_hash
        == FROZEN_SPLIT_SHA256
    )

    print(
        f"Manifest SHA256       : "
        f"{'PASS' if hash_ok else 'FAIL'}"
    )

    if not hash_ok:
        failures.append(
            "Split manifest SHA256: "
            f"expected {FROZEN_SPLIT_SHA256}, "
            f"found {split_hash}."
        )

    for cohort in (
        "A",
        "B",
    ):
        print()
        print(
            f"Cohort {cohort}"
        )
        print("-" * 40)

        cohort_frame = (
            manifest.loc[
                manifest["cohort"]
                == cohort
            ]
        )

        for split in (
            "train",
            "validation",
            "internal_test",
        ):
            actual = int(
                (
                    cohort_frame[
                        "split"
                    ]
                    == split
                ).sum()
            )

            expected = (
                FROZEN_SPLIT_COUNTS[
                    cohort
                ][
                    split
                ]
            )

            passed = (
                actual == expected
            )

            print(
                f"{split:<22}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

            if not passed:
                failures.append(
                    f"Cohort {cohort} "
                    f"{split}: expected "
                    f"{expected}, found "
                    f"{actual}."
                )

    print()
    print("=" * 72)

    if failures:
        print(
            "Frozen split verification: FAILED"
        )
    else:
        print(
            "Frozen split verification: PASS"
        )

    print("=" * 72)

    return failures

def main() -> int:
    args = parse_args()

    print("=" * 72)
    print(
        "DETERMINISTIC PATIENT-LEVEL SPLIT CREATION"
    )
    print("=" * 72)

    print(
        f"Split version : {SPLIT_VERSION}"
    )
    print(
        f"Seed          : {args.seed}"
    )

    paths = load_paths(
        args.config
    )

    manifests = []

    for cohort in ("A", "B"):
        patients = read_patient_labels(
            cohort=cohort,
            directory=paths[cohort],
        )

        manifest = assign_stratified_splits(
            patients=patients,
            cohort=cohort,
            seed=args.seed,
        )

        manifests.append(manifest)

    combined = pd.concat(
        manifests,
        ignore_index=True,
    ).sort_values(
        ["cohort", "patient_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Cross-cohort safety check
    # --------------------------------------------------------
    ids_a = set(
        combined.loc[
            combined["cohort"] == "A",
            "patient_id",
        ]
    )

    ids_b = set(
        combined.loc[
            combined["cohort"] == "B",
            "patient_id",
        ]
    )

    overlap = ids_a & ids_b

    if overlap:
        raise RuntimeError(
            "Patient IDs overlap across cohorts: "
            f"{sorted(overlap)[:10]}"
        )

    # --------------------------------------------------------
    # Local patient-level manifest
    # --------------------------------------------------------
    local_dir = Path(
        "data/interim/splits"
    )

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_manifest_path = (
        local_dir
        / "split_manifest.csv"
    )

    local_manifest_path.write_bytes(
        canonical_manifest_bytes(combined)
    )

    split_hash = manifest_sha256(
        combined
    )

    # --------------------------------------------------------
    # Aggregate Git-safe outputs
    # --------------------------------------------------------
    metadata_dir = Path(
        "data/metadata/splits"
    )

    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = build_summary(
        combined
    )

    summary.to_csv(
        metadata_dir
        / "split_summary.csv",
        index=False,
    )

    dataset_hashes = (
        load_dataset_hashes()
    )

    metadata = {
        "split_version": SPLIT_VERSION,
        "seed": args.seed,
        "algorithm": (
            "SHA256 deterministic stratified "
            "patient ordering"
        ),
        "fractions": {
            "train": 0.70,
            "validation": 0.15,
            "internal_test": 0.15,
        },
        "manifest_sha256": split_hash,
        "patients_total": int(
            len(combined)
        ),
        "cohort_A_patients": int(
            (combined["cohort"] == "A").sum()
        ),
        "cohort_B_patients": int(
            (combined["cohort"] == "B").sum()
        ),
        "dataset_tree_sha256": (
            dataset_hashes
        ),
        "python_version": (
            sys.version.split()[0]
        ),
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with (
        metadata_dir
        / "split_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print("=" * 72)
    print("SPLIT SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Manifest SHA256:"
    )
    print(
        f"  {split_hash}"
    )

    print()
    print(
        "Patient-level manifest:"
    )
    print(
        "  data/interim/splits/"
        "split_manifest.csv"
    )

    print(
        "Aggregate metadata:"
    )
    print(
        "  data/metadata/splits/"
    )

    failures = verify_frozen_split(
        manifest=combined,
        split_hash=split_hash,
        seed=args.seed,
    )

    if failures:
        print()

        for failure in failures:
            print(
                f"FAIL: {failure}"
            )

        return 1


    return 0


if __name__ == "__main__":
    raise SystemExit(main())