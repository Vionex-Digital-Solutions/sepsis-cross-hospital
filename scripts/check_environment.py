from __future__ import annotations

import os
import platform
import site
import sys
from importlib import metadata
from pathlib import Path

import numpy as np


REFERENCE_PYTHON = "3.11.15"

REFERENCE_PACKAGES = {
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "scipy": "1.17.1",
    "scikit-learn": "1.9.0",
    "joblib": "1.5.3",
    "matplotlib": "3.11.1",
    "pyarrow": "22.0.0",
    "pyyaml": "6.0.3",
    "pytest": "9.1.1",
    "torch": "2.13.0+cu126",
    "xgboost": "3.2.0",
    "pip": "26.2.1",
    "setuptools": "84.0.0",
}

REFERENCE_TORCH_CUDA = "12.6"


def package_version(
    package_name: str,
) -> str | None:
    try:
        return metadata.version(
            package_name
        )
    except metadata.PackageNotFoundError:
        return None


def print_header(
    title: str,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    failures: list[str] = []

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    print_header(
        "SEPSIS CROSS-HOSPITAL REFERENCE ENVIRONMENT CHECK"
    )

    python_version = (
        platform.python_version()
    )

    print(
        f"Python version       : "
        f"{python_version}"
    )
    print(
        f"Reference version    : "
        f"{REFERENCE_PYTHON}"
    )
    print(
        f"Python executable    : "
        f"{sys.executable}"
    )
    print(
        f"Platform             : "
        f"{platform.platform()}"
    )
    print(
        f"Machine              : "
        f"{platform.machine()}"
    )
    print(
        f"Conda environment    : "
        f"{os.environ.get('CONDA_DEFAULT_ENV', 'Not detected')}"
    )
    print(
        f"Conda prefix         : "
        f"{os.environ.get('CONDA_PREFIX', 'Not detected')}"
    )

    if python_version != REFERENCE_PYTHON:
        failures.append(
            "Python version mismatch: "
            f"expected {REFERENCE_PYTHON}, "
            f"found {python_version}."
        )

    # ======================================================
    # User-site isolation
    # ======================================================

    print_header(
        "PYTHON PACKAGE ISOLATION"
    )

    python_no_user_site = (
        os.environ.get(
            "PYTHONNOUSERSITE"
        )
    )

    print(
        f"PYTHONNOUSERSITE     : "
        f"{python_no_user_site or 'Not set'}"
    )
    print(
        f"User site enabled    : "
        f"{site.ENABLE_USER_SITE}"
    )

    if site.ENABLE_USER_SITE is False:
        print(
            "User-site isolation  : PASS"
        )
    else:
        print(
            "User-site isolation  : FAIL"
        )
        failures.append(
            "Python user-site packages are enabled."
        )

    # ======================================================
    # Exact package versions
    # ======================================================

    print_header(
        "LOCKED PACKAGE VERSIONS"
    )

    for package, expected in (
        REFERENCE_PACKAGES.items()
    ):
        actual = package_version(
            package
        )

        if actual is None:
            status = "MISSING"
            failures.append(
                f"{package} is not installed."
            )
        elif actual != expected:
            status = "MISMATCH"
            failures.append(
                f"{package} version mismatch: "
                f"expected {expected}, found {actual}."
            )
        else:
            status = "PASS"

        print(
            f"{package:<20} "
            f"expected={expected:<15} "
            f"actual={str(actual):<15} "
            f"{status}"
        )

    # ======================================================
    # PyTorch CUDA
    # ======================================================

    print_header(
        "PYTORCH CUDA"
    )

    try:
        import torch

        print(
            f"PyTorch version      : "
            f"{torch.__version__}"
        )
        print(
            f"CUDA build           : "
            f"{torch.version.cuda}"
        )
        print(
            f"CUDA available       : "
            f"{torch.cuda.is_available()}"
        )

        if (
            str(torch.version.cuda)
            != REFERENCE_TORCH_CUDA
        ):
            failures.append(
                "PyTorch CUDA build mismatch: "
                f"expected {REFERENCE_TORCH_CUDA}, "
                f"found {torch.version.cuda}."
            )

        if not torch.cuda.is_available():
            failures.append(
                "PyTorch cannot access a CUDA GPU."
            )
        else:
            gpu_name = (
                torch.cuda.get_device_name(0)
            )
            capability = (
                torch.cuda.get_device_capability(
                    0
                )
            )

            properties = (
                torch.cuda
                .get_device_properties(
                    0
                )
            )

            memory_gb = (
                properties.total_memory
                / (1024**3)
            )

            print(
                f"GPU                   : "
                f"{gpu_name}"
            )
            print(
                f"Compute capability    : "
                f"{capability[0]}."
                f"{capability[1]}"
            )
            print(
                f"GPU memory            : "
                f"{memory_gb:.2f} GB"
            )

            try:
                left = torch.tensor(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                    ],
                    device="cuda",
                )

                right = torch.tensor(
                    [
                        [5.0, 6.0],
                        [7.0, 8.0],
                    ],
                    device="cuda",
                )

                result = left @ right

                torch.cuda.synchronize()

                if not torch.isfinite(
                    result
                ).all():
                    raise RuntimeError(
                        "Non-finite CUDA result."
                    )

                print(
                    "PyTorch CUDA test     : PASS"
                )

            except Exception as exc:
                print(
                    "PyTorch CUDA test     : FAIL"
                )
                print(
                    f"Error                 : "
                    f"{exc}"
                )
                failures.append(
                    "PyTorch CUDA execution failed."
                )

    except Exception as exc:
        print(
            "PyTorch import/test   : FAIL"
        )
        print(
            f"Error                 : "
            f"{exc}"
        )
        failures.append(
            "PyTorch could not be imported "
            "or tested."
        )

    # ======================================================
    # XGBoost CUDA
    # ======================================================

    print_header(
        "XGBOOST CUDA"
    )

    try:
        import xgboost as xgb

        build_info = (
            xgb.build_info()
        )

        print(
            f"XGBoost version      : "
            f"{xgb.__version__}"
        )
        print(
            f"USE_CUDA             : "
            f"{build_info.get('USE_CUDA')}"
        )
        print(
            f"CUDA build           : "
            f"{build_info.get('CUDA_VERSION')}"
        )

        if not bool(
            build_info.get(
                "USE_CUDA",
                False,
            )
        ):
            failures.append(
                "XGBoost was built without CUDA support."
            )

        try:
            x = np.array(
                [
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            )

            y = np.array(
                [
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            )

            matrix = xgb.QuantileDMatrix(
                x,
                y,
            )

            model = xgb.train(
                {
                    "objective": (
                        "binary:logistic"
                    ),
                    "tree_method": "hist",
                    "device": "cuda",
                    "max_depth": 1,
                    "eta": 0.5,
                    "verbosity": 0,
                },
                matrix,
                num_boost_round=2,
            )

            predictions = (
                model.predict(
                    matrix
                )
            )

            if not np.isfinite(
                predictions
            ).all():
                raise RuntimeError(
                    "Non-finite XGBoost "
                    "predictions."
                )

            print(
                "XGBoost CUDA test     : PASS"
            )

        except Exception as exc:
            print(
                "XGBoost CUDA test     : FAIL"
            )
            print(
                f"Error                 : "
                f"{exc}"
            )
            failures.append(
                "XGBoost CUDA execution failed."
            )

    except Exception as exc:
        print(
            "XGBoost import/test   : FAIL"
        )
        print(
            f"Error                 : "
            f"{exc}"
        )
        failures.append(
            "XGBoost could not be imported "
            "or tested."
        )

    # ======================================================
    # Project import
    # ======================================================

    print_header(
        "PROJECT PACKAGE"
    )

    try:
        import sepsis_cross_hospital

        package_path = Path(
            sepsis_cross_hospital.__file__
        ).resolve()

        print(
            f"Package location     : "
            f"{package_path}"
        )

        try:
            package_path.relative_to(
                repo_root
            )
            from_current_repo = True
        except ValueError:
            from_current_repo = False

        if from_current_repo:
            print(
                "Current repository    : PASS"
            )
        else:
            print(
                "Current repository    : FAIL"
            )
            failures.append(
                "sepsis_cross_hospital was imported "
                "from another repository or installation."
            )

    except Exception as exc:
        print(
            "Project import        : FAIL"
        )
        print(
            f"Error                 : "
            f"{exc}"
        )
        failures.append(
            "Project package could not be imported. "
            "Install this repository with "
            "'python -m pip install -e .'."
        )

    # ======================================================
    # Result
    # ======================================================

    print_header(
        "RESULT"
    )

    if failures:
        for failure in failures:
            print(
                f"FAIL: {failure}"
            )

        print()
        print(
            "Environment check: FAILED"
        )

        return 1

    print(
        "Environment check: PASS"
    )
    print()
    print(
        "The locked Python/package versions, "
        "CUDA execution, and current project "
        "installation match the validated "
        "reference environment."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )