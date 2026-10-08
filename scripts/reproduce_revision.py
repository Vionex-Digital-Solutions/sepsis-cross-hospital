"""Reproduce revision analyses after the original frozen pipeline."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO)
    parser.add_argument("--original-evaluation", type=Path,
                        default=REPO / "runs/reproduction__external")
    parser.add_argument("--output-root", type=Path, default=REPO / "runs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    data = args.data_root.resolve()
    original = args.original_evaluation.resolve()
    output = args.output_root.resolve()
    if not (original / "run_metadata.json").is_file():
        raise FileNotFoundError("Complete the original frozen reproduction first.")
    environment = os.environ.copy()
    environment.update({
        "PYTHONNOUSERSITE": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })

    def run(script, arguments):
        subprocess.run(
            [sys.executable, "-X", "faulthandler", "-u", "-B",
             str(REPO / "scripts" / script), *map(str, arguments)],
            cwd=REPO, env=environment, check=True,
        )

    def created(script, arguments, parent, pattern):
        before = set(parent.glob(pattern))
        run(script, arguments)
        candidates = set(parent.glob(pattern)) - before
        if len(candidates) != 1:
            raise ValueError("Expected exactly one new invocation/result directory.")
        result = candidates.pop()
        completion = json.loads((result / "completion.json").read_text())
        if completion.get("status") != "complete":
            raise ValueError("Stage did not complete.")
        return result

    run("check_environment.py", [])
    subprocess.run(
        [sys.executable, "-X", "faulthandler", "-B", "-m", "pytest", "-q"],
        cwd=REPO, env=environment, check=True,
    )
    common = ["--data-root", data]
    resume = ["--resume"] if args.resume else []
    model_roots = {
        "rf": output / "revision_tree_models/rf",
        "extra_trees": output / "revision_tree_models/extra_trees",
        "mlp": output / "revision_mlp_models",
    }
    training = {}
    for family in ["rf", "extra_trees", "mlp"]:
        script = "train_revision_mlp.py" if family == "mlp" else "train_revision_trees.py"
        family_args = [] if family == "mlp" else ["--family", family]
        training[family] = created(
            script, [*family_args, *common, "--model-root", model_roots[family], *resume],
            model_roots[family] / "training-invocations", "*",
        )
    rf = created(
        "evaluate_rf_ensembles.py",
        [*common, "--model-root", model_roots["rf"], "--output-root", output],
        output, "rf-ensembles-*",
    )
    other = created(
        "evaluate_revision_ensembles.py",
        [*common, "--extra-trees-completion",
         training["extra_trees"] / "completion.json",
         "--mlp-completion", training["mlp"] / "completion.json",
         "--output-root", output],
        output, "revision-ensembles-*",
    )
    seeds = created(
        "evaluate_rf_seed_stability.py",
        [*common, "--model-root", model_roots["rf"],
         "--ensemble-protocol", rf / "frozen_protocol.json",
         "--output-root", output],
        output, "rf-seed-stability-*",
    )
    summary = created(
        "summarize_revision_results.py",
        ["--original-evaluation", original, "--rf-evaluation", rf,
         "--extra-mlp-evaluation", other, "--output-root", output],
        output, "revision-summary-*",
    )
    run("verify_revision_results.py", [
        "--rf-evaluation", rf, "--extra-mlp-evaluation", other,
        "--rf-seed-evaluation", seeds, "--summary", summary,
    ])
    print("REVISION REPRODUCTION: PASS", flush=True)

if __name__ == "__main__":
    main()
