import sys
from pathlib import Path

import pandas as pd

from src.data.loader import load_dataset
from src.pipeline.run_pipeline import data_pipeline
from src.pipeline.simulate import make_cutoffs

ROOT = Path(__file__).resolve().parents[2]


def main(name="household", n_runs=4):
    full = pd.concat([load_dataset(name, s) for s in ("train", "val", "test")])
    for i, cutoff in enumerate(make_cutoffs(full, n_runs), start=1):
        out = ROOT / "data" / "processed" / "runs" / f"run_{i}"
        print(f"=== Run {i}/{n_runs} : données jusqu'au {cutoff} ===")
        data_pipeline(datasets=[name], cutoff_date=str(cutoff), out_dir=str(out))


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "household"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(name, n_runs)