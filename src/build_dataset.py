"""
Build a combined feature dataset from processed .Q00 files.

Material groups:
- BI: bismuth films
- CR: chromium films
- FE: iron films

Reference group:
- XXX: modal reference film used to identify composition-independent noise
"""

from pathlib import Path

import numpy as np
import pandas as pd

from signal_processing import process_file


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "q00"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
DATASET_PATH = BASE_DIR / "data" / "processed" / "conductivity_features.csv"

SAMPLE_LENGTH = 128
OVERLAP_MODE = 2  # 50% overlap

FILE_NAMES = [
    "BI202C01",
    "BI304C01",
    "CR202C01",
    "CR304C01",
    "FE104C01",
    "FE202C01",
    "XXX01C01",
    "XXX01C03",
    "XXX01C05",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []

    for file_stem in FILE_NAMES:
        file_path = INPUT_DIR / f"{file_stem}.Q00"

        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        df = process_file(
            input_dir=INPUT_DIR,
            file_stem=file_stem,
            output_dir=OUTPUT_DIR,
            sample_length=SAMPLE_LENGTH,
            overlap_mode=OVERLAP_MODE,
        )

        all_frames.append(df)

    if not all_frames:
        raise RuntimeError("No files were processed.")

    dataset = pd.concat(all_frames, ignore_index=True)

    dataset = dataset.dropna(subset=["gamma", "energy"])

    dataset = dataset[
        np.isfinite(dataset["gamma"])
        & np.isfinite(dataset["energy"])
    ]

    dataset.to_csv(DATASET_PATH, index=False)

    print(f"Combined dataset saved: {DATASET_PATH}")
    print(f"Total samples: {len(dataset)}")
    print(dataset.groupby("material").size())


if __name__ == "__main__":
    main()
