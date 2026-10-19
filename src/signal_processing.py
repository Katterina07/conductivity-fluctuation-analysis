"""
Signal processing pipeline for conductivity fluctuation data.

Reads binary .Q00 files, splits records into overlapping samples,
removes linear trends, computes 1/f noise spectrum parameters and
average spectral energy, and saves features to CSV.

XXX files correspond to a modal reference film used to identify
composition-independent noise contributions.
"""

from pathlib import Path
import struct

import numpy as np
import pandas as pd
from scipy.fft import rfft
from scipy.signal.windows import blackman


SAMPLE_RATE_HZ = 20000  # 1 / 0.00005 s


def read_q00_file(file_path: Path):
    """
    Read binary .Q00 records.

    Yields dictionaries containing metadata and raw ADC codes.
    """
    with file_path.open("rb") as f:
        km = struct.unpack("<f", f.read(4))[0]
        ku = struct.unpack("<f", f.read(4))[0]

        while True:
            header = f.read(32)

            if len(header) < 32:
                break

            jd1, jd2, up1, up2, kc, n_values = struct.unpack(
                "<ddffii",
                header,
            )

            raw_codes = f.read(n_values * 2)

            if len(raw_codes) < n_values * 2:
                break

            codes = np.frombuffer(raw_codes, dtype="<i2")

            yield {
                "km": km,
                "ku": ku,
                "jd1": jd1,
                "jd2": jd2,
                "up1": up1,
                "up2": up2,
                "kc": kc,
                "codes": codes,
            }


def split_into_samples(
    codes: np.ndarray,
    sample_length: int = 128,
    overlap_mode: int = 2,
):
    """
    Split a 1D signal into overlapping samples.

    overlap_mode:
        0 - no overlap
        1 - 25% overlap
        2 - 50% overlap
        3 - 75% overlap
    """
    if overlap_mode == 0:
        step = sample_length
    elif overlap_mode == 1:
        step = int(sample_length * 3 / 4)
    elif overlap_mode == 2:
        step = int(sample_length / 2)
    elif overlap_mode == 3:
        step = int(sample_length / 4)
    else:
        raise ValueError("overlap_mode must be 0, 1, 2, or 3.")

    samples = []

    for start in range(0, len(codes) - sample_length + 1, step):
        samples.append(codes[start:start + sample_length])

    return samples


def compute_spectrum_features(
    sample: np.ndarray,
    sampling_rate: float = SAMPLE_RATE_HZ,
):
    """
    Compute spectral features for one detrended sample.

    Returns:
        gamma: slope of log(power) vs log(frequency)
        mean_energy: mean spectral power
    """
    n = len(sample)

    if n < 2:
        return np.nan, np.nan

    window = blackman(n)
    windowed_signal = sample * window

    spectrum = rfft(windowed_signal)
    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(n, d=1 / sampling_rate)

    # Exclude DC component and zero frequencies before log-log fitting
    mask = frequencies > 0
    frequencies = frequencies[mask]
    power = power[mask]

    # Avoid log(0)
    positive = power > 0
    frequencies = frequencies[positive]
    power = power[positive]

    if len(frequencies) < 2:
        return np.nan, np.nan

    gamma, _ = np.polyfit(
        np.log(frequencies),
        np.log(power),
        1,
    )

    mean_energy = float(np.mean(power))

    return gamma, mean_energy


def process_q00_file(
    file_path: Path,
    sample_length: int = 128,
    overlap_mode: int = 2,
    sampling_rate: float = SAMPLE_RATE_HZ,
):
    """
    Process one .Q00 file and return a DataFrame with spectral features.
    """
    records = list(read_q00_file(file_path))

    if not records:
        raise ValueError(f"No valid records found in {file_path.name}")

    features = []

    for record in records:
        scale = 2.0 / (
            record["kc"]
            * record["ku"]
            * record["km"]
            * (record["up1"] + record["up2"])
        )

        samples = split_into_samples(
            codes=record["codes"],
            sample_length=sample_length,
            overlap_mode=overlap_mode,
        )

        time_index = np.arange(sample_length)

        for sample in samples:
            sample = np.asarray(sample, dtype=float)

            linear_coeffs = np.polyfit(time_index, sample, 1)
            trend = np.polyval(linear_coeffs, time_index)

            detrended_sample = (sample - trend) * scale

            gamma, energy = compute_spectrum_features(
                detrended_sample,
                sampling_rate=sampling_rate,
            )

            features.append(
                {
                    "gamma": gamma,
                    "energy": energy,
                    "jd1": record["jd1"],
                    "jd2": record["jd2"],
                }
            )

    return pd.DataFrame(features)


def extract_material(file_stem: str) -> str:
    """
    Extract material group from the file name.

    XXX is a modal reference film, not a target material class.
    """
    name = file_stem.upper()

    if name.startswith("BI"):
        return "BI"
    if name.startswith("CR"):
        return "CR"
    if name.startswith("FE"):
        return "FE"
    if name.startswith("XXX"):
        return "XXX"

    return "UNKNOWN"


def process_file(
    input_dir: Path,
    file_stem: str,
    output_dir: Path,
    sample_length: int = 128,
    overlap_mode: int = 2,
    sampling_rate: float = SAMPLE_RATE_HZ,
):
    """
    Process one .Q00 file and save its feature dataset as CSV.
    """
    input_path = input_dir / f"{file_stem}.Q00"
    output_path = output_dir / f"{file_stem}_features.csv"

    df = process_q00_file(
        file_path=input_path,
        sample_length=sample_length,
        overlap_mode=overlap_mode,
        sampling_rate=sampling_rate,
    )

    df["file"] = file_stem
    df["material"] = extract_material(file_stem)

    df.to_csv(output_path, index=False)

    print(f"{file_stem}: {len(df)} samples processed -> {output_path}")

    return df
