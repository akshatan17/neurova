"""
A small reader for EDF files, the standard format for EEG recordings.

An EDF file is a text header followed by the samples as 16-bit integers. The
header says how many channels there are, what they are called, how fast each
one was sampled, and how to turn the stored integers into microvolts.
"""
import numpy as np


def read_header(raw: bytes) -> dict:
    if len(raw) < 256 or raw[:1] != b"0":
        raise ValueError("This is not an EDF file.")
    try:
        n = int(raw[252:256])                                   # number of channels
        record_seconds = float(raw[244:252])
        pos = 256

        def column(width, convert=str):
            nonlocal pos
            out = [convert(raw[pos + i * width: pos + (i + 1) * width].decode("latin-1").strip()) for i in range(n)]
            pos += n * width
            return out

        labels = column(16)
        column(80), column(8)                                   # sensor type and unit: not needed
        phys_min, phys_max = column(8, float), column(8, float)
        dig_min, dig_max = column(8, float), column(8, float)
        column(80)                                              # filter description: not needed
        per_record = column(8, int)                             # samples per data record, per channel
    except (ValueError, IndexError):
        raise ValueError("The EDF header is damaged or incomplete.") from None
    return {"labels": labels, "record_seconds": record_seconds or 1.0, "per_record": per_record,
            "phys_min": phys_min, "phys_max": phys_max, "dig_min": dig_min, "dig_max": dig_max,
            "data_start": 256 + 256 * n}


def read_channel(raw: bytes, header: dict, index: int) -> tuple[np.ndarray, float]:
    """Return one channel in physical units, and its sampling rate in Hz."""
    per_record = header["per_record"]
    record_len = sum(per_record)
    data = np.frombuffer(raw, dtype="<i2", offset=header["data_start"])
    records = len(data) // record_len                           # works even if the file was cut short
    if records == 0:
        raise ValueError("The EDF file contains no samples.")
    start = sum(per_record[:index])
    digital = data[: records * record_len].reshape(records, record_len)[:, start: start + per_record[index]].ravel()
    span = header["dig_max"][index] - header["dig_min"][index] or 1.0
    gain = (header["phys_max"][index] - header["phys_min"][index]) / span
    signal = (digital - header["dig_min"][index]) * gain + header["phys_min"][index]
    return signal.astype(np.float64), per_record[index] / header["record_seconds"]
