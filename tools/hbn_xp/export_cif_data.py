"""Build the XP offline scattering table from the pinned XrayDB database.

This is a build tool, never needed on the XP computer. Each f' interval uses
the same local seven-knot FITPACK interpolant as materials/optics.py.
"""

import argparse
import hashlib
import json
import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np
import xraydb
from scipy.interpolate import UnivariateSpline


def export(destination: Path) -> None:
    database = Path(xraydb.__file__).with_name("xraydb.sqlite")
    expected = "fd58bcc4213e3a7c6ecd32d1d8dfd7f4c74526ca468a3df62cce99afeb5ff5bf"
    if (
        xraydb.__version__ != "4.5.8"
        or hashlib.sha256(database.read_bytes()).hexdigest() != expected
    ):
        raise ValueError("Build requires XrayDB 4.5.8 with the documented database 9.2 SHA256")
    payload = bytearray()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        elastic = connection.execute(
            "SELECT atomic_number,ion,offset,scale,exponents FROM Waasmaier ORDER BY id"
        ).fetchall()
        anomalous = connection.execute(
            "SELECT id,energy,f1,f2 FROM Chantler ORDER BY id"
        ).fetchall()
    for atomic_number, ion, offset, scale, exponents in elastic:
        payload.extend(
            struct.pack(
                "<16sI11d",
                ion.encode("ascii"),
                atomic_number,
                offset,
                *json.loads(scale),
                *json.loads(exponents),
            )
        )
    for atomic_number, energy_json, f1_json, f2_json in anomalous:
        energy, f1, f2 = (
            np.asarray(json.loads(value), dtype=np.float64)
            for value in (energy_json, f1_json, f2_json)
        )
        f1[np.abs(f1) < 1e-99] = 1e-99
        f2[np.abs(f2) < 1e-99] = 1e-99
        first = int(np.searchsorted(energy, 250.0, side="right")) - 1
        rows = []
        for index in range(first, len(energy)):
            coefficients = [0.0] * 4
            if index + 1 < len(energy):
                low, high = max(0, index - 3), min(len(energy), index + 4)
                spline = UnivariateSpline(energy[low:high], f1[low:high], s=0)
                width = energy[index + 1] - energy[index]
                derivatives = spline.derivatives(energy[index])
                coefficients = [derivatives[k] * width**k / (1, 1, 2, 6)[k] for k in range(4)]
            rows.append((energy[index], *coefficients, np.log(f2[index])))
        table = np.asarray(rows, dtype="<f8")
        if not np.all(np.isfinite(table)) or not np.all(np.diff(table[:, 0]) > 0):
            raise ValueError("Invalid or duplicate Chantler knots in supported range")
        payload.extend(struct.pack("<II", atomic_number, len(rows)))
        payload.extend(table.tobytes())
    destination.write_bytes(
        struct.pack("<8sIII", b"SLCIF01\0", len(elastic), len(anomalous), zlib.crc32(payload))
        + payload
    )
    print(
        f"{destination.name}: {len(payload) + 20} bytes; SHA256 "
        f"{hashlib.sha256(destination.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    export(parser.parse_args().output)
