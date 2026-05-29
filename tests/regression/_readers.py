from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

ROOT = Path(__file__).parent


class ManifestEntry(TypedDict):
    """Committed manifest entry for one regression case."""

    case_id: str
    reference_code: str
    csv: str
    json: str


@dataclass(frozen=True)
class ReferenceCase:
    """Loaded external-reference case data."""

    case_id: str
    reference_code: str
    observable_type: str
    theta_cm_rad: np.ndarray
    dsdo: np.ndarray
    tolerance: dict[str, float]
    metadata: dict[str, Any]


def _read_csv_header(path: Path) -> tuple[dict[str, str], list[list[str]]]:
    header: dict[str, str] = {}
    rows: list[list[str]] = []

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#"):
                payload = first.removeprefix("#").strip()
                if ":" in payload:
                    key, value = payload.split(":", 1)
                    header[key.strip()] = value.strip()
                continue
            rows.append([cell.strip() for cell in row])

    return header, rows


def _validate_case(
    case: ManifestEntry, metadata: dict[str, Any], header: dict[str, str]
) -> None:
    checks = {
        "case_id": case["case_id"],
        "reference_code": case["reference_code"],
        "source_example": metadata["source_example"],
        "observable": metadata["observable_type"],
    }
    for key, expected in checks.items():
        actual = header.get(key)
        if actual != str(expected):
            raise ValueError(
                f"{case['case_id']} CSV header mismatch for {key}: "
                f"expected {expected!r}, found {actual!r}"
            )


def load_case(case: ManifestEntry) -> ReferenceCase:
    """Load one committed regression case from the manifest."""
    metadata_path = ROOT / case["json"]
    csv_path = ROOT / case["csv"]
    metadata = json.loads(metadata_path.read_text())
    header, rows = _read_csv_header(csv_path)
    _validate_case(case, metadata, header)

    if not rows or rows[0] != ["theta_cm_deg", "dsdo_mb_per_sr"]:
        raise ValueError(
            f"{case['case_id']} CSV must start with " "theta_cm_deg,dsdo_mb_per_sr"
        )

    data = np.asarray(rows[1:], dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"{case['case_id']} CSV must have exactly two data columns")
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"{case['case_id']} angle grid must be strictly increasing")

    tolerance = metadata["tolerances"]["dsdo_mb_per_sr"]
    return ReferenceCase(
        case_id=metadata["case_id"],
        reference_code=metadata["reference_code"],
        observable_type=metadata["observable_type"],
        theta_cm_rad=np.deg2rad(data[:, 0]),
        dsdo=data[:, 1],
        tolerance={
            "rtol": float(tolerance["rtol"]),
            "atol": float(tolerance["atol"]),
        },
        metadata=metadata,
    )
