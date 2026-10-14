from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANGLE_RE = re.compile(r"^\s*([0-9.]+) deg\.: X-S =\s*([0-9.E+-]+) mb/sr,")
POLS_MARKER = "& pols ="
# with kqmax=2 the tail after the marker is FORMAT F3.0,F9.5,3X,3F9.5 (frxx4.f):
# T10, iT11, then T20, T21, T22 (Madison frame, projectile, pp=0)
POLS_FIELDS = ((0, 3), (3, 12), (15, 24), (24, 33), (33, 42))
POLS_COLUMNS = ("iT11", "T20", "T21", "T22")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--min-angle-deg", type=float, default=0.0)
    parser.add_argument(
        "--polarizations",
        action="store_true",
        help="also parse iT11, T20, T21, T22 (deck run with kqmax=2, spin-1)",
    )
    return parser.parse_args()


def extract_block(text: str, case_index: int) -> str:
    starts = [
        match.start() for match in re.finditer("CROSS SECTIONS FOR OUTGOING", text)
    ]
    if case_index < 0 or case_index >= len(starts):
        raise ValueError(
            f"case-index {case_index} is out of range for "
            f"{len(starts)} cross-section blocks"
        )
    end = starts[case_index + 1] if case_index + 1 < len(starts) else len(text)
    return text[starts[case_index] : end]


def parse_polarizations(line: str) -> tuple[float, ...]:
    tail = line.split(POLS_MARKER, 1)[1]
    t10, *rest = (float(tail[a:b]) for a, b in POLS_FIELDS)
    if abs(t10) > 1e-5:
        raise ValueError(f"T10 should vanish for elastic scattering: {line!r}")
    return tuple(rest)


def parse_rows(
    block: str, min_angle_deg: float, polarizations: bool = False
) -> list[tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    for line in block.splitlines():
        match = ANGLE_RE.match(line)
        if match is None:
            continue
        angle_deg = float(match.group(1))
        if angle_deg < min_angle_deg:
            continue
        row = (angle_deg, float(match.group(2)))
        if polarizations:
            row += parse_polarizations(line)
        rows.append(row)
    if not rows:
        raise ValueError(
            "no angular-distribution rows were parsed from the requested block"
        )
    return rows


def write_csv(
    csv_out: Path, metadata: dict[str, object], rows: list[tuple[float, ...]]
) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["theta_cm_deg", "dsdo_mb_per_sr"]
    if len(rows[0]) > 2:
        columns += POLS_COLUMNS
    lines = [
        f"# case_id: {metadata['case_id']}",
        f"# reference_code: {metadata['reference_code']}",
        f"# source_example: {metadata['source_example']}",
        f"# observable: {metadata['observable_type']}",
        ",".join(columns),
    ]
    for angle_deg, dsdo, *pols in rows:
        fields = [f"{angle_deg:.6f}", f"{dsdo:.12e}"] + [f"{p:.5f}" for p in pols]
        lines.append(",".join(fields))
    csv_out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    block = extract_block(args.output.read_text(), args.case_index)
    rows = parse_rows(block, args.min_angle_deg, args.polarizations)
    write_csv(args.csv_out, metadata, rows)


if __name__ == "__main__":
    main()
