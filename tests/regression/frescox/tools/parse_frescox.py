from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ANGLE_RE = re.compile(r"^\s*([0-9.]+) deg\.: X-S =\s*([0-9.E+-]+) mb/sr,")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--min-angle-deg", type=float, default=0.0)
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


def parse_rows(block: str, min_angle_deg: float) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in block.splitlines():
        match = ANGLE_RE.match(line)
        if match is None:
            continue
        angle_deg = float(match.group(1))
        if angle_deg < min_angle_deg:
            continue
        rows.append((angle_deg, float(match.group(2))))
    if not rows:
        raise ValueError(
            "no angular-distribution rows were parsed from the requested block"
        )
    return rows


def write_csv(
    csv_out: Path, metadata: dict[str, object], rows: list[tuple[float, float]]
) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# case_id: {metadata['case_id']}",
        f"# reference_code: {metadata['reference_code']}",
        f"# source_example: {metadata['source_example']}",
        f"# observable: {metadata['observable_type']}",
        "theta_cm_deg,dsdo_mb_per_sr",
    ]
    lines.extend(f"{angle_deg:.6f},{dsdo:.12e}" for angle_deg, dsdo in rows)
    csv_out.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    block = extract_block(args.output.read_text(), args.case_index)
    rows = parse_rows(block, args.min_angle_deg)
    write_csv(args.csv_out, metadata, rows)


if __name__ == "__main__":
    main()
