from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--column", default="direct")
    return parser.parse_args()


def find_table(
    lines: list[str], column: str
) -> tuple[list[str], list[tuple[float, float]]]:
    requested = column.lower()
    for index, line in enumerate(lines):
        normalized = line.lstrip("#").lower().split()
        if "angle" not in normalized or requested not in normalized:
            continue
        angle_index = normalized.index("angle")
        value_index = normalized.index(requested)
        rows: list[tuple[float, float]] = []
        for data_line in lines[index + 1 :]:
            if data_line.lstrip().startswith("#"):
                continue
            if not data_line.strip():
                break
            parts = data_line.split()
            if len(parts) <= max(angle_index, value_index):
                break
            try:
                rows.append((float(parts[angle_index]), float(parts[value_index])))
            except ValueError:
                break
        if rows:
            return normalized, rows
    raise ValueError(
        f"could not find a TALYS angular-distribution table with column {column!r}"
    )


def write_csv(
    csv_out: Path, metadata: dict[str, object], rows: list[tuple[float, float]]
) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# case_id: {metadata['case_id']}"])
        writer.writerow([f"# reference_code: {metadata['reference_code']}"])
        writer.writerow([f"# source_example: {metadata['source_example']}"])
        writer.writerow([f"# observable: {metadata['observable_type']}"])
        writer.writerow(["theta_cm_deg", "dsdo_mb_per_sr"])
        for angle_deg, dsdo in rows:
            writer.writerow([f"{angle_deg:.6f}", f"{dsdo:.12e}"])


def main() -> None:
    args = parse_args()
    metadata = json.loads(args.metadata.read_text())
    _, rows = find_table(args.output.read_text().splitlines(), args.column)
    write_csv(args.csv_out, metadata, rows)


if __name__ == "__main__":
    main()
