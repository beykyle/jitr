"""Convert a directory of ASCII ``.rad`` density tables into one packaged ``.npz``.

Usage::

    python scripts/convert_density_tables.py <rad_dir> <out.npz>

``<rad_dir>`` holds one ``<Symbol>.rad`` file per element. Each file is a
sequence of blocks: a header line ``Z A n_points dr`` followed by ``n_points``
rows of 11 whitespace-separated columns, of which column 0 (radius in fm),
column 1 (proton density) and column 6 (neutron density) are used. Only those
densities are kept, as float32, on the uniform grid ``np.arange(n_points) * dr``.

The original ``bskg3`` and ``d1m`` tables were committed to the jitr repository
in commit ``71e4ea0`` under ``src/data/densities/<model>/`` and were replaced by
``src/data/densities/<model>.npz`` to stay under PyPI's file-size limit::

    git show 71e4ea0:src/data/densities/bskg3/Pb.rad

The model name used at runtime is the ``.npz`` file stem.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jitr.utils.density import DensityTable, read_rad_file, write_density_npz


def convert(rad_dir: Path, out_path: Path) -> int:
    """Read every ``*.rad`` in ``rad_dir`` and write ``out_path``.

    Returns:
        Number of nuclides written.
    """

    model = out_path.stem
    tables: list[DensityTable] = []
    rad_files = sorted(rad_dir.glob("*.rad"))
    if not rad_files:
        raise FileNotFoundError(f"No .rad files found in {rad_dir}")
    for path in rad_files:
        tables.extend(read_rad_file(path, model=model))
    write_density_npz(tables, out_path)
    return len(tables)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rad_dir", type=Path, help="directory of <Symbol>.rad files")
    parser.add_argument("out", type=Path, help="output .npz path; stem = model name")
    args = parser.parse_args(argv)
    n = convert(args.rad_dir, args.out)
    print(f"wrote {n} nuclides to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
