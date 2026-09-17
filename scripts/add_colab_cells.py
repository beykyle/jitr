"""Insert the "Open in Colab" badge and bootstrap cell into the example notebooks.

Colab hands the reader a bare ``.ipynb``: no ``jitr``, and none of the sibling data
files the notebook reads. So every notebook in ``examples/notebooks`` carries two
generated cells at the top -- a markdown badge, and a code cell that installs what
the notebook needs, but only when it detects a Colab runtime. The bootstrap cell
must come before any ``import numpy``/``import numba`` so that a fresh runtime
picks up whatever pip installs without needing a restart.

Run this after adding or renaming a notebook;
``tests/test_notebook_colab_cells.py`` guards against forgetting.

Usage::

    python scripts/add_colab_cells.py            # insert or refresh the cells
    python scripts/add_colab_cells.py --check    # exit 1 if any notebook is stale
"""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

REPO = "beykyle/jitr"
BRANCH = "main"
NOTEBOOK_SUBDIR = "examples/notebooks"

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / NOTEBOOK_SUBDIR
LINE_LENGTH = 88  # matches [tool.black] in pyproject.toml
BADGE_SVG = "https://colab.research.google.com/assets/colab-badge.svg"
COLAB_BASE = (
    f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{NOTEBOOK_SUBDIR}"
)
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{NOTEBOOK_SUBDIR}/"

# Markers used to recognise previously generated cells, so this script is idempotent
# and survives a change of branch or package list.
BADGE_MARKER = f"colab.research.google.com/github/{REPO}"
SETUP_MARKER = '"google.colab" in sys.modules'

# Packages Colab does not preinstall, and data files the notebook reads from its own
# directory. numpy, scipy, pandas, matplotlib, tqdm and IPython are already on Colab;
# periodictable, numba, sympy and mpmath arrive with jitr.
EXTRAS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "quickstart.ipynb": ((), ("alpha_ca_ratio_ruth.csv",)),
    "alpha_ca_calibration.ipynb": (
        ("dynesty", "corner", "matplotlib>=3.10.3"),
        ("alpha_ca_ratio_ruth.csv",),
    ),
    "alpha_ca48_ambiguity.ipynb": (
        ("dynesty", "corner", "matplotlib>=3.10.3"),
        ("alpha_ca_ratio_ruth.csv",),
    ),
    "chex_jitr_validation.ipynb": ((), ("chex_qepn_xs.txt",)),
    "builtin_omps_uq.ipynb": (("exfor-tools",), ()),
}

# Why a requirement above carries a version floor. Kept next to it in the generated
# cell, so nobody drops it again as redundant.
PACKAGE_NOTES = {
    "matplotlib>=3.10.3": (
        "# matplotlib 3.10.0-3.10.1, which Colab preinstalls, mangles an RGBA tuple",
        "# passed as a facecolor to ax.hist -- it breaks the corner plots below",
    ),
}

# Notebooks whose full run will not finish on a free Colab CPU runtime. They already
# read JITR_QUICK themselves; the badge cell says so, since it changes the result.
QUICK_NOTEBOOKS = ("alpha_ca_calibration.ipynb", "alpha_ca48_ambiguity.ipynb")

_QUICK_NOTE = (
    "Colab runs this notebook with `JITR_QUICK=1` -- fewer data points and live"
    " points -- so the posterior is coarser than the one rendered in the"
    " [documentation](https://beykyle.github.io/jitr/examples/index.html)."
    " Drop that line from the setup cell, on a runtime you can keep alive, to"
    " reproduce the full result."
)
_EXFOR_NOTE = (
    "On Colab the first `import exfor_tools` downloads and unpacks the EXFOR"
    " database, which takes a few minutes."
)
NOTES = {
    "alpha_ca_calibration.ipynb": _QUICK_NOTE,
    "alpha_ca48_ambiguity.ipynb": _QUICK_NOTE,
    "builtin_omps_uq.ipynb": _EXFOR_NOTE,
}


def badge_markdown(name: str) -> str:
    """Return the source of the badge cell for the notebook called ``name``."""
    lines = [f"[![Open in Colab]({BADGE_SVG})]({COLAB_BASE}/{name})"]
    note = NOTES.get(name)
    if note is not None:
        lines += ["", note]
    return "\n".join(lines)


def setup_source(name: str) -> str:
    """Return the source of the Colab bootstrap cell for the notebook ``name``."""
    packages, data_files = EXTRAS.get(name, ((), ()))
    quick = name in QUICK_NOTEBOOKS

    header = ["# Google Colab setup: installs jitr"]
    if packages:
        header[0] += " and this notebook's extra dependencies,"
    else:
        header[0] += " into the Colab runtime,"
    if data_files:
        plural = "s" if len(data_files) > 1 else ""
        header.append(f"# and downloads the data file{plural} it reads below.")
    else:
        header[0] = header[0].rstrip(",") + "."
    header.append("# Does nothing when the notebook is run anywhere else.")

    imports = ["subprocess"]
    if quick:
        imports.append("os")
    if data_files:
        imports.append("urllib.request")

    pip_specs = ("jitr", *packages)
    pip_args = '        [sys.executable, "-m", "pip", "install", "-q"'
    pip_args += "".join(f', "{spec}"' for spec in pip_specs) + "],"
    if len(pip_args) <= LINE_LENGTH:
        pip_lines = [pip_args]
    else:
        # black would explode the call one argument per line; a named list reads better
        pip_lines = [
            "    packages = [" + ", ".join(f'"{spec}"' for spec in pip_specs) + "]",
            '        [sys.executable, "-m", "pip", "install", "-q", *packages],',
        ]
    notes = [
        f"    {note}" for spec in pip_specs for note in PACKAGE_NOTES.get(spec, ())
    ]
    lines = [
        *header,
        "import sys",
        "",
        f"if {SETUP_MARKER}:",
        *(f"    import {module}" for module in sorted(imports)),
        "",
        *notes,
        *pip_lines[:-1],
        "    subprocess.run(",
        pip_lines[-1],
        "        check=True,",
        "    )",
    ]
    if data_files:
        names = ", ".join(f'"{data_file}"' for data_file in data_files)
        if len(data_files) == 1:
            names += ","
        lines += [
            f'    RAW = "{RAW_BASE}"',
            f"    for fname in ({names}):",
            "        urllib.request.urlretrieve(RAW + fname, fname)",
        ]
    if quick:
        lines += [
            "    # keep the nested-sampling run inside a free Colab CPU runtime",
            '    os.environ.setdefault("JITR_QUICK", "1")',
        ]
    return "\n".join(lines)


def _source_lines(text: str) -> list[str]:
    """Split ``text`` the way nbformat stores a cell source."""
    return text.splitlines(keepends=True)


def _is_badge(cell: dict) -> bool:
    return cell["cell_type"] == "markdown" and BADGE_MARKER in "".join(cell["source"])


def _is_setup(cell: dict) -> bool:
    return cell["cell_type"] == "code" and SETUP_MARKER in "".join(cell["source"])


def _make_cell(cell: dict, has_ids: bool, previous: dict | None) -> dict:
    """Add a cell id, reusing the previous one so repeated runs do not churn."""
    if has_ids:
        cell["id"] = (previous or {}).get("id") or uuid4().hex[:8]
    return cell


def expected_cells(name: str) -> tuple[dict, dict]:
    """Return the badge and bootstrap cells the notebook ``name`` should start with."""
    badge = {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(badge_markdown(name)),
    }
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["remove-cell"]},
        "outputs": [],
        "source": _source_lines(setup_source(name)),
    }
    return badge, setup


def rendered(notebook: dict) -> str:
    """Serialise a notebook the way nbformat writes it to disk."""
    return json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def updated(notebook: dict, name: str) -> dict:
    """Return ``notebook`` with a fresh badge cell and bootstrap cell at the top."""
    has_ids = notebook.get("nbformat_minor", 0) >= 5
    old_badge = next((cell for cell in notebook["cells"] if _is_badge(cell)), None)
    old_setup = next((cell for cell in notebook["cells"] if _is_setup(cell)), None)
    badge, setup = expected_cells(name)
    rest = [
        cell
        for cell in notebook["cells"]
        if not _is_badge(cell) and not _is_setup(cell)
    ]
    notebook = dict(notebook)
    notebook["cells"] = [
        _make_cell(badge, has_ids, old_badge),
        _make_cell(setup, has_ids, old_setup),
        *rest,
    ]
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any notebook is missing or stale",
    )
    args = parser.parse_args()

    stale = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        original = path.read_text()
        wanted = rendered(updated(json.loads(original), path.name))
        if wanted == original:
            continue
        stale.append(path.name)
        if not args.check:
            path.write_text(wanted)

    if args.check:
        if stale:
            print(
                "notebooks missing an up-to-date Colab badge or setup cell:\n  "
                + "\n  ".join(stale)
                + "\n\nrun: python scripts/add_colab_cells.py",
                file=sys.stderr,
            )
            return 1
        print(f"all {len(list(NOTEBOOK_DIR.glob('*.ipynb')))} notebooks are up to date")
        return 0

    if stale:
        print(f"updated {len(stale)} notebook(s): " + ", ".join(stale))
    else:
        print("all notebooks already up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
