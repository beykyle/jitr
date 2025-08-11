import os
import mkdocs_gen_files
from pathlib import Path

nav = mkdocs_gen_files.Nav()

# Base package name
package = "jitr"

# Get all Python modules and subpackages
src_path = Path("src") / package
modules = []
subpackages = []

for item in src_path.iterdir():
    if item.is_file() and item.suffix == ".py" and not item.name.startswith("__"):
        modules.append(f"{package}.{item.stem}")
    elif item.is_dir() and not item.name.startswith("__"):
        subpackages.append(f"{package}.{item.name}")

# Generate API documentation for modules
for module in modules:
    doc_path = f"docs/api/{module.replace('.', '/')}.md"
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w") as f:
        f.write(f"# `{module}`\n\n")
        f.write(f"::: {module}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: true\n")
        f.write("      show_root_full_path: false\n")
        f.write("      heading_level: 2\n")

    nav[module.split(".")[-1]] = doc_path

# Generate API documentation for subpackages
for subpackage in subpackages:
    doc_path = f"docs/api/{subpackage.replace('.', '/')}/index.md"
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    with open(doc_path, "w") as f:
        f.write(f"# `{subpackage}`\n\n")
        f.write(f"::: {subpackage}\n")
        f.write("    options:\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: true\n")
        f.write("      show_root_full_path: false\n")
        f.write("      heading_level: 2\n")
        f.write("      show_submodules: true\n")

    nav[subpackage.split(".")[-1]] = doc_path

# Write navigation file
with open("SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

print("API documentation generated successfully!")
