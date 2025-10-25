"""
Small use case script to generate meshes for a simple box-shaped hull.

This script generates:
- A buoyancy mesh (`_buoy.stl`) with its origin at the stern.
- A base mesh (`.stl`) for wave interaction analysis in Capytaine, centered at the origin.
- Two wetted surface meshes (`_<draft>m.stl`) for two different drafts.

Run with --grid-symmetry to generate half-meshes for symmetry analysis.

Requires the package pymeshup to be installed.
"""

import argparse
from pathlib import Path

from pymeshup import Box

# Constants for the box dimensions
BOX_LENGTH = 10
BOX_WIDTH = 4
BOX_HEIGHT = 3
DRAFTS = [1, 2]
REGRID_PERCENTAGE = 3
FILE_BASE = "defraction_box"


def main(grid_symmetry: bool, output_dir: Path, file_base: str, only_base: bool = False):
    """
    Generates STL meshes for a defraction box based on specified parameters.

    Args:
        grid_symmetry (bool): If True, enables grid symmetry, cutting the base mesh
                              at the xz plane (port/starboard symmetry).
        output_dir (Path): The directory where the generated STL files will be saved.
        file_base (str): The base name for the generated STL files.
        only_base (bool): If True, only the base mesh will be generated.
    """
    if grid_symmetry:
        print(f"Grid symmetry on with file base {file_base}")
    else:
        print(f"Grid symmetry off with file base {file_base}")

    output_dir.mkdir(exist_ok=True)

    box_buoy_filename = output_dir / f"{file_base}_buoy.stl"
    box_base_filename = output_dir / f"{file_base}.stl"

    half_width = BOX_WIDTH / 2
    half_length = BOX_LENGTH / 2

    # create the buoy mesh with the stern at x = 0
    box_buoy = Box(0, BOX_LENGTH, -half_width, half_width, 0, BOX_HEIGHT)
    print(f"Saving buoy mesh {box_buoy_filename}")
    box_buoy.save(str(box_buoy_filename))

    # create the base mesh for the wave interaction with x = 0 in the centre
    box_base = box_buoy.move(x=-half_length)
    if grid_symmetry:
        print("Cutting at xy plane")
        box_base = box_base.cut_at_xz()
    box_base_mesh = box_base.regrid(pct=REGRID_PERCENTAGE)
    print(f"Saving base mesh {box_base_filename}")
    box_base_mesh.save(str(box_base_filename))

    if only_base:
        return

    for draft in DRAFTS:
        box_draft = box_base.move(z=-draft)
        box_draft = box_draft.cut_at_waterline()
        box_draft_mesh = box_draft.regrid(pct=REGRID_PERCENTAGE)
        box_draft_filename = output_dir / f"{file_base}_{draft}m.stl"
        print(f"Saving draft mesh {box_draft_filename}")
        box_draft_mesh.save(str(box_draft_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate STL meshes for a defraction box, with optional grid symmetry."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="The directory to save the generated STL files.",
    )
    parser.add_argument("-f", "--file-base", type=str, default=FILE_BASE, help="The file base name for the STL files.")
    parser.add_argument(
        "--grid-symmetry",
        action="store_true",
        help="Enable grid symmetry (cuts the base mesh at the xz plane).",
    )
    parser.add_argument(
        "--only-base",
        action="store_true",
        help="Only generate the base mesh.",
    )
    args = parser.parse_args()
    main(
        grid_symmetry=args.grid_symmetry, output_dir=args.output_dir, file_base=args.file_base, only_base=args.only_base
    )
