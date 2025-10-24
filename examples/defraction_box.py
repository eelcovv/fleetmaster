import argparse
from pymeshup import Box

def main(grid_symmetry: bool):
    """
    Generates STL meshes for a defraction box based on specified parameters.

    Args:
        grid_symmetry (bool): If True, enables grid symmetry, cutting the base mesh at the xz plane.
    """
    file_base = "defraction_box"
    if grid_symmetry:
        file_base += "_half"


    length = 10
    width = 4
    height = 3
    drafts = [1, 2]

    box_buoy_filename = file_base + "_buoy.stl"
    box_base_filename = file_base + ".stl"

    half_width = width / 2
    half_length = length / 2

    # create the buoy mesh with the stern at x = 0
    box_buoy = Box(0, length, -half_width , half_width, 0, height)
    print(f"Saving buoy mesh {box_buoy_filename}")
    box_buoy.save(box_buoy_filename)

    # create the base mesh for the wave interaction with x = 0 in the centre
    box_base = box_buoy.move(x=-half_length)
    if grid_symmetry:
        print("Grid symmetry on: cutting mesh at xz")
        box_base = box_base.cut_at_xz()
    box_base_mesh = box_base.regrid(pct=3)
    print(f"Saving base mesh {box_base_filename}")
    box_base_mesh.save(box_base_filename)

    for draft in drafts:
        box_draft = box_base.move(z=-draft).cut_at_waterline()
        box_draft_mesh = box_draft.regrid(pct=3)
        box_draft_filename = file_base + f"_{draft}m.stl"
        print(f"Saving draft mesh {box_draft_filename}")
        box_draft_mesh.save(box_draft_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate STL meshes for a defraction box, with optional grid symmetry."
    )
    parser.add_argument(
        "--grid-symmetry",
        action="store_true",
        help="Enable grid symmetry (cuts the base mesh at the xz plane)."
    )
    args = parser.parse_args()
    main(args.grid_symmetry)
