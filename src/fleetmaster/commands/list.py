"""CLI command for listing meshes from HDF5 databases."""

import h5py
import io
import logging
from pathlib import Path

import click
import trimesh
from trimesh import Trimesh


logger = logging.getLogger(__name__)

def list_items_in_db(hdf5_paths: list[str], show_cases: bool):
    """
    Lists available meshes or simulation cases in one or more HDF5 database files.
    """
    for hdf5_path in hdf5_paths:
        db_file = Path(hdf5_path)
        if not db_file.exists():
            click.echo(f"❌ Error: Database file '{hdf5_path}' not found.", err=True)
            continue
        try:
            with h5py.File(db_file, "r") as f:
                if show_cases:
                    click.echo(f"\nAvailable cases in '{hdf5_path}':")
                    case_names = [name for name in f.keys() if name != "meshes"]
                    if not case_names:
                        click.echo("  No cases found.")
                        continue

                    for case_name in sorted(case_names):
                        case_group = f[case_name]
                        click.echo(f"\n- Case: {case_name}")

                        mesh_name = case_group.attrs.get("stl_mesh_name")
                        if not mesh_name:
                            click.echo("    Mesh: [Unknown]")
                            continue

                        click.echo(f"    Mesh: {mesh_name}")
                        mesh_info_group = f.get(f"meshes/{mesh_name}")
                        if mesh_info_group:
                            attrs = mesh_info_group.attrs
                            vol = attrs.get("volume", "N/A")
                            cog_x = attrs.get("cog_x", "N/A")
                            cog_y = attrs.get("cog_y", "N/A")
                            cog_z = attrs.get("cog_z", "N/A")
                            lx = attrs.get("bbox_lx", "N/A")
                            ly = attrs.get("bbox_ly", "N/A")
                            lz = attrs.get("bbox_lz", "N/A")

                            # Load mesh from stored content to get more details
                            num_faces = "N/A"
                            bounds = None
                            stl_content_dataset = mesh_info_group.get("stl_content")
                            if stl_content_dataset:
                                try:
                                    # When stored correctly, h5py returns a bytes object directly.
                                    stl_bytes = stl_content_dataset[()]
                                    mesh: Trimesh = trimesh.load_mesh(io.BytesIO(stl_bytes), file_type="stl") # type: ignore
                                    num_faces = len(mesh.faces)
                                    bounds = mesh.bounding_box.bounds
                                except (ValueError, IOError, TypeError) as e:
                                    logger.debug(f"Failed to parse STL content for mesh '{mesh_name}': {e}")
                                    click.echo(f"      Could not parse stored STL content: {e}")

                            click.echo(f"      Cells: {num_faces}")
                            click.echo(f"      Volume: {vol:.4f}" if isinstance(vol, float) else f"      Volume: {vol}")
                            click.echo(
                                f"      COG (x,y,z): ({cog_x:.3f}, {cog_y:.3f}, {cog_z:.3f})"
                                if all(isinstance(c, float) for c in [cog_x, cog_y, cog_z])
                                else f"      COG (x,y,z): ({cog_x}, {cog_y}, {cog_z})"
                            )
                            click.echo(
                                f"      BBox Dims (Lx,Ly,Lz): ({lx:.3f}, {ly:.3f}, {lz:.3f})"
                                if all(isinstance(d, float) for d in [lx, ly, lz])
                                else f"      BBox Dims (Lx,Ly,Lz): ({lx}, {ly}, {lz})"
                            )
                            if bounds is not None:
                                click.echo(f"      BBox Min (x,y,z): ({bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f})")
                                click.echo(f"      BBox Max (x,y,z): ({bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f})")
                        else:
                            click.echo("      Mesh properties not found in database.")
                else:
                    click.echo(f"\nAvailable meshes in '{hdf5_path}':")
                    meshes_group = f.get("meshes")
                    if meshes_group:
                        available_meshes = list(meshes_group.keys())
                        if available_meshes:
                            for name in sorted(available_meshes):
                                click.echo(f"  - {name}")
                        else:
                            click.echo("  No meshes found.")
                    else:
                        click.echo("  No 'meshes' group found.")
        except Exception as e:
            click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)


@click.command(name="list", help="List all meshes available in one or more HDF5 database files.")
@click.argument("files", nargs=-1, type=click.Path())
@click.option(
    "--file", "-f", "option_files", multiple=True, help="Path to one or more HDF5 database files. Can be specified multiple times."
)
@click.option("--cases", is_flag=True, help="List simulation cases and their properties instead of meshes.")
def list_command(files: tuple[str, ...], option_files: tuple[str, ...], cases: bool):
    """CLI command to list meshes."""
    # Combine positional arguments and optional --file arguments
    all_files = set(files) | set(option_files)

    # If no files are provided at all, use the default.
    if not all_files:
        final_files = ["results.hdf5"]
    else:
        final_files = list(all_files)

    list_items_in_db(final_files, show_cases=cases)