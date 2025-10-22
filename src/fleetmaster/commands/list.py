"""CLI command for listing meshes from HDF5 databases."""

import h5py
from pathlib import Path

import click


def list_meshes_in_db(hdf5_paths: list[str]):
    """
    Lists all available meshes in one or more HDF5 database files.
    """
    for hdf5_path in hdf5_paths:
        db_file = Path(hdf5_path)
        if not db_file.exists():
            click.echo(f"❌ Error: Database file '{hdf5_path}' not found.", err=True)
            continue

        click.echo(f"\nAvailable meshes in '{hdf5_path}':")
        try:
            with h5py.File(db_file, "r") as f:
                meshes_group = f.get("meshes")
                if meshes_group:
                    available_meshes = list(meshes_group.keys())
                    if available_meshes:
                        for name in available_meshes:
                            click.echo(f"  - {name}")
                    else:
                        click.echo("  No meshes found.")
                else:
                    click.echo("  No 'meshes' group found.")
        except Exception as e:
            click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)


@click.command(name="list", help="List all meshes available in one or more HDF5 database files.")
@click.option("--file", "-f", "hdf5_files", multiple=True, default=["results.hdf5"],
              help="Path to one or more HDF5 database files. Can be specified multiple times.")
def list_command(hdf5_files: tuple[str, ...]):
    """CLI command to list meshes."""
    list_meshes_in_db(list(hdf5_files))