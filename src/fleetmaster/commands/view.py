"""CLI command for visualizing meshes from the HDF5 database."""

import io
import logging
from pathlib import Path

import click
import h5py
import numpy as np
import trimesh

logger = logging.getLogger(__name__)

# Try to import vtk, but make it an optional dependency
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
    # The global import of numpy is sufficient.
except ImportError:
    VTK_AVAILABLE = False


def show_with_trimesh(mesh: trimesh.Trimesh):
    """Visualizes the mesh using the built-in trimesh viewer."""
    click.echo("🎨 Displaying mesh with trimesh viewer. Close the window to continue.")
    mesh.show()


def show_with_vtk(meshes: list[trimesh.Trimesh]):
    """Visualizes the mesh using a VTK pipeline."""
    if not VTK_AVAILABLE:
        click.echo("❌ Error: The 'vtk' library is not installed. Please install it with 'pip install vtk'.")
        return

    click.echo(f"🎨 Displaying {len(meshes)} mesh(es) with VTK viewer. Close the window to continue.")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue/gray

    # Define a list of colors for multiple meshes
    colors = [
        (0.8, 0.8, 1.0),  # Light Blue
        (1.0, 0.8, 0.8),  # Light Red
        (0.8, 1.0, 0.8),  # Light Green
        (1.0, 1.0, 0.8),  # Light Yellow
    ]

    # 2. Loop through each mesh, create an actor, and add it to the renderer
    for i, mesh in enumerate(meshes):
        # Convert trimesh data to VTK format
        vertices = mesh.vertices
        faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)).flatten()
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(vertices, deep=True))
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(mesh.faces), numpy_to_vtk(faces, deep=True, array_type=vtk.VTK_ID_TYPE))
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)
        poly_data.SetPolys(vtk_cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors[i % len(colors)])
        renderer.AddActor(actor)

    # Add a global axes actor at the origin
    axes_at_origin = vtk.vtkAxesActor()
    axes_at_origin.SetTotalLength(1.0, 1.0, 1.0)  # Set size of the axes
    renderer.AddActor(axes_at_origin)

    # Add an axes actor for context
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()  # This is the small one in the corner
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)

    # RenderWindow: The window on the screen
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("VTK Mesh Viewer")

    # Interactor: Handles mouse and keyboard interaction
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Couple the axes widget to the interactor
    widget.SetInteractor(render_window_interactor)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    # 4. Start the visualization
    render_window.Render()
    render_window_interactor.Start()


def visualize_meshes_from_db(hdf5_path: str, mesh_names_to_show: list[str], use_vtk: bool):
    """Loads one or more meshes from HDF5 databases and visualizes them in a single scene."""
    loaded_meshes = []

    db_file = Path(hdf5_path)
    if not db_file.exists():
        click.echo(f"❌ Error: Database file '{hdf5_path}' not found.", err=True)
        return

    for mesh_name in mesh_names_to_show:
        found_mesh_data = None
        mesh_group_path = f"meshes/{mesh_name}"
        try:
            logger.debug(f"Opening database {db_file}")
            with h5py.File(db_file, "r") as f:
                if mesh_group_path in f:
                    click.echo(f"📦 Loading mesh '{mesh_name}' from '{hdf5_path}'...")
                    found_mesh_data = f[mesh_group_path]["stl_content"][()]
        except Exception as e:
            logger.exception(f"Error reading mesh {mesh_group_path}' from '{hdf5_path}'")
            click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)
            continue

        if found_mesh_data:  # A non-empty numpy.void object evaluates to True.
            try:
                # The data is stored as a numpy.void object, which must be converted to bytes.
                stl_bytes = found_mesh_data.tobytes()
                mesh = trimesh.load_mesh(io.BytesIO(stl_bytes), file_type="stl")
                if mesh:
                    loaded_meshes.append(mesh)
            except Exception as e:
                logger.exception(f"Failed to parse mesh '{mesh_name}'")
                click.echo(f"❌ Error parsing mesh '{mesh_name}': {e}", err=True)
        else:
            click.echo(f"❌ Warning: Mesh '{mesh_name}' not found in any of the specified files.", err=True)

    if not loaded_meshes:
        click.echo("No meshes were loaded. Nothing to display.", err=True)
        return

    if use_vtk:
        show_with_vtk(loaded_meshes)
    else:
        click.echo(f"🎨 Displaying {len(loaded_meshes)} mesh(es) with trimesh viewer. Close the window to continue.")
        # To avoid potential rendering glitches with the scene object,
        # we create a scene with an axis and pass the meshes to show directly.
        axis = trimesh.creation.axis(origin_size=0.1)
        scene = trimesh.Scene([axis, *loaded_meshes])

        logger.debug("Showing with solid mode. Toggle with w/s to go to wireframe")
        scene.show()


@click.command(name="view", help="Visualize meshes from an HDF5 database file.")
@click.argument("hdf5_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("mesh_names", nargs=-1)
@click.option("--vtk", is_flag=True, help="Use the VTK viewer instead of the default trimesh viewer.")
@click.option("--show-all", is_flag=True, help="Visualize all meshes found in the specified files.")
def view(hdf5_file: str, mesh_names: tuple[str, ...], vtk: bool, show_all: bool):
    """
    CLI command to load and visualize meshes from HDF5 databases.

    HDF5_FILE: Path to the HDF5 database file. [MESH_NAMES]...: Optional names
    of meshes or cases to visualize.
    """
    # The HDF5 file is now a required positional argument.
    # Mesh names are optional positional arguments.
    names_to_resolve = set(mesh_names)
    resolved_mesh_names = set()

    if show_all:
        # If --show-all, we ignore any provided mesh names and find all meshes in the specified files.
        names_to_resolve = set()
        db_file = Path(hdf5_file)
        with h5py.File(db_file, "r") as f:
            resolved_mesh_names.update(f.get("meshes", {}).keys())

    elif names_to_resolve:
        # Resolve provided names: they can be mesh names or case names.
        with h5py.File(hdf5_file, "r") as f:
            for name in names_to_resolve:
                # Check if it's a direct mesh name
                if f.get(f"meshes/{name}"):
                    resolved_mesh_names.add(name)
                    logger.debug(f"Resolved '{name}' as a direct mesh name.")
                # Check if it's a case name
                elif (case_group := f.get(name)) and (mesh_name := case_group.attrs.get("stl_mesh_name")):
                    resolved_mesh_names.add(mesh_name)
                    logger.debug(f"Resolved case '{name}' to mesh '{mesh_name}'.")
                else:
                    click.echo(
                        f"❌ Warning: Could not resolve '{name}' as a mesh or a case name.",
                        err=True,
                    )

    if not resolved_mesh_names:
        click.echo("No mesh names provided and no meshes found with --show-all.", err=True)
        click.echo(
            "Usage: fleetmaster view <HDF5_FILE> [MESH_NAME...]  OR  fleetmaster view <HDF5_FILE> --show-all", err=True
        )
        return

    visualize_meshes_from_db(hdf5_file, sorted(resolved_mesh_names), vtk)
