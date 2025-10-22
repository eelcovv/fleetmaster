"""CLI command for visualizing meshes from the HDF5 database."""

import h5py
import io
import logging
from pathlib import Path

import click
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


def visualize_meshes_from_db(hdf5_paths: list[str], mesh_names_to_show: list[str], use_vtk: bool):
    """Loads one or more meshes from HDF5 databases and visualizes them in a single scene."""
    loaded_meshes = []

    for mesh_name in mesh_names_to_show:
        found_mesh_data = None
        for hdf5_path in hdf5_paths:
            db_file = Path(hdf5_path)
            if not db_file.exists():
                continue  # Skip non-existent files silently, list command can be used for checks

            mesh_group_path = f"meshes/{mesh_name}"
            try:
                with h5py.File(db_file, "r") as f:
                    if mesh_group_path in f:
                        click.echo(f"📦 Loading mesh '{mesh_name}' from '{hdf5_path}'...")
                        found_mesh_data = f[mesh_group_path]["stl_content"][()]
                        break  # Found the mesh, no need to check other files for this name
            except Exception as e:
                click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)
                continue

        if found_mesh_data is not None and found_mesh_data.size > 0: # Check for non-None and non-empty array
            stl_file_in_memory = io.BytesIO(found_mesh_data)
            mesh = trimesh.load_mesh(stl_file_in_memory, file_type="stl")
            loaded_meshes.append(mesh)
        else:
            click.echo(f"❌ Warning: Mesh '{mesh_name}' not found in any of the specified files.", err=True)

    if not loaded_meshes:
        click.echo("No meshes were loaded. Nothing to display.", err=True)
        return

    if use_vtk:
        show_with_vtk(loaded_meshes)
    else:
        click.echo(f"🎨 Displaying {len(loaded_meshes)} mesh(es) with trimesh viewer. Close the window to continue.")
        # Create a scene and add an axis marker at the origin
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.axis(origin_size=0.05))

        # Add all meshes to the scene
        scene.add_geometry(loaded_meshes)

        logger.debug("Showing with solid mode. Toggle with w/s to go to wireframe")
        scene.show()


@click.command(name="view", help="Visualize one or more meshes from HDF5 database files.")
@click.argument("mesh_names", nargs=-1)
@click.option("--file", "-f", "hdf5_files", multiple=True, default=["results.hdf5"],
              help="Path to one or more HDF5 database files. Can be specified multiple times.")
@click.option("--vtk", is_flag=True, help="Use the VTK viewer instead of the default trimesh viewer.")
@click.option("--show-all", is_flag=True, help="Visualize all meshes found in the specified files.")
def view(mesh_names: tuple[str, ...], hdf5_files: tuple[str, ...], vtk: bool, show_all: bool):
    """
    CLI command to load and visualize meshes from HDF5 databases.

    You can specify mesh names as arguments or use --show-all.
    """
    # --- Smartly separate file paths from mesh names ---
    all_args = list(mesh_names) + list(hdf5_files)
    
    files_to_check = {arg for arg in all_args if arg.endswith((".hdf5", ".h5"))}
    meshes_to_show = {arg for arg in all_args if not arg.endswith((".hdf5", ".h5"))}

    # If the user provided file paths but also left the default --file value, remove the default.
    # This happens if they provide a path as a positional argument without using -f.
    if files_to_check and "results.hdf5" in hdf5_files and len(hdf5_files) == 1:
        ctx = click.get_current_context()
        if ctx.get_parameter_source("hdf5_files") == click.core.ParameterSource.DEFAULT:
             # The user didn't explicitly type '--file results.hdf5', so we can ignore it
             # if other files were found.
             pass # The default is implicitly overridden by the positional file args.
    elif not files_to_check:
        files_to_check = set(hdf5_files) # Use the default or user-provided --file

    if show_all:
        all_found_meshes = set()
        for hdf5_path in files_to_check:
            db_file = Path(hdf5_path)
            if not db_file.exists():
                click.echo(f"❌ Warning: Database file '{hdf5_path}' not found. Skipping.", err=True)
                continue
            with h5py.File(db_file, "r") as f:
                meshes_to_show.update(f.get("meshes", {}).keys())

    # Remove duplicates
    final_meshes_to_show = sorted(list(meshes_to_show))
    final_files_to_check = list(files_to_check)
    
    if not meshes_to_show:
        click.echo("No mesh names provided and no meshes found with --show-all.", err=True)
        click.echo("Usage: fleetmaster view [MESH_NAME...] [--file <path>] or fleetmaster view --show-all", err=True)
        return

    visualize_meshes_from_db(final_files_to_check, final_meshes_to_show, vtk)