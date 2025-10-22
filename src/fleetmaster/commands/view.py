"""CLI command for visualizing meshes from the HDF5 database."""

import h5py
import io
from pathlib import Path

import click
import numpy as np
import trimesh

# Try to import vtk, but make it an optional dependency
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
    import numpy as np # numpy is needed for vtk conversion
except ImportError:
    VTK_AVAILABLE = False


def show_with_trimesh(mesh: trimesh.Trimesh):
    """Visualizes the mesh using the built-in trimesh viewer."""
    click.echo("🎨 Displaying mesh with trimesh viewer. Close the window to continue.")
    mesh.show()


def show_with_vtk(mesh: trimesh.Trimesh):
    """Visualizes the mesh using a VTK pipeline."""
    if not VTK_AVAILABLE:
        click.echo("❌ Error: The 'vtk' library is not installed. Please install it with 'pip install vtk'.")
        return

    click.echo("🎨 Displaying mesh with VTK viewer. Close the window to continue.")

    # 1. Convert trimesh data to VTK format
    # Get vertices and faces
    vertices = mesh.vertices
    # VTK requires a specific format for faces: [num_points, p1_idx, p2_idx, p3_idx, ...]
    faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces)).flatten()

    # Create vtkPoints for the vertices
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(vertices, deep=True))

    # Create vtkCellArray for the faces
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetCells(len(mesh.faces), numpy_to_vtk(faces, deep=True, array_type=vtk.VTK_ID_TYPE))

    # 2. Create the vtkPolyData (the actual geometry)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetPolys(vtk_cells)

    # 3. Set up the visualization pipeline
    # Mapper: Connects the geometry to the graphics hardware
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    # Actor: Represents the object in the scene (position, color, etc.)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.8, 1.0)  # Light blue
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.1, 0.1, 0.2)

    # Renderer: Manages the scene, camera, and lighting
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue/gray

    # Add an axes actor for context
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
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


def visualize_mesh_from_db(hdf5_paths: list[str], mesh_name: str, use_vtk: bool):
    """Loads a specific mesh from the HDF5 database and visualizes it."""
    found_mesh_data = None
    found_in_file = None

    for hdf5_path in hdf5_paths:
        db_file = Path(hdf5_path)
        if not db_file.exists():
            click.echo(f"❌ Warning: Database file '{hdf5_path}' not found. Skipping.", err=True)
            continue

        mesh_group_path = f"meshes/{mesh_name}"
        try:
            with h5py.File(db_file, "r") as f:
                if mesh_group_path in f:
                    click.echo(f"📦 Loading mesh '{mesh_name}' from '{hdf5_path}'...")
                    found_mesh_data = f[mesh_group_path]["stl_content"][()]
                    found_in_file = hdf5_path
                    break  # Found the mesh, no need to check other files
        except Exception as e:
            click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)
            continue

    if found_mesh_data is None:
        click.echo(f"❌ Error: Mesh '{mesh_name}' not found in any of the specified HDF5 files.", err=True)
        click.echo("Use 'fleetmaster list --file <your_file.hdf5>' to see available meshes.", err=True)
        return

    # If we found the mesh, proceed with visualization
    stl_binary_content = found_mesh_data

    stl_file_in_memory = io.BytesIO(stl_binary_content)
    mesh = trimesh.load_mesh(stl_file_in_memory, file_type="stl")

    if use_vtk:
        show_with_vtk(mesh)
    else:
        show_with_trimesh(mesh)


@click.command(name="view", help="Visualize a specific mesh from one or more HDF5 database files.")
@click.argument("mesh_name")
@click.option("--file", "-f", "hdf5_files", multiple=True, default=["results.hdf5"],
              help="Path to one or more HDF5 database files. Can be specified multiple times.")
@click.option("--vtk", is_flag=True, help="Use the VTK viewer instead of the default trimesh viewer.")
def view(mesh_name: str, hdf5_files: tuple[str, ...], vtk: bool):
    """
    CLI command to load and visualize a specific mesh from the HDF5 database.

    MESH_NAME: The name of the mesh to visualize (e.g., 'barge_draft_1.0').
    """
    if not mesh_name:
        click.echo("❌ Error: Please provide a MESH_NAME to visualize.", err=True)
        click.echo("Use 'fleetmaster list --file <your_file.hdf5>' to see available meshes.", err=True)
        return

    visualize_mesh_from_db(list(hdf5_files), mesh_name, vtk)