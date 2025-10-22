
import h5py
import trimesh
import io
import sys
import argparse
from pathlib import Path

# Try to import vtk, but make it an optional dependency
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

def show_with_trimesh(mesh: trimesh.Trimesh):
    """Visualizes the mesh using the built-in trimesh viewer."""
    print("🎨 Displaying mesh with trimesh viewer. Close the window to continue.")
    mesh.show()

def show_with_vtk(mesh: trimesh.Trimesh):
    """Visualizes the mesh using a VTK pipeline."""
    if not VTK_AVAILABLE:
        print("❌ Error: The 'vtk' library is not installed. Please install it with 'pip install vtk'.")
        return

    print("🎨 Displaying mesh with VTK viewer. Close the window to continue.")

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


def visualize_mesh_from_db(hdf5_path: str, mesh_name: str, use_vtk: bool):
    """
    Loads a specific mesh from the HDF5 database and visualizes it.
    """
    db_file = Path(hdf5_path)
    if not db_file.exists():
        print(f"❌ Error: Database file '{hdf5_path}' not found.")
        return

    mesh_group_path = f"meshes/{mesh_name}"

    with h5py.File(db_file, 'r') as f:
        if mesh_group_path not in f:
            print(f"❌ Error: Mesh '{mesh_name}' not found in {hdf5_path}.")
            available_meshes = list(f.get('meshes', {}).keys())
            if available_meshes:
                print("\nAvailable meshes are:")
                for name in available_meshes:
                    print(f"  - {name}")
            return

        print(f"📦 Loading mesh '{mesh_name}' from the database...")
        stl_binary_content = f[mesh_group_path]['stl_content'][()]

    stl_file_in_memory = io.BytesIO(stl_binary_content)
    mesh = trimesh.load_mesh(stl_file_in_memory, file_type='stl')

    if use_vtk:
        show_with_vtk(mesh)
    else:
        show_with_trimesh(mesh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize a mesh from the Fleetmaster HDF5 database.")
    parser.add_argument("mesh_name", help="The name of the mesh to visualize (e.g., 'barge_draft_1.0').")
    parser.add_argument("--file", default="results.hdf5", help="Path to the HDF5 database file.")
    parser.add_argument("--vtk", action="store_true", help="Use the VTK viewer instead of the default trimesh viewer.")
    
    args = parser.parse_args()

    visualize_mesh_from_db(args.file, args.mesh_name, args.vtk)
