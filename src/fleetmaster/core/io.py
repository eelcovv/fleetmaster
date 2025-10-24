import logging
import tempfile
from pathlib import Path

import h5py
import vtk

logger = logging.getLogger(__name__)


def load_meshes_from_hdf5(
    hdf5_path: Path,
    mesh_names: list[str],
) -> list[vtk.vtkPolyData]:
    """Load and return vtkPolyData objects for the given names from HDF5."""
    meshes: list[vtk.vtkPolyData] = []
    if not hdf5_path.exists():
        raise FileNotFoundError(f"{hdf5_path} not found")  # noqa: TRY003

    with h5py.File(hdf5_path, "r") as f:
        for name in mesh_names:
            group = f.get(f"meshes/{name}")
            if not group:
                logger.warning("Mesh %r not found", name)
                continue
            raw = group["stl_content"][()]
            try:
                # vtkSTLReader cannot read from memory. Write to a temporary file first.
                with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(raw.tobytes())

                try:
                    reader = vtk.vtkSTLReader()
                    reader.SetFileName(str(temp_path))
                    reader.Update()
                    poly_data = reader.GetOutput()
                    meshes.append(poly_data)
                finally:
                    # Ensure the temporary file is always deleted.
                    temp_path.unlink()

            except Exception:
                logger.exception("Failed to parse mesh %r", name)
    return meshes
