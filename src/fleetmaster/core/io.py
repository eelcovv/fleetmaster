import io
import logging
from pathlib import Path

import h5py
import trimesh

logger = logging.getLogger(__name__)


def load_meshes_from_hdf5(
    hdf5_path: Path,
    mesh_names: list[str],
) -> list[trimesh.Trimesh]:
    """Load and return trimesh objects for the given names from HDF5."""
    meshes: list[trimesh.Trimesh] = []
    if not hdf5_path.exists():
        raise FileNotFoundError(f"{hdf5_path} not found")

    with h5py.File(hdf5_path, "r") as f:
        for name in mesh_names:
            group = f.get(f"meshes/{name}")
            if not group:
                logger.warning("Mesh %r not found", name)
                continue
            raw = group["stl_content"][()]
            try:
                mesh = trimesh.load_mesh(io.BytesIO(raw.tobytes()), file_type="stl")
                if isinstance(mesh, trimesh.Trimesh):
                    meshes.append(mesh)
            except Exception:
                logger.exception("Failed to parse mesh %r", name)
    return meshes
