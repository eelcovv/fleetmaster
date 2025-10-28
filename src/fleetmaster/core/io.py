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
        raise FileNotFoundError(f"{hdf5_path} not found")  # noqa: TRY003

    with h5py.File(hdf5_path, "r") as f:
        for name in mesh_names:
            group = f.get(f"meshes/{name}")
            if not group:
                logger.debug("Mesh %r not found. Continue without failing", name)
                continue
            raw = group["stl_content"][()]
            try:
                mesh = trimesh.load_mesh(io.BytesIO(raw.tobytes()), file_type="stl")
                if isinstance(mesh, trimesh.Trimesh):
                    mesh.metadata["name"] = name  # Store the name for later identification
                    # Load attributes if they exist
                    mesh.metadata["translation"] = group.attrs.get("translation")
                    mesh.metadata["rotation"] = group.attrs.get("rotation")
                    mesh.metadata["cog"] = group.attrs.get("cog")
                    meshes.append(mesh)
            except Exception:
                logger.exception("Failed to parse mesh %r", name)
    return meshes
