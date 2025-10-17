import logging
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import trimesh

from .exceptions import LidAndSymmetryEnabledError
from .settings import MESH_GROUP_NAME, SimulationSettings

logger = logging.getLogger(__name__)


def make_database(
    body: Any,
    omegas: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depth: float,
    water_level: float,
    forward_speed: float,
) -> Any:
    """Create a dataset of BEM results for a given body and conditions."""
    bem_solver = cpt.BEMSolver()
    problems: list[Any] = []
    logger.debug(f"Solving for water_depth={water_depth} water_level={water_level} forward_speed={forward_speed}")
    for omega in omegas:
        logger.debug(f"RadiationProblem and DiffractionProblem for omega {omega}")
        problems.extend(
            cpt.RadiationProblem(
                omega=omega,
                body=body,
                radiating_dof=dof,
                water_depth=water_depth,
                free_surface=water_level,
                forward_speed=forward_speed,
            )
            for dof in body.dofs
        )
        for wave_direction in wave_directions:
            logger.debug(f"DiffractionProblem for wave_direction {wave_direction} ")
            problems.append(
                cpt.DiffractionProblem(
                    omega=omega,
                    body=body,
                    wave_direction=wave_direction,
                    water_depth=water_depth,
                    free_surface=water_level,
                    forward_speed=forward_speed,
                )
            )

    results = [bem_solver.solve(problem) for problem in problems]

    database = cpt.assemble_dataset(results)

    # Rename phony dimensions that might be created by capytaine.
    # Based on user feedback, we expect phony_dim_0, 1, and 2.
    rename_map = {
        "phony_dim_0": "i",  # Likely a 3x3 matrix row
        "phony_dim_1": "j",  # Likely a 3x3 matrix column
        "phony_dim_2": "mesh_nodes",  # Likely a mesh-related dimension
    }
    # Filter for dims that actually exist in the dataset to avoid errors
    dims_to_rename = {k: v for k, v in rename_map.items() if k in database.dims}
    if dims_to_rename:
        logger.info(f"Renaming phony dimensions: {dims_to_rename}")
        database = database.rename_dims(dims_to_rename)

    for coord_name, coord_data in database.coords.items():
        if isinstance(coord_data.dtype, pd.CategoricalDtype):
            logger.debug(f"Converting coordinate '{coord_name}' from Categorical to string dtype.")
            database[coord_name] = database[coord_name].astype(str)

    return database


def _setup_output_file(settings: SimulationSettings) -> Path:
    """
    Determine the output directory and prepare the HDF5 file.
    Deletes the file if it already exists.

    Returns:
        The full path to the HDF5 output file.
    """
    if not settings.stl_files:
        msg = "No STL files provided to process."
        raise ValueError(msg)

    output_dir = Path(settings.output_directory) if settings.output_directory else Path(settings.stl_files[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / settings.output_hdf5_file
    if output_file.exists() and settings.overwrite_meshes:
        logger.warning(f"Output file {output_file} already exists and will be overwritten as overwrite_meshes is True.")
        output_file.unlink()
    return output_file


def _prepare_capytaine_body(stl_file: str, lid: bool, grid_symmetry: bool) -> Any:
    """
    Load an STL file and configure a Capytaine FloatingBody object.
    """
    hull_mesh = cpt.load_mesh(stl_file)
    lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None

    if grid_symmetry:
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane, name=f"{Path(stl_file).stem}_mesh")

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh)
    boat.add_all_rigid_body_dofs()
    boat.keep_immersed_part()
    return boat


def add_mesh_to_database(output_file: Path, stl_file: str, overwrite: bool = False) -> None:
    """
    Adds a mesh and its geometric properties to the HDF5 database under the MESH_GROUP_NAME.

    Checks if the mesh already exists. If it does, it compares the mesh data.
    If the data is different, it will either raise a warning or overwrite if `overwrite` is True.
    """
    mesh_name = Path(stl_file).stem
    mesh_group_path = f"{MESH_GROUP_NAME}/{mesh_name}"
    new_mesh = trimesh.load_mesh(stl_file)

    with h5py.File(output_file, "a") as f:
        if mesh_group_path in f:
            logger.debug(f"Mesh '{mesh_name}' already exists in the database. Checking for consistency.")
            existing_group = f[mesh_group_path]
            existing_stl_content = existing_group["stl_content"][()]

            with open(stl_file, "rb") as stl_f:
                new_stl_content = stl_f.read()

            if new_stl_content == existing_stl_content:
                logger.info(f"Mesh '{mesh_name}' is identical to the existing one. Skipping.")
                return

            if not overwrite:
                logger.warning(
                    f"Mesh '{mesh_name}' is different from the one in the database. "
                    "Use --overwrite-meshes to overwrite."
                )
                return

            logger.warning(f"Overwriting existing mesh '{mesh_name}' as --overwrite-meshes is specified.")
            del f[mesh_group_path]

        logger.debug(f"Adding mesh '{mesh_name}' to group '{MESH_GROUP_NAME}'...")
        group = f.create_group(mesh_group_path)

        fingerprint_attrs = {
            "volume": new_mesh.volume,
            "cog_x": new_mesh.center_mass[0],
            "cog_y": new_mesh.center_mass[1],
            "cog_z": new_mesh.center_mass[2],
            "bbox_lx": new_mesh.bounding_box.extents[0],
            "bbox_ly": new_mesh.bounding_box.extents[1],
            "bbox_lz": new_mesh.bounding_box.extents[2],
        }
        for key, value in fingerprint_attrs.items():
            group.attrs[key] = value
        logger.debug(f"  - Wrote {len(fingerprint_attrs)} fingerprint attributes.")

        group.create_dataset("inertia_tensor", data=new_mesh.moment_inertia)
        logger.debug("  - Wrote dataset: inertia_tensor")

        with open(stl_file, "rb") as stl_f:
            stl_data = stl_f.read()
        group.create_dataset("stl_content", data=memoryview(stl_data))
        logger.debug("  - Wrote dataset: stl_content")


def _generate_case_group_name(mesh_name: str, water_depth: float, water_level: float, forward_speed: float) -> str:
    """Generates a descriptive group name for a specific simulation case."""
    return f"{mesh_name}_wd{water_depth}_wl{water_level}_fs{forward_speed}"


def _process_single_stl(stl_file: str, settings: SimulationSettings, output_file: Path) -> None:
    """
    Run the complete processing pipeline for a single STL file.
    """
    logger.info(f"Processing STL file: {stl_file}")

    # Add mesh to the database first
    add_mesh_to_database(output_file, stl_file, settings.overwrite_meshes)

    # --- Setup simulation parameters ---
    wave_periods = settings.wave_periods if isinstance(settings.wave_periods, list) else [settings.wave_periods]
    wave_frequencies = 2 * np.pi / np.array(wave_periods)
    wave_directions = (
        settings.wave_directions if isinstance(settings.wave_directions, list) else [settings.wave_directions]
    )
    wave_directions = np.deg2rad(wave_directions)
    water_depths = settings.water_depth if isinstance(settings.water_depth, list) else [settings.water_depth]
    water_levels = settings.water_level if isinstance(settings.water_level, list) else [settings.water_level]

    forwards_speeds = settings.forward_speed if isinstance(settings.forward_speed, list) else [settings.forward_speed]

    lid = settings.lid
    grid_symmetry = settings.grid_symmetry

    # check is done by Settings, so this should no happen anymore
    if lid and grid_symmetry:
        raise LidAndSymmetryEnabledError()

    output_file = output_file

    fmt_str = "%-40s: %s"
    logger.info(fmt_str % ("STL file", stl_file))
    logger.info(fmt_str % ("Output file", output_file))
    logger.info(fmt_str % ("Grid symmetry", grid_symmetry))
    logger.info(fmt_str % ("Use lid", lid))
    logger.info(fmt_str % ("Direction(s) [rad]", wave_directions))
    logger.info(fmt_str % ("Wave period(s) [s]", wave_periods))
    logger.info(fmt_str % ("Water depth(s) [m]", water_depths))
    logger.info(fmt_str % ("Water level(s) [m]", water_levels))
    logger.info(fmt_str % ("Forward speed(s) [m/s]", forwards_speeds))

    process_all_cases_for_one_stl(
        stl_file=stl_file,
        wave_frequencies=wave_frequencies,
        wave_directions=wave_directions,
        water_depths=water_depths,
        water_levels=water_levels,
        forwards_speeds=forwards_speeds,
        lid=lid,
        grid_symmetry=grid_symmetry,
        output_file=output_file,
    )


def process_all_cases_for_one_stl(
    stl_file: str,
    wave_frequencies: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depths: list | npt.NDArray[np.float64],
    water_levels: list | npt.NDArray[np.float64],
    forwards_speeds: list | npt.NDArray[np.float64],
    lid: bool,
    grid_symmetry: bool,
    output_file: Path,
):
    mesh_name = Path(stl_file).stem
    boat = _prepare_capytaine_body(stl_file, lid=lid, grid_symmetry=grid_symmetry)

    for water_level in water_levels:
        for water_depth in water_depths:
            for forward_speed in forwards_speeds:
                logger.info(
                    f"Starting BEM calculations for water_level={water_level}, water_depth={water_depth}, forward_speed={forward_speed}"
                )
                database = make_database(
                    body=boat,
                    omegas=wave_frequencies,
                    wave_directions=wave_directions,
                    water_level=water_level,
                    water_depth=water_depth,
                    forward_speed=forward_speed,
                )

                group_name = _generate_case_group_name(mesh_name, water_depth, water_level, forward_speed)
                logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")

                database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")

                # Add mesh name as attribute to the group for easy lookup
                with h5py.File(output_file, "a") as f:
                    if group_name in f:
                        f[group_name].attrs["stl_mesh_name"] = mesh_name

                logger.debug(f"Successfully wrote data for case to group {group_name}.")

    # The user also wanted to keep the option for multi-dimensional arrays.
    # The current logic saves each case separately. To implement the multi-dim storage,
    # we would need a separate function or a flag to combine datasets.
    # For now, this implementation follows the colleague's suggestion.
    # A future implementation could look like this:
    # if settings.combine_cases:
    #     all_datasets = []
    #     ... collect all ...
    #     combined_dataset = xr.combine_by_coords(all_datasets, combine_attrs="drop_conflicts")
    #     combined_group_name = f"{mesh_name}_multi_dim"
    #     combined_dataset.to_netcdf(output_file, mode="a", group=combined_group_name, engine="h5netcdf")
    #     with h5py.File(output_file, "a") as f:
    #         f[combined_group_name].attrs["stl_mesh_name"] = mesh_name

    logger.debug(f"Successfully wrote all data for {stl_file} to HDF5.")


def run_simulation_batch(settings: SimulationSettings) -> None:
    """
    Runs a batch of Capytaine simulations and saves all results to a single HDF5 file.
    Optionally, also exports results to individual NetCDF files.

    Args:
        settings: A SimulationSettings object with all necessary parameters.
    """
    logger.info("Starting simulation batch...")
    try:
        output_file = _setup_output_file(settings)
    except ValueError as e:
        logger.warning(e)
        return

    for stl_file in settings.stl_files:
        _process_single_stl(stl_file, settings, output_file)

    logger.info(f"✅ Simulation batch finished. Results saved to {output_file}")
