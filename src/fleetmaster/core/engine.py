import logging
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import trimesh
import xarray as xr

from .settings import SimulationSettings

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
    logger.debug("Collecting problems")
    for omega in omegas:
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
    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists and will be overwritten.")
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


def _write_geometric_data_to_hdf5(output_file: Path, group_name: str, stl_file: str) -> None:
    """
    Calculate the geometric fingerprint and append it along with the raw STL content
    to the specified group in the HDF5 file.
    """
    logger.debug(f"Calculating geometric fingerprint for {stl_file}...")
    mesh_for_props = trimesh.load_mesh(stl_file)

    fingerprint_attrs = {
        "volume": mesh_for_props.volume,
        "cog_x": mesh_for_props.center_mass[0],
        "cog_y": mesh_for_props.center_mass[1],
        "cog_z": mesh_for_props.center_mass[2],
        "bbox_lx": mesh_for_props.bounding_box.extents[0],
        "bbox_ly": mesh_for_props.bounding_box.extents[1],
        "bbox_lz": mesh_for_props.bounding_box.extents[2],
    }
    inertia_tensor_data = mesh_for_props.moment_inertia

    logger.debug(f"Appending geometric data to group '{group_name}'...")
    with h5py.File(output_file, "a") as f:
        group = f.require_group(group_name)

        for key, value in fingerprint_attrs.items():
            group.attrs[key] = value
        logger.debug(f"  - Wrote {len(fingerprint_attrs)} fingerprint attributes.")

        if "inertia_tensor" in group:
            del group["inertia_tensor"]
        group.create_dataset("inertia_tensor", data=inertia_tensor_data)
        logger.debug("  - Wrote dataset: inertia_tensor")

        if "stl_content" in group:
            del group["stl_content"]
        with open(stl_file, "rb") as stl_f:
            stl_data = stl_f.read()
        group.create_dataset("stl_content", data=memoryview(stl_data))
        logger.debug("  - Wrote dataset: stl_content")


def _process_single_stl(stl_file: str, settings: SimulationSettings, output_file: Path) -> None:
    """
    Run the complete processing pipeline for a single STL file.
    """
    logger.info(f"Processing STL file: {stl_file}")

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
    assert not (lid and grid_symmetry), "Cannot have both lid and grid_symmetry True simultaneously."  # noqa: S101

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
    group_name = Path(stl_file).stem
    logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
    boat = _prepare_capytaine_body(stl_file, lid=lid, grid_symmetry=grid_symmetry)

    all_datasets = []
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

                database = database.assign_coords(
                    water_level=water_level,
                    water_depth=water_depth,
                    forward_speed=forward_speed,
                ).expand_dims(["water_level", "water_depth", "forward_speed"])
                all_datasets.append(database)

    if not all_datasets:
        logger.warning("No datasets were generated. Nothing to write to HDF5.")
        return

    logger.info("Combining all datasets into a single xarray.Dataset...")
    combined_dataset = xr.combine_by_coords(all_datasets)

    logger.debug(f"Writing combined database to group '{group_name}'")
    combined_dataset.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")

    _write_geometric_data_to_hdf5(output_file, group_name, stl_file)
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
