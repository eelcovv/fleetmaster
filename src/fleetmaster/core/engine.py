import logging
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import trimesh
from capytaine.io.xarray import export_dataset

from .exceptions import SimulationConfigurationError
from .settings import SimulationSettings

logger = logging.getLogger(__name__)


def make_database(
    body: Any,
    omegas: npt.NDArray[np.float64],
    wave_directions: npt.NDArray[np.float64],
    water_depth: float,
    water_level: float,
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
                )
            )

    results = [bem_solver.solve(problem) for problem in problems]
    return cpt.assemble_dataset(results)


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
    wave_frequencies = 2 * np.pi / np.array(settings.wave_periods)
    wave_directions_list = settings.wave_directions if settings.wave_directions is not None else [0.0]
    wave_directions = np.deg2rad(wave_directions_list)

    lid = True
    grid_symmetry = False
    water_depth = np.inf
    water_level = 0.0
    if lid and grid_symmetry:
        raise SimulationConfigurationError(SimulationConfigurationError.LID_AND_SYMMETRY_ENABLED)

    logger.info(f"Directions [rad]: {wave_directions}")
    logger.info(f"Periods [s]: {settings.wave_periods}")
    logger.info(f"Frequencies (omega) [rad/s]: {wave_frequencies}")

    # --- Prepare Capytaine body and run BEM calculations ---
    boat = _prepare_capytaine_body(stl_file, lid=lid, grid_symmetry=grid_symmetry)
    logger.info("Starting BEM calculations...")
    database = make_database(
        body=boat,
        omegas=wave_frequencies,
        wave_directions=wave_directions,
        water_level=water_level,
        water_depth=water_depth,
    )

    # --- Pre-process data for storage ---
    for coord_name, coord_data in database.coords.items():
        if isinstance(coord_data.dtype, pd.CategoricalDtype):
            logger.debug(f"Converting coordinate '{coord_name}' from Categorical to string dtype.")
            database[coord_name] = database[coord_name].astype(str)

    # --- Write data to HDF5 file ---
    group_name = Path(stl_file).stem
    logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
    database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")
    _write_geometric_data_to_hdf5(output_file, group_name, stl_file)
    logger.debug(f"Successfully wrote all data for {stl_file} to HDF5.")

    # --- Handle optional legacy output ---
    if settings.export_to_netcdf:
        logger.info("Exporting to individual NetCDF and Tecplot files as requested.")
        output_dir = output_file.parent
        nc_file = output_dir / f"{group_name}.nc"
        tec_dir = output_dir / f"{group_name}_tecplot"
        tec_dir.mkdir(exist_ok=True)

        logger.info(f"Writing result as NetCDF file: {nc_file}")
        database.to_netcdf(nc_file)

        logger.info(f"Writing result to Tecplot directory: {tec_dir}")
        export_dataset(str(tec_dir), database, format="nemoh")


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
