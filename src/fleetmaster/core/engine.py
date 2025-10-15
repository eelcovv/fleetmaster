import logging
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
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


def run_simulation_batch(settings: SimulationSettings) -> None:
    """
    Runs a batch of Capytaine simulations and saves all results to a single HDF5 file.
    Optionally, also exports results to individual NetCDF files.

    Args:
        settings: A SimulationSettings object with all necessary parameters.
    """
    logger.info("Starting simulation batch...")

    output_dir = Path(settings.output_directory) if settings.output_directory else None

    output_dir = Path(settings.stl_files[0]).parent if output_dir is None else output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / settings.output_hdf5_file
    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists and will be overwritten.")
        output_file.unlink()

    for stl_file in settings.stl_files:
        logger.info(f"Processing STL file: {stl_file}")

        wave_frequencies = 2 * np.pi / np.array(settings.wave_periods)

        # Handle the case where wave_directions might be None
        wave_directions_list = settings.wave_directions if settings.wave_directions is not None else [0.0]
        wave_directions = np.deg2rad(wave_directions_list)

        # Example settings from the original script - decide how to handle them.
        # These could be added to SimulationSettings if they need to be configurable.
        lid = True
        grid_symmetry = False
        water_depth = np.inf
        water_level = 0.0

        if lid and grid_symmetry:
            raise SimulationConfigurationError(SimulationConfigurationError.LID_AND_SYMMETRY_ENABLED)

        logger.info(f"Directions [rad]: {wave_directions}")
        logger.info(f"Periods [s]: {settings.wave_periods}")
        logger.info(f"Frequencies (omega) [rad/s]: {wave_frequencies}")

        hull_mesh = cpt.load_mesh(stl_file)
        lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None

        if grid_symmetry:
            hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane, name=f"{Path(stl_file).stem}_mesh")

        boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh)
        boat.add_all_rigid_body_dofs()
        boat.keep_immersed_part()

        logger.info("Starting BEM calculations...")
        database = make_database(
            body=boat,
            omegas=wave_frequencies,
            wave_directions=wave_directions,
            water_level=water_level,
            water_depth=water_depth,
        )

        group_name = Path(stl_file).stem

        # Convert the Categorial items
        # This prevents the TypeError with the h5netcdf engine.
        for coord_name, coord_data in database.coords.items():
            if isinstance(coord_data.dtype, pd.CategoricalDtype):
                logger.debug(f"Converting coordinate '{coord_name}' from Categorical to string dtype.")
                database[coord_name] = database[coord_name].astype(str)

        # --- HDF5 output (always) ---
        logger.info(f"Writing results to group '{group_name}' in HDF5 file: {output_file}")
        database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")
        logger.debug(f"Opened netcdf outfile {output_file}")
        with h5py.File(output_file, "a") as f:
            group = f[group_name]
            logger.debug(f"Writin stl file {stl_file}")
            with open(stl_file, "rb") as stl_f:
                stl_data = stl_f.read()
            logger.debug("Call group create")
            group.create_dataset("stl_content", data=np.void(stl_data))
        logger.debug(f"Successfully wrote results for {stl_file} to HDF5 {output_file}.")

        # --- Legacy NetCDF/Tecplot output (optional) ---
        if settings.export_to_netcdf:
            logger.info("Exporting to individual NetCDF and Tecplot files as requested.")

            nc_file = output_dir / f"{group_name}.nc"
            tec_dir = output_dir / f"{group_name}_tecplot"
            tec_dir.mkdir(exist_ok=True)

            logger.info(f"Writing result as NetCDF file: {nc_file}")
            database.to_netcdf(nc_file)
            logger.debug("Success")

            logger.info(f"Writing result to Tecplot directory: {tec_dir}")
            export_dataset(str(tec_dir), database, format="nemoh")
            logger.debug("Success")

    logger.info(f"✅ Simulation batch finished. Results saved to {output_file}")
