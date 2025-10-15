import logging
from pathlib import Path

import capytaine as cpt
import numpy as np
from capytaine.io.xarray import export_dataset, save_dataset_as_netcdf

from .exceptions import SimulationConfigurationError
from .settings import SimulationSettings

logger = logging.getLogger(__name__)


def make_database(body, omegas, wave_directions, water_depth, water_level):
    """Create a dataset of BEM results for a given body and conditions."""
    bem_solver = cpt.BEMSolver()
    problems = []
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


def run_simulation_batch(settings: SimulationSettings, recalculate_if_exists: bool = True):
    """
    Runs a batch of Capytaine simulations based on the given settings.

    Args:
        settings: A SimulationSettings object with all necessary parameters.
        recalculate_if_exists: If True, re-run and overwrite existing results.
    """
    logger.info("Starting simulation batch...")
    logger.info(f"Processing STL file: {settings.stl_file}")

    file_base_name = Path(settings.stl_file).stem
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    nc_file = output_dir / f"{file_base_name}.nc"
    tec_dir = output_dir / f"{file_base_name}_tecplot"
    tec_dir.mkdir(exist_ok=True)

    wave_frequencies = 2 * np.pi / np.array(settings.wave_periods)
    wave_directions = np.deg2rad(settings.wave_directions)

    # Example settings from the original script - decide how to handle them.
    # These could be added to SimulationSettings if they need to be configurable.
    lid = True
    grid_symmetry = False
    water_depth = 0  # 0 for deep water
    water_level = 0

    if lid and grid_symmetry:
        raise SimulationConfigurationError(SimulationConfigurationError.LID_AND_SYMMETRY_ENABLED)

    logger.info(f"Directions [rad]: {wave_directions}")
    logger.info(f"Periods [s]: {settings.wave_periods}")
    logger.info(f"Frequencies (omega) [rad/s]: {wave_frequencies}")

    hull_mesh = cpt.load_mesh(settings.stl_file)
    lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None

    if grid_symmetry:
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane, name=f"{file_base_name}_mesh")

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh)
    boat.add_all_rigid_body_dofs()
    boat.keep_immersed_part()

    if not nc_file.exists() or recalculate_if_exists:
        logger.info("Starting BEM calculations...")
        database = make_database(
            body=boat,
            omegas=wave_frequencies,
            wave_directions=wave_directions,
            water_level=water_level,
            water_depth=water_depth,
        )

        logger.info(f"Writing result as NetCDF file: {nc_file}")
        save_dataset_as_netcdf(database, nc_file)

        logger.info(f"Writing result to Tecplot directory: {tec_dir}")
        export_dataset(database, tec_dir, format="nemoh")
    else:
        logger.info(f"Loading existing results from {nc_file}. Use --recalculate to force recalculation.")

    logger.info("✅ Simulation batch finished.")
