import logging
from pathlib import Path

# from pymeshup.gui.capytaine_runner import run_capytaine
import capytaine as cpt
import click
import numpy as np
import yaml
from capytaine.io.xarray import export_dataset, save_dataset_as_netcdf

from fleetmaster.core.engine import run_simulation_batch
from fleetmaster.core.settings import SimulationSettings

logger = logging.getLogger(__name__)

matlog = logging.getLogger("matplotlib")
matlog.setLevel(level=logging.WARNING)

caplog = logging.getLogger("capytaine")
caplog.setLevel(level=logging.WARNING)


@click.command()
@click.option("--settings-file", type=click.Path(exists=True), help="Path to a YAML settings file.")
@click.option("--stl-file", help="Override STL file from settings.")
def run(settings_file, stl_file, **kwargs):
    """Runs a set of capytaine simulations."""

    config = {}
    if settings_file:
        with open(settings_file, "r") as f:
            config = yaml.safe_load(f)

    # CLI-opties overschrijven de YAML-instellingen
    if stl_file:
        config["stl_file"] = stl_file
    # ... verwerk andere kwargs ...

    try:
        # Valideer en structureer de data met Pydantic
        settings = SimulationSettings(**config)

        # Roep de core logica aan met de gevalideerde settings
        run_simulation_batch(settings.model_dump())
        click.echo("Run completed successfully!")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)


def make_database(body, omegas, wave_directions, water_depth=0, water_level=0):
    bem_solver = cpt.BEMSolver()

    # SOLVE BEM PROBLEMS
    problems = []
    logger.debug("Collecting problems")
    for omega in omegas:
        problems += [
            cpt.RadiationProblem(
                omega=omega,
                body=body,
                radiating_dof=dof,
                water_depth=water_depth,
                free_surface=water_level,
            )
            for dof in body.dofs
        ]
        for wave_direction in wave_directions:
            problems += [
                cpt.DiffractionProblem(
                    omega=omega,
                    body=body,
                    wave_direction=wave_direction,
                    water_depth=water_depth,
                    free_surface=water_level,
                )
            ]
    results = []
    n_problems = len(problems)
    for cnt, problem in enumerate(problems):
        problem_type = type(problem).__name__
        try:
            problem_dof = problem.radiating_dof
        except AttributeError:
            problem_dof = "None"
        problem_ome = problem.omega
        problem_dir = problem.wave_direction
        logger.debug(
            f"Solving {cnt:02d}/{n_problems} : {problem_type:20s} {problem_dof:6s} {problem_ome:8.2f} {problem_dir:8.2f}"
        )

        result = bem_solver.solve(problem)
        results.append(result)
    # *radiation_results, diffraction_result = results
    dataset = cpt.assemble_dataset(results)

    # dataset['diffraction_result'] = diffraction_result

    return dataset
