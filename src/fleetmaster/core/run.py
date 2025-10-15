import logging
from pathlib import Path

# from pymeshup.gui.capytaine_runner import run_capytaine
import capytaine as cpt
import numpy as np
from capytaine.io.xarray import export_dataset, save_dataset_as_netcdf

logger = logging.getLogger(__name__)

matlog = logging.getLogger("matplotlib")
matlog.setLevel(level=logging.WARNING)

caplog = logging.getLogger("capytaine")
caplog.setLevel(level=logging.WARNING)


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


def run_simulations():
    logger.info("Welcome to capytaine!")

    file_base_name = "defraction_box"

    stl_file = Path(file_base_name).with_suffix(".stl")
    nc_file = Path(file_base_name).with_suffix(".nc")
    tec_dir = Path(file_base_name).with_suffix("")
    tec_dir.mkdir(exist_ok=True, parents=True)
    hyd_file = Path(file_base_name).with_suffix(".dhyd")

    wave_directions = np.deg2rad(np.linspace(0, 90, num=3))
    periods = np.linspace(5, 20, 4)
    wave_frequencies = 2 * np.pi / periods
    lid = True
    grid_symmetry = False
    heading_symmetry = False
    show_vtk_input = False
    show_capytaine_input = False
    water_depth = 10
    water_level = 0
    recalculate_if_exists = True

    if lid and grid_symmetry:
        raise ValueError("Can not have lid and symmetry True simultaniously.")

    logger.info("directions [rad] : %s", wave_directions)
    logger.info("periods [s]      : %s", periods)
    logger.info("omega [rad/s]    : %s", wave_frequencies)
    logger.info("Lid              : %s", lid)

    logger.debug("Writing to %s", stl_file)
    hull_mesh = cpt.load_mesh(str(stl_file))

    if lid:
        lid_mesh = hull_mesh.generate_lid(z=-0.01)
    else:
        lid_mesh = None

    if grid_symmetry:
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane, name=f"{file_base_name}_mesh")

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh)

    boat.add_all_rigid_body_dofs()
    boat.keep_immersed_part()

    if show_capytaine_input:
        logger.info("Showing the boat")
        boat.show()

    if not nc_file.exists() or recalculate_if_exists:
        logger.info("Start calculating")
        database = make_database(
            body=boat,
            omegas=wave_frequencies,
            wave_directions=wave_directions,
            water_level=water_level,
            water_depth=water_depth,
        )

        logger.info(f"Writing result as netcdf file: {nc_file}")
        save_dataset_as_netcdf(str(nc_file), database)
        logger.info(f"Writing result to tecplot director: {tec_dir}")
        export_dataset(str(tec_dir), database, format="nemoh")

        # sep = separate_complex_values(database)
        # for var_name in sep.variables:
        #     logger.debug(f"Checking {var_name}")
        #     if (
        #         hasattr(sep[var_name].dtype, "name")
        #         and sep[var_name].dtype.name == "category"
        #     ):
        #         logger.debug(f"Converting to string: {var_name}")
        #         sep[var_name] = sep[var_name].astype(str)

        # logger.info(f"Writing result to {nc_file}")
        # sep.to_netcdf(
        #     str(nc_file),
        #     encoding={
        #         "radiating_dof": {"dtype": "U"},
        #         "influenced_dof": {"dtype": "U"},
        #     },
        # )


def main():
    logging.basicConfig(level=logging.INFO)
    launch_capytaine()


if __name__ == "__main__":
    main()
