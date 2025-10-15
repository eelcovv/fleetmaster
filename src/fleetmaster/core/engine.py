import logging

logger = logging.getLogger(__name__)


def run_simulation_batch(settings: dict):
    """
    Runs a batch of Capytaine simulations based on the g    iven settings.
    Args:
        settings (dict): Een dictionary met alle benodigde parameters
                         (stl_file, wave_periods, etc.).
    """
    logger.info("Starting simulation batch...")
    stl_file = settings.get("stl_file")
    wave_directions = settings.get("wave_directions", [0])

    logger.info(f"Processing STL file: {stl_file}")
    for direction in wave_directions:
        logger.debug(f"Running for wave direction: {direction}")
        # resultaat = capytaine.solve(...)

    logger.info("Simulation batch finished.")
    # return resultaten

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
