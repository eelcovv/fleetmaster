import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import trimesh
import xarray as xr

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


def _prepare_capytaine_body(
    stl_file: str,
    lid: bool,
    grid_symmetry: bool,
    add_centre_of_mass: bool = False,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    translation_z: float = 0.0,
) -> tuple[Any, trimesh.Trimesh]:
    """
    Load an STL file and configure a Capytaine FloatingBody object.
    """
    hull_mesh = cpt.load_mesh(stl_file)

    # Apply translation if specified
    if translation_x != 0.0 or translation_y != 0.0 or translation_z != 0.0:
        translation_vector = np.array([translation_x, translation_y, translation_z])
        logger.debug(f"Applying mesh translation: {translation_vector}")
        hull_mesh = hull_mesh.transform(translation_vector=translation_vector)

    lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None

    if grid_symmetry:
        logger.debug("Applying grid symmetery")
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane, name=f"{Path(stl_file).stem}_mesh")

    cog = None
    if add_centre_of_mass:
        full_mesh = trimesh.load_mesh(stl_file)
        if translation_x != 0.0 or translation_y != 0.0 or translation_z != 0.0:
            transform_matrix = trimesh.transformations.translation_matrix([translation_x, translation_y, translation_z])
            full_mesh.apply_transform(transform_matrix)
        cog = full_mesh.center_mass
        logger.debug(f"Adding COG {cog}")

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh, center_of_mass=cog)
    boat.add_all_rigid_body_dofs()
    boat.keep_immersed_part()

    # Extract the final mesh that Capytaine will use, after all transformations and immersion.
    final_mesh_trimesh = trimesh.Trimesh(vertices=boat.mesh.vertices, faces=boat.mesh.faces)

    return boat, final_mesh_trimesh


def add_mesh_to_database(output_file: Path, mesh_to_add: trimesh.Trimesh, mesh_name: str, overwrite: bool = False) -> None:
    """
    Adds a mesh and its geometric properties to the HDF5 database under the MESH_GROUP_NAME.

    Checks if the mesh already exists by comparing SHA256 hashes.
    If the data is different, it will either raise a warning or overwrite if `overwrite` is True.

    Args:
        mesh_to_add: The trimesh object of the mesh to be added.
    """
    mesh_group_path = f"{MESH_GROUP_NAME}/{mesh_name}"

    # Export the trimesh to an in-memory STL binary string and compute its hash.
    new_stl_content = mesh_to_add.export(file_type="stl")
    new_hash = hashlib.sha256(new_stl_content).hexdigest()

    with h5py.File(output_file, "a") as f:
        if mesh_group_path in f:
            existing_group = f[mesh_group_path]
            stored_hash = existing_group.attrs.get("sha256")

            if stored_hash == new_hash:
                logger.info(f"Mesh '{mesh_name}' has the same SHA256 hash. Skipping.")
                return

            if not overwrite:
                logger.warning(
                    f"Mesh '{mesh_name}' is different from the one in the database (SHA256 mismatch). "
                    "Use --overwrite-meshes to overwrite."
                )
                return

            logger.warning(f"Overwriting existing mesh '{mesh_name}' as --overwrite-meshes is specified.")
            del f[mesh_group_path]

        logger.debug(f"Adding mesh '{mesh_name}' to group '{MESH_GROUP_NAME}'...")
        group = f.create_group(mesh_group_path)

        # Calculate geometric properties from the new mesh content
        fingerprint_attrs = {
            "volume": mesh_to_add.volume,
            "cog_x": mesh_to_add.center_mass[0],
            "cog_y": mesh_to_add.center_mass[1],
            "cog_z": mesh_to_add.center_mass[2],
            "bbox_lx": mesh_to_add.bounding_box.extents[0],
            "bbox_ly": mesh_to_add.bounding_box.extents[1],
            "bbox_lz": mesh_to_add.bounding_box.extents[2],
        }
        for key, value in fingerprint_attrs.items():
            group.attrs[key] = value
        logger.debug(f"  - Wrote {len(fingerprint_attrs)} fingerprint attributes.")

        # Add hash and original file name as attributes
        group.attrs["sha256"] = new_hash

        group.create_dataset("inertia_tensor", data=mesh_to_add.moment_inertia)
        logger.debug("  - Wrote dataset: inertia_tensor")

        # Store the binary content of the final, transformed STL
        group.create_dataset("stl_content", data=memoryview(new_stl_content))
        logger.debug("  - Wrote dataset: stl_content")


def _format_value_for_name(value: float) -> str:
    """Formats a float for use in a group name."""
    if value == np.inf:
        return "inf"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _generate_case_group_name(mesh_name: str, water_depth: float, water_level: float, forward_speed: float) -> str:  # noqa: E501
    """Generates a descriptive group name for a specific simulation case."""
    wd = _format_value_for_name(water_depth)
    wl = _format_value_for_name(water_level)
    fs = _format_value_for_name(forward_speed)
    return f"{mesh_name}_wd_{wd}_wl_{wl}_fs_{fs}"


def _process_single_stl(stl_file: str, settings: SimulationSettings, output_file: Path, mesh_name_override: str | None = None) -> None:
    """
    Run the complete processing pipeline for a single STL file.
    """
    logger.info(f"Processing STL file: {stl_file}")

    # check is done by Settings, so this should no happen anymore
    if settings.lid and settings.grid_symmetry:
        raise LidAndSymmetryEnabledError()

    wave_periods = settings.wave_periods if isinstance(settings.wave_periods, list) else [settings.wave_periods]
    wave_frequencies = (2 * np.pi / np.array(wave_periods)).tolist()
    wave_directions = (
        settings.wave_directions if isinstance(settings.wave_directions, list) else [settings.wave_directions]
    )
    wave_directions = np.deg2rad(wave_directions).tolist()
    water_depths = settings.water_depth if isinstance(settings.water_depth, list) else [settings.water_depth]
    water_levels = settings.water_level if isinstance(settings.water_level, list) else [settings.water_level]

    forwards_speeds = settings.forward_speed if isinstance(settings.forward_speed, list) else [settings.forward_speed]

    add_center_of_mass = settings.add_center_of_mass
    lid = settings.lid
    grid_symmetry = settings.grid_symmetry

    output_file = output_file

    fmt_str = "%-40s: %s"
    logger.info(fmt_str % ("Base STL file", stl_file))
    logger.info(fmt_str % ("Output file", output_file))
    logger.info(fmt_str % ("Grid symmetry", grid_symmetry))
    logger.info(fmt_str % ("Use lid", lid))
    logger.info(fmt_str % ("Add COG ", add_center_of_mass))
    logger.info(fmt_str % ("Direction(s) [rad]", wave_directions))
    logger.info(fmt_str % ("Wave period(s) [s]", wave_periods))
    logger.info(fmt_str % ("Water depth(s) [m]", water_depths))
    logger.info(fmt_str % ("Water level(s) [m]", water_levels))
    logger.info(fmt_str % ("Translation X", settings.translation_x))
    logger.info(fmt_str % ("Translation Y", settings.translation_y))
    logger.info(fmt_str % ("Translation Z", settings.translation_z))
    logger.info(fmt_str % ("Forward speed(s) [m/s]", forwards_speeds))

    process_all_cases_for_one_stl(
        stl_file=stl_file,
        wave_frequencies=wave_frequencies,
        wave_directions=wave_directions,
        water_depths=water_depths,
        water_levels=water_levels,
        forwards_speeds=forwards_speeds,
        lid=lid,
        add_center_of_mass=add_center_of_mass,
        grid_symmetry=grid_symmetry,
        output_file=output_file,
        update_cases=settings.update_cases,
        combine_cases=settings.combine_cases,
        translation_x=settings.translation_x,
        translation_y=settings.translation_y,
        translation_z=settings.translation_z,
        mesh_name_override=mesh_name_override,
    )


def process_all_cases_for_one_stl(
    stl_file: str,
    wave_frequencies: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depths: list | npt.NDArray[np.float64],
    water_levels: list | npt.NDArray[np.float64],
    forwards_speeds: list | npt.NDArray[np.float64],
    lid: bool,
    add_center_of_mass: bool,
    grid_symmetry: bool,
    output_file: Path,
    update_cases: bool = False,
    combine_cases: bool = False,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    translation_z: float = 0.0,
    mesh_name_override: str | None = None,
) -> None:
    mesh_name = mesh_name_override or Path(stl_file).stem
    boat, final_mesh = _prepare_capytaine_body(
        stl_file,
        lid=lid,
        grid_symmetry=grid_symmetry,
        add_center_of_mass=add_center_of_mass,
        translation_x=translation_x,
        translation_y=translation_y,
        translation_z=translation_z,
    )

    # Add the final, transformed, and immersed mesh to the database.
    add_mesh_to_database(output_file, final_mesh, mesh_name, overwrite=update_cases)

    all_datasets = []

    for water_level in water_levels:
        for water_depth in water_depths:
            for forward_speed in forwards_speeds:
                group_name = _generate_case_group_name(mesh_name, water_depth, water_level, forward_speed)

                with h5py.File(output_file, "a") as f:
                    if group_name in f:
                        if not update_cases:
                            logger.info(f"Case '{group_name}' already exists in the database. Skipping.")
                            continue
                        logger.info(f"Case '{group_name}' exists, but update_cases is True. Overwriting.")
                        del f[group_name]

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

                if combine_cases:
                    all_datasets.append(database)
                else:
                    logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
                    database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")
                    with h5py.File(output_file, "a") as f:
                        if group_name in f:
                            f[group_name].attrs["stl_mesh_name"] = mesh_name
                    logger.debug(f"Successfully wrote data for case to group {group_name}.")

    if combine_cases and all_datasets:
        logger.info("Combining all calculated cases into a single multi-dimensional dataset.")
        combined_dataset = xr.combine_by_coords(all_datasets, combine_attrs="drop_conflicts")
        combined_group_name = f"{mesh_name}_multi_dim"

        logger.info(f"Writing combined dataset to group '{combined_group_name}' in HDF5 file: {output_file}")
        with h5py.File(output_file, "a") as f:
            if combined_group_name in f:
                del f[combined_group_name]
        combined_dataset.to_netcdf(output_file, mode="a", group=combined_group_name, engine="h5netcdf")
        with h5py.File(output_file, "a") as f:
            f[combined_group_name].attrs["stl_mesh_name"] = mesh_name

    logger.debug(f"Successfully wrote all data for {stl_file} to HDF5.")


def run_simulation_batch(settings: SimulationSettings) -> None:
    """
    Runs a batch of Capytaine simulations and saves all results to a single HDF5 file.

    If `settings.drafts` is provided, it generates new meshes by translating a single
    base STL file for each draft. Otherwise, it processes the provided list of STL files.

    Args:
        settings: A SimulationSettings object with all necessary parameters.
    """
    logger.info("Starting simulation batch...")
    try:
        output_file = _setup_output_file(settings)
    except ValueError as e:
        logger.warning(e)
        return

    if settings.drafts:
        if len(settings.stl_files) != 1:
            msg = f"When using --drafts, exactly one base STL file must be provided, but {len(settings.stl_files)} were given."
            logger.error(msg)
            raise ValueError(msg)

        base_stl_file = settings.stl_files[0]
        base_mesh_name = Path(base_stl_file).stem
        logger.info(f"Starting draft generation mode for base mesh: {base_stl_file}")

        for draft in settings.drafts:
            logger.info(f"Processing for draft: {draft}")

            # Create a copy of the settings to modify for this specific draft
            draft_settings = settings.model_copy(deep=True)

            # Combine the draft with the existing z-translation
            # A positive draft means sinking the vessel, so we subtract it.
            draft_settings.translation_z -= draft

            # Create a unique name for this draft-specific mesh configuration
            draft_str = _format_value_for_name(draft)
            mesh_name_for_draft = f"{base_mesh_name}_draft_{draft_str}"

            # Process this specific configuration
            _process_single_stl(base_stl_file, draft_settings, output_file, mesh_name_override=mesh_name_for_draft)

    else:
        # Standard mode: process files as they are
        logger.info("Starting standard processing for provided STL files.")
        for stl_file in settings.stl_files:
            # In standard mode, also apply the translation settings
            _process_single_stl(stl_file, settings, output_file, mesh_name_override=None)

    logger.info(f"✅ Simulation batch finished. Results saved to {output_file}")
