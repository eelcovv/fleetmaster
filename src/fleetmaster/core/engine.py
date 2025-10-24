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
from .io import load_meshes_from_hdf5
from .settings import MESH_GROUP_NAME, MeshConfig, SimulationSettings

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

    first_stl_entry = settings.stl_files[0]
    first_stl_path = first_stl_entry if isinstance(first_stl_entry, str) else first_stl_entry.file
    output_dir = Path(settings.output_directory) if settings.output_directory else Path(first_stl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / settings.output_hdf5_file
    if output_file.exists() and settings.overwrite_meshes:
        logger.warning(f"Output file {output_file} already exists and will be overwritten as overwrite_meshes is True.")
        output_file.unlink()
    return output_file


def _prepare_trimesh_geometry(
    stl_file: str,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    translation_z: float = 0.0,
) -> trimesh.Trimesh:
    """
    Loads an STL file and applies specified translations.

    Returns:
        A trimesh.Trimesh object representing the transformed geometry.
    """
    transformed_mesh = trimesh.load_mesh(stl_file)

    # Apply translation if specified
    if translation_x != 0.0 or translation_y != 0.0 or translation_z != 0.0:
        translation_vector = np.array([translation_x, translation_y, translation_z])
        logger.debug(f"Applying mesh translation: {translation_vector}")
        transform_matrix = trimesh.transformations.translation_matrix(translation_vector)
        transformed_mesh.apply_transform(transform_matrix)

    return transformed_mesh


def _prepare_capytaine_body(
    source_mesh: trimesh.Trimesh,
    mesh_name: str,
    mesh_config: MeshConfig,
    lid: bool,
    grid_symmetry: bool,
    add_center_of_mass: bool = False,
) -> tuple[Any, trimesh.Trimesh | None]:
    """
    Configures a Capytaine FloatingBody from a pre-prepared trimesh object.

    The `center_of_mass` for Capytaine is determined by `mesh_config.cog`,
    falling back to the mesh's geometric center of mass. The `add_center_of_mass`
    flag can be used to disable this fallback and default to (0,0,0) if no
    `cog` is specified.
    """
    cog = None

    if mesh_config.cog:
        cog = np.array(mesh_config.cog)
        logger.debug(f"Using specified COG {cog} as the center of mass for Capytaine.")
    else:
        # If no local_origin is specified, use the center of mass of the (already translated) source_mesh.
        cog = source_mesh.center_mass
        logger.debug(f"Using geometric center of mass {cog} of the translated mesh for Capytaine.")

    # 1. Save the transformed mesh to a temporary file and load it with Capytaine.
    # This is more robust than creating a cpt.Mesh from vertices/faces directly.
    # We use NamedTemporaryFile to handle creation and cleanup automatically.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            # Step 1: Write to the temporary file.
            source_mesh.export(temp_file, file_type="stl")
            logger.debug(f"Exported transformed mesh to temporary file: {temp_path}")

        # Step 2: Read from the now-closed temporary file. This avoids race conditions.
        hull_mesh = cpt.load_mesh(str(temp_path), name=mesh_name)

    finally:
        # Step 3: Ensure the temporary file is always deleted, even if an error occurs.
        if temp_path and temp_path.exists():
            logger.debug(f"Deleting temporary file: {temp_path}")
            temp_path.unlink()

    # 4. Configure the Capytaine FloatingBody
    lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None
    if grid_symmetry:
        logger.debug("Applying grid symmetery")
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane)

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh, center_of_mass=cog)
    boat.add_all_rigid_body_dofs()
    boat.keep_immersed_part()

    # 5. Extract the final mesh that Capytaine will use for the database. After keep_immersed_part,
    # boat.mesh contains the correct vertices and faces for both regular and symmetric meshes.
    final_mesh_trimesh = trimesh.Trimesh(vertices=boat.mesh.vertices, faces=boat.mesh.faces)

    return boat, final_mesh_trimesh


def add_mesh_to_database(
    output_file: Path, mesh_to_add: trimesh.Trimesh, mesh_name: str, overwrite: bool = False
) -> None:
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
        # We must wrap the bytes in np.void to store it as opaque binary data,
        # otherwise h5py tries to interpret it as a string and fails on NULL bytes.
        group.create_dataset("stl_content", data=np.void(new_stl_content))
        logger.debug("  - Wrote dataset: stl_content")


def _format_value_for_name(value: float) -> str:
    """Formats a float for use in a group name."""
    if value == np.inf:
        return "inf"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _generate_case_group_name(mesh_name: str, water_depth: float, water_level: float, forward_speed: float) -> str:
    """Generates a descriptive group name for a specific simulation case."""
    wd = _format_value_for_name(water_depth)
    wl = _format_value_for_name(water_level)
    fs = _format_value_for_name(forward_speed)
    return f"{mesh_name}_wd_{wd}_wl_{wl}_fs_{fs}"


def _process_single_stl(
    mesh_config: MeshConfig,
    settings: SimulationSettings,
    output_file: Path,
    mesh_name_override: str | None = None,
    origin_translation: npt.NDArray[np.float64] | None = None,
) -> None:
    """
    Checks if a mesh exists in the database. If so, uses it.
    If not, generates it, saves it, and then uses it for the simulation pipeline.
    """
    mesh_name = mesh_name_override or Path(mesh_config.file).stem
    final_mesh_to_process: trimesh.Trimesh | None = None

    # --- Workflow to determine the mesh to process ---
    # 1. Prioritize loading from the HDF5 database if it already exists.
    try:
        existing_meshes = load_meshes_from_hdf5(output_file, [mesh_name])
        if existing_meshes:
            final_mesh_to_process = existing_meshes[0]
            logger.info(f"Found existing mesh '{mesh_name}' in the database. Using it directly.")
    except FileNotFoundError:
        # The HDF5 file doesn't exist yet, so no meshes can exist. This is expected on the first run.
        pass

    # 2. If not in DB, check if a pre-translated STL file exists.
    if final_mesh_to_process is None:
        target_stl_path = Path(mesh_config.file)
        if target_stl_path.exists():
            logger.info(
                f"Mesh '{mesh_name}' not in DB, but found STL file: '{target_stl_path}'. Loading and adding to DB."
            )
            # Load the existing, presumably pre-translated, STL file.
            final_mesh_to_process = _prepare_trimesh_geometry(stl_file=str(target_stl_path))
        else:
            # 3. If neither DB entry nor STL file exists, generate the mesh.
            logger.info(f"Mesh '{mesh_name}' not found in DB or as STL file. Attempting to generate it.")
            # Use the global base_mesh as the source for generation.
            source_file_path = settings.base_mesh
            if not source_file_path or not Path(source_file_path).exists():
                err_msg = (
                    f"Cannot generate mesh '{mesh_name}'. The source file '{target_stl_path}' does not exist, "
                    f"and no valid 'base_mesh' ('{source_file_path}') is configured to generate it from."
                )
                raise FileNotFoundError(err_msg)

            # Load the base STL and apply the specified translation.
            translated_mesh = _prepare_trimesh_geometry(
                stl_file=str(source_file_path),
                translation_x=mesh_config.translation[0],
                translation_y=mesh_config.translation[1],
                translation_z=mesh_config.translation[2],
            )
            # Save the newly generated, translated mesh to a separate STL file for inspection.
            logger.info(f"Saving newly generated translated mesh to: {target_stl_path}")
            translated_mesh.export(target_stl_path)
            final_mesh_to_process = translated_mesh

    # 4. Run the complete processing pipeline with the determined mesh.
    _run_pipeline_for_mesh(final_mesh_to_process, mesh_config, settings, output_file, mesh_name, origin_translation)


def _run_pipeline_for_mesh(
    source_mesh: trimesh.Trimesh,
    mesh_config: MeshConfig,
    settings: SimulationSettings,
    output_file: Path,
    mesh_name: str,
    origin_translation: npt.NDArray[np.float64] | None,
) -> None:
    """
    Run the complete processing pipeline for a single STL file.
    """
    logger.info(f"Processing STL file: {mesh_config.file}")

    # check is done by Settings, so this should no happen anymore
    if settings.lid and settings.grid_symmetry:
        raise LidAndSymmetryEnabledError()

    # Use mesh-specific wave periods and directions if provided, otherwise fall back to global settings.
    periods_to_use = mesh_config.wave_periods or settings.wave_periods
    wave_periods = periods_to_use if isinstance(periods_to_use, list) else [periods_to_use]
    wave_frequencies = (2 * np.pi / np.array(wave_periods)).tolist()

    directions_to_use = mesh_config.wave_directions or settings.wave_directions
    wave_directions_deg = directions_to_use if isinstance(directions_to_use, list) else [directions_to_use]
    wave_directions_rad = np.deg2rad(wave_directions_deg).tolist()

    water_depths = settings.water_depth if isinstance(settings.water_depth, list) else [settings.water_depth]
    water_levels = settings.water_level if isinstance(settings.water_level, list) else [settings.water_level]
    forwards_speeds = settings.forward_speed if isinstance(settings.forward_speed, list) else [settings.forward_speed]

    forwards_speeds = settings.forward_speed if isinstance(settings.forward_speed, list) else [settings.forward_speed]

    add_center_of_mass = settings.add_center_of_mass
    lid = settings.lid
    grid_symmetry = settings.grid_symmetry

    output_file = output_file

    fmt_str = "%-40s: %s"
    logger.info(fmt_str % ("Base STL file", mesh_config.file))
    logger.info(fmt_str % ("Output file", output_file))
    logger.info(fmt_str % ("Grid symmetry", grid_symmetry))
    logger.info(fmt_str % ("Use lid", lid))
    logger.info(fmt_str % ("Add COG ", add_center_of_mass))
    logger.info(fmt_str % ("Direction(s) [rad]", wave_directions_rad))
    logger.info(fmt_str % ("Wave period(s) [s]", wave_periods))
    logger.info(fmt_str % ("Water depth(s) [m]", water_depths))
    logger.info(fmt_str % ("Water level(s) [m]", water_levels))
    logger.info(fmt_str % ("Translation X", mesh_config.translation[0]))
    logger.info(fmt_str % ("Translation Y", mesh_config.translation[1]))
    logger.info(fmt_str % ("Translation Z", mesh_config.translation[2]))
    logger.info(fmt_str % ("Forward speed(s) [m/s]", forwards_speeds))

    process_all_cases_for_one_stl(
        source_mesh=source_mesh,
        wave_frequencies=wave_frequencies,
        wave_directions=wave_directions_rad,
        water_depths=water_depths,
        water_levels=water_levels,
        forwards_speeds=forwards_speeds,
        lid=lid,
        add_center_of_mass=add_center_of_mass,
        grid_symmetry=grid_symmetry,
        output_file=output_file,
        update_cases=settings.update_cases,
        combine_cases=settings.combine_cases,
        mesh_name=mesh_name,
        mesh_config=mesh_config,
        origin_translation=origin_translation,
    )


def _process_and_save_single_case(
    boat: Any,  # cpt.FloatingBody is not fully typed, use Any to satisfy mypy
    mesh_name: str,
    case_params: dict[str, Any],
    output_file: Path,
    origin_translation: npt.NDArray[np.float64] | None,
) -> Any:
    """Process a single simulation case and save its results to the HDF5 file."""
    group_name = _generate_case_group_name(
        mesh_name, case_params["water_depth"], case_params["water_level"], case_params["forward_speed"]
    )

    with h5py.File(output_file, "a") as f:
        if group_name in f:
            if not case_params["update_cases"]:
                logger.info(f"Case '{group_name}' already exists in the database. Skipping.")
                return None
            logger.info(f"Case '{group_name}' exists, but update_cases is True. Overwriting.")
            del f[group_name]

    # Calculate the transformation matrix for this specific case relative to the global origin
    transformation_matrix = None
    if origin_translation is not None:
        # The transformation is the translation from the global origin to the mesh's COG for this case.
        # Note: boat.center_of_mass is the COG used for calculation, not necessarily the geometric center.
        translation_vector = boat.center_of_mass - origin_translation
        transformation_matrix = trimesh.transformations.translation_matrix(translation_vector)

    logger.info(
        f"Starting BEM calculations for water_level={case_params['water_level']}, "
        f"water_depth={case_params['water_depth']}, forward_speed={case_params['forward_speed']}"
    )
    # Select only the parameters that make_database expects.
    db_params = {
        "omegas": case_params["omegas"],
        "wave_directions": case_params["wave_directions"],
        "water_depth": case_params["water_depth"],
        "water_level": case_params["water_level"],
        "forward_speed": case_params["forward_speed"],
    }
    database = make_database(body=boat, **db_params)

    if not case_params["combine_cases"]:
        logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
        database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")
        with h5py.File(output_file, "a") as f:
            if group_name in f:
                case_group = f[group_name]
                case_group.attrs["stl_mesh_name"] = mesh_name
                if transformation_matrix is not None:
                    case_group.attrs["transformation_matrix"] = transformation_matrix
                if boat.center_of_mass is not None:
                    case_group.attrs["cog_for_calculation"] = boat.center_of_mass

    logger.debug(f"Successfully wrote data for case to group {group_name}.")
    return database


def process_all_cases_for_one_stl(
    source_mesh: trimesh.Trimesh,
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
    mesh_name: str | None = None,
    mesh_config: MeshConfig | None = None,
    origin_translation: npt.NDArray[np.float64] | None = None,
) -> None:
    if mesh_name is None:
        # This should ideally not happen if called correctly, but it satisfies mypy.
        logger.error("process_all_cases_for_one_stl called without a mesh_name.")
        return

    if mesh_config is None:
        # This should ideally not happen if called correctly, but it satisfies mypy.
        logger.error("process_all_cases_for_one_stl called without a mesh_config.")
        return

    # 1. Use the prepared (and possibly translated) geometry to create the Capytaine body
    boat, final_mesh = _prepare_capytaine_body(
        source_mesh=source_mesh,
        mesh_name=mesh_name,
        mesh_config=mesh_config,
        lid=lid,
        grid_symmetry=grid_symmetry,
        add_center_of_mass=add_center_of_mass,
    )

    # 2. Add the final, immersed mesh geometry to the database. This version is now the translated one.
    add_mesh_to_database(output_file, final_mesh, mesh_name, overwrite=update_cases)

    all_datasets = []

    for water_level in water_levels:
        for water_depth in water_depths:
            for forward_speed in forwards_speeds:
                case_params = {
                    "omegas": wave_frequencies,
                    "wave_directions": wave_directions,
                    "water_level": water_level,
                    "water_depth": water_depth,
                    "forward_speed": forward_speed,
                    "update_cases": update_cases,
                    "combine_cases": combine_cases,
                }
                result_db = _process_and_save_single_case(boat, mesh_name, case_params, output_file, origin_translation)
                if combine_cases and result_db is not None:
                    all_datasets.append(result_db)

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

    logger.debug(f"Successfully wrote all data for mesh '{mesh_name}' to HDF5.")


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

    # Determine the base mesh and the origin translation
    all_mesh_configs = [MeshConfig.model_validate(mc) for mc in settings.stl_files]
    all_files = [mc.file for mc in all_mesh_configs]

    origin_translation = np.array([0.0, 0.0, 0.0])
    base_mesh_path = settings.base_mesh
    if not base_mesh_path and all_files:
        base_mesh_path = all_files[0]

    if base_mesh_path:
        if settings.base_origin:
            # If base_origin is specified, it's a point in the local coordinates of the base_mesh.
            # This point becomes the origin of our world coordinate system.
            origin_translation = np.array(settings.base_origin)
            logger.info(f"Using local point {origin_translation} from '{base_mesh_path}' as the world origin.")
        else:
            # Fallback to using the center of mass if base_origin is not provided.
            logger.info(f"Using center of mass of '{base_mesh_path}' to define the coordinate origin.")
            base_mesh_for_origin = _prepare_trimesh_geometry(base_mesh_path)
            origin_translation = base_mesh_for_origin.center_mass
            logger.info(f"Database origin (center of mass of base mesh) set to: {origin_translation}")

        # Store the base reference information in the root of the HDF5 file
        with h5py.File(output_file, "a") as f:
            f.attrs["base_mesh"] = Path(base_mesh_path).name
            if settings.base_origin:
                f.attrs["base_origin"] = settings.base_origin
            else:
                f.attrs["base_origin"] = origin_translation  # Store the calculated CoM as origin

    if settings.drafts and base_mesh_path:
        if len(all_files) != 1:
            msg = f"When using --drafts, exactly one base STL file must be provided, but {len(all_files)} were given."
            logger.error(msg)
            raise ValueError(msg)

        logger.info(f"Starting draft generation mode for base mesh: {base_mesh_path}")

        base_mesh_name = Path(base_mesh_path).stem
        for draft in settings.drafts:
            logger.info(f"Processing for draft: {draft}")

            # Create a copy of the settings to modify for this specific draft
            draft_settings = settings.model_copy(deep=True)

            # Create a MeshConfig for this specific draft
            base_mesh_config = next((mc for mc in all_mesh_configs if mc.file == base_mesh_path), None)
            draft_translation = base_mesh_config.translation.copy() if base_mesh_config else [0.0, 0.0, 0.0]
            draft_translation[2] -= draft  # Positive draft means sinking, so subtract from Z

            # Create a unique name for this draft-specific mesh configuration
            draft_str = _format_value_for_name(draft)
            mesh_name_for_draft = f"{base_mesh_name}_draft_{draft_str}"

            draft_mesh_config = MeshConfig(file=base_mesh_path, translation=draft_translation)

            # Process this specific configuration
            _process_single_stl(
                draft_mesh_config,
                draft_settings,
                output_file,
                mesh_name_override=mesh_name_for_draft,
                origin_translation=origin_translation,
            )

    else:
        # Standard mode: process files as they are
        logger.info("Starting standard processing for provided STL files.")
        for mesh_config in all_mesh_configs:
            _process_single_stl(
                mesh_config, settings, output_file, mesh_name_override=None, origin_translation=origin_translation
            )

    logger.info(f"✅ Simulation batch finished. Results saved to {output_file}")
