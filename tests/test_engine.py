import hashlib
import logging
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import h5py
import numpy as np
import pandas as pd
import pytest
import trimesh
import xarray as xr

from fleetmaster.core.engine import (
    _format_value_for_name,
    _generate_case_group_name,
    _prepare_capytaine_body,
    _process_single_stl,
    _setup_output_file,
    add_mesh_to_database,
    make_database,
    run_simulation_batch,
)
from fleetmaster.core.exceptions import LidAndSymmetryEnabledError
from fleetmaster.core.settings import SimulationSettings


@pytest.fixture
def mock_settings():
    """Fixture for SimulationSettings with default values."""
    return SimulationSettings(
        stl_files=["/path/to/dummy.stl"],
        output_directory=None,
        output_hdf5_file="output.hdf5",
        overwrite_meshes=False,
        wave_periods=[1.0],
        wave_directions=[0.0],
        water_depth=np.inf,
        water_level=0.0,
        forward_speed=0.0,
        add_center_of_mass=False,
        lid=False,
        grid_symmetry=False,
        update_cases=False,
        combine_cases=False,
        drafts=None,
    )


@patch("fleetmaster.core.engine.cpt")
def test_make_database_rename_and_convert(mock_cpt):
    """Test that make_database renames phony dims and converts categorical dtypes."""
    # Arrange
    mock_body = MagicMock()
    mock_body.dofs = ["Heave", "Pitch"]
    mock_cpt.assemble_dataset.return_value = xr.Dataset(
        coords={
            "phony_dim_0": [0, 1],
            "phony_dim_1": [0, 1],
            "radiating_dof": pd.Categorical(["Heave", "Pitch"]),
        }
    ).rename({"phony_dim_0": "phony_dim_0", "phony_dim_1": "phony_dim_1"})

    # Act
    dataset = make_database(mock_body, [1.0], [0.0], np.inf, 0.0, 0.0)

    # Assert
    assert "i" in dataset.dims
    assert "j" in dataset.dims
    assert "phony_dim_0" not in dataset.dims
    assert "phony_dim_1" not in dataset.dims
    assert dataset["radiating_dof"].dtype.kind == "U"  # converted to string


def test_setup_output_file_no_stl_files(mock_settings):
    """Test _setup_output_file raises ValueError when no STL files are provided."""
    mock_settings.stl_files = []
    with pytest.raises(ValueError, match=r"No STL files provided to process."):
        _setup_output_file(mock_settings)


def test_setup_output_file_overwrite(tmp_path, mock_settings):
    """Test _setup_output_file overwrites existing file when specified."""
    mock_settings.output_directory = str(tmp_path)
    mock_settings.overwrite_meshes = True
    output_file = tmp_path / "output.hdf5"
    output_file.touch()

    result_path = _setup_output_file(mock_settings)

    assert result_path == output_file
    assert not output_file.exists()


def test_setup_output_file_no_overwrite(tmp_path, mock_settings):
    """Test _setup_output_file does not delete existing file by default."""
    mock_settings.output_directory = str(tmp_path)
    mock_settings.overwrite_meshes = False
    output_file = tmp_path / "output.hdf5"
    output_file.touch()

    result_path = _setup_output_file(mock_settings)

    assert result_path == output_file
    assert output_file.exists()


@patch("fleetmaster.core.engine.cpt")
@patch("fleetmaster.core.engine.tempfile")
def test_prepare_capytaine_body(mock_tempfile, mock_cpt, tmp_path: Path):
    """Test _prepare_capytaine_body configures the body correctly."""
    # Arrange
    mock_source_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_source_mesh.center_mass = [1, 2, 3]

    # Create a real temporary file path for the test to use
    temp_file_path = tmp_path / "temp.stl"
    mock_tempfile.mkstemp.return_value = (123, str(temp_file_path))

    mock_hull_mesh = MagicMock()
    mock_cpt.load_mesh.return_value = mock_hull_mesh

    mock_body = MagicMock()
    mock_cpt.FloatingBody.return_value = mock_body

    # To make `isinstance(boat.mesh, cpt.meshes.ReflectionSymmetricMesh)` work,
    # we define a dummy class and configure the mock to use it. This avoids
    # `isinstance()` being called with a mock, which would raise a TypeError.
    class _DummySymmetricMesh:
        def to_mesh(self):
            return MagicMock()

    mock_cpt.meshes.ReflectionSymmetricMesh = _DummySymmetricMesh
    mock_cpt.ReflectionSymmetricMesh.return_value = _DummySymmetricMesh()

    # Act
    body, _ = _prepare_capytaine_body(
        source_mesh=mock_source_mesh, mesh_name="test_mesh", lid=True, grid_symmetry=True, add_center_of_mass=True
    )

    # Assert
    mock_source_mesh.export.assert_called_once()
    mock_cpt.load_mesh.assert_called_once()
    mock_hull_mesh.generate_lid.assert_called_once()
    mock_cpt.ReflectionSymmetricMesh.assert_called_once()
    mock_cpt.FloatingBody.assert_called_once_with(
        mesh=ANY, lid_mesh=mock_hull_mesh.generate_lid.return_value, center_of_mass=[1, 2, 3]
    )
    mock_body.add_all_rigid_body_dofs.assert_called_once()
    mock_body.keep_immersed_part.assert_called_once()
    assert body == mock_body


def test_add_mesh_to_database_new(tmp_path):
    """Test adding a new mesh to the HDF5 database."""
    output_file = tmp_path / "db.h5"
    stl_content = b"This is a dummy stl file content"

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.volume = 1.0
    mock_mesh.center_mass = [0.1, 0.2, 0.3]
    mock_mesh.bounding_box.extents = [1.0, 2.0, 3.0]
    mock_mesh.moment_inertia = np.eye(3)
    mock_mesh.export.return_value = stl_content

    add_mesh_to_database(output_file, mock_mesh, "mesh", overwrite=False)

    with h5py.File(output_file, "r") as f:
        group = f["meshes/mesh"]
        assert "sha256" in group.attrs
        assert group.attrs["volume"] == 1.0
        assert group.attrs["cog_x"] == 0.1
        assert "inertia_tensor" in group  # type: ignore[reportOperatorIssue]
        assert "stl_content" in group  # type: ignore[reportOperatorIssue]
        assert group["stl_content"][()].tobytes() == stl_content  # type: ignore[reportAttributeAccessIssue]


def test_add_mesh_to_database_skip_existing(tmp_path, caplog):
    """Test that an existing mesh with the same hash is skipped."""
    output_file = tmp_path / "db.h5"
    stl_content = b"dummy stl"
    file_hash = hashlib.sha256(stl_content).hexdigest()

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = stl_content

    with h5py.File(output_file, "w") as f:
        group = f.create_group("meshes/mesh")
        group.attrs["sha256"] = file_hash

    with caplog.at_level(logging.INFO):
        add_mesh_to_database(output_file, mock_mesh, "mesh", overwrite=False)

    assert "has the same SHA256 hash. Skipping." in caplog.text


def test_add_mesh_to_database_overwrite_warning(tmp_path, caplog):
    """Test warning when mesh is different and overwrite is False."""
    output_file = tmp_path / "db.h5"
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"new content"

    with h5py.File(output_file, "w") as f:
        group = f.create_group("meshes/mesh")
        group.attrs["sha256"] = "old_hash"

    with caplog.at_level(logging.WARNING):
        add_mesh_to_database(output_file, mock_mesh, "mesh", overwrite=False)

    assert "is different from the one in the database" in caplog.text


@pytest.mark.parametrize(
    "value, expected",
    [
        (10.0, "10"),
        (10.5, "10.5"),
        (10.55, "10.6"),
        (np.inf, "inf"),
    ],
)
def test_format_value_for_name(value, expected):
    assert _format_value_for_name(value) == expected


def test_generate_case_group_name():
    """Test the generation of a descriptive group name for a simulation case."""
    name = _generate_case_group_name("my_mesh", 100.0, 0.5, 2.0)
    assert name == "my_mesh_wd_100_wl_0.5_fs_2"


@patch("fleetmaster.core.engine.process_all_cases_for_one_stl")
def test_process_single_stl(mock_process_all, mock_settings):
    """Test the main processing pipeline for a single STL file."""
    stl_file = "/path/to/dummy.stl"
    output_file = Path("/fake/output.hdf5")

    _process_single_stl(stl_file, mock_settings, output_file)

    mock_process_all.assert_called_once()
    _, kwargs = mock_process_all.call_args
    assert kwargs["stl_file"] == stl_file
    assert kwargs["output_file"] == output_file


def test_process_single_stl_lid_and_symmetry_error(mock_settings):
    """Test that LidAndSymmetryEnabledError is raised if both are enabled."""
    mock_settings.lid = True
    mock_settings.grid_symmetry = True
    with pytest.raises(LidAndSymmetryEnabledError):
        _process_single_stl("file.stl", mock_settings, Path("out.h5"))


@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file")
def test_run_simulation_batch_standard(mock_setup, mock_process, mock_settings):
    """Test run_simulation_batch in standard mode."""
    mock_settings.stl_files = ["file1.stl", "file2.stl"]
    output_file = Path("/fake/output.hdf5")
    mock_setup.return_value = output_file

    run_simulation_batch(mock_settings)

    mock_setup.assert_called_once_with(mock_settings)
    assert mock_process.call_count == 2
    mock_process.assert_has_calls([
        call("file1.stl", mock_settings, output_file, mesh_name_override=None),
        call("file2.stl", mock_settings, output_file, mesh_name_override=None),
    ])


@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file", autospec=True)
def test_run_simulation_batch_drafts(mock_setup, mock_process, mock_settings, tmp_path: Path):
    """Test run_simulation_batch in draft generation mode."""
    # Arrange
    mock_setup.return_value = tmp_path / "output.hdf5"

    mock_settings.stl_files = ["base_mesh.stl"]
    mock_settings.drafts = [1.0, 2.5]
    mock_settings.translation_z = 5.0

    # Act
    run_simulation_batch(mock_settings)

    # Assert
    assert mock_process.call_count == 2

    # Check call for first draft
    args1, kwargs1 = mock_process.call_args_list[0]
    assert args1[0] == "base_mesh.stl"
    assert args1[1].translation_z == 4.0  # 5.0 - 1.0
    assert kwargs1["mesh_name_override"] == "base_mesh_draft_1"

    # Check call for second draft
    args2, kwargs2 = mock_process.call_args_list[1]
    assert args2[0] == "base_mesh.stl"
    assert args2[1].translation_z == 2.5  # 5.0 - 2.5
    assert kwargs2["mesh_name_override"] == "base_mesh_draft_2.5"


def test_run_simulation_batch_drafts_wrong_stl_count(mock_settings):
    """Test that draft mode raises an error if more than one STL is provided."""
    mock_settings.drafts = [1.0]
    mock_settings.stl_files = ["file1.stl", "file2.stl"]
    with pytest.raises(ValueError, match="exactly one base STL file must be provided"):
        run_simulation_batch(mock_settings)
