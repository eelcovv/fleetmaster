from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh

from fleetmaster.core.engine import (
    EngineMesh,
    _format_value_for_name,
    _generate_case_group_name,
    _prepare_capytaine_body,
    _process_single_stl,
    _run_pipeline_for_mesh,
    _setup_output_file,
    add_mesh_to_database,
    run_simulation_batch,
)
from fleetmaster.core.exceptions import LidAndSymmetryEnabledError
from fleetmaster.core.settings import MeshConfig, SimulationSettings


@pytest.fixture
def mock_settings():
    """Provides a default SimulationSettings object for tests."""
    return SimulationSettings(
        stl_files=[MeshConfig(file="/path/to/dummy.stl")],  # Default value, can be overridden
        water_depth=np.inf,
        water_level=0.0,
        wave_periods=[1.0],
        wave_directions=[0.0],
    )


def test_setup_output_file_no_stl_files(mock_settings):
    """Test that _setup_output_file raises ValueError if no STL files are provided."""
    mock_settings.stl_files = []
    with pytest.raises(ValueError, match=r"No STL files provided to process\."):
        _setup_output_file(mock_settings)


def test_setup_output_file_overwrite(tmp_path, mock_settings):
    """Test that an existing output file is deleted if overwrite_meshes is True."""
    mock_settings.overwrite_meshes = True
    mock_settings.output_hdf5_file = "test.h5"
    mock_settings.output_directory = str(tmp_path)
    output_file = tmp_path / "test.h5"
    output_file.touch()  # Create the file

    result_path = _setup_output_file(mock_settings)

    assert not output_file.exists()  # Should have been deleted
    assert result_path == tmp_path / "test.h5"


def test_setup_output_file_no_overwrite(tmp_path, mock_settings):
    """Test that an existing output file is kept if overwrite_meshes is False."""
    mock_settings.overwrite_meshes = False
    mock_settings.output_hdf5_file = "test.h5"
    mock_settings.stl_files = [MeshConfig(file=str(tmp_path / "dummy.stl"))]
    output_file = tmp_path / "test.h5"
    output_file.touch()  # Create the file

    _setup_output_file(mock_settings)

    assert output_file.exists()  # Should NOT have been deleted


@patch("fleetmaster.core.engine.cpt")
@patch("fleetmaster.core.engine.tempfile")
def test_prepare_capytaine_body(mock_tempfile, mock_cpt, tmp_path: Path):
    """Test _prepare_capytaine_body configures the body correctly."""
    # Arrange
    mock_source_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_source_mesh.center_mass = [1, 2, 3]
    mock_source_mesh.vertices = np.array([[1, 1, 1]])
    mock_source_mesh.faces = np.array([[0, 0, 0]])

    # Mock the temporary file creation
    mock_temp_file_handle = MagicMock()
    mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_handle.name = str(tmp_path / "temp.stl")

    mock_hull_mesh = MagicMock()
    mock_cpt.load_mesh.return_value = mock_hull_mesh

    mock_body = MagicMock()
    mock_cpt.FloatingBody.return_value = mock_body

    mock_mesh_config = MagicMock(spec=MeshConfig)
    mock_mesh_config.cog = None

    mock_engine_mesh = EngineMesh(name="test_mesh", mesh=mock_source_mesh, config=mock_mesh_config)

    # Act
    body, final_mesh = _prepare_capytaine_body(
        engine_mesh=mock_engine_mesh,
        lid=True,
        grid_symmetry=False,
    )

    # Assert
    assert body == mock_body
    mock_cpt.FloatingBody.assert_called_once()
    # Check that the center of mass from the mesh was used as a fallback
    assert np.array_equal(mock_cpt.FloatingBody.call_args.kwargs["center_of_mass"], [1, 2, 3])
    mock_hull_mesh.generate_lid.assert_called_once()
    body.keep_immersed_part.assert_called_once_with(free_surface=0.0)
    body.add_all_rigid_body_dofs.assert_called_once()
    assert final_mesh is not None


@patch("fleetmaster.core.engine.cpt")
@patch("fleetmaster.core.engine.tempfile")
def test_prepare_capytaine_body_with_symmetry(mock_tempfile, mock_cpt, tmp_path: Path):
    """Test that grid_symmetry correctly wraps the mesh in ReflectionSymmetricMesh."""
    # Arrange
    mock_source_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_source_mesh.center_mass = [1, 2, 3]
    mock_source_mesh.vertices = np.array([[1, 1, 1]])
    mock_source_mesh.faces = np.array([[0, 0, 0]])

    mock_temp_file_handle = MagicMock()
    mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_handle.name = str(tmp_path / "temp.stl")

    mock_base_hull_mesh = MagicMock()
    mock_cpt.load_mesh.return_value = mock_base_hull_mesh

    mock_symmetric_mesh = MagicMock()
    mock_cpt.ReflectionSymmetricMesh.return_value = mock_symmetric_mesh

    mock_mesh_config = MagicMock(spec=MeshConfig)
    mock_mesh_config.cog = None

    mock_engine_mesh = EngineMesh(name="test_mesh", mesh=mock_source_mesh, config=mock_mesh_config)

    # Act
    _prepare_capytaine_body(engine_mesh=mock_engine_mesh, lid=False, grid_symmetry=True)

    # Assert
    mock_cpt.ReflectionSymmetricMesh.assert_called_once_with(mock_base_hull_mesh, plane=mock_cpt.xOz_Plane)
    # Check that the FloatingBody was created with the symmetric mesh
    mock_cpt.FloatingBody.assert_called_once()
    assert mock_cpt.FloatingBody.call_args.kwargs["mesh"] == mock_symmetric_mesh


@patch("h5py.File")
def test_add_mesh_to_database_overwrite_warning(mock_h5py_file, caplog):
    """Test that a warning is logged when a different mesh with the same name exists and overwrite is False."""
    # Arrange
    mesh = trimesh.creation.box()

    mock_existing_group = MagicMock()
    mock_existing_group.attrs.get.return_value = "different_hash"
    mock_file = MagicMock()
    mock_file.__contains__.return_value = True
    mock_file.__getitem__.return_value = mock_existing_group
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    # Act
    add_mesh_to_database(Path("test.h5"), mesh, "test_mesh", overwrite=False)

    # Assert
    assert "is different from the one in the database" in caplog.text
    mock_file.create_group.assert_not_called()


@pytest.mark.parametrize(
    "value, expected",
    [(10.0, "10"), (10.5, "10.5"), (10.55, "10.6"), (np.inf, "inf")],
)
def test_format_value_for_name(value, expected):
    assert _format_value_for_name(value) == expected


def test_generate_case_group_name():
    name = _generate_case_group_name("mesh1", 100.0, -2.5, 5.0)
    assert name == "mesh1_wd_100_wl_-2.5_fs_5"


@patch("fleetmaster.core.engine._run_pipeline_for_mesh")
@patch("fleetmaster.core.engine.load_meshes_from_hdf5", return_value=[])
@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("pathlib.Path.exists", return_value=True)
def test_process_single_stl(mock_exists, mock_prepare, mock_load, mock_run_pipeline):
    """Test the main processing pipeline for a single STL file."""
    mesh_config = MeshConfig(file="/path/to/dummy.stl")
    settings = SimulationSettings(stl_files=[mesh_config])
    output_file = Path("/fake/output.hdf5")

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_prepare.return_value = mock_mesh

    _process_single_stl(mesh_config, settings, output_file)

    mock_run_pipeline.assert_called_once()
    # Check that the EngineMesh passed to the pipeline has the correct components
    call_args = mock_run_pipeline.call_args[0]
    engine_mesh_arg = call_args[0]
    assert isinstance(engine_mesh_arg, EngineMesh)
    assert engine_mesh_arg.name == "dummy"
    assert engine_mesh_arg.mesh == mock_mesh
    assert engine_mesh_arg.config == mesh_config


def test_process_single_stl_lid_and_symmetry_error(mock_settings):
    """Test that LidAndSymmetryEnabledError is raised if both are enabled."""
    mock_settings.lid = True
    mock_settings.grid_symmetry = True
    mesh_config = MeshConfig(file="file.stl")
    # This test doesn't need to go deep into the pipeline, it should fail early.
    # We just need to check that _run_pipeline_for_mesh raises the error.
    with pytest.raises(LidAndSymmetryEnabledError):
        _run_pipeline_for_mesh(
            EngineMesh(name="test", mesh=MagicMock(), config=mesh_config),
            mock_settings,
            Path("out.h5"),
            origin_translation=None,
        )


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file")
def test_run_simulation_batch_standard(mock_setup, mock_process, mock_prepare, mock_settings, tmp_path: Path):
    mock_settings.stl_files = [MeshConfig(file="file1.stl"), MeshConfig(file="file2.stl")]
    output_file = tmp_path / "output.hdf5"
    mock_setup.return_value = output_file

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"
    mock_prepare.return_value = mock_mesh

    run_simulation_batch(mock_settings)

    assert mock_process.call_count == 2


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file", autospec=True)
def test_run_simulation_batch_drafts(mock_setup, mock_process, mock_prepare, mock_settings, tmp_path: Path):
    """Test run_simulation_batch in draft generation mode."""
    # Arrange
    mock_setup.return_value = tmp_path / "output.hdf5"
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"
    mock_prepare.return_value = mock_mesh

    mock_settings.stl_files = [MeshConfig(file="base_mesh.stl", translation=[0, 0, 5])]
    mock_settings.drafts = [1.0, 2.5]

    # Act
    run_simulation_batch(mock_settings)

    # Assert
    assert mock_process.call_count == 2
    # Check the first call for draft 1.0
    call1 = mock_process.call_args_list[0]
    config_draft1 = call1.args[0]
    assert isinstance(config_draft1, MeshConfig)
    assert config_draft1.translation == [0.0, 0.0, 4.0]  # 5.0 - 1.0
    assert call1.kwargs["mesh_name_override"] == "base_mesh_draft_1"

    # Check the second call for draft 2.5
    call2 = mock_process.call_args_list[1]
    config_draft2 = call2.args[0]
    assert config_draft2.translation == [0.0, 0.0, 2.5]  # 5.0 - 2.5
    assert call2.kwargs["mesh_name_override"] == "base_mesh_draft_2.5"


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
def test_run_simulation_batch_drafts_wrong_stl_count(mock_prepare, mock_settings):
    """Test that draft mode raises an error if more than one STL is provided."""
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"
    mock_prepare.return_value = mock_mesh

    mock_settings.drafts = [1.0]
    mock_settings.stl_files = [MeshConfig(file="file1.stl"), MeshConfig(file="file2.stl")]
    with pytest.raises(ValueError, match="exactly one base STL file must be provided"):
        run_simulation_batch(mock_settings)
