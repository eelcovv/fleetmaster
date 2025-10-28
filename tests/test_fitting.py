"""Unit tests for the mesh fitting functionality."""

import logging
from pathlib import Path

import pytest

from fleetmaster.core.fitting import find_best_matching_mesh

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def hdf5_path() -> Path:
    """
    Provides the path to the HDF5 database file and skips tests if it doesn't exist.

    This fixture ensures that tests depending on the pre-generated database
    are only run when the file is available.
    """
    # The HDF5 file is expected to be in the 'examples' directory, relative to the project root.
    # Assuming tests are run from the project root.
    path = Path("examples/boxship.hdf5")
    if not path.exists():
        pytest.skip(
            f"Database file not found at: {path.resolve()}. "
            "Run 'fleetmaster -v run --settings-file examples/settings_rotations.yml' to generate it."
        )
    return path


# Define test cases using pytest.mark.parametrize
# Each tuple represents a test case:
# (case_description, target_translation, target_rotation, water_level, expected_match, expected_distance_check)
# The `expected_distance_check` is a lambda function to validate the distance.
TEST_CASES = [
    (
        "Case 1: Exact Match Draft 1 meter",
        [0.0, 0.0, -1.0],
        [20.0, 20.0, 0.0],
        0.0,
        "boxship_t_1_r_20_20_00",
        lambda dist: dist == pytest.approx(0.0, abs=1e-9),
    ),
    (
        "Case 2: Match with irrelevant translation/rotation noise (draft 1.0)",
        [2.5, -4.2, -1.0],  # dx, dy noise
        [20.0, 20.0, 15.0],  # yaw noise
        0.0,
        "boxship_t_1_r_20_20_00",
        lambda dist: dist == pytest.approx(0.0, abs=1e-9),  # Distance should still be near zero
    ),
    (
        "Case 3: Different match due to significant rotation deviation (draft 1.0)",
        [2.5, -4.2, -1.1],
        [23.0, 19.0, 15.0],  # Deviations in roll and pitch
        0.0,
        "boxship_t_1_r_20_20_00",  # This is still the closest match
        lambda dist: dist > 0.001,  # Distance should be clearly non-zero
    ),
    (
        "Case 4: Exact Match for draft 2.0",
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 0.0],
        0.0,
        "boxship_t_2_r_00_00_00",
        lambda dist: dist == pytest.approx(0.0, abs=1e-9),
    ),
    (
        "Case 5: Exact Match for draft 2.0 with irrelevant xy-plane and yaw deviation",
        [10.0, -20.0, -2.0],
        [0.0, 0.0, 15.0],
        0.0,
        "boxship_t_2_r_00_00_00",
        lambda dist: dist == pytest.approx(0.0, abs=1e-9),
    ),
    (
        "Case 6: Match for draft 2.0 with noise in all axes",
        [10.0, -20.0, -2.2],
        [4.0, -1.0, 15.0],
        0.0,
        "boxship_t_2_r_00_00_00",
        lambda dist: dist > 0.01,  # Distance should be clearly non-zero
    ),
]


@pytest.mark.parametrize(
    "description, target_translation, target_rotation, water_level, expected_match, distance_check",
    TEST_CASES,
    ids=[case[0] for case in TEST_CASES],
)
def test_find_best_matching_mesh(
    hdf5_path: Path,
    description: str,
    target_translation: list[float],
    target_rotation: list[float],
    water_level: float,
    expected_match: str,
    distance_check,
):
    """Tests the find_best_matching_mesh function with various scenarios."""
    logger.info(f"Running test: {description}")
    best_match, distance = find_best_matching_mesh(hdf5_path, target_translation, target_rotation, water_level)

    assert best_match is not None, "A best match should have been found."
    assert best_match == expected_match
    assert distance_check(distance), f"Distance check failed for {description}. Got distance: {distance}"
