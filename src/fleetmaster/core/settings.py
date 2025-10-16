import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from fleetmaster.core.exceptions import (
    LidAndSymmetryEnabledError,
    NegativeForwardSpeedError,
    NonPositivePeriodError,
)


class SimulationSettings(BaseModel):
    """Defines all possible settings for a simulation."""

    stl_files: list[str] = Field(description="Path to the STL mesh files.")
    output_directory: str | None = Field(default=None, description="Directory to save the output files.")
    output_hdf5_file: str = Field(default="results.hdf5", description="Path to the HDF5 output file.")
    export_to_netcdf: bool = Field(default=False, description="Also export results to individual NetCDF files.")
    wave_periods: list[float] = Field(default=[5, 10, 15, 20])
    wave_directions: list[float] = Field(default=[0, 45, 90, 135, 180])
    forward_speed: float | list[float] = 0.0
    lid: bool = True
    grid_symmetry: bool = False
    water_depth: float | list[float] = np.inf
    water_level: float | list[float] = 0.0

    @field_validator("forward_speed")
    def speed_must_be_non_negative(cls, v: float | list[float]) -> float | list[float]:
        """Validate that forward speed is non-negative."""
        if isinstance(v, list):
            if any(speed < 0 for speed in v):
                raise NegativeForwardSpeedError()
        elif v < 0:
            raise NegativeForwardSpeedError()
        return v

    @field_validator("wave_periods")
    def periods_must_be_positive(cls, v: list[float]) -> list[float]:
        """Validate that wave periods are positive."""
        if any(p <= 0 for p in v):
            raise NonPositivePeriodError()
        return v

    @model_validator(mode="after")
    def check_lid_and_symmetry(self) -> "SimulationSettings":
        """Validate that lid and grid_symmetry are not both enabled."""
        if self.lid and self.grid_symmetry:
            raise LidAndSymmetryEnabledError()
        return self
