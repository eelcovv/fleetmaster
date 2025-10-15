import numpy as np
from pydantic import BaseModel, Field, field_validator


class SimulationSettings(BaseModel):
    """Defines all possible settings for a simulation."""

    stl_files: list[str] = Field(description="Path to the STL mesh files.")
    wave_periods: list[float] = Field(default=[5, 10, 15, 20])
    wave_directions: list[float] = Field(default=[0, 45, 90, 135, 180])
    forward_speed: float | list[float] = 0.0
    lid: bool = True
    grid_symmetry: bool = False
    water_depth: float | list[float] = np.inf
    water_level: float | list[float] = 0.0

    @field_validator("forward_speed")
    def speed_must_be_positive(cls, v):
        error_message = "Forward speed must be positive"
        if isinstance(v, list):
            if any(speed < 0 for speed in v):
                raise ValueError(error_message)
        elif v < 0:
            raise ValueError(error_message)
        return v
