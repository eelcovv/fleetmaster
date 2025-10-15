from typing import List, Optional

from pydantic import BaseModel, Field


class SimulationSettings(BaseModel):
    """Definieert alle mogelijke instellingen voor een simulatie."""

    stl_file: str = Field(description="Path to the STL mesh file.")
    wave_periods: List[float]
    wave_directions: Optional[List[float]] = [0.0]
    forward_speed: float = 0.0

    # Je kunt hier ook validatie toevoegen
    # @validator('forward_speed')
    # def speed_must_be_positive(cls, v):
    #    ...
