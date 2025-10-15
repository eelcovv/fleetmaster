class SimulationConfigurationError(ValueError):
    """Custom exception for simulation configuration errors."""

    LID_AND_SYMMETRY_ENABLED = "Cannot have both lid and grid_symmetry True simultaneously."
