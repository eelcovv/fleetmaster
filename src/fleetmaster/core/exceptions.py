"""Custom exceptions for the Fleetmaster application's core logic."""


class SimulationConfigurationError(ValueError):
    """Base exception for simulation configuration errors."""


class LidAndSymmetryEnabledError(SimulationConfigurationError):
    """Raised when both lid and grid_symmetry are enabled simultaneously."""

    def __init__(
        self,
        message: str = "Cannot have both lid and grid_symmetry True simultaneously.",
    ) -> None:
        super().__init__(message)


class NegativeForwardSpeedError(SimulationConfigurationError):
    """Raised when forward speed is negative."""

    def __init__(self, message: str = "Forward speed must be non-negative.") -> None:
        super().__init__(message)


class NonPositivePeriodError(SimulationConfigurationError):
    """Raised when a simulation period is not positive."""

    def __init__(self, message: str = "Periods must be larger than 0.") -> None:
        super().__init__(message)
