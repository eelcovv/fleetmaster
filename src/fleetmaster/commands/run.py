import glob
import logging
import types
from collections.abc import Callable
from typing import Any, TypeVar, Union, get_args, get_origin

import click
import yaml
from pydantic import BaseModel, ValidationError

from fleetmaster.core.engine import run_simulation_batch
from fleetmaster.core.settings import SimulationSettings

logger = logging.getLogger(__name__)

# Define a TypeVar for decorator typing
F = TypeVar("F", bound=Callable[..., Any])

# Set logging levels for external libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)


def _expand_stl_files(stl_files: tuple[str, ...]) -> list[str]:
    """Expand glob patterns for STL files and return a list of paths."""
    if not stl_files:
        return []

    expanded_files = [path for pattern in stl_files for path in glob.glob(pattern)]

    if not expanded_files:
        err_msg = f"No files found matching the provided STL patterns: {", ".join(stl_files)}"
        raise click.UsageError(err_msg)
    return expanded_files


def _process_cli_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Process CLI arguments, filtering None values and converting tuples."""
    cli_args = {k: v for k, v in kwargs.items() if v is not None and v != ()}
    for key, value in cli_args.items():
        if isinstance(value, tuple):
            try:
                cli_args[key] = [float(i) for i in value]
            except (ValueError, TypeError):
                cli_args[key] = list(value)
    return cli_args


def _load_and_validate_settings(
    settings_file: str | None,
    stl_files: tuple[str, ...],
    kwargs: dict[str, Any],
) -> SimulationSettings:
    """Load settings from file or CLI, merge them, and validate."""
    expanded_stl_files = _expand_stl_files(stl_files)

    if not settings_file and not expanded_stl_files:
        err_msg = "Either a settings file or at least one STL file must be provided."
        raise click.UsageError(err_msg)
    if settings_file and expanded_stl_files:
        err_msg = "Provide either a settings file or STL files, not both."
        raise click.UsageError(err_msg)

    config: dict[str, Any] = {}
    if settings_file:
        with open(settings_file) as f:
            config = yaml.safe_load(f) or {}
    elif expanded_stl_files:
        config["stl_files"] = expanded_stl_files

    cli_args = _process_cli_args(kwargs)
    config.update(cli_args)

    try:
        settings = SimulationSettings(**config)
    except ValidationError as e:
        click.echo("❌ Error: Invalid settings provided.", err=True)
        click.echo(e, err=True)
        raise click.Abort() from e
    else:
        logger.info("Successfully validated simulation settings.")
        logger.debug(f"Running with settings: {settings.model_dump_json(indent=2)}")
        return settings


def create_cli_options(model: type[BaseModel]) -> Callable[[F], F]:
    """Dynamically create click options from a Pydantic model."""

    def decorator(f: F) -> F:
        # Decorators are applied bottom-up, so reverse the order of fields
        for name, field in reversed(model.model_fields.items()):
            # Skip stl_files as it's handled as a direct argument
            if name == "stl_files":
                continue

            option_name = f"--{name.replace("_", "-")}"
            option_type = field.annotation

            # Handle Union types (e.g., int | None)
            if get_origin(option_type) in (types.UnionType, Union):
                # Use the first non-None type from the union
                args = get_args(option_type)
                non_none_types = [t for t in args if t is not type(None)]
                option_type = non_none_types[0] if non_none_types else str

            # Handle List types in Click
            if get_origin(option_type) is list:
                option_type = str  # Treat list inputs as comma-separated strings

            f = click.option(
                option_name,
                type=option_type,
                default=None,  # Default to None to distinguish between not set and set to a default value
                help=field.description or f"Set the {name}.",
                multiple=get_origin(option_type) is list,
            )(f)
        return f

    return decorator


@click.command(context_settings={"ignore_unknown_options": False})
@click.argument("stl_files", required=False, nargs=-1)
@click.option("--settings-file", type=click.Path(exists=True), help="Path to a YAML settings file.")
@create_cli_options(SimulationSettings)
def run(stl_files: tuple[str, ...], settings_file: str | None, **kwargs: Any) -> None:
    """Runs a set of capytaine simulations based on provided settings."""
    try:
        settings = _load_and_validate_settings(settings_file, stl_files, kwargs)
        run_simulation_batch(settings)
        click.echo("✅ Run completed successfully!")
    except (click.UsageError, click.Abort):
        raise  # Re-raise to let click handle the error and exit
    except Exception as e:
        click.echo(f"❌ An unexpected error occurred: {e}", err=True)
