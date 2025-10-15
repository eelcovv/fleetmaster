import logging
import types
from typing import Any, Callable, TypeVar, Union, get_args, get_origin

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


def create_cli_options(model: type[BaseModel]) -> Callable[[F], F]:
    """Dynamically create click options from a Pydantic model."""

    def decorator(f: F) -> F:
        # Decorators are applied bottom-up, so reverse the order of fields
        for name, field in reversed(model.model_fields.items()):
            # Skip stl_file as it's handled as a direct argument
            if name == "stl_file":
                continue

            option_name = f"--{name.replace('_', '-')}"
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


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("stl_file", type=click.Path(exists=True), required=False)
@click.option("--settings-file", type=click.Path(exists=True), help="Path to a YAML settings file.")
@create_cli_options(SimulationSettings)
def run(stl_file: str | None, settings_file: str | None, **kwargs: Any) -> None:
    """Runs a set of capytaine simulations based on provided settings."""

    # Validate that either settings_file or stl_file is provided, but not both.
    if not settings_file and not stl_file:
        err_msg = "Either a settings file or an STL file must be provided."
        raise click.UsageError(err_msg)
    if settings_file and stl_file:
        err_msg = "Provide either a settings file or an STL file, not both."
        raise click.UsageError(err_msg)

    # 1. Load settings from YAML file if provided
    config: dict[str, Any] = {}
    if settings_file:
        with open(settings_file) as f:
            config = yaml.safe_load(f) or {}
    elif stl_file:
        config["stl_file"] = stl_file

    # 2. Override with CLI options
    # Filter out None values so they don't override YAML settings
    cli_args = {k: v for k, v in kwargs.items() if v is not None and v != ()}

    # Convert tuple from 'multiple=True' to list for Pydantic
    for key, value in cli_args.items():
        if isinstance(value, tuple):
            # Attempt to convert string numbers to float
            try:
                cli_args[key] = [float(i) for i in value]
            except (ValueError, TypeError):
                cli_args[key] = list(value)

    config.update(cli_args)

    try:
        # 3. Validate settings with Pydantic
        settings = SimulationSettings(**config)
        logger.info("Successfully validated simulation settings.")
        logger.debug(f"Running with settings: {settings.model_dump_json(indent=2)}")

        # 4. Run the simulation
        # Assuming run_simulation_batch expects the Pydantic model or a dict
        run_simulation_batch(settings)
        click.echo("✅ Run completed successfully!")

    except ValidationError as e:
        click.echo("❌ Error: Invalid settings provided.", err=True)
        click.echo(e, err=True)
    except Exception as e:
        click.echo(f"❌ An unexpected error occurred: {e}", err=True)
