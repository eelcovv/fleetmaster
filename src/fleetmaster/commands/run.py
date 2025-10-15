import logging
import types
from typing import get_origin

import click
import yaml
from pydantic import ValidationError

from fleetmaster.core.engine import run_simulation_batch
from fleetmaster.core.settings import SimulationSettings

logger = logging.getLogger(__name__)

# Set logging levels for external libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)


def create_cli_options(model):
    """Dynamically create click options from a Pydantic model."""

    def decorator(f):
        for name, field in model.model_fields.items():
            option_name = f"--{name.replace('_', '-')}"
            option_type = field.annotation

            # Handle Union types (e.g., int | None)
            if isinstance(option_type, types.UnionType):
                # Use the first non-None type from the union
                non_none_types = [t for t in option_type.__args__ if t is not type(None)]
                option_type = non_none_types[0] if non_none_types else str

            # Handle List types in Click
            if get_origin(option_type) in (list, list):
                option_type = str  # Treat list inputs as comma-separated strings

            f = click.option(
                option_name,
                type=option_type,
                default=None,  # Default to None to distinguish between not set and set to a default value
                help=field.description or f"Set the {name}.",
                multiple=get_origin(option_type) in (list, list),
            )(f)
        return f

    return decorator


@click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--settings-file", type=click.Path(exists=True), help="Path to a YAML settings file.")
@create_cli_options(SimulationSettings)
def run(settings_file, **kwargs):
    """Runs a set of capytaine simulations based on provided settings."""

    # 1. Load settings from YAML file if provided
    config = {}
    if settings_file:
        with open(settings_file) as f:
            config = yaml.safe_load(f) or {}

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
