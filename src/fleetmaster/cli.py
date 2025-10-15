"""Command-line interface (CLI) for Python build utilities.

This module uses the `click` library to create a CLI group with commands for
renaming wheel files and removing tarballs.

Functions:
    cli(): CLI entrypoint that registers available commands.

Commands:
    rename_wheel_files: Rename wheel files to match a standardized format.
    remove_tarballs: Remove `.tar.gz` source distributions from the current directory.
"""

import logging

import click
from rich.logging import RichHandler

from . import __version__
from .commands import run
from .logging_setup import setup_general_logger

logger = setup_general_logger()


@click.group(help="Register CLI tools for fleetmaster.", invoke_without_command=True)
@click.version_option(
    __version__,
    "--version",
    message="Version: %(version)s",
    help="Show the version and exit.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity level. Use -v for info, -vv for debug.",
)
def cli(verbose: int) -> None:
    """Register CLI tools for Python build utilities.

    This function is the entrypoint for the CLI. It adjusts the logging level based
    on verbosity flags and registers all available subcommands.
    """
    # Get the root logger of the package and set its level
    package_logger = logging.getLogger("fleetmaster")

    # These constants are not defined, let's define them for clarity
    VERBOSITY_DEBUG, VERBOSITY_INFO, LOGLEVEL_DEBUG, LOGLEVEL_INFO, LOGLEVEL_DEFAULT = (
        2,
        1,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
    )

    if verbose >= VERBOSITY_DEBUG:
        log_level = LOGLEVEL_DEBUG
    elif verbose == VERBOSITY_INFO:
        log_level = LOGLEVEL_INFO
    else:
        log_level = LOGLEVEL_DEFAULT

    package_logger.setLevel(log_level)
    for handler in package_logger.handlers:
        handler.setLevel(log_level)

    if log_level <= logging.INFO:
        logger.info(
            "🚀 Fleetmaster CLI — ready to start your capytaine simulations.",
        )


# Register all subcommands
cli.add_command(run, name="run")


if __name__ == "__main__":
    cli()
