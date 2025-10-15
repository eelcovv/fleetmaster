"""Command to launch the Fleetmaster GUI."""

import click


@click.command()
def gui() -> None:
    """Launch the Fleetmaster GUI.

    This command attempts to launch the graphical user interface (GUI) for Fleetmaster.
    The GUI requires the 'PySide6' package to be installed. If the package is not
    found, it will print an error message with instructions on how to install it.
    """
    try:
        from fleetmaster.gui.main_window import main as main_gui

        main_gui()
    except ImportError:
        click.echo("The GUI dependencies are not installed.")
        click.echo("Please install them using: pip install fleetmaster[gui]")
