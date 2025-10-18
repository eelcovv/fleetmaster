# fleetmaster

[![Release](https://img.shields.io/github/v/release/eelcovv/fleetmaster)](https://img.shields.io/github/v/release/eelcovv/fleetmaster)
[![Build status](https://img.shields.io/github/actions/workflow/status/eelcovv/fleetmaster/main.yml?branch=main)](https://github.com/eelcovv/fleetmaster/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/eelcovv/fleetmaster/branch/main/graph/badge.svg)](https://codecov.io/gh/eelcovv/fleetmaster)
[![Commit activity](https://img.shields.io/github/commit-activity/m/eelcovv/fleetmaster)](https://img.shields.io/github/commit-activity/m/eelcovv/fleetmaster)
[![License](https://img.shields.io/github/license/eelcovv/fleetmaster)](https://img.shields.io/github/license/eelcovv/fleetmaster)

A wrapper to run capytaine from theandline
A CLI for running hydrodynamic simulations with Fleetmaster.

- **Github repository**: <https://github.com/eelcovv/fleetmaster/>
- **Documentation** <https://eelcovv.github.io/fleetmaster/>

## Installation

You can install `fleetmaster` from PyPI using pip (or any other PEP 517 compliant installer like uv):

```bash
pip install fleetmaster
```

## What is Fleetmaster?

`fleetmaster` is a command-line interface (CLI) wrapper for the Capytaine library. It simplifies the process of running hydrodynamic simulations by allowing you to define and execute simulation batches directly from the terminal or through a YAML configuration file. This makes it easy to automate and replicate your simulations.

The package also includes an optional GUI for a more interactive experience.

## Examples

Here are a few examples of how to use the `fleetmaster` CLI.

### Show the help message

To see all available commands and options, run:

```bash
fleetmaster --help
```

### Run a simulation for a single geometry

You can run a simulation for a single STL file and specify parameters like draft directly.

```bash
fleetmaster run path/to/your/geometry.stl --drafts 0.5 1.0 1.5
```

### Run a batch simulation using a settings file

For more complex scenarios, you can define all your settings in a YAML file and pass it to the `run` command.

```bash
fleetmaster run --settings-file path/to/your/settings.yml
```

### Launch the GUI

If you have installed the GUI dependencies (`pip install fleetmaster[gui]`), you can launch the graphical user interface.

```bash
fleetmaster gui
```
