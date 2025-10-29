# General description of `fleetmaster`

## Summary

`fleetmaster` is a command-line tool designed to simplify running batch processes with [Capytaine](https://capytaine.github.io/), an open-source Python library for simulating wave-structure interactions. While Capytaine provides powerful tools for hydrodynamic analysis, `fleetmaster` streamlines the process of running multiple simulations with varying parameters, managing inputs, and organizing outputs. It acts as a wrapper, allowing users to define a fleet of simulations in a structured way and execute them with a single command.

## Core Concepts

The main goal of `fleetmaster` is to automate the execution of multiple hydrodynamic simulations using Capytaine. This is achieved through a few core concepts:

- **Settings File**: The user defines a batch of simulations using a YAML settings file. This file specifies the mesh files, water depth, wave directions, and other parameters for each case to be run.
- **Batch Engine**: The core engine of `fleetmaster` reads the settings file, prepares each individual Capytaine simulation, runs it, and stores the results to an hdf5-database.
- **Command-Line Interface (CLI)**: All operations are handled through the `fleetmaster` command. This allows for easy integration into scripts and automated workflows.

## Solution Database

The soluation for each mesh and simulation settings are stored in a database. The database supsquently can be used by external programs to quickly access each Capytaine solution. For more details on the database, see [Database](./database.md).

## Mesh Fitting

In addition to running batch simulations, `fleetmaster` also provides a powerful mesh fitting capability. This feature allows you to find the best matching mesh from a database of pre-calculated meshes based on a target transformation (translation and rotation). This is particularly useful for finding the most relevant hydrodynamic data for a specific loading condition without running a new simulation.

For more details, see the [Mesh Fitting](./fitting.md) documentation.

## Typical Workflow

A typical workflow for using `fleetmaster` involves the following steps:

1.  **Prepare Meshes**: Create or obtain the mesh files (e.g., `.obj`, `.stl`) for the floating bodies you want to analyze.
2.  **Create a Settings File**: Write a YAML file that defines the parameters for your batch of simulations. This includes pointing to the mesh files and specifying the desired environmental conditions.
3.  **Run `fleetmaster`**: Execute the tool from your terminal, pointing it to your settings file.
4.  **Analyze Results**: `fleetmaster` will generate output files (typically in NetCDF format) containing the hydrodynamic data for each simulation in the batch.

Below is a diagram illustrating this workflow.

```mermaid
graph LR
    A[Prepare Meshes] --> B[Create Settings File];
    B --> C[Run fleetmaster];
    C --> D[Analyze Results];
```
