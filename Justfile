# Default task
default: all

# Run all code quality checks
all: fix type test

# Generate meshes for the defraction box example
generate-box-meshes:
    uv run python examples/defraction_box.py --output-dir examples
    uv run python examples/defraction_box.py --output-dir examples --grid-symmetry