#!/bin/bash

# Exit on error, undefined variables, and pipe failures
set -euo pipefail
IFS=$'\n\t'

# Variables for easy adjustments
WORKSPACE_DIR="spatter_tests"
APP_NAME="spatter"
TESTS_DIR="tests/basic_tests"
REPO_URL="https://raw.githubusercontent.com/hpcgarage/spatter/refs/heads/main"

main() {
    echo "--- Initializing Ramble Application ---"
    
    # Create and enter working directory
    mkdir -p "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"

    # Initialize Ramble repo
    # Using || true in case the repo already exists in some environments
    ramble repo create "$APP_NAME" || echo "Repo already exists, skipping create..."
    ramble repo add "$APP_NAME" || echo "Repo already added, skipping add..."

    # Setup application directory and fetch python script
    mkdir -p "$APP_NAME/applications/$APP_NAME"
    echo "Downloading application.py..."
    curl -fsSL -o "$APP_NAME/applications/$APP_NAME/application.py" \
        "$REPO_URL/tests/ramble-tests/application.py"

    echo "--- Creating Experiment Workspace ---"
    ramble workspace create -d tests -a

    # Download test files
    mkdir -p "$TESTS_DIR"
    
    echo "Downloading test JSON files..."
    curl -fsSL -o "$TESTS_DIR/cpu-ustride.json" "$REPO_URL/standard-suite/basic-tests/cpu-ustride.json"
    curl -fsSL -o "$TESTS_DIR/cpu-stream.json" "$REPO_URL/standard-suite/basic-tests/cpu-stream.json"

    echo "--- Defining Experiments ---"
    # Note: Using $PWD ensures Ramble gets the full absolute path to the JSON files
    ramble workspace manage experiments "$APP_NAME" --overwrite \
        -e UniformStride \
        -v f="$PWD/$TESTS_DIR/cpu-ustride.json" \

    ramble workspace manage experiments "$APP_NAME" --overwrite \
        -e Stream \
        -v f="$PWD/$TESTS_DIR/cpu-stream.json"

    echo "--- Setting up and Concretizing ---"
    ramble workspace setup
    ramble workspace concretize

    echo "--------------------------------------------------------"
    echo "Setup Complete."
    echo "Output will be generated in: experiments/$APP_NAME/$APP_NAME/{experiment name}/{experiment name}.out"
    echo "--------------------------------------------------------"
}

# Dependency Check
if ! command -v ramble &> /dev/null; then
    echo "Error: 'ramble' command not found. Please ensure Ramble is in your PATH." >&2
    exit 1
fi

main "$@"