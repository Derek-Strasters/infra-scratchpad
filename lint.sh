#!/usr/bin/env bash

set -eux

poetry run black .
poetry run isort .
poetry run flake8 .
poetry run pydocstyle
# Find all top level modules and send to pylint
find ./src -maxdepth 2 -name '__init__.py' -printf '%h ' | xargs poetry run pylint
poetry run mypy .
poetry run pytest