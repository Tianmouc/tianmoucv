#!/bin/sh
set -eu
cd "$(dirname "$0")"

python -m pip install 'setuptools>=64' wheel 'pybind11>=2.12' -r requirements.txt
ONLY_UPDATE_PYTHON=1 python -m pip install --no-build-isolation --no-deps --no-cache-dir .
