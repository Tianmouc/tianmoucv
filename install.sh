#!/bin/sh
set -eu
cd "$(dirname "$0")"

python -m pip install 'setuptools>=64' wheel 'pybind11>=2.12' -r requirements.txt
if ! (cd tianmoucv/camera && sh compile.sh); then
    echo "Optional camera SDK build failed; continuing with the Python package." >&2
fi
python -m pip install --no-build-isolation --no-deps .
