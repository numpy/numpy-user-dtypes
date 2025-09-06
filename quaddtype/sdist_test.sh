#!/bin/bash
set -x

if [ -d "build/" ]; then
    rm -rf dist/
fi

python -m pip uninstall -y numpy_quaddtype
python -m build --sdist --outdir dist/
python -m pip install dist/numpy_quaddtype-0.1.0.tar.gz -v