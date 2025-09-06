#!/bin/bash
set -x

if [ -d "build/" ]; then
    rm -r build
    rm -rf dist/
    rm -rf subprojects/qblas
    rm -rf subprojects/sleef
fi


python -m pip uninstall -y numpy_quaddtype
python -m pip install . -v