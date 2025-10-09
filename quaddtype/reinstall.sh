#!/bin/bash
set -x

if [ -d "build/" ]; then
    rm -r build
    rm -rf dist/
    rm -rf subprojects/qblas
    rm -rf subprojects/sleef
fi

# export CFLAGS="-g -O0" 
# export CXXFLAGS="-g -O0"
python -m pip uninstall -y numpy_quaddtype
python -m pip install . -v --no-build-isolation 2>&1 | tee build_log.txt