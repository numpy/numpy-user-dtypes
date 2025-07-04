#!/bin/bash
set -xeuo pipefail
IFS=$'\n\t'

if [ -d "build/" ]
then
    rm -r build
fi

export CC=clang
export CXX=clang++
export SLEEF_DIR=$PWD/sleef/build
export LIBRARY_PATH=$SLEEF_DIR/lib
export C_INCLUDE_PATH=$SLEEF_DIR/include
export CPLUS_INCLUDE_PATH=$SLEEF_DIR/include

# Set RPATH via LDFLAGS
export LDFLAGS="-Wl,-rpath,$SLEEF_DIR/lib"

python -m pip uninstall -y numpy_quaddtype
python -m pip install . -v --no-build-isolation -Cbuilddir=build -C'compile-args=-v'