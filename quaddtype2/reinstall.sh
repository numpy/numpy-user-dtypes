#!/bin/bash
set -xeuo pipefail
IFS=$'\n\t'

if [ -d "build/" ]
then
    rm -r build
fi

#meson setup build -Db_sanitize=address,undefined
meson setup build
python -m pip uninstall -y quaddtype
python -m pip install . -v --no-build-isolation --global-option="build_ext" --global-option="-v" --global-option="--build-dir=build" --global-option="--debug"