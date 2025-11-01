#!/bin/bash
set -x

if [ -d "build/" ]; then
    rm -r build
    rm -rf dist/
    rm -rf subprojects/qblas
    rm -rf subprojects/sleef
fi

python -m pip uninstall -y numpy_quaddtype
python -m pip install . -vv --no-build-isolation 2>&1 | tee build_log.txt

# for debugging and TSAN builds, comment the above line and uncomment all below:
# export CFLAGS="-fsanitize=thread -g -O0" 
# export CXXFLAGS="-fsanitize=thread -g -O0"
# export LDFLAGS="-fsanitize=thread"
# python -m pip install . -vv --no-build-isolation -Csetup-args=-Db_sanitize=thread 2>&1 | tee build_log.txt