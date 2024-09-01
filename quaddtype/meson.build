project('numpy_quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

sleef_dep = c.find_library('sleef')
sleefquad_dep = c.find_library('sleefquad')

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; import os; print(os.path.relpath(numpy.get_include()))'
  ],
  check: true
).stdout().strip()

includes = include_directories(
    [
        incdir_numpy,
        'numpy_quaddtype/src',
    ]
)

srcs = [
    'numpy_quaddtype/src/casts.h',
    'numpy_quaddtype/src/casts.cpp',
    'numpy_quaddtype/src/scalar.h',
    'numpy_quaddtype/src/scalar.c',
    'numpy_quaddtype/src/dtype.h',
    'numpy_quaddtype/src/dtype.c',
    'numpy_quaddtype/src/quaddtype_main.c',
    'numpy_quaddtype/src/scalar_ops.h',
    'numpy_quaddtype/src/scalar_ops.cpp',
    'numpy_quaddtype/src/ops.hpp',
    # 'numpy_quaddtype/src/umath.h',
    # 'numpy_quaddtype/src/umath.cpp'
]

py.install_sources(
    [
        'numpy_quaddtype/__init__.py',
    ],
    subdir: 'numpy_quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
srcs,
c_args: ['-g', '-O0', '-lsleef', '-lsleefquad'],
dependencies: [sleef_dep, sleefquad_dep],
install: true,
subdir: 'numpy_quaddtype',
include_directories: includes
)