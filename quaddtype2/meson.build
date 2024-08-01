project('quaddtype2', 'c')

py_mod = import('python')
py = py_mod.find_installation('/home/rootacess/numpy-user-dtypes/quaddtype2/.venv/bin/python3')

c = meson.get_compiler('c')
sleef = c.find_library('sleef')
sleefquad_math =c.find_library('sleefquad')

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
        'quaddtype/src'
    ]
)

srcs = [
    'quaddtype/src/scalar.h',
    'quaddtype/src/scalar.c',
    'quaddtype/src/dtype.h',
    'quaddtype/src/dtype.c',
    'quaddtype/src/quaddtype_main.c'
]

py.install_sources(
    [
        'quaddtype/__init__.py',
    ],
    subdir: 'quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
srcs,
c_args: ['-g', '-O0', '-lsleef', '-lsleefquad'],
install: true,
subdir: 'quaddtype',
include_directories: includes
)