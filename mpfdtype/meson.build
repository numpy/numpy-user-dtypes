project(
  'mpfdtype',
  'c',
  'cpp',
)

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')
mpfr = c.find_library('mpfr')

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

includes = include_directories(
  [
    incdir_numpy,
    'mpfdtype/src',
  ],
)

srcs = [
  'mpfdtype/src/casts.cpp',
  'mpfdtype/src/casts.h',
  'mpfdtype/src/dtype.c',
  'mpfdtype/src/dtype.h',
  'mpfdtype/src/mpfdtype_main.c',
  'mpfdtype/src/numbers.cpp',
  'mpfdtype/src/numbers.h',
  'mpfdtype/src/ops.hpp',
  'mpfdtype/src/scalar.c',
  'mpfdtype/src/scalar.h',
  'mpfdtype/src/terrible_hacks.c',
  'mpfdtype/src/terrible_hacks.h',
  'mpfdtype/src/umath.cpp',
  'mpfdtype/src/umath.h',
]

py.install_sources(
  [
    'mpfdtype/__init__.py',
  ],
  subdir: 'mpfdtype'
)

py.extension_module(
  '_mpfdtype_main',
  srcs,
  install: true,
  subdir: 'mpfdtype',
  include_directories: includes,
  dependencies: [mpfr],
)
