project(
  'stringdtype',
  'c',
)

py_mod = import('python')
py = py_mod.find_installation()

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
    'stringdtype/src'
  ]
)

srcs = [
  'stringdtype/src/casts.c',
  'stringdtype/src/casts.h',
  'stringdtype/src/dtype.c',
  'stringdtype/src/main.c',
  'stringdtype/src/static_string.c',
  'stringdtype/src/static_string.h',
  'stringdtype/src/umath.c',
  'stringdtype/src/umath.h',
]

py.install_sources(
  [
    'stringdtype/__init__.py',
    'stringdtype/scalar.py'
  ],
  subdir: 'stringdtype'
)

py.extension_module(
  '_main',
  srcs,
  c_args: ['-g', '-O0', '-pg'],
  install: true,
  subdir: 'stringdtype',
  include_directories: includes
)
