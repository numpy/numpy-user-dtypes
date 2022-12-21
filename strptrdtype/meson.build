project(
  'strptrdtype',
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
    'strptrdtype/src'
  ]
)

srcs = [
  'strptrdtype/src/casts.c',
  'strptrdtype/src/casts.h',
  'strptrdtype/src/dtype.c',
  'strptrdtype/src/_main.c',
  'strptrdtype/src/umath.c',
  'strptrdtype/src/umath.h',
]

py.install_sources(
  [
    'strptrdtype/__init__.py',
    'strptrdtype/scalar.py'
  ],
  subdir: 'strptrdtype'
)

py.extension_module(
  '_main',
  srcs,
  c_args: ['-g', '-O0'],
  install: true,
  subdir: 'strptrdtype',
  include_directories: includes
)
