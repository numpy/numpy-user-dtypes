project(
  'unytdtype',
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
    'unytdtype/src'
  ]
)

srcs = [
  'unytdtype/src/casts.c',
  'unytdtype/src/casts.h',
  'unytdtype/src/dtype.c',
  'unytdtype/src/unytdtype_main.c',
  'unytdtype/src/umath.c',
  'unytdtype/src/umath.h',
]

py.install_sources(
  [
    'unytdtype/__init__.py',
    'unytdtype/scalar.py'
  ],
  subdir: 'unytdtype',
  pure: false
)

py.extension_module(
  '_unytdtype_main',
  srcs,
  c_args: ['-g', '-O0'],
  install: true,
  subdir: 'unytdtype',
  include_directories: includes
)
