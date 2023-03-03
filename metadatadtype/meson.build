project(
  'metadatadtype',
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
    'metadatadtype/src'
  ]
)

srcs = [
  'metadatadtype/src/casts.c',
  'metadatadtype/src/casts.h',
  'metadatadtype/src/dtype.c',
  'metadatadtype/src/metadatadtype_main.c',
  'metadatadtype/src/umath.c',
  'metadatadtype/src/umath.h',
]

py.install_sources(
  [
    'metadatadtype/__init__.py',
    'metadatadtype/scalar.py'
  ],
  subdir: 'metadatadtype'
)

py.extension_module(
  '_metadatadtype_main',
  srcs,
  install: true,
  subdir: 'metadatadtype',
  include_directories: includes
)
