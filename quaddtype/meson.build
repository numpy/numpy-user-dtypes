project(
  'quaddtype',
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
    'quaddtype/src'
  ]
)

srcs = [
  'quaddtype/src/umath.c',
  'quaddtype/src/casts.c',
  'quaddtype/src/dtype.c',
  'quaddtype/src/quaddtype_main.c',
]

py.install_sources(
  [
    'quaddtype/__init__.py',
    'quaddtype/quadscalar.py'
  ],
  subdir: 'quaddtype'
)

py.extension_module(
  '_quaddtype_main',
  srcs,
  c_args: ['-g', '-O0'],
  install: true,
  subdir: 'quaddtype',
  include_directories: includes
)
