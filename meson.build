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

incdir_python = run_command(py,
  [
    '-c',
    'import sysconfig; print(sysconfig.get_path("include"))'
  ],
  check: true
).stdout().strip()

message(incdir_python)

includes = include_directories(
  [
    incdir_python,
    incdir_numpy,
    'quaddtype/src'
  ]
)

srcs = [
  'quaddtype/src/dtype.c',
  'quaddtype/src/quaddtype_main.c',
]

py.install_sources(
  [
    'quaddtype/__init__.py'
  ],
  subdir: 'quaddtype'
)

py.extension_module(
  '_quaddtype_main',
  srcs,
  install: true,
  subdir: 'quaddtype',
  include_directories: includes
)
