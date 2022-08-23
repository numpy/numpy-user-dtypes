project(
  'quaddtype',
  'c',
)

py_mod = import('python')
py = py_mod.find_installation()

cc = meson.get_compiler('c')

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

srcs = [
  'quaddtype/src/casts.c',
  'quaddtype/src/dtype.c',
  'quaddtype/src/scalar.c',
  'quaddtype/src/umath.c',
  'quaddtype/src/unitdtype_main.c',
]

py.extension_module(
  'quaddtype',
  srcs,
  install: true,
  include_directories: inc_np
)
