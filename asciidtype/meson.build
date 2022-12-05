project(
  'asciidtype',
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
    'asciidtype/src'
  ]
)

srcs = [
  'asciidtype/src/casts.c',
  'asciidtype/src/casts.h',
  'asciidtype/src/dtype.c',
  'asciidtype/src/asciidtype_main.c',
  'asciidtype/src/umath.c',
  'asciidtype/src/umath.h',
]

py.install_sources(
  [
    'asciidtype/__init__.py',
    'asciidtype/scalar.py'
  ],
  subdir: 'asciidtype'
)

py.extension_module(
  '_asciidtype_main',
  srcs,
  c_args: ['-g', '-O0'],
  install: true,
  subdir: 'asciidtype',
  include_directories: includes
)
