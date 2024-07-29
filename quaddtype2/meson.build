project(
  'quaddtype2',
  'c',
)

py_mod = import('python')
py = py_mod.find_installation('/home/rootacess/numpy-user-dtypes/quaddtype2/.venv/bin/python3')

# incdir_numpy = run_command(py,
#   [
#     '-c',
#     'import numpy; print(numpy.get_include())'
#   ],
#   check: true
# ).stdout().strip()

includes = include_directories(
  [
    '.venv/lib/python3.12/site-packages/numpy/_core/include',
    'quaddtype/src'
  ]
)

srcs = [
  'quaddtype/src/dtype.c',
  'quaddtype/src/quaddtype_main.c',
]

py.install_sources(
  [
    'quaddtype/__init__.py',
    'quaddtype/quadscalar.py'
  ],
  subdir: 'quaddtype',
  pure: false
)

py.extension_module(
  '_quaddtype_main',
  srcs,
  c_args: ['-g', '-O0' ,'-lsleef', '-lsleefquad'],
  install: true,
  subdir: 'quaddtype',
  include_directories: includes
)
