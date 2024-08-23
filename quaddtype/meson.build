project('quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

py_version = py.language_version()
message('Python version: ' + py_version)

if py_version.version_compare('>=3.13')
  py_dep = declare_dependency(
    include_directories: py.get_path('include'),
    link_args: ['-L' + py.get_path('stdlib'), '-lpython' + py_version]
  )
else
  py_dep = py.dependency(required: true)
endif

c = meson.get_compiler('c')

# Get the conda prefix
conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()
is_macos = build_machine.system() == 'darwin'

# Add conda lib directory to library path
add_project_link_arguments('-L' + conda_prefix + '/lib', language: ['c', 'cpp'])

if is_macos
  sleef_include_dir = run_command('bash', '-c', 'echo $SLEEF_INCLUDE_DIR', check: true).stdout().strip()
  sleef_library_dir = run_command('bash', '-c', 'echo $SLEEF_LIBRARY', check: true).stdout().strip()
  
  add_project_arguments('-I' + sleef_include_dir, language: ['c', 'cpp'])
  add_project_link_arguments('-L' + sleef_library_dir, language: ['c', 'cpp'])
  
  sleef_dep = c.find_library('sleef', dirs: [sleef_library_dir])
  sleefquad_dep = c.find_library('sleefquad', dirs: [sleef_library_dir])
else
  sleef_dep = c.find_library('sleef', dirs: [conda_prefix + '/lib'])
  sleefquad_dep = c.find_library('sleefquad', dirs: [conda_prefix + '/lib'])
endif

if not sleef_dep.found() or not sleefquad_dep.found()
    error('SLEEF library not found. Please ensure it is installed in your conda environment\nconda install sleef.')
endif

incdir_numpy = run_command(py,
  [
    '-c',
    'import numpy; import os; print(os.path.relpath(numpy.get_include()))'
  ],
  check: true
).stdout().strip()

includes = include_directories(
    [
        incdir_numpy,
        'quaddtype/src',
    ]
)

srcs = [
    'quaddtype/src/casts.h',
    'quaddtype/src/casts.cpp',
    'quaddtype/src/scalar.h',
    'quaddtype/src/scalar.c',
    'quaddtype/src/dtype.h',
    'quaddtype/src/dtype.c',
    'quaddtype/src/quaddtype_main.c',
    'quaddtype/src/scalar_ops.h',
    'quaddtype/src/scalar_ops.cpp',
    'quaddtype/src/ops.hpp',
    'quaddtype/src/umath.h',
    'quaddtype/src/umath.cpp'
]

py.install_sources(
    [
        'quaddtype/__init__.py',
    ],
    subdir: 'quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
    srcs,
    c_args: ['-g', '-O0', '-lsleef', '-lsleefquad'],
    dependencies: [sleef_dep, sleefquad_dep, py_dep],
    install: true,
    subdir: 'quaddtype',
    include_directories: includes
)