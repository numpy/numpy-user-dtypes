project('quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

# Get the conda prefix
conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()
is_windows = build_machine.system() == 'windows'
is_macos = build_machine.system() == 'darwin'

# Add conda lib directory to library path
add_project_link_arguments('-L' + conda_prefix + '/lib', language: ['c', 'cpp'])

if is_windows
  add_project_arguments('-DWIN32', '-D_WINDOWS', language : ['c', 'cpp'])
  
  conda_env_dir = run_command('cmd', '/c', 'echo %CONDA_PREFIX%', check: true).stdout().strip()
  sleef_include_dir = conda_env_dir + '\\Library\\include'
  sleef_library_dir = conda_env_dir + '\\Library\\lib'
  
  add_project_arguments('-I' + sleef_include_dir, language: ['c', 'cpp'])
  add_project_link_arguments('-L' + sleef_library_dir, language: ['c', 'cpp'])
  
  sleef_lib = c.find_library('sleef', dirs: [sleef_library_dir])
  sleefquad_lib = c.find_library('sleefquad', dirs: [sleef_library_dir])
  
  sleef_dep = declare_dependency(include_directories: include_directories(sleef_include_dir),
                                 dependencies: [sleef_lib, sleefquad_lib])
elif is_macos
  # Existing macOS configuration
  conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()
  sleef_include_dir = run_command('bash', '-c', 'echo $SLEEF_INCLUDE_DIR', check: true).stdout().strip()
  sleef_library_dir = run_command('bash', '-c', 'echo $SLEEF_LIBRARY', check: true).stdout().strip()
  
  add_project_arguments('-I' + sleef_include_dir, language: ['c', 'cpp'])
  add_project_link_arguments('-L' + sleef_library_dir, language: ['c', 'cpp'])
  
  sleef_dep = c.find_library('sleef', dirs: [sleef_library_dir])
  sleefquad_dep = c.find_library('sleefquad', dirs: [sleef_library_dir])
else
  conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()
  add_project_link_arguments('-L' + conda_prefix + '/lib', language: ['c', 'cpp'])
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
  c_args: ['-g', '-O0'] + (is_windows ? [] : ['-lsleef', '-lsleefquad']),
  cpp_args: ['-g', '-O0'] + (is_windows ? [] : ['-lsleef', '-lsleefquad']),
  link_args: is_windows ? ['-DEFAULTLIB:sleef', '-DEFAULTLIB:sleefquad'] : [],
  dependencies: [sleef_dep] + (is_windows ? [] : [sleefquad_dep]),
  install: true,
  subdir: 'quaddtype',
  include_directories: includes
)