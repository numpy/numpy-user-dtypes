project('numpy_quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

is_windows = build_machine.system() == 'windows'

incdir_numpy = ''
if is_windows
  add_project_arguments('-DWIN32', '-D_WINDOWS', language : ['c', 'cpp'])
  
  conda_env_dir = run_command('cmd', '/c', 'echo %CONDA_PREFIX%', check: true).stdout().strip()
  sleef_include_dir = conda_env_dir + '\\Library\\include'
  sleef_library_dir = conda_env_dir + '\\Library\\lib'
  incdir_numpy = run_command('cmd', '/c', 'echo %NUMPY_INCLUDE_DIR%', check: true).stdout().strip()
  
  add_project_arguments('-I' + sleef_include_dir, language: ['c', 'cpp'])
  add_project_link_arguments('-L' + sleef_library_dir, language: ['c', 'cpp'])
  
  sleef_lib = c.find_library('sleef', dirs: [sleef_library_dir])
  sleefquad_lib = c.find_library('sleefquad', dirs: [sleef_library_dir])
  
  sleef_dep = declare_dependency(include_directories: include_directories(sleef_include_dir),
                                 dependencies: [sleef_lib, sleefquad_lib])
else
  # Linux and macOS configuration

  # Get the SLEEF path
  sleef_path = run_command('bash', '-c', 'echo $SLEEF_PATH', check: false).stdout().strip()
  if sleef_path == ''
    sleef_path = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: false).stdout().strip()
  endif

  if sleef_path == ''
    error('SLEEF_PATH or CONDA_PREFIX environment variable is not set')
  endif

  add_project_link_arguments('-L' + sleef_path + '/lib', language: ['c', 'cpp'])

  sleef_include_dir = sleef_path + '/include'
  sleef_library_dir = sleef_path + '/lib'

  add_project_arguments('-I' + sleef_include_dir, language: ['c', 'cpp'])
  add_project_link_arguments('-L' + sleef_library_dir, language: ['c', 'cpp'])
  
  sleef_dep = c.find_library('sleef', dirs: [sleef_library_dir])
  sleefquad_dep = c.find_library('sleefquad', dirs: [sleef_library_dir])
endif

if not sleef_dep.found() or (not is_windows and not sleefquad_dep.found())
  error('SLEEF library not found. Please ensure it is installed in your conda environment.')
endif

# Try to get NumPy include path from environment variable first
if incdir_numpy == ''
  incdir_numpy = run_command(py,
    [
      '-c',
      'import numpy; import os; print(os.path.relpath(numpy.get_include()))'
    ],
    check: false
  ).stdout().strip()
  
  if incdir_numpy == ''
    error('NumPy include directory not found. Please set NUMPY_INCLUDE_DIR or ensure NumPy is installed.')
  endif
endif

includes = include_directories(
    [
        incdir_numpy,
        'numpy_quaddtype/src',
    ]
)

srcs = [
    'numpy_quaddtype/src/quad_common.h',
    'numpy_quaddtype/src/casts.h',
    'numpy_quaddtype/src/casts.cpp',
    'numpy_quaddtype/src/scalar.h',
    'numpy_quaddtype/src/scalar.c',
    'numpy_quaddtype/src/dtype.h',
    'numpy_quaddtype/src/dtype.c',
    'numpy_quaddtype/src/quaddtype_main.c',
    'numpy_quaddtype/src/scalar_ops.h',
    'numpy_quaddtype/src/scalar_ops.cpp',
    'numpy_quaddtype/src/ops.hpp',
    'numpy_quaddtype/src/umath.h',
    'numpy_quaddtype/src/umath.cpp',
    'numpy_quaddtype/src/dragon4.h',
    'numpy_quaddtype/src/dragon4.c'
]

py.install_sources(
    [
        'numpy_quaddtype/__init__.py',
    ],
    subdir: 'numpy_quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
srcs,
c_args: ['-g', '-O0', '-lsleef', '-lsleefquad'],
dependencies: [sleef_dep, sleefquad_dep],
install: true,
subdir: 'numpy_quaddtype',
include_directories: includes
)