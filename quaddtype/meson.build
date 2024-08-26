project('quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

# Get the SLEEF path
sleef_path = run_command('bash', '-c', 'echo $SLEEF_PATH', check: false).stdout().strip()
if sleef_path == ''
  sleef_path = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: false).stdout().strip()
endif

if sleef_path == ''
  error('SLEEF_PATH or CONDA_PREFIX environment variable is not set')
endif

is_windows = build_machine.system() == 'windows'

# Add SLEEF lib directory to library path
add_project_link_arguments('-L' + sleef_path + '/lib', language: ['c', 'cpp'])

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
else
  # Linux and macOS configuration
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
numpy_include = run_command('cmd', '/c', 'echo %NUMPY_INCLUDE_DIR%', check: true).stdout().strip()
if numpy_include == ''
  numpy_include = run_command(py,
    [
      '-c',
      'import numpy; import os; print(os.path.relpath(numpy.get_include()))'
    ],
    check: false
  ).stdout().strip()
  
  if numpy_include == ''
    error('NumPy include directory not found. Please set NUMPY_INCLUDE_DIR or ensure NumPy is installed.')
  endif
endif

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
  c_args: is_windows ? ['/DWIN32', '/D_WINDOWS'] : ['-g', '-O0'],
  cpp_args: is_windows ? ['/DWIN32', '/D_WINDOWS', '/EHsc'] : ['-g', '-O0'],
  link_args: is_windows ? ['/DEFAULTLIB:sleef', '/DEFAULTLIB:sleefquad'] : ['-lsleef', '-lsleefquad'],
  dependencies: [sleef_dep] + (is_windows ? [] : [sleefquad_dep]),
  link_language: 'cpp',
  install: true,
  subdir: 'quaddtype',
  include_directories: includes
)