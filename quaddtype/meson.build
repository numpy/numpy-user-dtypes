project('quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

# Get the conda prefix
conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()

# Add conda lib and include directories
add_project_link_arguments('-L' + conda_prefix + '/lib', language: ['c', 'cpp'])
add_project_arguments('-I' + conda_prefix + '/include', language: ['c', 'cpp'])

# Use pkg-config to find SLEEF
pkg = import('pkgconfig')
sleef_dep = pkg.dependency('sleef', required: false)

if not sleef_dep.found()
    # Fallback to manual library detection
    sleef_dep = c.find_library('sleef', dirs: [conda_prefix + '/lib'])
    sleefquad_dep = c.find_library('sleefquad', dirs: [conda_prefix + '/lib'])
    
    if not sleef_dep.found() or not sleefquad_dep.found()
        error('SLEEF library not found. Please ensure it is installed in your conda environment.')
    endif
else
    sleefquad_dep = sleef_dep
endif

incdir_numpy = run_command(py,
  ['-c', 'import numpy; import os; print(os.path.relpath(numpy.get_include()))'],
  check: true
).stdout().strip()

includes = include_directories(
    [incdir_numpy, 'quaddtype/src']
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
    ['quaddtype/__init__.py'],
    subdir: 'quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
    srcs,
    dependencies: [sleef_dep, sleefquad_dep],
    install: true,
    subdir: 'quaddtype',
    include_directories: includes
)