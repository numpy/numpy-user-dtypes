project('quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++17', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()

c = meson.get_compiler('c')

# Get the conda prefix
conda_prefix = run_command('bash', '-c', 'echo $CONDA_PREFIX', check: true).stdout().strip()

message('Conda prefix: ' + conda_prefix)

# Add conda lib and include directories
add_project_link_arguments('-L' + conda_prefix + '/lib', language: ['c', 'cpp'])
add_project_arguments('-I' + conda_prefix + '/include', language: ['c', 'cpp'])

# Try to find SLEEF using pkg-config first
sleef_dep = dependency('sleef', required: false)

if not sleef_dep.found()
    # If pkg-config fails, try to find the library manually
    sleef_dep = c.find_library('sleef', dirs: [conda_prefix + '/lib'])
    sleefquad_dep = c.find_library('sleefquad', dirs: [conda_prefix + '/lib'])
    
    if not sleef_dep.found() or not sleefquad_dep.found()
        error('SLEEF library not found. Please ensure it is installed in your conda environment\nconda install sleef.')
    endif
else
    sleefquad_dep = sleef_dep
endif

message('SLEEF library found: ' + sleef_dep.found().to_string())

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
        conda_prefix + '/include',
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
    c_args: ['-g', '-O0'],
    link_args: ['-lsleef', '-lsleefquad'],
    dependencies: [sleef_dep, sleefquad_dep],
    install: true,
    subdir: 'quaddtype',
    include_directories: includes
)