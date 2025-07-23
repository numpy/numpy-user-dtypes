project('numpy_quaddtype', 'c', 'cpp', default_options : ['cpp_std=c++20', 'b_pie=true'])

py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

c = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

is_windows = build_machine.system() == 'windows'

if is_windows
  add_project_arguments('-DWIN32', '-D_WINDOWS', language : ['c', 'cpp'])
endif

sleef_dep = [
  c.find_library('sleef', required : true),
  c.find_library('sleefquad', required : true)
]

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# Add OpenMP dependency (optional, for threading)
openmp_dep = dependency('openmp', required: false)
dependencies = [sleef_dep, py_dep]
if openmp_dep.found()
    dependencies += openmp_dep
endif

includes = include_directories(
    [
        incdir_numpy,
        'numpy_quaddtype/QBLAS/include',
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
    'numpy_quaddtype/src/dragon4.h',
    'numpy_quaddtype/src/dragon4.c',
    'numpy_quaddtype/src/quadblas_interface.h',
    'numpy_quaddtype/src/quadblas_interface.cpp',
    'numpy_quaddtype/src/umath/umath.h',
    'numpy_quaddtype/src/umath/umath.cpp',
    'numpy_quaddtype/src/umath/binary_ops.h',
    'numpy_quaddtype/src/umath/binary_ops.cpp',
    'numpy_quaddtype/src/umath/unary_ops.h',
    'numpy_quaddtype/src/umath/unary_ops.cpp',
    'numpy_quaddtype/src/umath/unary_props.h',
    'numpy_quaddtype/src/umath/unary_props.cpp',
    'numpy_quaddtype/src/umath/comparison_ops.h',
    'numpy_quaddtype/src/umath/comparison_ops.cpp',
    'numpy_quaddtype/src/umath/promoters.hpp',
    'numpy_quaddtype/src/umath/matmul.h',
    'numpy_quaddtype/src/umath/matmul.cpp',
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
  link_args: is_windows ? ['/DEFAULTLIB:sleef', '/DEFAULTLIB:sleefquad'] : ['-lsleef', '-lsleefquad'],
  link_language: 'cpp',
  dependencies: dependencies,
  install: true,
  subdir: 'numpy_quaddtype',
  include_directories: includes
)