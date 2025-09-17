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

qblas_dep = dependency('qblas', fallback: ['qblas', 'qblas_dep'])

sleef_subproj = subproject('sleef', required: true)
sleef_dep = sleef_subproj.get_variable('sleef_dep')
sleefquad_dep = sleef_subproj.get_variable('sleefquad_dep')

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

npymath_path = incdir_numpy / '..' / 'lib'
npymath_lib = c.find_library('npymath', dirs: npymath_path)

dependencies = [py_dep, qblas_dep, sleef_dep, sleefquad_dep, npymath_lib]

# Add OpenMP dependency (optional, for threading)
openmp_dep = dependency('openmp', required: false)
if openmp_dep.found()
    dependencies += openmp_dep
endif

# compiler flags for QBLAS compatibility
if not is_windows
    # QBLAS requires extended numeric literals for Q suffix support
    # if compiler supports (usually gcc)
    if cpp.has_argument('-fext-numeric-literals')
        add_project_arguments('-fext-numeric-literals', language: 'cpp')
    endif
endif

# Thread-local storage detection (borrowed from NumPy)
optional_variable_attributes = [
  ['thread_local', 'HAVE_THREAD_LOCAL'],    # C23
  ['_Thread_local', 'HAVE__THREAD_LOCAL'],  # C11/C17
  ['__thread', 'HAVE___THREAD'],           # GCC/Clang
  ['__declspec(thread)', 'HAVE___DECLSPEC_THREAD_']  # MSVC
]

if not is_variable('cdata')
  cdata = configuration_data()
endif

foreach optional_attr: optional_variable_attributes
  attr = optional_attr[0]
  code = '''
    #pragma GCC diagnostic error "-Wattributes"
    #pragma clang diagnostic error "-Wattributes"

    int @0@ foo;

    int main() {
      return 0;
    }
  '''.format(attr)

  if c.compiles(code, name: optional_attr[0])
    cdata.set10(optional_attr[1], true)
    message('Thread-local storage support found: @0@'.format(attr))
  endif
endforeach

configure_file(
  output: 'quaddtype_config.h',
  configuration: cdata
)

build_includes = include_directories('.') # compile time generated headers as per system
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
        'numpy_quaddtype/__init__.pyi',
        'numpy_quaddtype/_quaddtype_main.pyi',
        'numpy_quaddtype/py.typed',
    ],
    subdir: 'numpy_quaddtype',
    pure: false
)

py.extension_module('_quaddtype_main',
  srcs,
  link_language: 'cpp',
  dependencies: dependencies,
  install: true,
  subdir: 'numpy_quaddtype',
  include_directories: [includes, build_includes],
)
