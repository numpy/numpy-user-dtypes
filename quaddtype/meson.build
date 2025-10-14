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

# Try to find SLEEF system-wide first, fall back to subproject if not found
sleef_dep = dependency('sleef', fallback: ['sleef', 'sleef_dep'], required: false)

use_system_sleef = false
fallback_reason = ''

if sleef_dep.found() and sleef_dep.type_name() != 'internal'
  # SLEEF found system-wide - verify quad-precision support
  cpp = meson.get_compiler('cpp')
  sleefquad_lib = cpp.find_library('sleefquad', required: false)
  
  if sleefquad_lib.found()
    sleefquad_test_code = '''
    #include <sleefquad.h>

    int main(void) {
      Sleef_quad q1 = Sleef_cast_from_doubleq1(1.0);
      Sleef_quad q2 = Sleef_cast_from_doubleq1(2.0);
      Sleef_quad result = Sleef_addq1_u05(q1, q2);
      return 0;
    }
    '''
    # this should compile and link
    quad_works = cpp.links(
      sleefquad_test_code,
      dependencies: [sleef_dep, sleefquad_lib],
      name: 'SLEEF quad-precision support'
    )
    
    if quad_works
      sleefquad_dep = declare_dependency(
        dependencies: [sleef_dep, sleefquad_lib]
      )
      use_system_sleef = true
    else
      fallback_reason = 'quad-precision support is not working'
    endif
  else
    fallback_reason = 'sleefquad library is missing'
  endif
else
  fallback_reason = 'SLEEF not found system-wide'
endif

if use_system_sleef
  message('Using system-wide SLEEF installation with quad-precision support')
else
  warning('Falling back to SLEEF subproject: ' + fallback_reason)
  sleef_subproj = subproject('sleef')
  sleef_dep = sleef_subproj.get_variable('sleef_dep')
  sleefquad_dep = sleef_subproj.get_variable('sleefquad_dep')
  message('Using SLEEF subproject')
endif

incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

# print numpy version used
numpy_version = run_command(py,
  ['-c', 'import numpy; print(numpy.__version__)'],
  check : true
).stdout().strip()
message('Using NumPy version: @0@'.format(numpy_version))

npymath_path = incdir_numpy / '..' / 'lib'
npymath_lib = c.find_library('npymath', dirs: npymath_path)

dependencies = [py_dep, qblas_dep, sleef_dep, sleefquad_dep, npymath_lib]

# Add OpenMP dependency (optional, for threading)
openmp_dep = dependency('openmp', required: false, static: false)
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
    'numpy_quaddtype/src/constants.hpp',
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
