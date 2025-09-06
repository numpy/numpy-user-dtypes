project('sleef', version: '3.8')

cmake = find_program('cmake')
ninja = find_program('ninja', 'make', required: false)

sleef_build_dir = 'sleef_build'
sleef_install_dir = 'sleef_install'

# This can be done manually from user side, but lets handle it here as well
libdir = 'lib'
if host_machine.system() == 'linux' and host_machine.cpu_family() == 'x86_64'
  libdir = 'lib64'
endif

sleef_configure = run_command([
    cmake, 
    '-S', meson.current_source_dir(),
    '-B', meson.current_build_dir() / sleef_build_dir,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DSLEEF_BUILD_QUAD=ON',
    '-DSLEEF_BUILD_SHARED_LIBS=OFF',
    '-DSLEEF_BUILD_TESTS=OFF',
    '-DSLEEF_BUILD_INLINE_HEADERS=OFF',
    '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    '-DCMAKE_INSTALL_PREFIX=' + meson.current_build_dir() / sleef_install_dir
], check: false, capture: true)

if sleef_configure.returncode() != 0
    error('SLEEF CMake configuration failed: ' + sleef_configure.stderr())
endif

sleef_build_target = custom_target('sleef_build',
    command: [cmake, '--build', meson.current_build_dir() / sleef_build_dir, '--target', 'install', '--parallel'],
    output: 'sleef_built.stamp',  # Dummy stamp file
    console: true,
    build_always_stale: true,
    build_by_default: true
)

sleef_include_path = meson.current_build_dir() / sleef_install_dir / 'include'
sleef_lib_path = meson.current_build_dir() / sleef_install_dir / libdir

sleef_build_dep = declare_dependency(sources: [sleef_build_target])
sleef_dep = declare_dependency(
    dependencies: [sleef_build_dep], 
    compile_args: ['-I' + sleef_include_path],
    link_args: ['-L' + sleef_lib_path, '-lsleef']
)
sleefquad_dep = declare_dependency(
    dependencies: [sleef_build_dep],
    compile_args: ['-I' + sleef_include_path],
    link_args: ['-L' + sleef_lib_path, '-lsleefquad']
)