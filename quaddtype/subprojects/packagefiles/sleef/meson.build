project('sleef', version: '3.8')

cmake = find_program('cmake')
ninja = find_program('ninja', 'make', required: false)

sleef_build_dir = 'sleef_build'
sleef_install_dir = 'sleef_install'

# turning off parallel build in windows
parallel_flag = ['--parallel']
if host_machine.system() == 'windows'
  parallel_flag = []
endif

# uncomment below lines for TSAN builds (in case compiler flags are not picked up from meson)
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
    # '-DCMAKE_C_FLAGS=-fsanitize=thread -g',
    # '-DCMAKE_CXX_FLAGS=-fsanitize=thread -g',
    # '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread',
    # '-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread',
    '-DCMAKE_INSTALL_PREFIX=' + meson.current_build_dir() / sleef_install_dir
], check: false, capture: true)

if sleef_configure.returncode() != 0
    error('SLEEF CMake configuration failed: ' + sleef_configure.stderr())
endif

sleef_build_target = custom_target('sleef_build',
    command: [cmake, '--build', meson.current_build_dir() / sleef_build_dir, '--target', 'install'] + parallel_flag,
    output: 'sleef_built.stamp',
    console: true,
    build_always_stale: true,
    build_by_default: true
)

sleef_include_path = meson.current_build_dir() / sleef_install_dir / 'include'

sleef_build_dep = declare_dependency(sources: [sleef_build_target])

sleef_static_define = ''
if host_machine.system() == 'windows'
  sleef_static_define = '-DSLEEF_STATIC_LIBS'
endif

compile_args_list = ['-I' + sleef_include_path]
if sleef_static_define != ''
  compile_args_list += sleef_static_define
endif

sleef_dep = declare_dependency(
    dependencies: [sleef_build_dep],
    compile_args: compile_args_list,
    link_args: ['-L' + meson.current_build_dir() / sleef_install_dir / 'lib', '-L' + meson.current_build_dir() / sleef_install_dir / 'lib64', '-lsleef'] #both lib and lib64 because of ubuntu vs redhat compatibility
)

sleefquad_dep = declare_dependency(
    dependencies: [sleef_build_dep],
    compile_args: compile_args_list,
    link_args: ['-L' + meson.current_build_dir() / sleef_install_dir / 'lib', '-L' + meson.current_build_dir() / sleef_install_dir / 'lib64', '-lsleefquad']
)