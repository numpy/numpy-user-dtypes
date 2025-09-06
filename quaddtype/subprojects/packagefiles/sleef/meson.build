project('sleef', version: '3.8')

cmake = find_program('cmake')
ninja = find_program('ninja', 'make', required: false)

# Use relative paths from build directory
sleef_build_dir = 'sleef_build'
sleef_install_dir = 'sleef_install'

# Configure SLEEF at configuration time
sleef_configure = run_command([
    cmake, 
    '-S', meson.current_source_dir(),
    '-B', meson.current_build_dir() / sleef_build_dir,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DSLEEF_BUILD_QUAD=ON',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DSLEEF_BUILD_TESTS=OFF',
    '-DSLEEF_BUILD_INLINE_HEADERS=OFF',
    '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    '-DCMAKE_INSTALL_PREFIX=' + meson.current_build_dir() / sleef_install_dir
], check: false, capture: true)

if sleef_configure.returncode() != 0
    error('SLEEF CMake configuration failed: ' + sleef_configure.stderr())
endif

# Build target for SLEEF libraries - create a dummy output file
sleef_build_target = custom_target('sleef_build',
    command: [cmake, '--build', meson.current_build_dir() / sleef_build_dir, '--target', 'install', '--parallel'],
    output: 'sleef_built.stamp',  # Dummy stamp file
    console: true,
    build_always_stale: true,
    build_by_default: true
)

# Path variables
sleef_include_path = meson.current_build_dir() / sleef_install_dir / 'include'
sleef_lib_path = meson.current_build_dir() / sleef_install_dir / 'lib'

# Create a dependency that ensures the build happens but doesn't link the dummy file
sleef_build_dep = declare_dependency(sources: [sleef_build_target])

# Create the actual linking dependencies
sleef_dep = declare_dependency(
    dependencies: [sleef_build_dep],  # Ensures build happens first
    compile_args: ['-I' + sleef_include_path],
    link_args: ['-L' + sleef_lib_path, '-lsleef']
)

sleefquad_dep = declare_dependency(
    dependencies: [sleef_build_dep],  # Ensures build happens first
    compile_args: ['-I' + sleef_include_path],
    link_args: ['-L' + sleef_lib_path, '-lsleefquad']
)