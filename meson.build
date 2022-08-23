project(
  'quaddtype',
  'c',
  default_options [
    'c_std=c99',
  ],
)

cc = meson.get_compiler('c')

py_mod = import('python')
py = py_mod.find_installation()
py_dep = py.dependency()

subdir('quaddtype')
