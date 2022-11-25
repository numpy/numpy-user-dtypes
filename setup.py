from os import path


def configuration(parent_package='', top_path=None):
    import numpy
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    cfiles = ["mpfdtype_main.c", "scalar.c", "dtype.c", "casts.cpp", "umath.c"]
    cfiles = [path.join('mpfdtype/src', f) for f in cfiles]

    config.add_subpackage('mpfdtype')

    config.add_extension(
        'mpfdtype._mpfdtype_main',
        sources=cfiles,
        libraries=["mpfr"],
        include_dirs=[numpy.get_include(), "mpfdtype/src"],)
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name="mpfdtype",
        configuration=configuration,)

