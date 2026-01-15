#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>
#include <cstdio>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
}
#include "../quad_common.h"
#include "../scalar.h"
#include "../dtype.h"
#include "umath.h"
#include "../ops.hpp"
#include "unary_ops.h"
#include "unary_props.h"
#include "binary_ops.h"
#include "comparison_ops.h"
#include "matmul.h"

int
init_quad_umath(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) {
        PyErr_SetString(PyExc_ImportError, "Failed to import numpy module");
        return -1;
    }

    if (init_quad_unary_ops(numpy) < 0) {
        goto err;
    }

    if (init_quad_unary_props(numpy) < 0) {
        goto err;
    }

    if (init_quad_binary_ops(numpy) < 0) {
        goto err;
    }

    if (init_quad_comps(numpy) < 0) {
        goto err;
    }

    if (init_matmul_ops(numpy) < 0) {
        goto err;
    }

    Py_DECREF(numpy);
    return 0;

err:
    // Already raises appropriate error
    Py_DECREF(numpy);
    return -1;
}