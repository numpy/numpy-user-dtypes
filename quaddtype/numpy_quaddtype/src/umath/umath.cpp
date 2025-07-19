#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
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
#include "binary_ops.h"
#include "comparison_ops.h"
#include "matmul.h"

// helper debugging function
static const char *
get_dtype_name(PyArray_DTypeMeta *dtype)
{
    if (dtype == &QuadPrecDType) {
        return "QuadPrecDType";
    }
    else if (dtype == &PyArray_BoolDType) {
        return "BoolDType";
    }
    else if (dtype == &PyArray_ByteDType) {
        return "ByteDType";
    }
    else if (dtype == &PyArray_UByteDType) {
        return "UByteDType";
    }
    else if (dtype == &PyArray_ShortDType) {
        return "ShortDType";
    }
    else if (dtype == &PyArray_UShortDType) {
        return "UShortDType";
    }
    else if (dtype == &PyArray_IntDType) {
        return "IntDType";
    }
    else if (dtype == &PyArray_UIntDType) {
        return "UIntDType";
    }
    else if (dtype == &PyArray_LongDType) {
        return "LongDType";
    }
    else if (dtype == &PyArray_ULongDType) {
        return "ULongDType";
    }
    else if (dtype == &PyArray_LongLongDType) {
        return "LongLongDType";
    }
    else if (dtype == &PyArray_ULongLongDType) {
        return "ULongLongDType";
    }
    else if (dtype == &PyArray_FloatDType) {
        return "FloatDType";
    }
    else if (dtype == &PyArray_DoubleDType) {
        return "DoubleDType";
    }
    else if (dtype == &PyArray_LongDoubleDType) {
        return "LongDoubleDType";
    }
    else {
        return "UnknownDType";
    }
}

int
init_quad_umath(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) {
        PyErr_SetString(PyExc_ImportError, "Failed to import numpy module");
        return -1;
    }

    if (init_quad_unary_ops(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad unary operations");
        goto err;
    }

    if (init_quad_binary_ops(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad binary operations");
        goto err;
    }

    if (init_quad_comps(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad comparison operations");
        goto err;
    }

    if (init_matmul_ops(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad matrix multiplication operations");
        goto err;
    }

    Py_DECREF(numpy);
    return 0;

err:
    Py_DECREF(numpy);
    return -1;
}