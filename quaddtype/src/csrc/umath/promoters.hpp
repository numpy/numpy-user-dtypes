#ifndef _QUADDTYPE_PROMOTERS
#define _QUADDTYPE_PROMOTERS

#include <Python.h>
#include <cstdio>
#include <cassert>
#include "numpy/arrayobject.h"
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

#include "../dtype.h"

inline int
quad_ufunc_promoter(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                    PyArray_DTypeMeta *signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    int nin = ufunc->nin;
    int nargs = ufunc->nargs;
    PyArray_DTypeMeta *common = NULL;
    bool has_quad = false;

    // Handle the special case for reductions
    if (op_dtypes[0] == NULL) {
        assert(nin == 2 && ufunc->nout == 1); /* must be reduction */
        for (int i = 0; i < 3; i++) {
            Py_INCREF(op_dtypes[1]);
            new_op_dtypes[i] = op_dtypes[1];
        }
        return 0;
    }

    // Check if any input or signature is QuadPrecision
    for (int i = 0; i < nin; i++) {
        if (op_dtypes[i] == &QuadPrecDType) {
            has_quad = true;
        }
    }

    if (has_quad) {
        common = &QuadPrecDType;
    }
    else {
        for (int i = nin; i < nargs; i++) {
            if (signature[i] != NULL) {
                if (common == NULL) {
                    Py_INCREF(signature[i]);
                    common = signature[i];
                }
                else if (common != signature[i]) {
                    Py_CLEAR(common);  // Not homogeneous, unset common
                    break;
                }
            }
        }
    }
    // If no common output dtype, use standard promotion for inputs
    if (common == NULL) {
        common = PyArray_PromoteDTypeSequence(nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();  // Do not propagate normal promotion errors
            }

            return -1;
        }
    }

    // Set all new_op_dtypes to the common dtype
    for (int i = 0; i < nargs; i++) {
        if (signature[i]) {
            // If signature is specified for this argument, use it
            Py_INCREF(signature[i]);
            new_op_dtypes[i] = signature[i];
        }
        else {
            // Otherwise, use the common dtype
            Py_INCREF(common);

            new_op_dtypes[i] = common;
        }
    }

    Py_XDECREF(common);

    return 0;
}


#endif