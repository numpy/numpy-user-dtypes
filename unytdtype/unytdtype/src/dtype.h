#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

// clang-format off
#include <Python.h>
#include "structmember.h"
// clang-format on

#define PY_ARRAY_UNIQUE_SYMBOL unytdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

typedef struct {
    PyArray_Descr base;
    PyObject *unit;
} UnytDTypeObject;

extern PyArray_DTypeMeta UnytDType;
extern PyTypeObject *UnytScalar_Type;

UnytDTypeObject *
new_unytdtype_instance(PyObject *unit);

int
init_unyt_dtype(void);

#endif /*_NPY_DTYPE_H*/
