#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL quaddtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

typedef struct {
    PyArray_Descr base;
} QuadDTypeObject;

extern PyArray_DTypeMeta QuadDType;
extern PyTypeObject *QuadScalar_Type;

QuadDTypeObject *
new_quaddtype_instance(void);

int
init_quad_dtype(void);

#endif /*_NPY_DTYPE_H*/
