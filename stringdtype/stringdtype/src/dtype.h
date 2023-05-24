#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

// clang-format off
#include <Python.h>
#include "structmember.h"
// clang-format on

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"
#include "numpy/npy_math.h"

typedef struct {
    PyArray_Descr base;
    PyObject *na_object;
} StringDTypeObject;

extern PyArray_DTypeMeta StringDType;
extern PyTypeObject *StringScalar_Type;
extern PyObject *NA_OBJ;

StringDTypeObject *
new_stringdtype_instance(void);

int
init_string_dtype(void);

int
compare(void *, void *, void *);

int
init_string_na_object(PyObject *mod);

// from dtypemeta.h, not public in numpy
#define NPY_DTYPE(descr) ((PyArray_DTypeMeta *)Py_TYPE(descr))

#endif /*_NPY_DTYPE_H*/
