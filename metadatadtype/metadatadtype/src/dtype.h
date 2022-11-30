#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

#include <Python.h>
#include "structmember.h"

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

typedef struct {
    PyArray_Descr base;
    PyObject *metadata;
} MetadataDTypeObject;

extern PyArray_DTypeMeta MetadataDType;
extern PyTypeObject *MetadataScalar_Type;

MetadataDTypeObject *
new_metadatadtype_instance(PyObject *metadata);

int
init_metadata_dtype(void);

#endif /*_NPY_DTYPE_H*/
