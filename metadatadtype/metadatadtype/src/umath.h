#ifndef _NPY_UFUNC_H
#define _NPY_UFUNC_H

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

int
init_ufuncs(PyArray_DTypeMeta *MetadataDType);

#endif /*_NPY_UFUNC_H */
