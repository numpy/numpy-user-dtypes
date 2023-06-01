#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

PyArrayMethod_Spec **
get_casts(PyArray_DTypeMeta *this_dtype, PyArray_DTypeMeta *other_dtype);

size_t
utf8_char_to_ucs4_code(unsigned char *, Py_UCS4 *);

#endif /* _NPY_CASTS_H */
