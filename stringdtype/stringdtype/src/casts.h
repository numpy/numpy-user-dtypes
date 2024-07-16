#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

// needed for Py_UCS4
#include <Python.h>

// need these defines and includes for PyArrayMethod_Spec
#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

PyArrayMethod_Spec **
get_casts();

#endif /* _NPY_CASTS_H */
