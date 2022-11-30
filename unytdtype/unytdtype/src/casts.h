#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unytdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

/* Gets the conversion between two units: */
int
get_conversion_factor(PyObject *from_unit, PyObject *to_unit, double *factor,
                      double *offset);

PyArrayMethod_Spec **
get_casts(void);

int
UnitConverter(PyObject *obj, PyObject **unit);

#endif /* _NPY_CASTS_H */
