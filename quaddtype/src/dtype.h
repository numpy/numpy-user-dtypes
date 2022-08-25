#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"


typedef struct {
    PyArray_Descr base;
    PyObject *unit;
} QuadDTypeObject;

extern PyArray_DTypeMeta QuadDType;

QuadDTypeObject * new_quaddtype_instance(PyObject *unit);

// int init_unit_dtype(void);

#endif  /*_NPY_DTYPE_H*/
