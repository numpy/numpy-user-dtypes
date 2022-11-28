#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

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
