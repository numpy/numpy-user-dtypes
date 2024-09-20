#ifndef _QUADDTYPE_SCALAR_H
#define _QUADDTYPE_SCALAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <sleef.h>
#include "quad_common.h"

typedef union {
    Sleef_quad sleef_value;
    long double longdouble_value;
} quad_value;

typedef struct {
    PyObject_HEAD
    quad_value value;
    QuadBackendType backend;
} QuadPrecisionObject;

extern PyTypeObject QuadPrecision_Type;

QuadPrecisionObject *
QuadPrecision_raw_new(QuadBackendType backend);

QuadPrecisionObject *
QuadPrecision_from_object(PyObject *value, QuadBackendType backend);

int
init_quadprecision_scalar(void);

#define PyArray_IsScalar(obj, QuadPrecDType) PyObject_TypeCheck(obj, &QuadPrecision_Type)
#define PyArrayScalar_VAL(obj, QuadPrecDType) (((QuadPrecisionObject *)obj)->value)

#ifdef __cplusplus
}
#endif

#endif