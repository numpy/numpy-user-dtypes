#ifndef _QUADDTYPE_SCALAR_H
#define _QUADDTYPE_SCALAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <sleef.h>
#include "quad_common.h"

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

#ifdef __cplusplus
}
#endif

#endif
