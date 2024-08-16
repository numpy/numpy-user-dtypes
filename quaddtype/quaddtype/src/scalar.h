#ifndef _QUADDTYPE_SCALAR_H
#define _QUADDTYPE_SCALAR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <sleef.h>

typedef struct {
    Sleef_quad value;
} quad_field;

typedef struct {
    PyObject_HEAD
    quad_field quad;
} QuadPrecisionObject;

extern PyTypeObject QuadPrecision_Type;

QuadPrecisionObject *
QuadPrecision_raw_new(void);

QuadPrecisionObject *
QuadPrecision_from_object(PyObject *value);

int
init_quadprecision_scalar(void);

#ifdef __cplusplus
}
#endif

#endif