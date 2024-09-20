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

PyObject* QuadPrecision_get_pi(PyObject* self, void* closure);
PyObject* QuadPrecision_get_e(PyObject* self, void* closure);

#define PyArray_IsScalar(obj, QuadPrecDType) PyObject_TypeCheck(obj, &QuadPrecision_Type)
#define PyArrayScalar_VAL(obj, QuadPrecDType) (((QuadPrecisionObject *)obj)->value)

QuadPrecisionObject* initialize_constants(const Sleef_quad value, QuadBackendType backend);

// constant objects
extern QuadPrecisionObject *QuadPrecision_pi;
extern QuadPrecisionObject *QuadPrecision_e;
extern QuadPrecisionObject *QuadPrecision_log2e;
extern QuadPrecisionObject *QuadPrecision_log10e;
extern QuadPrecisionObject *QuadPrecision_ln2;
extern QuadPrecisionObject *QuadPrecision_ln10;
extern QuadPrecisionObject *QuadPrecision_sqrt2;
extern QuadPrecisionObject *QuadPrecision_sqrt3;
extern QuadPrecisionObject *QuadPrecision_egamma;
extern QuadPrecisionObject *QuadPrecision_phi;
extern QuadPrecisionObject *QuadPrecision_quad_max;
extern QuadPrecisionObject *QuadPrecision_quad_min;
extern QuadPrecisionObject *QuadPrecision_quad_epsilon;
extern QuadPrecisionObject *QuadPrecision_quad_denorm_min;

#ifdef __cplusplus
}
#endif

#endif