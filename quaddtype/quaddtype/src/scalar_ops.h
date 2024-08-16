#ifndef _QUADDTYPE_SCALAR_OPS_H
#define _QUADDTYPE_SCALAR_OPS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "dtype.h"

PyObject *
quad_richcompare(QuadPrecisionObject *self, PyObject *other, int cmp_op);

extern PyNumberMethods quad_as_scalar;

#ifdef __cplusplus
}
#endif

#endif
