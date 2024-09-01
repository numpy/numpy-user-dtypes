#ifndef _QUADDTYPE_DTYPE_H
#define _QUADDTYPE_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include <numpy/ndarraytypes.h>
#include <numpy/dtype_api.h>
#include "quad_common.h"

typedef struct {
    PyArray_Descr base;
    QuadBackendType backend;
} QuadPrecDTypeObject;

extern PyArray_DTypeMeta QuadPrecDType;

QuadPrecDTypeObject *
new_quaddtype_instance(QuadBackendType backend);

int
init_quadprec_dtype(void);

#ifdef __cplusplus
}
#endif

#endif