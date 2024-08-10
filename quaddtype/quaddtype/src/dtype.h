#ifndef _QUADDTYPE_DTYPE_H
#define _QUADDTYPE_DTYPE_H
#ifdef __cplusplus
extern "C" {
#endif

#include<Python.h>
#include<sleef.h>
#include<numpy/ndarraytypes.h>
#include<numpy/dtype_api.h>

#include "scalar.h"

typedef struct
{
    PyArray_Descr base;
    
} QuadPrecDTypeObject;

extern PyArray_DTypeMeta QuadPrecDType;

QuadPrecDTypeObject * new_quaddtype_instance(void);

int init_quadprec_dtype(void);

#ifdef __cplusplus
}
#endif

#endif