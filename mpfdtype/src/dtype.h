#ifndef _MPRFDTYPE_DTYPE_H
#define _MPRFDTYPE_DTYPE_H

#include "mpfr.h"

#include "scalar.h"


typedef struct {
    PyArray_Descr base;
    mpfr_prec_t precision;
} MPFDTypeObject;

extern PyArray_DTypeMeta MPFDType;

MPFDTypeObject *
new_MPFDType_instance(mpfr_prec_t precision);

int
init_mpf_dtype(void);

#endif  /*_MPRFDTYPE_DTYPE_H*/
