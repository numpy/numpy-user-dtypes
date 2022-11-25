#ifndef _MPRFDTYPE_SCALAR_H
#define _MPRFDTYPE_SCALAR_H

#include <Python.h>

#include "mpfr.h"


typedef struct {
    mpfr_t x;
    mp_limb_t significand[];
} mpf_field;


typedef struct {
    PyObject_HEAD;
    mpf_field mpf;
} MPFloatObject;


extern PyTypeObject MPFloat_Type;

MPFloatObject *
MPFLoat_raw_new(mpfr_prec_t prec);

mpfr_prec_t
get_prec_from_object(PyObject *value);

PyObject *
MPFloat_from_object(PyObject *value, Py_ssize_t prec);

int
init_mpf_scalar(void);

#endif  /* _MPRFDTYPE_SCALAR_H */