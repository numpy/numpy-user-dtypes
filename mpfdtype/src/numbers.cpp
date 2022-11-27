#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY

extern "C" {
    #include <Python.h>

    #include "numpy/arrayobject.h"
    #include "numpy/ndarraytypes.h"
    #include "numpy/ufuncobject.h"

    #include "numpy/experimental_dtype_api.h"
}

#include <algorithm>

#include "scalar.h"
#include "ops.hpp"

#include "numbers.h"


template<unary_op_def unary_op>
PyObject *
unary_func(MPFloatObject *self)
{
    MPFloatObject *res = MPFLoat_raw_new(mpfr_get_prec(self->mpf.x));
    if (res == NULL) {
        return NULL;
    }

    unary_op(self->mpf.x, res->mpf.x);
    return (PyObject *)res;
}


template<binop_def binary_op>
PyObject *
binary_func(PyObject *op1, PyObject *op2)
{
    MPFloatObject *self;
    PyObject *other;
    MPFloatObject *other_mpf = NULL;
    mpfr_prec_t precision;

    int is_forward;  /* We may be a forward operation (so can defer) */
    if (PyObject_TypeCheck(op1, &MPFloat_Type)) {
        is_forward = 1;
        self = (MPFloatObject *)op2;
        other = Py_NewRef(op1);
    }
    else {
        is_forward = 0;
        self = (MPFloatObject *)op1;
        other = Py_NewRef(op2);
    }

    precision = mpfr_get_prec(self->mpf.x);

    if (PyObject_TypeCheck(other, &MPFloat_Type)) {
        /* All good, we can continue, both are MPFloats */
        Py_INCREF(other);
        other_mpf = (MPFloatObject *)other;
        precision = std::max(precision, mpfr_get_prec(other_mpf->mpf.x));
    }
    else if (PyLong_CheckExact(other) || PyFloat_CheckExact(other) ||
                (!is_forward &&
                    (PyLong_Check(other) || PyFloat_Check(other)))) {
        // TODO: We want week handling, so truncate precision.  But is it
        //       correct to do it here? (not that it matters much...)
        other_mpf = MPFloat_from_object(other, precision);
        if (other_mpf == NULL) {
            return NULL;
        }
    }
    else {
        /* Defer to the other (NumPy types are handled through array path) */
        Py_RETURN_NOTIMPLEMENTED;
    }

    MPFloatObject *res = MPFLoat_raw_new(precision);
    if (res == NULL) {
        Py_DECREF(other_mpf);
        return NULL;
    }
    if (is_forward) {
        binary_op(res->mpf.x, self->mpf.x, other_mpf->mpf.x);
    }
    else{
        binary_op(res->mpf.x, other_mpf->mpf.x, self->mpf.x);
    }

    Py_DECREF(other_mpf);
    return (PyObject *)res;
}


PyNumberMethods mpf_as_number = {
    .nb_add = (binaryfunc)binary_func<add>,
    .nb_subtract = (binaryfunc)binary_func<sub>,
    .nb_multiply = (binaryfunc)binary_func<mul>,
    //.nb_remainder = (binaryfunc)gentype_remainder,
    //.nb_divmod = (binaryfunc)gentype_divmod,
    //.nb_power = (ternaryfunc)gentype_power,
    .nb_negative = (unaryfunc)unary_func<negative>,
    .nb_positive = (unaryfunc)unary_func<positive>,
    .nb_absolute = (unaryfunc)unary_func<absolute>,
    //.nb_bool = (inquiry)gentype_nonzero_number,
    //.nb_int = (unaryfunc)gentype_int,
    //.nb_float = (unaryfunc)gentype_float,
    //.nb_floor_divide = (binaryfunc)gentype_floor_divide,
    .nb_true_divide = (binaryfunc)binary_func<div>,
};
