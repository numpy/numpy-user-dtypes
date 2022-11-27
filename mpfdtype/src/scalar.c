#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "scalar.h"
#include "numbers.h"


mpfr_prec_t
get_prec_from_object(PyObject *value) {
    Py_ssize_t prec;
    if (PyFloat_Check(value)) {
        prec = 53;
    }
    else if (PyLong_Check(value)) {
        prec = 8 * sizeof(Py_ssize_t);
    }
    else if (PyObject_TypeCheck(value, &MPFloat_Type)) {
        prec = mpfr_get_prec(((MPFloatObject *)value)->mpf.x);
    }
    else if (PyUnicode_CheckExact(value)) {
        PyErr_Format(PyExc_TypeError,
                "MPFloat: precsion must currently be given to parse string.");
        return -1;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "MPFloat value must be an MPF, float, or string");
                return -1;
    }
    return prec;
}


/*
 * Get a 0 initialized new scalar, the value may be changed immediately after
 * getting the new scalar.  (otherwise scalars are immutable)
 */
MPFloatObject *
MPFLoat_raw_new(mpfr_prec_t prec)
{
    size_t n_limb = mpfr_custom_get_size(prec) / sizeof(mp_limb_t);
    MPFloatObject *new = PyObject_NewVar(MPFloatObject, &MPFloat_Type, n_limb);
    if (new == NULL) {
        return NULL;
    }
    mpfr_custom_init_set(
        new->mpf.x, MPFR_ZERO_KIND, 0, prec, new->mpf.significand);

    return new;
}


MPFloatObject *
MPFloat_from_object(PyObject *value, Py_ssize_t prec)
{
    if (prec != -1) {
        if (prec < MPFR_PREC_MIN || prec > MPFR_PREC_MAX) {
            PyErr_Format(PyExc_ValueError,
                    "precision must be between %d and %d.",
                    MPFR_PREC_MIN, MPFR_PREC_MAX);
            return NULL;
        }
    }
    else {
        prec = get_prec_from_object(value);
        if (prec < 0) {
            return NULL;
        }
    }

    MPFloatObject *self = MPFLoat_raw_new(prec);
    if (self == NULL) {
        return NULL;
    }

    if (PyFloat_Check(value)) {
        mpfr_set_d(self->mpf.x, PyFloat_AsDouble(value), MPFR_RNDN);
    }
    else if (PyLong_Check(value)) {
        Py_ssize_t val = PyLong_AsSsize_t(value);
        if (val == -1 && PyErr_Occurred()) {
            return NULL;
        }
        int ternary = mpfr_set_sj(self->mpf.x, val, MPFR_RNDN);
        if (ternary != 0) {
            // TODO: Not sure this should always raise, since a float will
            //       still be close to correct.
            PyErr_Format(PyExc_ValueError,
                "%zd could not be converted to MPFloat with precision %zd",
                val, (Py_ssize_t)prec);
            Py_DECREF(self);
            return NULL;
        }
    }
    else if (PyObject_TypeCheck(value, &MPFloat_Type)) {
        mpfr_set(self->mpf.x, ((MPFloatObject *)value)->mpf.x, MPFR_RNDN);
    }
    else if (PyUnicode_CheckExact(value)) {
        // TODO: Might be better to use mpfr_strtofr
        Py_ssize_t s_length;
        const char *s = PyUnicode_AsUTF8AndSize(value, &s_length);
        char *end;
        mpfr_strtofr(self->mpf.x, s, &end, 10, MPFR_RNDN);
        if (s + s_length != end) {
            PyErr_SetString(PyExc_ValueError,
                    "unable to parse string to MPFloat.");
            Py_DECREF(self);
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                "MPFloat value must be an MPF, float, or string");
        Py_DECREF(self);
        return NULL;
    }

    return self;
}


static PyObject *
MPFloat_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    char *keywords[] = {"", "prec", NULL};
    PyObject *value;
    Py_ssize_t prec = -1;  /* default precision may be discovered */

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "O|$n", keywords, &value, &prec)) {
        return NULL;
    }

    return (PyObject *)MPFloat_from_object(value, prec);
}


static PyObject *
MPFloat_str(MPFloatObject* self)
{
    char *val_repr;
    char format[100];

    mpfr_prec_t precision = mpfr_min_prec(self->mpf.x);

    // TODO: I am not 100% sure this ensures round-tripping.
    size_t ndigits = mpfr_get_str_ndigits(10, precision);
    ndigits -= 1;
    if (ndigits < 0) {
        ndigits = 0;
    }
    snprintf(format, 99, "%%.%zdRE", ndigits);

    /* Note: For format characters other than "e" precision is needed */
    if (mpfr_asprintf(&val_repr, format, self->mpf.x) < 0) {
        /* Lets assume errors must be memory errors. */
        PyErr_NoMemory();
        return NULL;
    }
    PyObject *str = PyUnicode_FromString(val_repr);
    mpfr_free_str(val_repr);
    return str;
}


static PyObject *
MPFloat_repr(MPFloatObject* self)
{
    PyObject *val_repr = MPFloat_str(self);
    if (val_repr == NULL) {
        return NULL;
    }

    PyObject *res = PyUnicode_FromFormat(
            "MPFloat('%S', prec=%zd)", val_repr, mpfr_get_prec(self->mpf.x));
    Py_DECREF(val_repr);
    return res;
}


int
init_mpf_scalar(void)
{
    return PyType_Ready(&MPFloat_Type);
}


PyTypeObject MPFloat_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "MPFDType.MPFDType",
    .tp_basicsize = sizeof(MPFloatObject),
    .tp_itemsize = sizeof(mp_limb_t),
    .tp_new = MPFloat_new,
    .tp_repr = (reprfunc)MPFloat_repr,
    .tp_str = (reprfunc)MPFloat_str,
    .tp_as_number = &mpf_as_number,
    .tp_richcompare = &mpf_richcompare,
};
