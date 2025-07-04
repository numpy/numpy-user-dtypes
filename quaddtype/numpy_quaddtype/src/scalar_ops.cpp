#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY

extern "C" {
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "numpy/dtype_api.h"
}

#include "scalar.h"
#include "ops.hpp"
#include "scalar_ops.h"
#include "quad_common.h"

template <unary_op_quad_def sleef_op, unary_op_longdouble_def longdouble_op>
static PyObject *
quad_unary_func(QuadPrecisionObject *self)
{
    QuadPrecisionObject *res = QuadPrecision_raw_new(self->backend);
    if (!res) {
        return NULL;
    }

    if (self->backend == BACKEND_SLEEF) {
        res->value.sleef_value = sleef_op(&self->value.sleef_value);
    }
    else {
        res->value.longdouble_value = longdouble_op(&self->value.longdouble_value);
    }
    return (PyObject *)res;
}

PyObject *
quad_nonzero(QuadPrecisionObject *self)
{
    if (self->backend == BACKEND_SLEEF) {
        return PyBool_FromLong(Sleef_icmpneq1(self->value.sleef_value, Sleef_cast_from_int64q1(0)));
    }
    else {
        return PyBool_FromLong(self->value.longdouble_value != 0.0L);
    }
}

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
static PyObject *
quad_binary_func(PyObject *op1, PyObject *op2)
{
    QuadPrecisionObject *self;
    PyObject *other;
    QuadPrecisionObject *other_quad = NULL;
    int is_forward;
    QuadBackendType backend;

    if (PyObject_TypeCheck(op1, &QuadPrecision_Type)) {
        is_forward = 1;
        self = (QuadPrecisionObject *)op1;
        other = Py_NewRef(op2);
        backend = self->backend;
    }
    else {
        is_forward = 0;
        self = (QuadPrecisionObject *)op2;
        other = Py_NewRef(op1);
        backend = self->backend;
    }

    if (PyObject_TypeCheck(other, &QuadPrecision_Type)) {
        Py_INCREF(other);
        other_quad = (QuadPrecisionObject *)other;
        if (other_quad->backend != backend) {
            PyErr_SetString(PyExc_TypeError, "Cannot mix QuadPrecision backends");
            Py_DECREF(other);
            return NULL;
        }
    }
    else if (PyLong_Check(other) || PyFloat_Check(other)) {
        other_quad = QuadPrecision_from_object(other, backend);
        if (!other_quad) {
            Py_DECREF(other);
            return NULL;
        }
    }
    else {
        Py_DECREF(other);
        Py_RETURN_NOTIMPLEMENTED;
    }

    QuadPrecisionObject *res = QuadPrecision_raw_new(backend);
    if (!res) {
        Py_DECREF(other_quad);
        Py_DECREF(other);
        return NULL;
    }

    if (backend == BACKEND_SLEEF) {
        if (is_forward) {
            res->value.sleef_value =
                    sleef_op(&self->value.sleef_value, &other_quad->value.sleef_value);
        }
        else {
            res->value.sleef_value =
                    sleef_op(&other_quad->value.sleef_value, &self->value.sleef_value);
        }
    }
    else {
        if (is_forward) {
            res->value.longdouble_value = longdouble_op(&self->value.longdouble_value,
                                                        &other_quad->value.longdouble_value);
        }
        else {
            res->value.longdouble_value = longdouble_op(&other_quad->value.longdouble_value,
                                                        &self->value.longdouble_value);
        }
    }

    Py_DECREF(other_quad);
    Py_DECREF(other);
    return (PyObject *)res;
}

PyObject *
quad_richcompare(QuadPrecisionObject *self, PyObject *other, int cmp_op)
{
    QuadPrecisionObject *other_quad = NULL;
    QuadBackendType backend = self->backend;

    if (PyObject_TypeCheck(other, &QuadPrecision_Type)) {
        Py_INCREF(other);
        other_quad = (QuadPrecisionObject *)other;
        if (other_quad->backend != backend) {
            PyErr_SetString(PyExc_TypeError,
                            "Cannot compare QuadPrecision objects with different backends");
            Py_DECREF(other_quad);
            return NULL;
        }
    }
    else if (PyLong_CheckExact(other) || PyFloat_CheckExact(other)) {
        other_quad = QuadPrecision_from_object(other, backend);
        if (other_quad == NULL) {
            return NULL;
        }
    }
    else {
        Py_RETURN_NOTIMPLEMENTED;
    }

    int cmp;
    if (backend == BACKEND_SLEEF) {
        switch (cmp_op) {
            case Py_LT:
                cmp = Sleef_icmpltq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            case Py_LE:
                cmp = Sleef_icmpleq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            case Py_EQ:
                cmp = Sleef_icmpeqq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            case Py_NE:
                cmp = Sleef_icmpneq1(self->value.sleef_value, other_quad->value.sleef_value) || Sleef_iunordq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            case Py_GT:
                cmp = Sleef_icmpgtq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            case Py_GE:
                cmp = Sleef_icmpgeq1(self->value.sleef_value, other_quad->value.sleef_value);
                break;
            default:
                Py_DECREF(other_quad);
                Py_RETURN_NOTIMPLEMENTED;
        }
    }
    else {
        switch (cmp_op) {
            case Py_LT:
                cmp = self->value.longdouble_value < other_quad->value.longdouble_value;
                break;
            case Py_LE:
                cmp = self->value.longdouble_value <= other_quad->value.longdouble_value;
                break;
            case Py_EQ:
                cmp = self->value.longdouble_value == other_quad->value.longdouble_value;
                break;
            case Py_NE:
                cmp = self->value.longdouble_value != other_quad->value.longdouble_value;
                break;
            case Py_GT:
                cmp = self->value.longdouble_value > other_quad->value.longdouble_value;
                break;
            case Py_GE:
                cmp = self->value.longdouble_value >= other_quad->value.longdouble_value;
                break;
            default:
                Py_DECREF(other_quad);
                Py_RETURN_NOTIMPLEMENTED;
        }
    }
    Py_DECREF(other_quad);

    return PyBool_FromLong(cmp);
}

static PyObject *
QuadPrecision_float(QuadPrecisionObject *self)
{
    if (self->backend == BACKEND_SLEEF) {
        return PyFloat_FromDouble(Sleef_cast_to_doubleq1(self->value.sleef_value));
    }
    else {
        return PyFloat_FromDouble((double)self->value.longdouble_value);
    }
}

static PyObject *
QuadPrecision_int(QuadPrecisionObject *self)
{
    if (self->backend == BACKEND_SLEEF) {
        return PyLong_FromLongLong(Sleef_cast_to_int64q1(self->value.sleef_value));
    }
    else {
        return PyLong_FromLongLong((long long)self->value.longdouble_value);
    }
}

PyNumberMethods quad_as_scalar = {
        .nb_add = (binaryfunc)quad_binary_func<quad_add, ld_add>,
        .nb_subtract = (binaryfunc)quad_binary_func<quad_sub, ld_sub>,
        .nb_multiply = (binaryfunc)quad_binary_func<quad_mul, ld_mul>,
        .nb_power = (ternaryfunc)quad_binary_func<quad_pow, ld_pow>,
        .nb_negative = (unaryfunc)quad_unary_func<quad_negative, ld_negative>,
        .nb_positive = (unaryfunc)quad_unary_func<quad_positive, ld_positive>,
        .nb_absolute = (unaryfunc)quad_unary_func<quad_absolute, ld_absolute>,
        .nb_bool = (inquiry)quad_nonzero,
        .nb_int = (unaryfunc)QuadPrecision_int,
        .nb_float = (unaryfunc)QuadPrecision_float,
        .nb_true_divide = (binaryfunc)quad_binary_func<quad_div, ld_div>,
};