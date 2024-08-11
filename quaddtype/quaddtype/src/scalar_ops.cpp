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

template <unary_op_def unary_op>
static PyObject *
quad_unary_func(QuadPrecisionObject *self)
{
    QuadPrecisionObject *res = QuadPrecision_raw_new();
    if (!res) {
        return NULL;
    }

    unary_op(&self->quad.value, &res->quad.value);
    return (PyObject *)res;
}

PyObject *
quad_nonzero(QuadPrecisionObject *self)
{
    return PyBool_FromLong(Sleef_icmpneq1(self->quad.value, Sleef_cast_from_int64q1(0)));
}

template <binop_def binary_op>
static PyObject *
quad_binary_func(PyObject *op1, PyObject *op2)
{
    QuadPrecisionObject *self;
    PyObject *other;
    QuadPrecisionObject *other_quad = NULL;
    int is_forward;

    if (PyObject_TypeCheck(op1, &QuadPrecision_Type)) {
        is_forward = 1;
        self = (QuadPrecisionObject *)op1;
        other = Py_NewRef(op2);
    }
    else {
        is_forward = 0;
        self = (QuadPrecisionObject *)op2;
        other = Py_NewRef(op1);
    }

    if (PyObject_TypeCheck(other, &QuadPrecision_Type)) {
        Py_INCREF(other);
        other_quad = (QuadPrecisionObject *)other;
    }
    else if (PyLong_Check(other) || PyFloat_Check(other)) {
        other_quad = QuadPrecision_raw_new();
        if (!other_quad) {
            Py_DECREF(other);
            return NULL;
        }

        if (PyLong_Check(other)) {
            long long value = PyLong_AsLongLong(other);
            if (value == -1 && PyErr_Occurred()) {
                Py_DECREF(other);
                Py_DECREF(other_quad);
                return NULL;
            }
            other_quad->quad.value = Sleef_cast_from_int64q1(value);
        }
        else {
            double value = PyFloat_AsDouble(other);
            if (value == -1.0 && PyErr_Occurred()) {
                Py_DECREF(other);
                Py_DECREF(other_quad);
                return NULL;
            }
            other_quad->quad.value = Sleef_cast_from_doubleq1(value);
        }
    }
    else {
        Py_DECREF(other);
        Py_RETURN_NOTIMPLEMENTED;
    }

    QuadPrecisionObject *res = QuadPrecision_raw_new();
    if (!res) {
        Py_DECREF(other_quad);
        Py_DECREF(other);
        return NULL;
    }

    if (is_forward) {
        binary_op(&res->quad.value, &self->quad.value, &other_quad->quad.value);
    }
    else {
        binary_op(&res->quad.value, &other_quad->quad.value, &self->quad.value);
    }

    Py_DECREF(other_quad);
    Py_DECREF(other);
    return (PyObject *)res;
}

// todo: add support with float and int
PyObject *
quad_richcompare(QuadPrecisionObject *self, PyObject *other, int cmp_op)
{
    QuadPrecisionObject *other_quad = NULL;

    if (PyObject_TypeCheck(other, &QuadPrecision_Type)) {
        Py_INCREF(other);
        other_quad = (QuadPrecisionObject *)other;
    }
    else if (PyLong_CheckExact(other) || PyFloat_CheckExact(other)) {
        other_quad = QuadPrecision_from_object(other);
        if (other_quad == NULL) {
            return NULL;
        }
    }
    else {
        Py_RETURN_NOTIMPLEMENTED;
    }
    int cmp;
    switch (cmp_op) {
        case Py_LT:
            cmp = Sleef_icmpltq1(self->quad.value, other_quad->quad.value);
            break;
        case Py_LE:
            cmp = Sleef_icmpleq1(self->quad.value, other_quad->quad.value);
            break;
        case Py_EQ:
            cmp = Sleef_icmpeqq1(self->quad.value, other_quad->quad.value);
            break;
        case Py_NE:
            cmp = Sleef_icmpneq1(self->quad.value, other_quad->quad.value);
            break;
        case Py_GT:
            cmp = Sleef_icmpgtq1(self->quad.value, other_quad->quad.value);
            break;
        case Py_GE:
            cmp = Sleef_icmpgeq1(self->quad.value, other_quad->quad.value);
            break;
        default:
            Py_DECREF(other_quad);
            Py_RETURN_NOTIMPLEMENTED;
    }
    Py_DECREF(other_quad);

    return PyBool_FromLong(cmp);
}

PyNumberMethods quad_as_scalar = {
        .nb_add = (binaryfunc)quad_binary_func<quad_add>,
        .nb_subtract = (binaryfunc)quad_binary_func<quad_sub>,
        .nb_multiply = (binaryfunc)quad_binary_func<quad_mul>,
        .nb_true_divide = (binaryfunc)quad_binary_func<quad_div>,
        .nb_power = (ternaryfunc)quad_binary_func<quad_pow>,
        .nb_negative = (unaryfunc)quad_unary_func<quad_negative>,
        .nb_positive = (unaryfunc)quad_unary_func<quad_positive>,
        .nb_absolute = (unaryfunc)quad_unary_func<quad_absolute>,
        .nb_bool = (inquiry)quad_nonzero,
};