#include <Python.h>
#include <sleef.h>
#include <sleefquad.h>
#include <stdlib.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "scalar.h"
#include "scalar_ops.h"
#include "dragon4.h"
#include "dtype.h"

// For IEEE 754 binary128 (quad precision), we need 36 decimal digits 
// to guarantee round-trip conversion (string -> parse -> equals original value)
// Formula: ceil(1 + MANT_DIG * log10(2)) = ceil(1 + 113 * 0.30103) = 36
// src: https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
#define SLEEF_QUAD_DECIMAL_DIG 36


QuadPrecisionObject *
QuadPrecision_raw_new(QuadBackendType backend)
{
    QuadPrecisionObject *new;
    new = PyObject_New(QuadPrecisionObject, &QuadPrecision_Type);

    if (!new)
        return NULL;
    new->backend = backend;
    if (backend == BACKEND_SLEEF) {
        new->value.sleef_value = Sleef_cast_from_doubleq1(0.0);
    }
    else {
        new->value.longdouble_value = 0.0L;
    }
    return new;
}

QuadPrecisionObject *
QuadPrecision_from_object(PyObject *value, QuadBackendType backend)
{   
    // Handle numpy scalars (np.int32, np.float32, etc.) before arrays
    // We need to check this before PySequence_Check because some numpy scalars are sequences
    if (PyArray_CheckScalar(value)) {
        QuadPrecisionObject *self = QuadPrecision_raw_new(backend);
        if (!self)
            return NULL;
        
        // Try as floating point first
        if (PyArray_IsScalar(value, Floating)) {
            PyObject *py_float = PyNumber_Float(value);
            if (py_float == NULL) {
                Py_DECREF(self);
                return NULL;
            }
            double dval = PyFloat_AsDouble(py_float);
            Py_DECREF(py_float);
            
            if (backend == BACKEND_SLEEF) {
                self->value.sleef_value = Sleef_cast_from_doubleq1(dval);
            }
            else {
                self->value.longdouble_value = (long double)dval;
            }
            return self;
        }
        // Try as integer
        else if (PyArray_IsScalar(value, Integer)) {
            PyObject *py_int = PyNumber_Long(value);
            if (py_int == NULL) {
                Py_DECREF(self);
                return NULL;
            }
            long long lval = PyLong_AsLongLong(py_int);
            Py_DECREF(py_int);
            
            if (backend == BACKEND_SLEEF) {
                self->value.sleef_value = Sleef_cast_from_int64q1(lval);
            }
            else {
                self->value.longdouble_value = (long double)lval;
            }
            return self;
        }
        // For other scalar types, fall through to error handling
        Py_DECREF(self);
    }
    
    // this checks arrays and sequences (array, tuple)
    // rejects strings; they're parsed below
    if (PyArray_Check(value) || (PySequence_Check(value) && !PyUnicode_Check(value) && !PyBytes_Check(value))) 
    {
        QuadPrecDTypeObject *dtype_descr = new_quaddtype_instance(backend);
        if (dtype_descr == NULL) {
            return NULL;
        }
        
        // steals reference to the descriptor
        PyObject *result = PyArray_FromAny(
            value,
            (PyArray_Descr *)dtype_descr,
            0,
            0,
            NPY_ARRAY_ENSUREARRAY, // this should handle the casting if possible
            NULL
        );
        
        // PyArray_FromAny steals the reference to dtype_descr, so no need to DECREF
        return (QuadPrecisionObject *)result;
    }

    QuadPrecisionObject *self = QuadPrecision_raw_new(backend);
    if (!self)
        return NULL;

    if (PyFloat_Check(value)) {
        double dval = PyFloat_AsDouble(value);
        if (backend == BACKEND_SLEEF) {
            self->value.sleef_value = Sleef_cast_from_doubleq1(dval);
        }
        else {
            self->value.longdouble_value = (long double)dval;
        }
    }
    else if (PyUnicode_CheckExact(value)) {
        const char *s = PyUnicode_AsUTF8(value);
        char *endptr = NULL;
        if (backend == BACKEND_SLEEF) {
            self->value.sleef_value = Sleef_strtoq(s, &endptr);
        }
        else {
            self->value.longdouble_value = strtold(s, &endptr);
        }
        if (*endptr != '\0' || endptr == s) {
            PyErr_SetString(PyExc_ValueError, "Unable to parse string to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
    }
    else if (PyLong_Check(value)) {
        long long val = PyLong_AsLongLong(value);
        if (val == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_OverflowError, "Overflow Error, value out of range");
            Py_DECREF(self);
            return NULL;
        }
        if (backend == BACKEND_SLEEF) {
            self->value.sleef_value = Sleef_cast_from_int64q1(val);
        }
        else {
            self->value.longdouble_value = (long double)val;
        }
    }
    else if (Py_TYPE(value) == &QuadPrecision_Type) {
        Py_DECREF(self);  // discard the default one
        QuadPrecisionObject *quad_obj = (QuadPrecisionObject *)value;
        // create a new one with the same backend
        QuadPrecisionObject *self = QuadPrecision_raw_new(quad_obj->backend);
        if (quad_obj->backend == BACKEND_SLEEF) {
            self->value.sleef_value = quad_obj->value.sleef_value;
        }
        else {
            self->value.longdouble_value = quad_obj->value.longdouble_value;
        }

        return self;
    }
    else {
        PyObject *type_str = PyObject_Str((PyObject *)Py_TYPE(value));
        if (type_str != NULL) {
            const char *type_cstr = PyUnicode_AsUTF8(type_str);
            if (type_cstr != NULL) {
                PyErr_Format(PyExc_TypeError,
                             "QuadPrecision value must be a quad, float, int, string, array or sequence, but got %s "
                             "instead",
                             type_cstr);
            }
            else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "QuadPrecision value must be a quad, float, int, string, array or sequence, but got an "
                        "unknown type instead");
            }
            Py_DECREF(type_str);
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "QuadPrecision value must be a quad, float, int, string, array or sequence, but got an "
                            "unknown type instead");
        }
        Py_DECREF(self);
        return NULL;
    }

    return self;
}

static PyObject *
QuadPrecision_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    PyObject *value;
    const char *backend_str = "sleef";
    static char *kwlist[] = {"value", "backend", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &value, &backend_str)) {
        return NULL;
    }

    QuadBackendType backend = BACKEND_SLEEF;
    if (strcmp(backend_str, "longdouble") == 0) {
        backend = BACKEND_LONGDOUBLE;
    }
    else if (strcmp(backend_str, "sleef") != 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid backend. Use 'sleef' or 'longdouble'.");
        return NULL;
    }

    return (PyObject *)QuadPrecision_from_object(value, backend);
}

static PyObject *
QuadPrecision_str_dragon4(QuadPrecisionObject *self)
{
    Dragon4_Options opt = {.scientific = 0,
                           .digit_mode = DigitMode_Unique,
                           .cutoff_mode = CutoffMode_TotalLength,
                           .precision = SLEEF_QUAD_DECIMAL_DIG,
                           .sign = 1,
                           .trim_mode = TrimMode_LeaveOneZero,
                           .digits_left = 1,
                           .digits_right = 0};

    if (self->backend == BACKEND_SLEEF) {
        return Dragon4_Positional_QuadDType(
                &self->value.sleef_value, opt.digit_mode, opt.cutoff_mode, opt.precision,
                opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left, opt.digits_right);
    }
    else {
        Sleef_quad sleef_val = Sleef_cast_from_doubleq1(self->value.longdouble_value);
        return Dragon4_Positional_QuadDType(&sleef_val, opt.digit_mode, opt.cutoff_mode,
                                            opt.precision, opt.min_digits, opt.sign, opt.trim_mode,
                                            opt.digits_left, opt.digits_right);
    }
}

static PyObject *
QuadPrecision_str(QuadPrecisionObject *self)
{
    char buffer[128];
    if (self->backend == BACKEND_SLEEF) {
        Sleef_snprintf(buffer, sizeof(buffer), "%.*Qe", SLEEF_QUAD_DIG, self->value.sleef_value);
    }
    else {
        snprintf(buffer, sizeof(buffer), "%.35Le", self->value.longdouble_value);
    }
    return PyUnicode_FromString(buffer);
}

static PyObject *
QuadPrecision_repr(QuadPrecisionObject *self)
{
    PyObject *str = QuadPrecision_str(self);
    if (str == NULL) {
        return NULL;
    }
    const char *backend_str = (self->backend == BACKEND_SLEEF) ? "sleef" : "longdouble";
    PyObject *res = PyUnicode_FromFormat("QuadPrecision('%S', backend='%s')", str, backend_str);
    Py_DECREF(str);
    return res;
}

static PyObject *
QuadPrecision_repr_dragon4(QuadPrecisionObject *self)
{
    Dragon4_Options opt = {.scientific = 1,
                           .digit_mode = DigitMode_Unique,
                           .cutoff_mode = CutoffMode_TotalLength,
                           .precision = SLEEF_QUAD_DECIMAL_DIG,
                           .sign = 1,
                           .trim_mode = TrimMode_LeaveOneZero,
                           .digits_left = 1,
                           .exp_digits = 3};

    PyObject *str;
    if (self->backend == BACKEND_SLEEF) {
        str = Dragon4_Scientific_QuadDType(&self->value.sleef_value, opt.digit_mode, opt.precision,
                                           opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left,
                                           opt.exp_digits);
    }
    else {
        Sleef_quad sleef_val = Sleef_cast_from_doubleq1(self->value.longdouble_value);
        str = Dragon4_Scientific_QuadDType(&sleef_val, opt.digit_mode, opt.precision,
                                           opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left,
                                           opt.exp_digits);
    }

    if (str == NULL) {
        return NULL;
    }

    const char *backend_str = (self->backend == BACKEND_SLEEF) ? "sleef" : "longdouble";
    PyObject *res = PyUnicode_FromFormat("QuadPrecision('%S', backend='%s')", str, backend_str);
    Py_DECREF(str);
    return res;
}

static void
QuadPrecision_dealloc(QuadPrecisionObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyTypeObject QuadPrecision_Type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_quaddtype.QuadPrecision",
        .tp_basicsize = sizeof(QuadPrecisionObject),
        .tp_itemsize = 0,
        .tp_new = QuadPrecision_new,
        .tp_dealloc = (destructor)QuadPrecision_dealloc,
        .tp_repr = (reprfunc)QuadPrecision_repr_dragon4,
        .tp_str = (reprfunc)QuadPrecision_str_dragon4,
        .tp_as_number = &quad_as_scalar,
        .tp_richcompare = (richcmpfunc)quad_richcompare,
};

int
init_quadprecision_scalar(void)
{
    QuadPrecision_Type.tp_base = &PyFloatingArrType_Type;
    return PyType_Ready(&QuadPrecision_Type);
}