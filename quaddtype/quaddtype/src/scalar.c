#include<Python.h>
#include<sleef.h>
#include<sleefquad.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "scalar.h"
#include "scalar_ops.h"


// static PyTypeObject QuadPrecision_Type;


QuadPrecisionObject * QuadPrecision_raw_new(void)
{
    QuadPrecisionObject * new = PyObject_New(QuadPrecisionObject, &QuadPrecision_Type);
    if(!new)
        return NULL;
    new->quad.value = Sleef_cast_from_doubleq1(0.0); // initialize to 0
    return new;
}

QuadPrecisionObject * QuadPrecision_from_object(PyObject * value)
{
    QuadPrecisionObject *self = QuadPrecision_raw_new();
    if(!self)
        return NULL;
    
    if(PyFloat_Check(value))
        self->quad.value = Sleef_cast_from_doubleq1(PyFloat_AsDouble(value));
    
    else if(PyUnicode_CheckExact(value))
    {
        const char * s = PyUnicode_AsUTF8(value);
        char *endptr = NULL;
        self->quad.value = Sleef_strtoq(s, &endptr);
        if (*endptr != '\0' || endptr == s)
        {
            PyErr_SetString(PyExc_ValueError, "Unable to parse string to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
    }
    else if(PyLong_Check(value))
    {
        self->quad.value = Sleef_cast_from_int64q1(PyLong_AsLong(value));
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "QuadPrecision value must be a float or string");
        Py_DECREF(self);
        return NULL;
    }

    return self;
}

static PyObject *
QuadPrecision_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    PyObject *value;

    if (!PyArg_ParseTuple(args, "O", &value)) {
        return NULL;
    }

    return (PyObject *)QuadPrecision_from_object(value);
}

static PyObject * QuadPrecision_str(QuadPrecisionObject * self)
{
    char buffer[128];
    Sleef_snprintf(buffer, sizeof(buffer), "%.*Qe", SLEEF_QUAD_DIG, self->quad.value);
    return PyUnicode_FromString(buffer);
}

static PyObject * QuadPrecision_repr(QuadPrecisionObject* self)
{
    PyObject *str = QuadPrecision_str(self);
    if (str == NULL) {
        return NULL;
    }
    PyObject *res = PyUnicode_FromFormat("QuadPrecision('%S')", str);
    Py_DECREF(str);
    return res;
}

PyTypeObject QuadPrecision_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "QuadPrecType.QuadPrecision",
    .tp_basicsize = sizeof(QuadPrecisionObject),
    .tp_itemsize = 0,
    .tp_new = QuadPrecision_new,
    .tp_repr = (reprfunc)QuadPrecision_repr,
    .tp_str = (reprfunc)QuadPrecision_str,
    .tp_as_number = &quad_as_scalar,
    .tp_richcompare = (richcmpfunc)quad_richcompare

};

int
init_quadprecision_scalar(void)
{
    return PyType_Ready(&QuadPrecision_Type);
}