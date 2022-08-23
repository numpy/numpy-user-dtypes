/*
 * This file defines a very basic QuantityScalar object to work together with
 * our DType.
 * Note that it is possible that we do not require such a scalar in the future
 * (Getting an element from an array would always return a 0-D array.)
 *
 * For now we need it.  Not the least because NumPy currently prints arrays
 * by printing the scalars.
 * We do not inherit from `np.generic` because it comes with more expectations
 * than I like.  There should likely be a new `np.scalar` base class (that
 * `np.generic` also inherits) to provide a "no strings attached" base.
 * (See also NEP 41-42.)
 */

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"

#include "scalar.h"


/*
 * Helper function also used elsewhere to make sure the unit is a unit.
 *
 * NOTE: Supports if `obj` is NULL (meaning nothing is passed on)
 */
int
UnitConverter(PyObject *obj, PyObject **unit)
{
    static PyObject *get_unit = NULL;
    if (NPY_UNLIKELY(get_unit == NULL)) {
        PyObject *mod = PyImport_ImportModule("unitdtype._helpers");
        if (mod == NULL) {
            return 0;
        }
        get_unit = PyObject_GetAttrString(mod, "get_unit");
        Py_DECREF(mod);
        if (get_unit == NULL) {
            return 0;
        }
    }
    *unit = PyObject_CallFunctionObjArgs(get_unit, obj, NULL);
    if (*unit == NULL) {
        return 0;
    }
    return 1;
}


static PyObject *
quantityscalar_new(PyTypeObject *cls, PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"value", "unit", NULL};
    double value;
    PyObject *unit;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "dO&:QuantityScalar", kwargs_strs,
            &value, &UnitConverter, &unit)) {
        return NULL;
    }

    QuantityScalarObject *new = PyObject_New(QuantityScalarObject, cls);
    if (new == NULL) {
        return NULL;
    }
    new->value = value;
    new->unit = unit;

    return (PyObject *)new;
}


static PyObject *
quantityscalar_repr(QuantityScalarObject *self)
{
    char *val_repr = PyOS_double_to_string(self->value, 'r', 0, 0, NULL);
    if (val_repr == NULL) {
        return NULL;
    }

    PyObject *res = PyUnicode_FromFormat("%s*%R", val_repr, self->unit);
    PyMem_Free(val_repr);
    return res;
}


PyTypeObject QuantityScalar_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "unitdtype.QuantityScalar",
    .tp_basicsize = sizeof(QuantityScalarObject),
    .tp_new = quantityscalar_new,
    .tp_repr = (reprfunc)quantityscalar_repr,
    .tp_str = (reprfunc)quantityscalar_repr,
};