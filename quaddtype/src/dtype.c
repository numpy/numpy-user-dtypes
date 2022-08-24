#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "casts.h"
#include "dtype.h"


PyObject *QuantityScalar_Type = NULL;

/*
 * `get_value` and `get_unit` are small helpers to deal with the scalar.
 */
static double
get_value(PyObject *scalar) {
    if (Py_TYPE(scalar) != QuantityScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store QuantityScalar in a UnitDType array.");
        return -1;
    }

    PyObject *value = PyObject_GetAttrString(scalar, "value");
    if (value == NULL) {
        return -1;
    }
    double res = PyFloat_AsDouble(value);
    Py_DECREF(value);
    return res;
}


static PyObject *
get_unit(PyObject *scalar) {
    if (Py_TYPE(scalar) != QuantityScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store QuantityScalar in a UnitDType array.");
        return NULL;
    }

    return PyObject_GetAttrString(scalar, "unit");
}


/*
 * Internal helper to create new instances, does not check unit for validity.
 */
UnitDTypeObject *
new_unitdtype_instance(PyObject *unit)
{
    UnitDTypeObject *new = (UnitDTypeObject *)PyArrayDescr_Type.tp_new(
            /* TODO: Using NULL for args here works, but seems not clean? */
            (PyTypeObject *)&UnitDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    Py_INCREF(unit);
    new->unit = unit;
    new->base.elsize = sizeof(double);
    new->base.alignment = _Alignof(double);  /* is there a better spelling? */
    /* do not support byte-order for now */

    return new;
}


/*
 * For now, give the more precise unit as the "common" one, but just bail and
 * give the first one if there is an offset (e.g. Celsius and Fahrenheit?).
 * It might also make sense to give the more "standard" one, but that depends?
 */
static UnitDTypeObject *
common_instance(UnitDTypeObject *dtype1, UnitDTypeObject *dtype2)
{
    double factor, offset;
    if (get_conversion_factor(
            dtype1->unit, dtype2->unit, &factor, &offset) < 0) {
        return NULL;
    }
    if (offset != 0 || fabs(factor) > 1.) {
        Py_INCREF(dtype1);
        return dtype1;
    }
    else {
        Py_INCREF(dtype2);
        return dtype2;
    }
}


static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /*
     * Typenum is useful for NumPy, but there it can still be convenient.
     * (New-style user dtypes will probably get -1 as type number...)
     */
    if (other->type_num >= 0
            && PyTypeNum_ISNUMBER(other->type_num)
            && !PyTypeNum_ISCOMPLEX(other->type_num)
            && other != &PyArray_LongDoubleDType) {
        /*
         * A (simple) builtin numeric type that is not a complex or longdouble
         * will always promote to the Double Unit (cls).
         */
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}


/*
 * Functions dealing with scalar logic
 */

static PyArray_Descr *
unit_discover_descriptor_from_pyobject(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    if (Py_TYPE(obj) != QuantityScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store QuantityScalar in a UnitDType array.");
        return NULL;
    }
    PyObject *unit = get_unit(obj);
    if (unit == NULL) {
        return NULL;
    }
    return (PyArray_Descr *)new_unitdtype_instance(unit);
}


/*
 * Note, for correct support in HPy, this function will probably need to get
 * an `owner` object.  This object would be opaque and possibly ephemeral
 * (you are not allowed to hold on to it) but "owns" the data where things get
 * stored.
 *
 * NumPy allows you to set an "is known scalar" function that normally lets
 * the simple Python types pass (int, float, complex, bytes, strings) as well
 * as the scalar type itself.
 * This can be customized, but `setitem` may still be called with arbitrary
 * objects on a "best effort" basis.
 */
static int
unit_setitem(UnitDTypeObject *descr, PyObject *obj, char *dataptr)
{
    double value = get_value(obj);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }
    PyObject *unit = get_unit(obj);
    if (unit == NULL) {
        return -1;
    }

    double factor, offset;
    if (get_conversion_factor(unit, descr->unit, &factor, &offset) < 0) {
        return -1;
    }
    value = factor * (value + offset);
    memcpy(dataptr, &value, sizeof(double));
    return 0;
}

/*
 * Note, same as above (but more).  For correct support in HPy we likely need
 * to pass an `owner` here.  But, we probably also need to pass a "base",
 * because that is how structured scalars work (they return a view...).
 * Those two might have subtly different logic, though?
 * (Realistically, maybe we can special case void to not pass the base, I do
 * not think that a scalar should ever be a view, such a scalar should not
 * exist.  E.g. in that case, better not have a scalar at all to begin with.)
 */
static PyObject *
unit_getitem(UnitDTypeObject *descr, char *dataptr)
{
    double val;
    /* get the value */
    memcpy(&val, dataptr, sizeof(double));

    PyObject *val_obj = PyFloat_FromDouble(val);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs(
            QuantityScalar_Type, val_obj, descr->unit, NULL);
    Py_DECREF(val_obj);
    return res;
}


static PyType_Slot UnitDType_Slots[] = {
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject, &unit_discover_descriptor_from_pyobject},
    /* The header is wrong on main :(, so we add 1 */
    {NPY_DT_setitem, &unit_setitem},
    {NPY_DT_getitem, &unit_getitem},
    {0, NULL}
};


/*
 * The following defines everything type object related (i.e. not NumPy
 * specific).
 *
 * Note that this function is by default called without any arguments to fetch
 * a default version of the descriptor (in principle at least).  During init
 * we fill in `cls->singleton` though for the dimensionless unit.
 */
static PyObject *
unitdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"unit", NULL};

    PyObject *unit = NULL;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|O&:UntDType", kwargs_strs,
            &UnitConverter, &unit)) {
        return NULL;
    }
    if (unit == NULL) {
        if (!UnitConverter(NULL, &unit)) {
            return NULL;
        }
    }

    PyObject *res = (PyObject *)new_unitdtype_instance(unit);
    Py_DECREF(unit);
    return res;
}


static void
unitdtype_dealloc(UnitDTypeObject *self)
{
    Py_CLEAR(self->unit);
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}


static PyObject *
unitdtype_repr(UnitDTypeObject *self)
{
    PyObject *res = PyUnicode_FromFormat(
            "UnitDType(%R)", self->unit);
    return res;
}


/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta UnitDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "unitdtype.UnitDType",
        .tp_basicsize = sizeof(UnitDTypeObject),
        .tp_new = unitdtype_new,
        .tp_dealloc = (destructor)unitdtype_dealloc,
        .tp_repr = (reprfunc)unitdtype_repr,
        .tp_str = (reprfunc)unitdtype_repr,
    }},
    /* rest, filled in during DTypeMeta initialization */
};


int
init_unit_dtype(void)
{
    /*
     * To create our DType, we have to use a "Spec" that tells NumPy how to
     * do it.  You first have to create a static type, but see the note there!
     */
    PyArrayMethod_Spec *casts[] = {
            &UnitToUnitCastSpec,
            NULL};

    PyArrayDTypeMeta_Spec UnitDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .casts = casts,
            .typeobj = QuantityScalar_Type,
            .slots = UnitDType_Slots,
    };
    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&UnitDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&UnitDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&UnitDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(
            &UnitDType, &UnitDType_DTypeSpec) < 0) {
        return -1;
    }

    /*
     * Ensure that `singleton` is filled in (we rely on that).  It is possible
     * to provide a custom `default_descr`, but it is filled in automatically
     * to just call `DType()` -- however, it does not cache the result
     * automatically (right now).  This is because it can make sense for a
     * DType to requiring a new one each time (e.g. a Categorical that needs
     * to be able to add new Categories).
     * TODO: Consider flipping this around, so that if you need a new one
     *       each time, you have to provide a custom `default_descr`.
     */
    UnitDType.singleton = PyArray_GetDefaultDescr(&UnitDType);

    return 0;
}
