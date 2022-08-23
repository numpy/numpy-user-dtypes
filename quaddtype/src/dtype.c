#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "scalar.h"
#include "casts.h"
#include "dtype.h"


/*
 * Internal helper to create new instances, does not check unit for validity.
 */
UnitDTypeObject *
new_unitdtype_instance(PyObject *unit)
{
    UnitDTypeObject *new = (UnitDTypeObject *)PyArrayDescr_Type.tp_new(
            /* TODO: Using NULL for args here works, but seems not clean? */
            (PyTypeObject *)&UnitDType_Type, NULL, NULL);
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
    if (Py_TYPE(obj) != &QuantityScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store QuantityScalar in a Float64Unit array.");
        return NULL;
    }
    return (PyArray_Descr *)new_unitdtype_instance(((QuantityScalarObject *)obj)->unit);
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
    if (Py_TYPE(obj) != &QuantityScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store QuantityScalar in a Float64Unit array.");
        return -1;
    }
    double value = ((QuantityScalarObject *)obj)->value;
    PyObject *unit = ((QuantityScalarObject *)obj)->unit;

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
    // TODO: should create and use a helper for this.
    QuantityScalarObject *new = PyObject_New(
            QuantityScalarObject, &QuantityScalar_Type);
    if (new == NULL) {
        return NULL;
    }
    memcpy(&new->value, dataptr, sizeof(double));
    Py_INCREF(descr->unit);
    new->unit = descr->unit;
    return (PyObject *)new;
}


static PyType_Slot UnitDType_Slots[] = {
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject, &unit_discover_descriptor_from_pyobject},
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
            args, kwds, "|O&:Float64Unit", kwargs_strs,
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
            "Float64Unit(%R)", self->unit);
    return res;
}


PyArray_DTypeMeta UnitDType_Type = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "unitdtype.Float64Unit",
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
    PyArrayMethod_Spec *casts[] = {
            &UnitToUnitCastSpec,
            &UnitToBoolCastSpec, &BoolToUnitCastSpec,
            /* units */
            &UnitToUByteCastSpec, &UByteToUnitCastSpec,
            &UnitToUShortCastSpec, &UShortToUnitCastSpec,
            &UnitToUIntCastSpec, &UIntToUnitCastSpec,
            &UnitToULongCastSpec, &ULongToUnitCastSpec,
            &UnitToULongLongCastSpec, &ULongLongToUnitCastSpec,
            /* ints */
            &UnitToByteCastSpec, &ByteToUnitCastSpec,
            &UnitToShortCastSpec, &ShortToUnitCastSpec,
            &UnitToIntCastSpec, &IntToUnitCastSpec,
            &UnitToLongCastSpec, &LongToUnitCastSpec,
            &UnitToLongLongCastSpec, &LongLongToUnitCastSpec,
            /* floats */
            &UnitToFloatCastSpec, &FloatToUnitCastSpec,
            &UnitToDoubleCastSpec, &DoubleToUnitCastSpec,
            NULL};
    /*
     * The registration machinery is OK with NULL being the new DType, but
     * the other DType is dynamic information we cannot hardcode unfortunately:
     */
    UnitToBoolCastSpec.dtypes[1] = &PyArray_BoolDType;
    BoolToUnitCastSpec.dtypes[0] = &PyArray_BoolDType;

    UnitToUByteCastSpec.dtypes[1] = &PyArray_UByteDType;
    UByteToUnitCastSpec.dtypes[0] = &PyArray_UByteDType;
    UnitToUShortCastSpec.dtypes[1] = &PyArray_UShortDType;
    UShortToUnitCastSpec.dtypes[0] = &PyArray_UShortDType;
    UnitToUIntCastSpec.dtypes[1] = &PyArray_UIntDType;
    UIntToUnitCastSpec.dtypes[0] = &PyArray_UIntDType;
    UnitToULongCastSpec.dtypes[1] = &PyArray_ULongDType;
    ULongToUnitCastSpec.dtypes[0] = &PyArray_ULongDType;
    UnitToULongLongCastSpec.dtypes[1] = &PyArray_ULongLongDType;
    ULongLongToUnitCastSpec.dtypes[0] = &PyArray_ULongLongDType;

    UnitToByteCastSpec.dtypes[1] = &PyArray_ByteDType;
    ByteToUnitCastSpec.dtypes[0] = &PyArray_ByteDType;
    UnitToShortCastSpec.dtypes[1] = &PyArray_ShortDType;
    ShortToUnitCastSpec.dtypes[0] = &PyArray_ShortDType;
    UnitToIntCastSpec.dtypes[1] = &PyArray_IntDType;
    IntToUnitCastSpec.dtypes[0] = &PyArray_IntDType;
    UnitToLongCastSpec.dtypes[1] = &PyArray_LongDType;
    LongToUnitCastSpec.dtypes[0] = &PyArray_LongDType;
    UnitToLongLongCastSpec.dtypes[1] = &PyArray_LongLongDType;
    LongLongToUnitCastSpec.dtypes[0] = &PyArray_LongLongDType;

    UnitToFloatCastSpec.dtypes[1] = &PyArray_FloatDType;
    FloatToUnitCastSpec.dtypes[0] = &PyArray_FloatDType;
    UnitToDoubleCastSpec.dtypes[1] = &PyArray_DoubleDType;
    DoubleToUnitCastSpec.dtypes[0] = &PyArray_DoubleDType;

    PyArrayDTypeMeta_Spec UnitDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .casts = casts,
            .typeobj = &QuantityScalar_Type,
            .slots = UnitDType_Slots,
    };

    ((PyObject *)&UnitDType_Type)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&UnitDType_Type)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&UnitDType_Type) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(
            &UnitDType_Type, &UnitDType_DTypeSpec) < 0) {
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
    UnitDType_Type.singleton = PyArray_GetDefaultDescr(&UnitDType_Type);

    return 0;
}
