#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "mpfr.h"

#include "scalar.h"
#include "casts.h"
#include "dtype.h"



/*
 * Internal helper to create new instances.
 */
MPFDTypeObject *
new_MPFDType_instance(mpfr_prec_t precision)
{
    /*
     * The docs warn that temporary ops may need an increased max prec, so
     * there could a point in reducing it.  But lets assume errors will get
     * set in that case.
     */
    if (precision < MPFR_PREC_MIN || precision > MPFR_PREC_MAX) {
        PyErr_Format(PyExc_ValueError,
                "precision must be between %d and %d.",
                MPFR_PREC_MIN, MPFR_PREC_MAX);
        return NULL;
    }

    MPFDTypeObject *new = (MPFDTypeObject *)PyArrayDescr_Type.tp_new(
            /* TODO: Using NULL for args here works, but seems not clean? */
            (PyTypeObject *)&MPFDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->precision = precision;
    size_t size = mpfr_custom_get_size(precision);
    if (size > NPY_MAX_INT - sizeof(mpf_field)) {
        PyErr_SetString(PyExc_TypeError,
                "storage of single float would be too large for precision.");
    }
    new->base.elsize = sizeof(mpf_storage) + size;
    new->base.alignment = _Alignof(mpf_field);
    new->base.flags |= NPY_NEEDS_INIT;

    return new;
}


static MPFDTypeObject *
ensure_canonical(MPFDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}


static MPFDTypeObject *
common_instance(MPFDTypeObject *dtype1, MPFDTypeObject *dtype2)
{
    if (dtype1->precision > dtype2->precision) {
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
            && !PyTypeNum_ISCOMPLEX(other->type_num)) {
        /*
         * A (simple) builtin numeric type (not complex) promotes to fixed
         * precision.
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
mpf_discover_descriptor_from_pyobject(
        PyArray_DTypeMeta *NPY_UNUSED(cls), PyObject *obj)
{
    if (Py_TYPE(obj) != &MPFloat_Type) {
        PyErr_SetString(PyExc_TypeError,
                "Can only store MPFloat in a MPFDType array.");
        return NULL;
    }
    mpfr_prec_t prec = get_prec_from_object(obj);
    if (prec < 0) {
        return NULL;
    }
    return (PyArray_Descr *)new_MPFDType_instance(prec);
}


static int
mpf_setitem(MPFDTypeObject *descr, PyObject *obj, char *dataptr)
{
    MPFloatObject *value;
    if (PyObject_TypeCheck(obj, &MPFloat_Type)) {
        Py_INCREF(obj);
        value = (MPFloatObject *)obj;
    }
    else {
        value = MPFloat_from_object(obj, -1);
        if (value == NULL) {
            return -1;
        }
    }
    // TODO: This doesn't support unaligned access, maybe we should just
    //       allow DTypes to say that they cannot be unaligned?!

    mpfr_ptr res;
    mpf_load(res, dataptr, descr->precision);
    mpfr_set(res, value->mpf.x, MPFR_RNDN);
    mpf_store(dataptr, res);

    Py_DECREF(value);
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
mpf_getitem(MPFDTypeObject *descr, char *dataptr)
{
    MPFloatObject *new = MPFLoat_raw_new(descr->precision);
    if (new == NULL) {
        return NULL;
    }
    mpfr_ptr val;
    mpf_load(val, dataptr, descr->precision);
    mpfr_set(new->mpf.x, val, MPFR_RNDN);

    return (PyObject *)new;
}


static PyType_Slot MPFDType_Slots[] = {
    {NPY_DT_ensure_canonical, &ensure_canonical},
    {NPY_DT_common_instance, &common_instance},
    {NPY_DT_common_dtype, &common_dtype},
    {NPY_DT_discover_descr_from_pyobject,
            &mpf_discover_descriptor_from_pyobject},
    {NPY_DT_setitem, &mpf_setitem},
    {NPY_DT_getitem, &mpf_getitem},
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
MPFDType_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"precision", NULL};

    Py_ssize_t precision;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "n:MPFDType", kwargs_strs, &precision)) {
        return NULL;
    }

    return (PyObject *)new_MPFDType_instance(precision);
}


static PyObject *
MPFDType_repr(MPFDTypeObject *self)
{
    PyObject *res = PyUnicode_FromFormat(
            "MPFDType(%ld)", (long)self->precision);
    return res;
}


PyObject *
MPFDType_get_prec(MPFDTypeObject *self)
{
    return PyLong_FromLong(self->precision);
}


NPY_NO_EXPORT PyGetSetDef mpfdtype_getsetlist[] = {
    {"prec",
        (getter)MPFDType_get_prec,
        NULL,
        NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},  /* Sentinel */
};


/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta MPFDType = {{{
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "MPFDType.MPFDType",
        .tp_basicsize = sizeof(MPFDTypeObject),
        .tp_new = MPFDType_new,
        .tp_repr = (reprfunc)MPFDType_repr,
        .tp_str = (reprfunc)MPFDType_repr,
        .tp_getset = mpfdtype_getsetlist,
    }},
    /* rest, filled in during DTypeMeta initialization */
};


int
init_mpf_dtype(void)
{
    /*
     * To create our DType, we have to use a "Spec" that tells NumPy how to
     * do it.  You first have to create a static type, but see the note there!
     */
    PyArrayMethod_Spec **casts = init_casts();

    PyArrayDTypeMeta_Spec MPFDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .casts = casts,
            .typeobj = &MPFloat_Type,
            .slots = MPFDType_Slots,
    };
    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&MPFDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&MPFDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&MPFDType) < 0) {
        free_casts();
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(
            &MPFDType, &MPFDType_DTypeSpec) < 0) {
        free_casts();
        return -1;
    }

    free_casts();
    return 0;
}
