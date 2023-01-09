#include "dtype.h"

#include "casts.h"

PyTypeObject *MetadataScalar_Type = NULL;

/*
 * `get_value` and `get_unit` are small helpers to deal with the scalar.
 */

static double
get_value(PyObject *scalar)
{
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    if (scalar_type != MetadataScalar_Type) {
        double res = PyFloat_AsDouble(scalar);
        if (res == -1 && PyErr_Occurred()) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "Can only store MetadataScalar in a MetadataDType array.");
            return -1;
        }
        return res;
    }

    PyObject *value = PyObject_GetAttrString(scalar, "value");
    if (value == NULL) {
        return -1;
    }
    double res = PyFloat_AsDouble(value);
    if (res == -1 && PyErr_Occurred()) {
        return -1;
    }
    Py_DECREF(value);
    return res;
}

static PyObject *
get_metadata(PyObject *scalar)
{
    if (Py_TYPE(scalar) != MetadataScalar_Type) {
        PyErr_SetString(
                PyExc_TypeError,
                "Can only store MetadataScalar in a MetadataDType array.");
        return NULL;
    }

    MetadataDTypeObject *dtype =
            (MetadataDTypeObject *)PyObject_GetAttrString(scalar, "dtype");
    if (dtype == NULL) {
        return NULL;
    }

    PyObject *metadata = dtype->metadata;
    Py_DECREF(dtype);
    if (metadata == NULL) {
        return NULL;
    }
    Py_INCREF(metadata);
    return metadata;
}

/*
 * Internal helper to create new instances
 */
MetadataDTypeObject *
new_metadatadtype_instance(PyObject *metadata)
{
    MetadataDTypeObject *new = (MetadataDTypeObject *)PyArrayDescr_Type.tp_new(
            /* TODO: Using NULL for args here works, but seems not clean? */
            (PyTypeObject *)&MetadataDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    Py_INCREF(metadata);
    new->metadata = metadata;
    new->base.elsize = sizeof(double);
    new->base.alignment = _Alignof(double); /* is there a better spelling? */
    /* do not support byte-order for now */

    return new;
}

/*
 * This is used to determine the correct dtype to return when operations mix
 * dtypes (I think?). For now just return the first one.
 */
static MetadataDTypeObject *
common_instance(MetadataDTypeObject *dtype1, MetadataDTypeObject *dtype2)
{
    Py_INCREF(dtype1);
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    /*
     * Typenum is useful for NumPy, but there it can still be convenient.
     * (New-style user dtypes will probably get -1 as type number...)
     */
    if (other->type_num >= 0 && PyTypeNum_ISNUMBER(other->type_num) &&
        !PyTypeNum_ISCOMPLEX(other->type_num) &&
        other != &PyArray_LongDoubleDType) {
        /*
         * A (simple) builtin numeric type that is not a complex or longdouble
         * will always promote to double (cls).
         */
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static PyArray_Descr *
metadata_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                           PyObject *obj)
{
    if (Py_TYPE(obj) != MetadataScalar_Type) {
        PyErr_SetString(
                PyExc_TypeError,
                "Can only store MetadataScalar in a MetadataDType array.");
        return NULL;
    }

    PyObject *metadata = get_metadata(obj);
    if (metadata == NULL) {
        return NULL;
    }
    PyArray_Descr *ret = (PyArray_Descr *)PyObject_GetAttrString(obj, "dtype");
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

static int
metadatadtype_setitem(MetadataDTypeObject *descr, PyObject *obj, char *dataptr)
{
    double value = get_value(obj);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }

    memcpy(dataptr, &value, sizeof(double));  // NOLINT

    return 0;
}

static PyObject *
metadatadtype_getitem(MetadataDTypeObject *descr, char *dataptr)
{
    double val;
    /* get the value */
    memcpy(&val, dataptr, sizeof(double));  // NOLINT

    PyObject *val_obj = PyFloat_FromDouble(val);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs(
            (PyObject *)MetadataScalar_Type, val_obj, descr, NULL);
    if (res == NULL) {
        return NULL;
    }
    Py_DECREF(val_obj);

    return res;
}

static MetadataDTypeObject *
metadatadtype_ensure_canonical(MetadataDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyType_Slot MetadataDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &metadata_discover_descriptor_from_pyobject},
        /* The header is wrong on main :(, so we add 1 */
        {NPY_DT_setitem, &metadatadtype_setitem},
        {NPY_DT_getitem, &metadatadtype_getitem},
        {NPY_DT_ensure_canonical, &metadatadtype_ensure_canonical},
        {0, NULL}};

static PyObject *
metadatadtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args,
                  PyObject *kwds)
{
    static char *kwargs_strs[] = {"metadata", NULL};

    PyObject *metadata = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:MetadataDType",
                                     kwargs_strs, &metadata)) {
        return NULL;
    }
    if (metadata == NULL) {
        metadata = Py_None;
    }

    return (PyObject *)new_metadatadtype_instance(metadata);
}

static void
metadatadtype_dealloc(MetadataDTypeObject *self)
{
    Py_CLEAR(self->metadata);
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
metadatadtype_repr(MetadataDTypeObject *self)
{
    PyObject *res = PyUnicode_FromFormat("MetadataDType(%R)", self->metadata);
    return res;
}

static PyMemberDef MetadataDType_members[] = {
        {"_metadata", T_OBJECT_EX, offsetof(MetadataDTypeObject, metadata),
         READONLY, "some metadata"},
        {NULL},
};

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta MetadataDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "metadatadtype.MetadataDType",
                .tp_basicsize = sizeof(MetadataDTypeObject),
                .tp_new = metadatadtype_new,
                .tp_dealloc = (destructor)metadatadtype_dealloc,
                .tp_repr = (reprfunc)metadatadtype_repr,
                .tp_str = (reprfunc)metadatadtype_repr,
                .tp_members = MetadataDType_members,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_metadata_dtype(void)
{
    /*
     * To create our DType, we have to use a "Spec" that tells NumPy how to
     * do it.  You first have to create a static type, but see the note there!
     */
    PyArrayMethod_Spec **casts = get_casts();

    PyArrayDTypeMeta_Spec MetadataDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .casts = casts,
            .typeobj = MetadataScalar_Type,
            .slots = MetadataDType_Slots,
    };
    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&MetadataDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&MetadataDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&MetadataDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&MetadataDType,
                                      &MetadataDType_DTypeSpec) < 0) {
        return -1;
    }

    MetadataDType.singleton = PyArray_GetDefaultDescr(&MetadataDType);

    free(MetadataDType_DTypeSpec.casts[1]->dtypes);
    free(MetadataDType_DTypeSpec.casts[1]);
    free(MetadataDType_DTypeSpec.casts);

    return 0;
}
