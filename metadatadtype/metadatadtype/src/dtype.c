#include "dtype.h"

#include "casts.h"

/*
 * `get_value` and `get_unit` are small helpers to deal with the scalar.
 */

static double
get_value(PyObject *scalar, const PyTypeObject *MetadataScalar_Type)
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
get_metadata(PyObject *scalar, const PyTypeObject *MetadataScalar_Type)
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
new_metadatadtype_instance(metadatadtype_state *state, PyObject *metadata)
{
    if (state == NULL) {
        return NULL;
    }
    MetadataDTypeObject *new = (MetadataDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)state->MetadataDType, NULL, NULL);
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

PyArray_Descr *
common_instance(MetadataDTypeObject *dtype1,
                MetadataDTypeObject *NPY_UNUSED(dtype2))
{
    Py_INCREF(dtype1);
    return (PyArray_Descr *)dtype1;
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
metadata_discover_descriptor_from_pyobject(PyArray_DTypeMeta *cls,
                                           PyObject *obj)
{
    metadatadtype_state *state = PyType_GetModuleState((PyTypeObject *)cls);
    if (Py_TYPE(obj) != state->MetadataScalar_Type) {
        PyErr_SetString(
                PyExc_TypeError,
                "Can only store MetadataScalar in a MetadataDType array.");
        return NULL;
    }

    PyObject *metadata = get_metadata(obj, state->MetadataScalar_Type);
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
    metadatadtype_state *state = PyType_GetModuleState(Py_TYPE(descr));
    double value = get_value(obj, state->MetadataScalar_Type);
    if (value == -1 && PyErr_Occurred()) {
        return -1;
    }

    memcpy(dataptr, &value, sizeof(double));  // NOLINT

    return 0;
}

static PyObject *
metadatadtype_getitem(MetadataDTypeObject *descr, char *dataptr)
{
    metadatadtype_state *state = PyType_GetModuleState(Py_TYPE(descr));
    double val;
    /* get the value */
    memcpy(&val, dataptr, sizeof(double));  // NOLINT

    PyObject *val_obj = PyFloat_FromDouble(val);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs(
            (PyObject *)state->MetadataScalar_Type, val_obj, descr, NULL);
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
metadatadtype_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    metadatadtype_state *state = PyType_GetModuleState(type);

    static char *kwargs_strs[] = {"metadata", NULL};

    PyObject *metadata = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:MetadataDType",
                                     kwargs_strs, &metadata)) {
        return NULL;
    }
    if (metadata == NULL) {
        metadata = Py_None;
    }

    return (PyObject *)new_metadatadtype_instance(state, metadata);
}

/* MetadataDType finalization */

static int
metadatadtype_traverse(PyObject *self_obj, visitproc visit, void *arg)
{
    // Visit the type
    Py_VISIT(Py_TYPE(self_obj));

    // Visit the metadata attribute
    Py_VISIT(((MetadataDTypeObject *)self_obj)->metadata);

    return 0;
}

static int
metadatadtype_clear(MetadataDTypeObject *self)
{
    Py_CLEAR(self->metadata);
    return 0;
}

static void
metadatadtype_finalize(PyObject *self_obj)
{
    Py_CLEAR(((MetadataDTypeObject *)self_obj)->metadata);
}

static void
metadatadtype_dealloc(MetadataDTypeObject *self)
{
    PyObject_GC_UnTrack(self);
    metadatadtype_finalize((PyObject *)self);
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
        {NULL, 0, 0, 0, ""},
};

static PyType_Slot MetadataDType_Type_slots[] = {
        {Py_tp_traverse, metadatadtype_traverse},
        {Py_tp_clear, metadatadtype_clear},
        {Py_tp_finalize, metadatadtype_finalize},
        {Py_tp_dealloc, metadatadtype_dealloc},
        {Py_tp_str, (reprfunc)metadatadtype_repr},
        {Py_tp_repr, (reprfunc)metadatadtype_repr},
        {Py_tp_members, MetadataDType_members},
        {Py_tp_new, metadatadtype_new},
        {0, 0}, /* sentinel */
};

static PyType_Spec MetadataDType_Type_spec = {
        .name = "metadatadtype.MetadataDType",
        .basicsize = sizeof(MetadataDTypeObject),
        .flags = Py_TPFLAGS_DEFAULT,
        .slots = MetadataDType_Type_slots,
};

int
init_metadata_dtype(PyObject *m)
{
    metadatadtype_state *state = PyModule_GetState(m);

    PyObject *bases = PyTuple_Pack(1, (PyObject *)&PyArrayDescr_Type);

    state->MetadataDType = (PyArray_DTypeMeta *)PyType_FromModuleAndSpec(
            m, &MetadataDType_Type_spec, bases);
    if (state->MetadataDType == NULL) {
        return -1;
    }

    // manually set type so it has the correct metaclass
    ((PyObject *)state->MetadataDType)->ob_type = &PyArrayDTypeMeta_Type;

    // manually null remaining fields, is there a more forward compatible
    // way to do this with e.g. memset?
    state->MetadataDType->singleton = NULL;
    state->MetadataDType->type_num = 0;
    state->MetadataDType->scalar_type = NULL;
    state->MetadataDType->flags = 0;
    state->MetadataDType->dt_slots = 0;

    /*
     * To create our DType, we have to use a "Spec" that tells NumPy how to
     * do it.  You first have to create a static type, but see the note there!
     */
    PyArrayMethod_Spec **casts = get_casts();

    PyArrayDTypeMeta_Spec MetadataDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC | NPY_DT_NUMERIC,
            .casts = casts,
            .typeobj = state->MetadataScalar_Type,
            .slots = MetadataDType_Slots,
    };

    if (PyArrayInitDTypeMeta_FromSpec(state->MetadataDType,
                                      &MetadataDType_DTypeSpec) < 0) {
        return -1;
    }

    state->MetadataDType->singleton =
            PyArray_GetDefaultDescr(state->MetadataDType);

    free(MetadataDType_DTypeSpec.casts[1]->dtypes);
    free(MetadataDType_DTypeSpec.casts[1]);
    free(MetadataDType_DTypeSpec.casts);

    return 0;
}
