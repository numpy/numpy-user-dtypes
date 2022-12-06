#include "dtype.h"

#include "casts.h"

PyTypeObject *ASCIIScalar_Type = NULL;

static char *
get_value(PyObject *scalar)
{
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    if (scalar_type != ASCIIScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store ASCIIScalar in a ASCIIDType array.");
        return NULL;
    }

    PyObject *value = PyObject_GetAttrString(scalar, "value");
    if (value == NULL) {
        return NULL;
    }
    PyObject *res_bytes = PyUnicode_AsASCIIString(value);
    Py_DECREF(value);
    char *res = PyBytes_AsString(res_bytes);
    if (res == NULL) {
        return NULL;
    }
    return res;
}

/*
 * Internal helper to create new instances
 */
ASCIIDTypeObject *
new_asciidtype_instance(PyObject *size)
{
    ASCIIDTypeObject *new = (ASCIIDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&ASCIIDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    long size_l = PyLong_AsLong(size);
    if (size_l == -1 && PyErr_Occurred()) {
        return NULL;
    }
    new->size = size_l;
    new->base.elsize = size_l * sizeof(char);
    new->base.alignment = size_l *_Alignof(char);

    return new;
}

/*
 * This is used to determine the correct dtype to return when operations mix
 * dtypes (I think?). For now just return the first one.
 */
static ASCIIDTypeObject *
common_instance(ASCIIDTypeObject *dtype1, ASCIIDTypeObject *dtype2)
{
    if (!PyObject_RichCompareBool((PyObject *)dtype1, (PyObject *)dtype2,
                                  Py_EQ)) {
        PyErr_SetString(
                PyExc_RuntimeError,
                "common_instance called on unequal ASCIIDType instances");
        return NULL;
    }
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // for now always raise an error here until we can figure out
    // how to deal with strings here

    PyErr_SetString(PyExc_RuntimeError, "common_dtype called in ASCIIDType");
    return NULL;

    // Py_INCREF(Py_NotImplemented);
    // return (PyArray_DTypeMeta *)Py_NotImplemented;
}

static PyArray_Descr *
ascii_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                        PyObject *obj)
{
    if (Py_TYPE(obj) != ASCIIScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store ASCIIScalar in a ASCIIDType array.");
        return NULL;
    }

    PyArray_Descr *ret = (PyArray_Descr *)PyObject_GetAttrString(obj, "dtype");
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

static int
asciidtype_setitem(ASCIIDTypeObject *descr, PyObject *obj, char *dataptr)
{
    char *value = get_value(obj);
    if (value == NULL) {
        return -1;
    }

    memcpy(dataptr, value, descr->size * sizeof(char));  // NOLINT

    return 0;
}

static PyObject *
asciidtype_getitem(ASCIIDTypeObject *descr, char *dataptr)
{
    char *val = NULL;
    /* get the value */
    memcpy(val, dataptr, descr->size * sizeof(char));  // NOLINT

    PyObject *val_obj = PyUnicode_FromStringAndSize(val, descr->size);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)ASCIIScalar_Type,
                                                 val_obj, descr, NULL);
    if (res == NULL) {
        return NULL;
    }
    Py_DECREF(val_obj);

    return res;
}

static ASCIIDTypeObject *
asciidtype_ensure_canonical(ASCIIDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyType_Slot ASCIIDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &ascii_discover_descriptor_from_pyobject},
        /* The header is wrong on main :(, so we add 1 */
        {NPY_DT_setitem, &asciidtype_setitem},
        {NPY_DT_getitem, &asciidtype_getitem},
        {NPY_DT_ensure_canonical, &asciidtype_ensure_canonical},
        {0, NULL}};

static PyObject *
asciidtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"size", NULL};

    PyObject *size = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O:ASCIIDType", kwargs_strs,
                                     &size)) {
        return NULL;
    }
    if (size == NULL) {
        size = PyLong_FromLong(0);
    }

    PyObject *ret = (PyObject *)new_asciidtype_instance(size);
    Py_DECREF(size);
    return ret;
}

static void
asciidtype_dealloc(ASCIIDTypeObject *self)
{
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
asciidtype_repr(ASCIIDTypeObject *self)
{
    PyObject *res = PyUnicode_FromFormat("ASCIIDType(%ld)", self->size);
    return res;
}

static PyMemberDef ASCIIDType_members[] = {
        {"size", T_OBJECT_EX, offsetof(ASCIIDTypeObject, size), READONLY,
         "The number of characters per array element"},
        {NULL},
};

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta ASCIIDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "asciidtype.ASCIIDType",
                .tp_basicsize = sizeof(ASCIIDTypeObject),
                .tp_new = asciidtype_new,
                .tp_dealloc = (destructor)asciidtype_dealloc,
                .tp_repr = (reprfunc)asciidtype_repr,
                .tp_str = (reprfunc)asciidtype_repr,
                .tp_members = ASCIIDType_members,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_ascii_dtype(void)
{
    static PyArrayMethod_Spec *casts[] = {&ASCIIToASCIICastSpec, NULL};

    PyArrayDTypeMeta_Spec ASCIIDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .typeobj = ASCIIScalar_Type,
            .slots = ASCIIDType_Slots,
            .casts = casts,
    };
    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&ASCIIDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&ASCIIDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&ASCIIDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&ASCIIDType, &ASCIIDType_DTypeSpec) <
        0) {
        return -1;
    }

    PyArray_Descr *singleton = PyArray_GetDefaultDescr(&ASCIIDType);

    if (singleton == NULL) {
        return -1;
    }

    ASCIIDType.singleton = singleton;

    return 0;
}
