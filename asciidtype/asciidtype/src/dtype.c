// clang-format off
#include <Python.h>
#include "structmember.h"
// clang-format on

#define PY_ARRAY_UNIQUE_SYMBOL asciidtype_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL asciidtype_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

#include "dtype.h"

#include "casts.h"

PyTypeObject *ASCIIScalar_Type = NULL;

static PyObject *
get_value(PyObject *scalar)
{
    PyObject *ret_bytes = NULL;
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    if (scalar_type == &PyUnicode_Type) {
        // attempt to decode as ASCII
        ret_bytes = PyUnicode_AsASCIIString(scalar);
        if (ret_bytes == NULL) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "Can only store ASCII text in a ASCIIDType array.");
            return NULL;
        }
    }
    else if (scalar_type != ASCIIScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store ASCII text in a ASCIIDType array.");
        return NULL;
    }
    else {
        ret_bytes = PyUnicode_AsASCIIString(scalar);
        if (ret_bytes == NULL) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "Can only store ASCII text in a ASCIIDType array.");
            return NULL;
        }
    }
    return ret_bytes;
}

/*
 * Internal helper to create new instances
 */
ASCIIDTypeObject *
new_asciidtype_instance(long size)
{
    ASCIIDTypeObject *new = (ASCIIDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&ASCIIDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->size = size;
    new->base.elsize = size * sizeof(char);
    new->base.alignment = size *_Alignof(char);

    return new;
}

/*
 * This is used to determine the correct dtype to return when dealing
 * with a mix of different dtypes (for example when creating an array
 * from a list of scalars). Always return the dtype with the biggest
 * size.
 */
static ASCIIDTypeObject *
common_instance(ASCIIDTypeObject *dtype1, ASCIIDTypeObject *dtype2)
{
    if (dtype1->size >= dtype2->size) {
        Py_INCREF(dtype1);
        return dtype1;
    }
    Py_INCREF(dtype2);
    return dtype2;
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
    PyTypeObject *obj_type = Py_TYPE(obj);
    PyArray_Descr *ret = NULL;
    if (obj_type != ASCIIScalar_Type) {
        if (PyUnicode_Check(obj)) {
            if (!PyUnicode_IS_ASCII(obj)) {
                PyErr_SetString(
                        PyExc_TypeError,
                        "Can only store strings or bytes convertible to ASCII "
                        "in a ASCIIDType array.");
                return NULL;
            }
            ret = (PyArray_Descr *)new_asciidtype_instance(
                    (long)PyUnicode_GetLength(obj));
        }
        // could do bytes too if we want
        PyErr_SetString(PyExc_TypeError,
                        "Can only store strings or bytes convertible to ASCII "
                        "in a ASCIIDType array.");
        return NULL;
    }
    else {
        ret = (PyArray_Descr *)PyObject_GetAttrString(obj, "dtype");
        if (ret == NULL) {
            return NULL;
        }
    }
    return ret;
}

static int
asciidtype_setitem(ASCIIDTypeObject *descr, PyObject *obj, char *dataptr)
{
    PyObject *value = get_value(obj);
    if (value == NULL) {
        return -1;
    }

    Py_ssize_t len = PyBytes_Size(value);

    long copysize;

    if (len > descr->size) {
        copysize = descr->size;
    }
    else {
        copysize = len;
    }

    char *char_value = PyBytes_AsString(value);

    memcpy(dataptr, char_value, copysize * sizeof(char));  // NOLINT

    for (int i = copysize; i < descr->size; i++) {
        dataptr[i] = '\0';
    }

    Py_DECREF(value);

    return 0;
}

static PyObject *
asciidtype_getitem(ASCIIDTypeObject *descr, char *dataptr)
{
    char scalar_buffer[descr->size + 1];

    memcpy(scalar_buffer, dataptr, descr->size * sizeof(char));

    scalar_buffer[descr->size] = '\0';

    PyObject *val_obj = PyUnicode_FromString(scalar_buffer);
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

    long size = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|l:ASCIIDType", kwargs_strs,
                                     &size)) {
        return NULL;
    }

    PyObject *ret = (PyObject *)new_asciidtype_instance(size);
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
        {"size", T_LONG, offsetof(ASCIIDTypeObject, size), READONLY,
         "The number of characters per array element"},
        {NULL},
};

static int PICKLE_VERSION = 1;

static PyObject *
asciidtype__reduce__(ASCIIDTypeObject *self)
{
    PyObject *ret, *mod, *obj, *state;

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }

    mod = PyImport_ImportModule("asciidtype");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    obj = PyObject_GetAttrString(mod, "ASCIIDType");
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    PyTuple_SET_ITEM(ret, 0, obj);

    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(l)", self->size));

    state = PyTuple_New(1);

    PyTuple_SET_ITEM(state, 0, PyLong_FromLong(PICKLE_VERSION));

    PyTuple_SET_ITEM(ret, 2, state);

    return ret;
}

static PyObject *
asciidtype__setstate__(ASCIIDTypeObject *NPY_UNUSED(self), PyObject *args)
{
    if (PyTuple_GET_SIZE(args) != 1 ||
        !(PyLong_Check(PyTuple_GET_ITEM(args, 0)))) {
        PyErr_BadInternalCall();
        return NULL;
    }

    long version = PyLong_AsLong(PyTuple_GET_ITEM(args, 0));

    if (version != PICKLE_VERSION) {
        PyErr_Format(PyExc_ValueError,
                     "Pickle version mismatch. Got version %d but expected "
                     "version %d.",
                     version, PICKLE_VERSION);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef ASCIIDType_methods[] = {
        {
                "__reduce__",
                (PyCFunction)asciidtype__reduce__,
                METH_NOARGS,
                "Reduction method for an ASCIIDType object",
        },
        {
                "__setstate__",
                (PyCFunction)asciidtype__setstate__,
                METH_O,
                "Unpickle an ASCIIDType object",
        },
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
                .tp_methods = ASCIIDType_methods,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_ascii_dtype(void)
{
    PyArrayMethod_Spec **casts = get_casts();

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

    free(ASCIIDType_DTypeSpec.casts[1]->dtypes);
    free(ASCIIDType_DTypeSpec.casts[1]);
    free(ASCIIDType_DTypeSpec.casts[2]->dtypes);
    free(ASCIIDType_DTypeSpec.casts[2]);
    free(ASCIIDType_DTypeSpec.casts);

    return 0;
}
