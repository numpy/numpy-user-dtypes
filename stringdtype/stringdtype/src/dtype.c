#include "dtype.h"

#include "casts.h"

PyTypeObject *StringScalar_Type = NULL;

/*
 * Internal helper to create new instances
 */
StringDTypeObject *
new_stringdtype_instance(void)
{
    StringDTypeObject *new = (StringDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&StringDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->base.elsize = sizeof(char *);
    new->base.alignment = _Alignof(char *);

    return new;
}

/*
 * This is used to determine the correct dtype to return when dealing
 * with a mix of different dtypes (for example when creating an array
 * from a list of scalars). Since StringDType doesn't have any parameters,
 * we can safely always return the first one.
 */
static StringDTypeObject *
common_instance(StringDTypeObject *dtype1, StringDTypeObject *dtype2)
{
    Py_INCREF(dtype1);
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // for now always raise an error here until we can figure out
    // how to deal with strings here
    PyErr_SetString(PyExc_RuntimeError, "common_dtype called in StringDType");
    return NULL;
}

// For a given python object, this function returns a borrowed reference
// to the dtype property of the array
static PyArray_Descr *
string_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    if (Py_TYPE(obj) != StringScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store StringScalar in a StringDType array.");
        return NULL;
    }

    PyArray_Descr *ret = (PyArray_Descr *)PyObject_GetAttrString(obj, "dtype");
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

static PyObject *
get_value(PyObject *scalar)
{
    PyObject *ret_bytes = NULL;
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    // FIXME: handle bytes too
    if ((scalar_type == &PyUnicode_Type) ||
        (scalar_type == StringScalar_Type)) {
        // attempt to decode as UTF8
        ret_bytes = PyUnicode_AsUTF8String(scalar);
        if (ret_bytes == NULL) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "Can only store UTF8 text in a StringDType array.");
            return NULL;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store String text in a StringDType array.");
        return NULL;
    }
    return ret_bytes;
}

// Take a python object `obj` and insert it into the array of dtype `descr` at
// the position given by dataptr.
static int
stringdtype_setitem(StringDTypeObject *descr, PyObject *obj, char **dataptr)
{
    PyObject *val_obj = get_value(obj);
    char *val = NULL;
    Py_ssize_t length = 0;
    if (PyBytes_AsStringAndSize(val_obj, &val, &length) == -1) {
        return -1;
    }

    *dataptr = malloc(sizeof(char) * length + 1);
    strncpy(*dataptr, val, length + 1);
    Py_DECREF(val_obj);
    return 0;
}

static PyObject *
stringdtype_getitem(StringDTypeObject *descr, char **dataptr)
{
    PyObject *val_obj = PyUnicode_FromString(*dataptr);

    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)StringScalar_Type,
                                                 val_obj, descr, NULL);

    if (res == NULL) {
        return NULL;
    }

    Py_DECREF(val_obj);

    return res;
}

static StringDTypeObject *
stringdtype_ensure_canonical(StringDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyType_Slot StringDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &string_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &stringdtype_setitem},
        {NPY_DT_getitem, &stringdtype_getitem},
        {NPY_DT_ensure_canonical, &stringdtype_ensure_canonical},
        {0, NULL}};

static PyObject *
stringdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"size", NULL};

    long size = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|l:StringDType", kwargs_strs,
                                     &size)) {
        return NULL;
    }

    return (PyObject *)new_stringdtype_instance();
}

static void
stringdtype_dealloc(StringDTypeObject *self)
{
    // Need to deallocate all the memory allocated during setitem.

    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
stringdtype_repr(StringDTypeObject *self)
{
    return PyUnicode_FromString("StringDType");
}

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta StringDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "stringdtype.StringDType",
                .tp_basicsize = sizeof(StringDTypeObject),
                .tp_new = stringdtype_new,
                .tp_dealloc = (destructor)stringdtype_dealloc,
                .tp_repr = (reprfunc)stringdtype_repr,
                .tp_str = (reprfunc)stringdtype_repr,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_string_dtype(void)
{
    PyArrayMethod_Spec **casts = get_casts();

    PyArrayDTypeMeta_Spec StringDType_DTypeSpec = {
            .typeobj = StringScalar_Type,
            .slots = StringDType_Slots,
            .casts = casts,
    };

    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&StringDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&StringDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&StringDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&StringDType, &StringDType_DTypeSpec) <
        0) {
        return -1;
    }

    PyArray_Descr *singleton = PyArray_GetDefaultDescr(&StringDType);

    if (singleton == NULL) {
        return -1;
    }

    StringDType.singleton = singleton;

    return 0;
}
