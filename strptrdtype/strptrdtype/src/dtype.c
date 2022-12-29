#include "dtype.h"

#include "casts.h"

PyTypeObject *StrPtrScalar_Type = NULL;

/*
 * Internal helper to create new instances
 */
StrPtrDTypeObject *
new_strptrdtype_instance(void)
{
    StrPtrDTypeObject *new = (StrPtrDTypeObject *)PyArrayDescr_Type.tp_new(
            (PyTypeObject *)&StrPtrDType, NULL, NULL);
    if (new == NULL) {
        return NULL;
    }
    new->base.elsize = sizeof(char *);
    new->base.alignment = _Alignof(char *);

    return new;
}

//
// This is used to determine the correct dtype to return when operations mix
// dtypes (I think?). For now just return the first one.
//
static StrPtrDTypeObject *
common_instance(StrPtrDTypeObject *dtype1, StrPtrDTypeObject *dtype2)
{
    if (!PyObject_RichCompareBool((PyObject *)dtype1, (PyObject *)dtype2,
                                  Py_EQ)) {
        PyErr_SetString(
                PyExc_RuntimeError,
                "common_instance called on unequal StrPtrDType instances");
        return NULL;
    }
    return dtype1;
}

static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    // for now always raise an error here until we can figure out
    // how to deal with strings here
    PyErr_SetString(PyExc_RuntimeError, "common_dtype called in StrPtrDType");
    return NULL;
}

// For a given python object, this function returns a borrowed reference
// to the dtype property of the array
static PyArray_Descr *
strptr_discover_descriptor_from_pyobject(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    if (Py_TYPE(obj) != StrPtrScalar_Type) {
        PyErr_SetString(PyExc_TypeError,
                        "Can only store StrPtrScalar in a StrPtrDType array.");
        return NULL;
    }

    PyArray_Descr *ret = (PyArray_Descr *)PyObject_GetAttrString(obj, "dtype");
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

// Take a python object `obj` and insert it into the array of dtype `descr` at
// the position given by dataptr.
static int
strptrdtype_setitem(StrPtrDTypeObject *descr, PyObject *obj, char **dataptr)
{
    char *val = PyBytes_AsString(obj);
    if (val == NULL) {
        return -1;
    }

    *dataptr = malloc(sizeof(char) * strlen(val));
    strcpy(*dataptr, val);
    return 0;
}

static PyObject *
strptrdtype_getitem(StrPtrDTypeObject *descr, char **dataptr)
{
    PyObject *val_obj = PyBytes_FromString(*dataptr);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)StrPtrScalar_Type,
                                                 val_obj, NULL);
    if (res == NULL) {
        return NULL;
    }
    Py_DECREF(val_obj);

    return res;
}

static StrPtrDTypeObject *
strptrdtype_ensure_canonical(StrPtrDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static PyType_Slot StrPtrDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &strptr_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &strptrdtype_setitem},
        {NPY_DT_getitem, &strptrdtype_getitem},
        {NPY_DT_ensure_canonical, &strptrdtype_ensure_canonical},
        {0, NULL}};

static PyObject *
strptrdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    return (PyObject *)new_strptrdtype_instance();
}

static void
strptrdtype_dealloc(StrPtrDTypeObject *self)
{
    // Need to deallocate all the memory allocated during setitem.

    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
strptrdtype_repr(StrPtrDTypeObject *self)
{
    return PyUnicode_FromString("StrPtrDType");
}

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta StrPtrDType = {
        {{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "strptrdtype.StrPtrDType",
                .tp_basicsize = sizeof(StrPtrDTypeObject),
                .tp_new = strptrdtype_new,
                .tp_dealloc = (destructor)strptrdtype_dealloc,
                .tp_repr = (reprfunc)strptrdtype_repr,
                .tp_str = (reprfunc)strptrdtype_repr,
        }},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_strptr_dtype(void)
{
    PyArrayMethod_Spec **casts = get_casts();

    PyArrayDTypeMeta_Spec StrPtrDType_DTypeSpec = {
            .typeobj = StrPtrScalar_Type,
            .slots = StrPtrDType_Slots,
            .casts = casts,
    };

    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&StrPtrDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&StrPtrDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&StrPtrDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec(&StrPtrDType, &StrPtrDType_DTypeSpec) <
        0) {
        return -1;
    }

    PyArray_Descr *singleton = PyArray_GetDefaultDescr(&StrPtrDType);

    if (singleton == NULL) {
        return -1;
    }

    StrPtrDType.singleton = singleton;

    return 0;
}
