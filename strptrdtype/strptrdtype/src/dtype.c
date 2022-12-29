#include "dtype.h"

// #include "casts.h"

PyTypeObject *StrPtrScalar_Type = NULL;

// static PyObject *
// get_value(PyObject *scalar)
// {
//     PyObject *ret_bytes = NULL;
//     PyTypeObject *scalar_type = Py_TYPE(scalar);
//     if (scalar_type == &PyUnicode_Type) {
//         // attempt to decode as ASCII
//         ret_bytes = PyUnicode_AsASCIIString(scalar);
//         if (ret_bytes == NULL) {
//             PyErr_SetString(
//                     PyExc_TypeError,
//                     "Can only store ASCII text in a ASCIIDType array.");
//             return NULL;
//         }
//     }
//     else if (scalar_type != StrPtrScalar_Type) {
//         PyErr_SetString(PyExc_TypeError,
//                         "Can only store ASCII text in a ASCIIDType array.");
//         return NULL;
//     }
//     else {
//         PyObject *value = PyObject_GetAttrString(scalar, "value");
//         if (value == NULL) {
//             return NULL;
//         }
//         ret_bytes = PyUnicode_AsASCIIString(value);
//         if (ret_bytes == NULL) {
//             PyErr_SetString(
//                     PyExc_TypeError,
//                     "Can only store ASCII text in a ASCIIDType array.");
//             return NULL;
//         }
//         Py_DECREF(value);
//     }
//     return ret_bytes;
// }

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

    PyErr_SetString(PyExc_RuntimeError, "common_dtype called in ASCIIDType");
    return NULL;

    // Py_INCREF(Py_NotImplemented);
    // return (PyArray_DTypeMeta *)Py_NotImplemented;
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

static int
strptrdtype_setitem(StrPtrDTypeObject *descr, PyObject *obj, char *dataptr)
{
    // PyObject *value = get_value(obj);
    // if (value == NULL) {
    //     return -1;
    // }
    //
    // Py_ssize_t len = PyBytes_Size(value);
    //
    // long copysize;
    //
    // if (len > descr->size) {
    //     copysize = descr->size;
    // }
    // else {
    //     copysize = len;
    // }
    //
    // char *char_value = PyBytes_AsString(value);
    //
    // memcpy(dataptr, char_value, copysize * sizeof(char));  // NOLINT
    //
    // for (int i = copysize; i < descr->size; i++) {
    //     dataptr[i] = '\0';
    // }
    //
    // Py_DECREF(value);
    //

    char *val = PyBytes_AsString(obj);


    return 0;
}

static PyObject *
strptrdtype_getitem(StrPtrDTypeObject *descr, char *dataptr)
{
    // dataptr points to an element of the array; but each element is itself a pointer
    // to a charcter array in memory, so we probably need to dereference this
    PyObject *val_obj = PyBytes_FromString(dataptr);
    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)StrPtrScalar_Type, val_obj, NULL);
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
        /* The header is wrong on main :(, so we add 1 */
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
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
strptrdtype_repr(ASCIIDTypeObject *self)
{
    return PyUnicode_FromString("StrPtrDType");
}

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
PyArray_DTypeMeta ASCIIDType = {
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
    // PyArrayMethod_Spec **casts = get_casts();

    PyArrayDTypeMeta_Spec StrPtrDType_DTypeSpec = {
            .typeobj = StrPtrScalar_Type,
            .slots = StrPtrDType_Slots,
            // .casts = casts,
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

    // free(StrPtrDType_DTypeSpec.casts[1]->dtypes);
    // free(StrPtrDType_DTypeSpec.casts[1]);
    // free(StrPtrDType_DTypeSpec.casts[2]->dtypes);
    // free(StrPtrDType_DTypeSpec.casts[2]);
    // free(StrPtrDType_DTypeSpec.casts);

    return 0;
}
