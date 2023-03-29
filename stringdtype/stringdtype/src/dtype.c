#include "dtype.h"

#include "casts.h"
#include "static_string.h"

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
    new->base.elsize = sizeof(ss);
    new->base.alignment = _Alignof(ss);
    new->base.flags |= NPY_NEEDS_INIT;
    new->base.flags |= NPY_LIST_PICKLE;
    new->base.flags |= NPY_ITEM_REFCOUNT;

    return new;
}

/*
 * This is used to determine the correct dtype to return when dealing
 * with a mix of different dtypes (for example when creating an array
 * from a list of scalars). Since StringDType doesn't have any parameters,
 * we can safely always return the first one.
 */
static StringDTypeObject *
common_instance(StringDTypeObject *dtype1,
                StringDTypeObject *NPY_UNUSED(dtype2))
{
    Py_INCREF(dtype1);
    return dtype1;
}

/*
 *  Used to determine the correct "common" dtype for promotion.
 *  cls is always StringDType, other is an arbitrary other DType
 */
static PyArray_DTypeMeta *
common_dtype(PyArray_DTypeMeta *cls, PyArray_DTypeMeta *other)
{
    if (other->type_num == NPY_UNICODE) {
        /*
         *  We have a cast from unicode, so allow unicode to promote
         *  to StringDType
         */
        Py_INCREF(cls);
        return cls;
    }
    Py_INCREF(Py_NotImplemented);
    return (PyArray_DTypeMeta *)Py_NotImplemented;
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

    PyArray_Descr *ret = (PyArray_Descr *)new_stringdtype_instance();
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
stringdtype_setitem(StringDTypeObject *NPY_UNUSED(descr), PyObject *obj,
                    char **dataptr)
{
    PyObject *val_obj = get_value(obj);
    if (val_obj == NULL) {
        return -1;
    }

    char *val = NULL;
    Py_ssize_t length = 0;
    if (PyBytes_AsStringAndSize(val_obj, &val, &length) == -1) {
        return -1;
    }

    // free if dataptr holds preexisting string data,
    // ssfree does a NULL check
    ssfree((ss *)dataptr);

    // copies contents of val into item_val->buf
    int res = ssnewlen(val, length, (ss *)dataptr);

    // val_obj must stay alive until here to ensure *val* doesn't get
    // deallocated
    Py_DECREF(val_obj);

    if (res == -1) {
        PyErr_NoMemory();
        return -1;
    }
    else if (res == -2) {
        // this should never happen
        assert(0);
    }

    return 0;
}

static PyObject *
stringdtype_getitem(StringDTypeObject *NPY_UNUSED(descr), char **dataptr)
{
    char *data;

    if (*dataptr == NULL) {
        data = "\0";
    }
    else {
        data = ((ss *)dataptr)->buf;
    }

    PyObject *val_obj = PyUnicode_FromString(data);

    if (val_obj == NULL) {
        return NULL;
    }

    PyObject *res = PyObject_CallFunctionObjArgs((PyObject *)StringScalar_Type,
                                                 val_obj, NULL);

    Py_DECREF(val_obj);

    return res;
}

// PyArray_NonzeroFunc
// Unicode strings are nonzero if their length is nonzero.
npy_bool
nonzero(void *data, void *NPY_UNUSED(arr))
{
    return ((ss *)data)->len != 0;
}

// Implementation of PyArray_CompareFunc.
// Compares unicode strings by their code points.
int
compare(void *a, void *b, void *NPY_UNUSED(arr))
{
    ss *ss_a = NULL;
    ss *ss_b = NULL;
    load_string(a, &ss_a);
    load_string(b, &ss_b);
    return strcmp(ss_a->buf, ss_b->buf);
}

// PyArray_ArgFunc
// The max element is the one with the highest unicode code point.
int
argmax(void *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    ss *dptr = (ss *)data;
    *max_ind = 0;
    for (int i = 1; i < n; i++) {
        if (compare(&dptr[i], &dptr[*max_ind], arr) > 0) {
            *max_ind = i;
        }
    }
    return 0;
}

// PyArray_ArgFunc
// The min element is the one with the lowest unicode code point.
int
argmin(void *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    ss *dptr = (ss *)data;
    *min_ind = 0;
    for (int i = 1; i < n; i++) {
        if (compare(&dptr[i], &dptr[*min_ind], arr) < 0) {
            *min_ind = i;
        }
    }
    return 0;
}

// PyArray_FillWithScalarFunc
// Fill an array with a scalar value.
int
fillwithscalar(void *buffer, npy_intp length, void *value,
               void *NPY_UNUSED(arr))
{
    ss *dptr = (ss *)buffer;
    for (int i = 0; i < length; i++) {
        dptr[i] = *(ss *)value;
    }
    return 0;
}

static StringDTypeObject *
stringdtype_ensure_canonical(StringDTypeObject *self)
{
    Py_INCREF(self);
    return self;
}

static int
stringdtype_clear_loop(void *NPY_UNUSED(traverse_context),
                       PyArray_Descr *NPY_UNUSED(descr), char *data,
                       npy_intp size, npy_intp stride,
                       NpyAuxData *NPY_UNUSED(auxdata))
{
    while (size--) {
        if (data != NULL) {
            ssfree((ss *)data);
            memset(data, 0, sizeof(ss));
        }
        data += stride;
    }

    return 0;
}

static int
stringdtype_get_clear_loop(void *NPY_UNUSED(traverse_context),
                           PyArray_Descr *NPY_UNUSED(descr),
                           int NPY_UNUSED(aligned),
                           npy_intp NPY_UNUSED(fixed_stride),
                           traverse_loop_function **out_loop,
                           NpyAuxData **NPY_UNUSED(out_auxdata),
                           NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &stringdtype_clear_loop;
    return 0;
}

static PyType_Slot StringDType_Slots[] = {
        {NPY_DT_common_instance, &common_instance},
        {NPY_DT_common_dtype, &common_dtype},
        {NPY_DT_discover_descr_from_pyobject,
         &string_discover_descriptor_from_pyobject},
        {NPY_DT_setitem, &stringdtype_setitem},
        {NPY_DT_getitem, &stringdtype_getitem},
        {NPY_DT_ensure_canonical, &stringdtype_ensure_canonical},
        {NPY_DT_PyArray_ArrFuncs_nonzero, &nonzero},
        {NPY_DT_PyArray_ArrFuncs_compare, &compare},
        {NPY_DT_PyArray_ArrFuncs_argmax, &argmax},
        {NPY_DT_PyArray_ArrFuncs_argmin, &argmin},
        {NPY_DT_PyArray_ArrFuncs_fillwithscalar, &fillwithscalar},
        {NPY_DT_get_clear_loop, &stringdtype_get_clear_loop},
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
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
stringdtype_repr(StringDTypeObject *NPY_UNUSED(self))
{
    return PyUnicode_FromString("StringDType()");
}

static int PICKLE_VERSION = 1;

static PyObject *
stringdtype__reduce__(StringDTypeObject *NPY_UNUSED(self))
{
    PyObject *ret, *mod, *obj, *state;

    ret = PyTuple_New(3);
    if (ret == NULL) {
        return NULL;
    }

    mod = PyImport_ImportModule("stringdtype");
    if (mod == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    obj = PyObject_GetAttrString(mod, "StringDType");
    Py_DECREF(mod);
    if (obj == NULL) {
        Py_DECREF(ret);
        return NULL;
    }

    PyTuple_SET_ITEM(ret, 0, obj);

    PyTuple_SET_ITEM(ret, 1, PyTuple_New(0));

    state = PyTuple_New(1);

    PyTuple_SET_ITEM(state, 0, PyLong_FromLong(PICKLE_VERSION));

    PyTuple_SET_ITEM(ret, 2, state);

    return ret;
}

static PyObject *
stringdtype__setstate__(StringDTypeObject *NPY_UNUSED(self), PyObject *args)
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

static PyMethodDef StringDType_methods[] = {
        {
                "__reduce__",
                (PyCFunction)stringdtype__reduce__,
                METH_NOARGS,
                "Reduction method for an StringDType object",
        },
        {
                "__setstate__",
                (PyCFunction)stringdtype__setstate__,
                METH_O,
                "Unpickle an StringDType object",
        },
        {NULL},
};

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
                .tp_methods = StringDType_methods,
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

    for (int i = 0; casts[i] != NULL; i++) {
        free(casts[i]->dtypes);
        free(casts[i]);
    }

    return 0;
}
