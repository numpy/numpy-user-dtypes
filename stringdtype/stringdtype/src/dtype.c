#include "dtype.h"

#include "casts.h"
#include "static_string.h"

PyTypeObject *StringScalar_Type = NULL;
static PyTypeObject *StringNA_Type = NULL;
PyObject *NA_OBJ = NULL;

/*
 * Internal helper to create new instances
 */
PyObject *
new_stringdtype_instance(PyObject *na_object)
{
    PyObject *new =
            PyArrayDescr_Type.tp_new((PyTypeObject *)&StringDType, NULL, NULL);

    if (new == NULL) {
        return NULL;
    }

    Py_INCREF(na_object);
    ((StringDTypeObject *)new)->na_object = na_object;

    PyArray_Descr *base = (PyArray_Descr *)new;
    base->elsize = sizeof(ss);
    base->alignment = _Alignof(ss);
    base->flags |= NPY_NEEDS_INIT;
    base->flags |= NPY_LIST_PICKLE;
    base->flags |= NPY_ITEM_REFCOUNT;

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

// returns a new reference to the string "value" of
// `scalar`. If scalar is not already a string, __str__
// is called to convert it to a string. If the scalar
// is the na_object for the dtype class, return
// a new reference to the na_object.

static PyObject *
get_value(PyObject *scalar)
{
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    if (!((scalar_type == &PyUnicode_Type) ||
          (scalar_type == StringScalar_Type))) {
        // attempt to coerce to str
        scalar = PyObject_Str(scalar);
        if (scalar == NULL) {
            // __str__ raised an exception
            return NULL;
        }
    }
    // attempt to decode as UTF8
    return PyUnicode_AsUTF8String(scalar);
}

static PyArray_Descr *
string_discover_descriptor_from_pyobject(PyTypeObject *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    PyObject *val = get_value(obj);
    if (val == NULL) {
        return NULL;
    }

    PyArray_Descr *ret = (PyArray_Descr *)new_stringdtype_instance(NA_OBJ);
    if (ret == NULL) {
        return NULL;
    }
    return ret;
}

// Take a python object `obj` and insert it into the array of dtype `descr` at
// the position given by dataptr.
int
stringdtype_setitem(StringDTypeObject *descr, PyObject *obj, char **dataptr)
{
    ss *sdata = (ss *)dataptr;

    // free if dataptr holds preexisting string data,
    // ssfree does a NULL check
    ssfree(sdata);

    // borrow reference
    PyObject *na_object = descr->na_object;

    // setting NA *must* check pointer equality since NA types might not
    // allow equality
    if (obj == na_object) {
        // do nothing, ssfree already NULLed the struct ssdata points to
        // so it already contains a NA value
    }
    else {
        PyObject *val_obj = get_value(obj);

        if (val_obj == NULL) {
            return -1;
        }

        char *val = NULL;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(val_obj, &val, &length) == -1) {
            Py_DECREF(val_obj);
            return -1;
        }

        // copies contents of val into item_val->buf
        int res = ssnewlen(val, length, sdata);

        if (res == -1) {
            PyErr_NoMemory();
            Py_DECREF(val_obj);
            return -1;
        }
        else if (res == -2) {
            // this should never happen
            assert(0);
            Py_DECREF(val_obj);
            return -1;
        }
    }

    return 0;
}

static PyObject *
stringdtype_getitem(StringDTypeObject *descr, char **dataptr)
{
    PyObject *val_obj = NULL;
    ss *sdata = (ss *)dataptr;

    if (ss_isnull(sdata)) {
        PyObject *na_object = descr->na_object;
        Py_INCREF(na_object);
        val_obj = na_object;
    }
    else {
        char *data = sdata->buf;
        size_t len = sdata->len;
        val_obj = PyUnicode_FromStringAndSize(data, len);
        if (val_obj == NULL) {
            return NULL;
        }
    }

    /*
     * In principle we should return a StringScalar instance here, but
     * creating a StringScalar via PyObject_CallFunctionObjArgs has
     * approximately 4 times as much overhead than just returning a str
     * here. This is due to Python overhead as well as copying the string
     * buffer from val_obj to the StringScalar we'd like to return. In
     * principle we could avoid this by making a C function like
     * PyUnicode_FromStringAndSize that fills a StringScalar instead of a
     * str. For now (4-11-23) we are punting on that with the expectation that
     * eventually the scalar type for this dtype will be str.
     */
    return val_obj;
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
    ss *ss_a = (ss *)a;
    ss *ss_b = (ss *)b;
    int a_is_null = ss_isnull(ss_a);
    int b_is_null = ss_isnull(ss_b);
    if (a_is_null) {
        return 1;
    }
    else if (b_is_null) {
        return -1;
    }
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

static int
stringdtype_fill_zero_loop(void *NPY_UNUSED(traverse_context),
                           PyArray_Descr *NPY_UNUSED(descr), char *data,
                           npy_intp size, npy_intp stride,
                           NpyAuxData *NPY_UNUSED(auxdata))
{
    while (size--) {
        if (ssnewlen("", 0, (ss *)(data)) < 0) {
            return -1;
        }
        data += stride;
    }
    return 0;
}

static int
stringdtype_get_fill_zero_loop(void *NPY_UNUSED(traverse_context),
                               PyArray_Descr *NPY_UNUSED(descr),
                               int NPY_UNUSED(aligned),
                               npy_intp NPY_UNUSED(fixed_stride),
                               traverse_loop_function **out_loop,
                               NpyAuxData **NPY_UNUSED(out_auxdata),
                               NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &stringdtype_fill_zero_loop;
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
        {NPY_DT_get_clear_loop, &stringdtype_get_clear_loop},
        {NPY_DT_get_fill_zero_loop, &stringdtype_get_fill_zero_loop},
        {0, NULL}};

static PyObject *
stringdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"size", "na_object", NULL};

    long size = 0;
    PyObject *na_object = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|lO:StringDType",
                                     kwargs_strs, &size, &na_object)) {
        return NULL;
    }

    if (na_object == NULL) {
        na_object = NA_OBJ;
    }

    PyObject *ret = new_stringdtype_instance(na_object);

    return ret;
}

static void
stringdtype_dealloc(StringDTypeObject *self)
{
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
stringdtype_repr(StringDTypeObject *self)
{
    PyObject *ret = NULL;
    // borrow reference
    PyObject *na_object = self->na_object;

    // TODO: handle non-default NA
    if (na_object != NA_OBJ) {
        ret = PyUnicode_FromFormat("StringDType(na_object=%R)",
                                   self->na_object);
    }
    else {
        ret = PyUnicode_FromString("StringDType()");
    }

    return ret;
}

static int PICKLE_VERSION = 1;

static PyObject *
stringdtype__reduce__(StringDTypeObject *self)
{
    PyObject *ret, *mod, *obj;

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

    PyTuple_SET_ITEM(
            ret, 1,
            Py_BuildValue("(NO)", PyLong_FromLong(0), self->na_object));

    PyTuple_SET_ITEM(ret, 2, Py_BuildValue("(l)", PICKLE_VERSION));

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
        {NULL, NULL, 0, NULL},
};

static PyMemberDef StringDType_members[] = {
        {"na_object", T_OBJECT_EX, offsetof(StringDTypeObject, na_object),
         READONLY,
         "The missing value object associated with the dtype instance"},
        {NULL, 0, 0, 0, NULL},
};

/*
 * This is the basic things that you need to create a Python Type/Class in C.
 * However, there is a slight difference here because we create a
 * PyArray_DTypeMeta, which is a larger struct than a typical type.
 * (This should get a bit nicer eventually with Python >3.11.)
 */
StringDType_type StringDType = {
        {{{
                PyVarObject_HEAD_INIT(NULL, 0).tp_name =
                        "stringdtype.StringDType",
                .tp_basicsize = sizeof(StringDTypeObject),
                .tp_new = stringdtype_new,
                .tp_dealloc = (destructor)stringdtype_dealloc,
                .tp_repr = (reprfunc)stringdtype_repr,
                .tp_str = (reprfunc)stringdtype_repr,
                .tp_methods = StringDType_methods,
                .tp_members = StringDType_members,
        }}},
        /* rest, filled in during DTypeMeta initialization */
};

int
init_string_dtype(void)
{
    PyArrayMethod_Spec **StringDType_casts = get_casts();

    PyArrayDTypeMeta_Spec StringDType_DTypeSpec = {
            .flags = NPY_DT_PARAMETRIC,
            .typeobj = StringScalar_Type,
            .slots = StringDType_Slots,
            .casts = StringDType_casts,
    };

    /* Loaded dynamically, so may need to be set here: */
    ((PyObject *)&StringDType)->ob_type = &PyArrayDTypeMeta_Type;
    ((PyTypeObject *)&StringDType)->tp_base = &PyArrayDescr_Type;
    if (PyType_Ready((PyTypeObject *)&StringDType) < 0) {
        return -1;
    }

    if (PyArrayInitDTypeMeta_FromSpec((PyArray_DTypeMeta *)&StringDType,
                                      &StringDType_DTypeSpec) < 0) {
        return -1;
    }

    PyArray_Descr *singleton =
            PyArray_GetDefaultDescr((PyArray_DTypeMeta *)&StringDType);

    if (singleton == NULL) {
        return -1;
    }

    StringDType.base.singleton = singleton;

    for (int i = 0; StringDType_casts[i] != NULL; i++) {
        free(StringDType_casts[i]->dtypes);
        free(StringDType_casts[i]);
    }

    return 0;
}

int
init_string_na_object(PyObject *mod)
{
    NA_OBJ = PyObject_GetAttrString(mod, "NA");
    StringNA_Type = Py_TYPE(NA_OBJ);
    Py_INCREF(StringNA_Type);
    return 0;
}
