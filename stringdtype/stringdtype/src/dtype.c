#include "dtype.h"

#include "casts.h"
#include "static_string.h"

PyTypeObject *StringScalar_Type = NULL;

/*
 * Internal helper to create new instances
 */
PyObject *
new_stringdtype_instance(PyObject *na_object, int coerce)
{
    PyObject *new =
            PyArrayDescr_Type.tp_new((PyTypeObject *)&StringDType, NULL, NULL);

    if (new == NULL) {
        return NULL;
    }

    Py_XINCREF(na_object);
    ((StringDTypeObject *)new)->na_object = na_object;
    npy_static_string na_name = NULL_STRING;
    int hasnull = na_object != NULL;
    int has_nan_na = 0;
    int has_string_na = 0;
    npy_static_string default_string = EMPTY_STRING;
    if (hasnull) {
        // first check for a string
        if (PyUnicode_Check(na_object)) {
            has_string_na = 1;
            Py_ssize_t size = 0;
            const char *buf = PyUnicode_AsUTF8AndSize(na_object, &size);
            default_string = NULL_STRING;
            int res = npy_string_newsize(buf, (size_t)size, &default_string);
            if (res == -1) {
                PyErr_NoMemory();
                Py_DECREF(new);
                return NULL;
            }
            else if (res == -2) {
                // this should never happen
                assert(0);
                Py_DECREF(new);
                return NULL;
            }
        }
        else {
            // treat as nan-like if != comparison returns a object whose truth
            // value raises an error (pd.NA) or a truthy value (e.g. a
            // NaN-like object)
            PyObject *eq = PyObject_RichCompare(na_object, na_object, Py_NE);
            if (eq == NULL) {
                Py_DECREF(new);
                return NULL;
            }
            int is_truthy = PyObject_IsTrue(na_object);
            if (is_truthy == -1) {
                PyErr_Clear();
                has_nan_na = 1;
            }
            else if (is_truthy == 1) {
                has_nan_na = 1;
            }
            Py_DECREF(eq);
        }
        PyObject *na_pystr = PyObject_Str(na_object);
        if (na_pystr == NULL) {
            Py_DECREF(new);
            return NULL;
        }

        Py_ssize_t size = 0;
        const char *utf8_ptr = PyUnicode_AsUTF8AndSize(na_pystr, &size);
        int res = npy_string_newsize(utf8_ptr, (size_t)size, &na_name);
        if (res == -1) {
            PyErr_NoMemory();
            Py_DECREF(new);
            return NULL;
        }
        else if (res == -2) {
            // this should never happen
            assert(0);
            Py_DECREF(new);
            return NULL;
        }
        Py_DECREF(na_pystr);
    }
    ((StringDTypeObject *)new)->has_nan_na = has_nan_na;
    ((StringDTypeObject *)new)->has_string_na = has_string_na;
    ((StringDTypeObject *)new)->default_string = default_string;
    ((StringDTypeObject *)new)->na_name = na_name;
    ((StringDTypeObject *)new)->coerce = coerce;

    PyArray_Descr *base = (PyArray_Descr *)new;
    base->elsize = sizeof(npy_static_string);
    base->alignment = _Alignof(npy_static_string);
    base->flags |= NPY_NEEDS_INIT;
    base->flags |= NPY_LIST_PICKLE;
    base->flags |= NPY_ITEM_REFCOUNT;
    // this is only because of error propagation in sorting, once this dtype
    // lives inside numpy we can relax this and patch the sorting code
    // directly.
    if (hasnull && !(has_string_na && has_nan_na)) {
        base->flags |= NPY_NEEDS_PYAPI;
    }

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
// `scalar`. If scalar is not already a string and
// coerce is nonzero, __str__ is called to convert it
// to a string. If coerce is zero, raises an error for
// non-string or non-NA input. If the scalar is the
// na_object for the dtype class, return a new
// reference to the na_object.

static PyObject *
get_value(PyObject *scalar, int coerce)
{
    PyTypeObject *scalar_type = Py_TYPE(scalar);
    if (!((scalar_type == &PyUnicode_Type) ||
          (scalar_type == StringScalar_Type))) {
        if (coerce == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "StringDType only allows string data");
            return NULL;
        }
        else {
            // attempt to coerce to str
            scalar = PyObject_Str(scalar);
            if (scalar == NULL) {
                // __str__ raised an exception
                return NULL;
            }
        }
    }
    // attempt to decode as UTF8
    return PyUnicode_AsUTF8String(scalar);
}

static PyArray_Descr *
string_discover_descriptor_from_pyobject(PyTypeObject *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    PyObject *val = get_value(obj, 1);
    if (val == NULL) {
        return NULL;
    }

    PyArray_Descr *ret = (PyArray_Descr *)new_stringdtype_instance(NULL, 1);
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
    npy_static_string *sdata = (npy_static_string *)dataptr;

    // free if dataptr holds preexisting string data,
    // npy_string_free does a NULL check
    npy_string_free(sdata);

    // borrow reference
    PyObject *na_object = descr->na_object;

    // setting NA *must* check pointer equality since NA types might not
    // allow equality
    if (na_object != NULL && obj == na_object) {
        // do nothing, npy_string_free already NULLed the struct ssdata points
        // to so it already contains a NA value
    }
    else {
        PyObject *val_obj = get_value(obj, descr->coerce);

        if (val_obj == NULL) {
            return -1;
        }

        char *val = NULL;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(val_obj, &val, &length) == -1) {
            Py_DECREF(val_obj);
            return -1;
        }

        int res = npy_string_newsize(val, length, sdata);

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
    npy_static_string *sdata = (npy_static_string *)dataptr;
    int hasnull = descr->na_object != NULL;

    if (npy_string_isnull(sdata)) {
        if (hasnull) {
            PyObject *na_object = descr->na_object;
            Py_INCREF(na_object);
            val_obj = na_object;
        }
        else {
            val_obj = PyUnicode_FromStringAndSize("", 0);
        }
    }
    else {
        char *data = npy_string_buf(sdata);
        size_t size = npy_string_size(sdata);
        val_obj = PyUnicode_FromStringAndSize(data, size);
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
    return npy_string_size((npy_static_string *)data) != 0;
}

// Implementation of PyArray_CompareFunc.
// Compares unicode strings by their code points.
int
compare(void *a, void *b, void *arr)
{
    StringDTypeObject *descr = (StringDTypeObject *)PyArray_DESCR(arr);
    return _compare(a, b, descr);
}

int
_compare(void *a, void *b, StringDTypeObject *descr)
{
    int hasnull = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    int has_nan_na = descr->has_nan_na;
    if (hasnull && !(has_string_na && has_nan_na)) {
        // check if an error occured already to avoid setting an error again
        if (PyErr_Occurred()) {
            return 0;
        }
    }
    const npy_static_string *default_string = &descr->default_string;
    const npy_static_string *ss_a = (npy_static_string *)a;
    const npy_static_string *ss_b = (npy_static_string *)b;
    int a_is_null = npy_string_isnull(ss_a);
    int b_is_null = npy_string_isnull(ss_b);
    if (NPY_UNLIKELY(a_is_null || b_is_null)) {
        if (hasnull && !has_string_na) {
            if (has_nan_na) {
                if (a_is_null) {
                    return 1;
                }
                else if (b_is_null) {
                    return -1;
                }
            }
            else {
                // we must hold the GIL in this branch
                PyErr_SetString(
                        PyExc_ValueError,
                        "Cannot compare null this is not a nan-like value");
                return 0;
            }
        }
        else {
            if (a_is_null) {
                ss_a = default_string;
            }
            if (b_is_null) {
                ss_b = default_string;
            }
        }
    }
    return npy_string_cmp(ss_a, ss_b);
}

// PyArray_ArgFunc
// The max element is the one with the highest unicode code point.
int
argmax(void *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    npy_static_string *dptr = (npy_static_string *)data;
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
    npy_static_string *dptr = (npy_static_string *)data;
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
            npy_string_free((npy_static_string *)data);
            memset(data, 0, sizeof(npy_static_string));
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
        if (npy_string_newsize("", 0, (npy_static_string *)(data)) < 0) {
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

static int
stringdtype_is_known_scalar_type(PyArray_DTypeMeta *NPY_UNUSED(cls),
                                 PyTypeObject *pytype)
{
    if (pytype == &PyFloat_Type) {
        return 1;
    }
    if (pytype == &PyLong_Type) {
        return 1;
    }
    if (pytype == &PyBool_Type) {
        return 1;
    }
    if (pytype == &PyComplex_Type) {
        return 1;
    }
    if (pytype == &PyUnicode_Type) {
        return 1;
    }
    if (pytype == &PyBytes_Type) {
        return 1;
    }
    if (pytype == &PyBoolArrType_Type) {
        return 1;
    }
    if (pytype == &PyByteArrType_Type) {
        return 1;
    }
    if (pytype == &PyShortArrType_Type) {
        return 1;
    }
    if (pytype == &PyIntArrType_Type) {
        return 1;
    }
    if (pytype == &PyLongArrType_Type) {
        return 1;
    }
    if (pytype == &PyLongLongArrType_Type) {
        return 1;
    }
    if (pytype == &PyUByteArrType_Type) {
        return 1;
    }
    if (pytype == &PyUShortArrType_Type) {
        return 1;
    }
    if (pytype == &PyUIntArrType_Type) {
        return 1;
    }
    if (pytype == &PyULongArrType_Type) {
        return 1;
    }
    if (pytype == &PyULongLongArrType_Type) {
        return 1;
    }
    if (pytype == &PyHalfArrType_Type) {
        return 1;
    }
    if (pytype == &PyFloatArrType_Type) {
        return 1;
    }
    if (pytype == &PyDoubleArrType_Type) {
        return 1;
    }
    if (pytype == &PyLongDoubleArrType_Type) {
        return 1;
    }
    if (pytype == &PyCFloatArrType_Type) {
        return 1;
    }
    if (pytype == &PyCDoubleArrType_Type) {
        return 1;
    }
    if (pytype == &PyCLongDoubleArrType_Type) {
        return 1;
    }
    if (pytype == &PyIntpArrType_Type) {
        return 1;
    }
    if (pytype == &PyUIntpArrType_Type) {
        return 1;
    }
    if (pytype == &PyDatetimeArrType_Type) {
        return 1;
    }
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
        {_NPY_DT_is_known_scalar_type, &stringdtype_is_known_scalar_type},
        {0, NULL}};

static PyObject *
stringdtype_new(PyTypeObject *NPY_UNUSED(cls), PyObject *args, PyObject *kwds)
{
    static char *kwargs_strs[] = {"size", "coerce", "na_object", NULL};

    long size = 0;
    PyObject *na_object = NULL;
    int coerce = 1;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|lpO:StringDType",
                                     kwargs_strs, &size, &coerce,
                                     &na_object)) {
        return NULL;
    }

    PyObject *ret = new_stringdtype_instance(na_object, coerce);

    return ret;
}

static void
stringdtype_dealloc(StringDTypeObject *self)
{
    Py_XDECREF(self->na_object);
    npy_string_free(&self->default_string);
    npy_string_free(&self->na_name);
    PyArrayDescr_Type.tp_dealloc((PyObject *)self);
}

static PyObject *
stringdtype_repr(StringDTypeObject *self)
{
    PyObject *ret = NULL;
    // borrow reference
    PyObject *na_object = self->na_object;
    int coerce = self->coerce;

    if (na_object != NULL && coerce == 0) {
        ret = PyUnicode_FromFormat("StringDType(na_object=%R, coerce=False)",
                                   na_object);
    }
    else if (na_object != NULL) {
        ret = PyUnicode_FromFormat("StringDType(na_object=%R)", na_object);
    }
    else if (coerce == 0) {
        ret = PyUnicode_FromFormat("StringDType(coerce=False)", coerce);
    }
    else {
        ret = PyUnicode_FromString("StringDType()");
    }

    return ret;
}

static int PICKLE_VERSION = 2;

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

    if (self->na_object != NULL) {
        PyTuple_SET_ITEM(ret, 1,
                         Py_BuildValue("(NiO)", PyLong_FromLong(0),
                                       self->coerce, self->na_object));
    }
    else {
        PyTuple_SET_ITEM(
                ret, 1,
                Py_BuildValue("(Ni)", PyLong_FromLong(0), self->coerce));
    }

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
        {"coerce", T_INT, offsetof(StringDTypeObject, coerce), READONLY,
         "Controls hether non-string values should be coerced to string"},
        {NULL, 0, 0, 0, NULL},
};

static PyObject *
StringDType_richcompare(PyObject *self, PyObject *other, int op)
{
    if (!((op == Py_EQ) || (op == Py_NE)) ||
        (Py_TYPE(other) != Py_TYPE(self))) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }

    // we know both are instances of StringDType so this is safe
    StringDTypeObject *sself = (StringDTypeObject *)self;
    StringDTypeObject *sother = (StringDTypeObject *)other;

    int eq = 0;
    PyObject *sna = sself->na_object;
    PyObject *ona = sother->na_object;

    if (sself->coerce != sother->coerce) {
        eq = 0;
    }
    else if (sna == ona) {
        // pointer equality catches pandas.NA and other NA singletons
        eq = 1;
    }
    else if (PyFloat_Check(sna) && PyFloat_Check(ona)) {
        // nan check catches np.nan and float('nan')
        double sna_float = PyFloat_AsDouble(sna);
        if (sna_float == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        double ona_float = PyFloat_AsDouble(ona);
        if (ona_float == -1.0 && PyErr_Occurred()) {
            return NULL;
        }
        if (npy_isnan(sna_float) && npy_isnan(ona_float)) {
            eq = 1;
        }
    }
    else {
        // finally check if a python equals comparison returns True
        eq = PyObject_RichCompareBool(sna, ona, Py_EQ);
        if (eq == -1) {
            return NULL;
        }
    }

    PyObject *ret = Py_NotImplemented;
    if ((op == Py_EQ && eq) || (op == Py_NE && !eq)) {
        ret = Py_True;
    }
    else {
        ret = Py_False;
    }

    Py_INCREF(ret);
    return ret;
}

static Py_hash_t
StringDType_hash(PyObject *self)
{
    StringDTypeObject *sself = (StringDTypeObject *)self;
    PyObject *hash_tup = NULL;
    if (sself->na_object != NULL) {
        hash_tup = Py_BuildValue("(iO)", sself->coerce, sself->na_object);
    }
    else {
        hash_tup = Py_BuildValue("(i)", sself->coerce);
    }

    Py_hash_t ret = PyObject_Hash(hash_tup);
    Py_DECREF(hash_tup);
    return ret;
}

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
                .tp_richcompare = StringDType_richcompare,
                .tp_hash = StringDType_hash,
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
        PyMem_Free(StringDType_casts[i]->dtypes);
        PyMem_Free(StringDType_casts[i]);
    }

    PyMem_Free(StringDType_casts);

    return 0;
}

void
gil_error(PyObject *type, const char *msg)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyErr_SetString(type, msg);
    PyGILState_Release(gstate);
}
