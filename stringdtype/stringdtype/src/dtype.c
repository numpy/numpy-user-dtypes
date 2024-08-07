#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL stringdtype_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/halffloat.h"
#include "numpy/npy_math.h"

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

    npy_string_allocator *allocator = NULL;
    PyThread_type_lock *allocator_lock = NULL;

    char *default_string_buf = NULL;
    char *na_name_buf = NULL;

    allocator = _NpyString_new_allocator(PyMem_RawMalloc, PyMem_RawFree,
                                        PyMem_RawRealloc);
    if (allocator == NULL) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to create string allocator");
        goto fail;
    }

    allocator_lock = PyThread_allocate_lock();
    if (allocator_lock == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate thread lock");
        goto fail;
    }

    _npy_static_string default_string = {0, NULL};
    _npy_static_string na_name = {0, NULL};

    Py_XINCREF(na_object);
    ((StringDTypeObject *)new)->na_object = na_object;
    int hasnull = na_object != NULL;
    int has_nan_na = 0;
    int has_string_na = 0;
    if (hasnull) {
        // first check for a string
        if (PyUnicode_Check(na_object)) {
            has_string_na = 1;
            Py_ssize_t size = 0;
            const char *buf = PyUnicode_AsUTF8AndSize(na_object, &size);
            default_string.buf = PyMem_RawMalloc(size);
            memcpy((char *)default_string.buf, buf, size);
            default_string.size = size;
        }
        else {
            // treat as nan-like if != comparison returns a object whose truth
            // value raises an error (pd.NA) or a truthy value (e.g. a
            // NaN-like object)
            PyObject *eq = PyObject_RichCompare(na_object, na_object, Py_NE);
            if (eq == NULL) {
                goto fail;
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
            goto fail;
        }

        Py_ssize_t size = 0;
        const char *utf8_ptr = PyUnicode_AsUTF8AndSize(na_pystr, &size);
        if (utf8_ptr == NULL) {
            Py_DECREF(na_pystr);
            goto fail;
        }
        na_name.buf = PyMem_RawMalloc(size);
        memcpy((char *)na_name.buf, utf8_ptr, size);
        na_name.size = size;
        Py_DECREF(na_pystr);
    }

    StringDTypeObject *snew = (StringDTypeObject *)new;

    snew->has_nan_na = has_nan_na;
    snew->has_string_na = has_string_na;
    snew->coerce = coerce;
    snew->allocator_lock = allocator_lock;
    snew->allocator = allocator;
    snew->array_owned = 0;
    snew->na_name = na_name;
    snew->default_string = default_string;

    PyArray_Descr *base = (PyArray_Descr *)new;
    base->elsize = SIZEOF_NPY_PACKED_STATIC_STRING;
    base->alignment = ALIGNOF_NPY_PACKED_STATIC_STRING;
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

fail:
    // this only makes sense if the allocator isn't attached to new yet
    Py_DECREF(new);
    if (default_string_buf != NULL) {
        PyMem_RawFree(default_string_buf);
    }
    if (na_name_buf != NULL) {
        PyMem_RawFree(na_name_buf);
    }
    if (allocator != NULL) {
        _NpyString_free_allocator(allocator);
    }
    if (allocator_lock != NULL) {
        PyThread_free_lock(allocator_lock);
    }
    return NULL;
}

// sets the logical rules for determining equality between dtype instances
int
_eq_comparison(int scoerce, int ocoerce, PyObject *sna, PyObject *ona)
{
    if (scoerce != ocoerce) {
        return 0;
    }
    else if (sna == ona) {
        // Pointer equality catches pandas.NA and other NA singletons.
        // Also much faster when comparing two dtype instances that share
        // the same na_object.
        return 1;
    }
    else if (PyFloat_Check(sna) && PyFloat_Check(ona)) {
        // nan check catches np.nan and float('nan')
        double sna_float = PyFloat_AsDouble(sna);
        if (sna_float == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        double ona_float = PyFloat_AsDouble(ona);
        if (ona_float == -1.0 && PyErr_Occurred()) {
            return -1;
        }
        if (npy_isnan(sna_float) && npy_isnan(ona_float)) {
            return 1;
        }
    }
    // could have two distinct instances that compare equal
    return PyObject_RichCompareBool(sna, ona, Py_EQ);
}

/*
 * This is used to determine the correct dtype to return when dealing
 * with a mix of different dtypes (for example when creating an array
 * from a list of scalars).
 */
static StringDTypeObject *
common_instance(StringDTypeObject *dtype1, StringDTypeObject *dtype2)
{
    int eq = _eq_comparison(dtype1->coerce, dtype2->coerce, dtype1->na_object,
                            dtype2->na_object);

    if (eq <= 0) {
        PyErr_SetString(
                PyExc_ValueError,
                "Cannot find common instance for unequal dtype instances");
        return NULL;
    }

    return (StringDTypeObject *)new_stringdtype_instance(dtype1->na_object,
                                                         dtype1->coerce);
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
// non-string or non-NA input.
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
    else {
        Py_INCREF(scalar);
    }

    return scalar;
}

static PyArray_Descr *
string_discover_descriptor_from_pyobject(PyTypeObject *NPY_UNUSED(cls),
                                         PyObject *obj)
{
    PyObject *val = get_value(obj, 1);
    if (val == NULL) {
        return NULL;
    }

    Py_DECREF(val);

    PyArray_Descr *ret = (PyArray_Descr *)new_stringdtype_instance(NULL, 1);

    return ret;
}

// Take a python object `obj` and insert it into the array of dtype `descr` at
// the position given by dataptr.
int
stringdtype_setitem(StringDTypeObject *descr, PyObject *obj, char **dataptr)
{
    npy_packed_static_string *sdata = (npy_packed_static_string *)dataptr;

    npy_string_allocator *allocator = _NpyString_acquire_allocator(descr);

    // borrow reference
    PyObject *na_object = descr->na_object;

    // setting NA *must* check pointer equality since NA types might not
    // allow equality
    if (na_object != NULL && obj == na_object) {
        if (_NpyString_pack_null(allocator, sdata) < 0) {
            PyErr_SetString(PyExc_MemoryError,
                            "Failed to pack null string during StringDType "
                            "setitem");
            goto fail;
        }
    }
    else {
        PyObject *val_obj = get_value(obj, descr->coerce);

        if (val_obj == NULL) {
            goto fail;
        }

        Py_ssize_t length = 0;
        const char *val = PyUnicode_AsUTF8AndSize(val_obj, &length);
        if (val == NULL) {
            Py_DECREF(val_obj);
            goto fail;
        }

        if (_NpyString_pack(allocator, sdata, val, length) < 0) {
            PyErr_SetString(PyExc_MemoryError,
                            "Failed to pack string during StringDType "
                            "setitem");
            Py_DECREF(val_obj);
            goto fail;
        }
        Py_DECREF(val_obj);
    }

    _NpyString_release_allocator(descr);

    return 0;

fail:
    _NpyString_release_allocator(descr);

    return -1;
}

static PyObject *
stringdtype_getitem(StringDTypeObject *descr, char **dataptr)
{
    PyObject *val_obj = NULL;
    npy_packed_static_string *psdata = (npy_packed_static_string *)dataptr;
    _npy_static_string sdata = {0, NULL};
    int hasnull = descr->na_object != NULL;
    npy_string_allocator *allocator = _NpyString_acquire_allocator(descr);
    int is_null = _NpyString_load(allocator, psdata, &sdata);

    if (is_null < 0) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to load string in StringDType getitem");
        goto fail;
    }
    else if (is_null == 1) {
        if (hasnull) {
            PyObject *na_object = descr->na_object;
            Py_INCREF(na_object);
            val_obj = na_object;
        }
        else {
            // cannot fail
            val_obj = PyUnicode_FromStringAndSize("", 0);
        }
    }
    else {
        val_obj = PyUnicode_FromStringAndSize(sdata.buf, sdata.size);
        if (val_obj == NULL) {
            goto fail;
        }
    }

    _NpyString_release_allocator(descr);

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

fail:

    _NpyString_release_allocator(descr);

    return NULL;
}

// PyArray_NonzeroFunc
// Unicode strings are nonzero if their length is nonzero.
npy_bool
nonzero(void *data, void *NPY_UNUSED(arr))
{
    return _NpyString_size((npy_packed_static_string *)data) != 0;
}

// Implementation of PyArray_CompareFunc.
// Compares unicode strings by their code points.
int
compare(void *a, void *b, void *arr)
{
    StringDTypeObject *descr = (StringDTypeObject *)PyArray_DESCR(arr);
    // ignore the allocator returned by this function
    // since _compare needs the descr anyway
    _NpyString_acquire_allocator(descr);
    int ret = _compare(a, b, descr, descr);
    _NpyString_release_allocator(descr);
    return ret;
}

int
_compare(void *a, void *b, StringDTypeObject *descr_a,
         StringDTypeObject *descr_b)
{
    npy_string_allocator *allocator_a = descr_a->allocator;
    npy_string_allocator *allocator_b = descr_b->allocator;
    // descr_a and descr_b are either the same object or objects
    // the are equal, so we can refer only to descr_a safely
    // this is enforced in the resolve_descriptors for comparisons
    int hasnull = descr_a->na_object != NULL;
    int has_string_na = descr_a->has_string_na;
    int has_nan_na = descr_a->has_nan_na;
    _npy_static_string *default_string = &descr_a->default_string;
    const npy_packed_static_string *ps_a = (npy_packed_static_string *)a;
    _npy_static_string s_a = {0, NULL};
    int a_is_null = _NpyString_load(allocator_a, ps_a, &s_a);
    const npy_packed_static_string *ps_b = (npy_packed_static_string *)b;
    _npy_static_string s_b = {0, NULL};
    int b_is_null = _NpyString_load(allocator_b, ps_b, &s_b);
    if (NPY_UNLIKELY(a_is_null == -1 || b_is_null == -1)) {
        char *msg = "Failed to load string in string comparison";
        if (hasnull && !(has_string_na && has_nan_na)) {
            // we hold the gil in this branch
            if (PyErr_Occurred()) {
                return 0;
            }
            PyErr_SetString(PyExc_MemoryError, msg);
        }
        else {
            // has a check for PyErr_Occurred so error
            // only gets set once
            gil_error(PyExc_MemoryError, msg);
        }
        return 0;
    }
    else if (NPY_UNLIKELY(a_is_null || b_is_null)) {
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
                s_a = *default_string;
            }
            if (b_is_null) {
                s_b = *default_string;
            }
        }
    }
    return _NpyString_cmp(&s_a, &s_b);
}

// PyArray_ArgFunc
// The max element is the one with the highest unicode code point.
int
argmax(char *data, npy_intp n, npy_intp *max_ind, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR(arr);
    npy_intp elsize = descr->elsize;
    *max_ind = 0;
    for (int i = 1; i < n; i++) {
        if (compare(data + i * elsize, data + (*max_ind) * elsize, arr) > 0) {
            *max_ind = i;
        }
    }
    return 0;
}

// PyArray_ArgFunc
// The min element is the one with the lowest unicode code point.
int
argmin(char *data, npy_intp n, npy_intp *min_ind, void *arr)
{
    PyArray_Descr *descr = PyArray_DESCR(arr);
    npy_intp elsize = descr->elsize;
    *min_ind = 0;
    for (int i = 1; i < n; i++) {
        if (compare(data + i * elsize, data + (*min_ind) * elsize, arr) < 0) {
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
                       const PyArray_Descr *descr, char *data, npy_intp size,
                       npy_intp stride, NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *sdescr = (StringDTypeObject *)descr;
    npy_string_allocator *allocator = _NpyString_acquire_allocator(sdescr);
    while (size--) {
        npy_packed_static_string *sdata = (npy_packed_static_string *)data;
        if (data != NULL && _NpyString_free(sdata, allocator) < 0) {
            gil_error(PyExc_MemoryError,
                      "String deallocation failed in clear loop");
            goto fail;
        }
        data += stride;
    }
    _NpyString_release_allocator(sdescr);
    return 0;

fail:
    _NpyString_release_allocator(sdescr);
    return -1;
}

static int
stringdtype_get_clear_loop(void *NPY_UNUSED(traverse_context),
                           PyArray_Descr *NPY_UNUSED(descr),
                           int NPY_UNUSED(aligned),
                           npy_intp NPY_UNUSED(fixed_stride),
                           PyArrayMethod_TraverseLoop **out_loop,
                           NpyAuxData **NPY_UNUSED(out_auxdata),
                           NPY_ARRAYMETHOD_FLAGS *flags)
{
    *flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    *out_loop = &stringdtype_clear_loop;
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

PyArray_Descr *
stringdtype_finalize_descr(PyArray_Descr *dtype)
{
    StringDTypeObject *sdtype = (StringDTypeObject *)dtype;
    if (sdtype->array_owned == 0) {
        sdtype->array_owned = 1;
        Py_INCREF(dtype);
        return dtype;
    }
    StringDTypeObject *ret = (StringDTypeObject *)new_stringdtype_instance(
            sdtype->na_object, sdtype->coerce);
    ret->array_owned = 1;
    return (PyArray_Descr *)ret;
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
        {NPY_DT_finalize_descr, &stringdtype_finalize_descr},
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

    return new_stringdtype_instance(na_object, coerce);
}

static void
stringdtype_dealloc(StringDTypeObject *self)
{
    Py_XDECREF(self->na_object);
    // this can be null if an error happens while initializing an instance
    if (self->allocator != NULL) {
        // can we assume the destructor for an instance will only get called
        // inside of one C thread?
        _NpyString_free_allocator(self->allocator);
        PyThread_free_lock(self->allocator_lock);
    }
    PyMem_RawFree((char *)self->na_name.buf);
    PyMem_RawFree((char *)self->default_string.buf);
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

    int eq = _eq_comparison(sself->coerce, sother->coerce, sself->na_object,
                            sother->na_object);

    if (eq == -1) {
        return NULL;
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
    if (!PyErr_Occurred()) {
        PyErr_SetString(type, msg);
    }
    PyGILState_Release(gstate);
}

int
free_and_copy(npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator,
              const npy_packed_static_string *in,
              npy_packed_static_string *out, const char *location)
{
    if (_NpyString_free(out, out_allocator) < 0) {
        char message[200];
        snprintf(message, sizeof(message), "Failed to deallocate string in %s",
                 location);
        gil_error(PyExc_MemoryError, message);
        return -1;
    }
    if (_NpyString_dup(in, out, in_allocator, out_allocator) < 0) {
        char message[200];
        snprintf(message, sizeof(message), "Failed to allocate string in %s",
                 location);
        gil_error(PyExc_MemoryError, message);
        return -1;
    }
    return 0;
}
