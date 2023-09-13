#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "static_string.h"
#include "umath.h"

static PyObject *
_memory_usage(PyObject *NPY_UNUSED(self), PyObject *obj)
{
    if (!PyArray_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "can only be called with ndarray object");
        return NULL;
    }

    PyArrayObject *arr = (PyArrayObject *)obj;

    PyArray_Descr *descr = PyArray_DESCR(arr);
    PyArray_DTypeMeta *dtype = NPY_DTYPE(descr);

    if (dtype != (PyArray_DTypeMeta *)&StringDType) {
        PyErr_SetString(PyExc_TypeError,
                        "can only be called with a StringDType array");
        return NULL;
    }

    NpyIter *iter = NpyIter_New(
            arr, NPY_ITER_READONLY | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
            NPY_KEEPORDER, NPY_NO_CASTING, NULL);

    if (iter == NULL) {
        return NULL;
    }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);

    if (iternext == NULL) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    char **dataptr = NpyIter_GetDataPtrArray(iter);
    npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
    npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    // initialize with the size of the internal buffer
    size_t memory_usage = PyArray_NBYTES(arr);

    do {
        char *in = dataptr[0];
        npy_intp stride = *strideptr;
        npy_intp count = *innersizeptr;

        while (count--) {
            memory_usage += ((npy_static_string *)in)->size;
            in += stride;
        }

    } while (iternext(iter));

    NpyIter_Deallocate(iter);

    PyObject *ret = PyLong_FromSize_t(memory_usage);

    return ret;
}

static PyMethodDef string_methods[] = {
        {"_memory_usage", _memory_usage, METH_O,
         "get memory usage for an array"},
        {NULL},
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "stringdtype_main",
        .m_size = -1,
        .m_methods = string_methods,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__main(void)
{
    import_array();

    if (import_experimental_dtype_api(13) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    PyObject *mod = PyImport_ImportModule("stringdtype");
    if (mod == NULL) {
        goto error;
    }

    StringScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "StringScalar");

    if (StringScalar_Type == NULL) {
        goto error;
    }

    Py_DECREF(mod);

    if (init_string_dtype() < 0) {
        goto error;
    }

    Py_INCREF((PyObject *)&StringDType);
    if (PyModule_AddObject(m, "StringDType", (PyObject *)&StringDType) < 0) {
        Py_DECREF((PyObject *)&StringDType);
        goto error;
    }

    if (init_ufuncs() < 0) {
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
