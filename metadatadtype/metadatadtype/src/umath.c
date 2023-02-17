#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "dtype.h"
#include "umath.h"

static int
translate_given_descrs_to_double(
        int nin, int nout, PyArray_DTypeMeta *NPY_UNUSED(wrapped_dtypes[]),
        PyArray_Descr *given_descrs[], PyArray_Descr *new_descrs[])
{
    assert(nin == 2 && nout == 1);
    for (int i = 0; i < 3; i++) {
        if (given_descrs[i] == NULL) {
            new_descrs[i] = NULL;
        }
        else {
            new_descrs[i] = PyArray_DescrFromType(NPY_DOUBLE);
        }
    }
    return 0;
}

static int
translate_loop_descrs(int nin, int nout,
                      PyArray_DTypeMeta *NPY_UNUSED(new_dtypes[]),
                      PyArray_Descr *given_descrs[],
                      PyArray_Descr *NPY_UNUSED(original_descrs[]),
                      PyArray_Descr *loop_descrs[])
{
    assert(nin == 2 && nout == 1);
    loop_descrs[0] = common_instance((MetadataDTypeObject *)given_descrs[0],
                                     (MetadataDTypeObject *)given_descrs[1]);
    if (loop_descrs[0] == NULL) {
        return -1;
    }
    Py_INCREF(loop_descrs[0]);
    loop_descrs[1] = loop_descrs[0];
    Py_INCREF(loop_descrs[0]);
    loop_descrs[2] = loop_descrs[0];
    return 0;
}

static PyObject *
get_ufunc(const char *ufunc_name)
{
    PyObject *mod = PyImport_ImportModule("numpy");
    if (mod == NULL) {
        return NULL;
    }
    PyObject *ufunc = PyObject_GetAttrString(mod, ufunc_name);
    Py_DECREF(mod);
    if (ufunc == NULL) {
        return NULL;
    }
    return ufunc;
}

static int
add_wrapping_loop(const char *ufunc_name, PyArray_DTypeMeta *dtypes[3])
{
    PyObject *ufunc = get_ufunc(ufunc_name);
    PyArray_DTypeMeta *wrapped_dtypes[3] = {
            &PyArray_DoubleDType, &PyArray_DoubleDType, &PyArray_DoubleDType};
    return PyUFunc_AddWrappingLoop(ufunc, dtypes, wrapped_dtypes,
                                   &translate_given_descrs_to_double,
                                   &translate_loop_descrs);
}

int
init_ufuncs(void)
{
    PyArray_DTypeMeta *dtypes[3] = {&MetadataDType, &MetadataDType,
                                    &MetadataDType};
    if (add_wrapping_loop("multiply", dtypes) == -1) {
        goto error;
    }

    return 0;
error:
    return -1;
}
