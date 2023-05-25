#include <Python.h>

#include "umath.h"

#include "dtype.h"

static int
translate_given_descrs(int nin, int nout,
                       PyArray_DTypeMeta *NPY_UNUSED(wrapped_dtypes[]),
                       PyArray_Descr *given_descrs[],
                       PyArray_Descr *new_descrs[])
{
    for (int i = 0; i < nin + nout; i++) {
        if (given_descrs[i] == NULL) {
            new_descrs[i] = NULL;
        }
        else {
            if (NPY_DTYPE(given_descrs[i]) == &PyArray_BoolDType) {
                new_descrs[i] = PyArray_DescrFromType(NPY_BOOL);
            }
            else {
                new_descrs[i] = PyArray_DescrFromType(NPY_DOUBLE);
            }
        }
    }
    return 0;
}

static int
translate_loop_descrs(int nin, int NPY_UNUSED(nout),
                      PyArray_DTypeMeta *NPY_UNUSED(new_dtypes[]),
                      PyArray_Descr *given_descrs[],
                      PyArray_Descr *original_descrs[],
                      PyArray_Descr *loop_descrs[])
{
    if (nin == 2) {
        loop_descrs[0] =
                common_instance((MetadataDTypeObject *)given_descrs[0],
                                (MetadataDTypeObject *)given_descrs[1]);
        if (loop_descrs[0] == NULL) {
            return -1;
        }
        Py_INCREF(loop_descrs[0]);
        loop_descrs[1] = loop_descrs[0];
        Py_INCREF(loop_descrs[1]);
        if (NPY_DTYPE(original_descrs[2]) == &PyArray_BoolDType) {
            loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);
        }
        else {
            loop_descrs[2] = loop_descrs[0];
        }
        Py_INCREF(loop_descrs[2]);
    }
    else if (nin == 1) {
        loop_descrs[0] = given_descrs[0];
        Py_INCREF(loop_descrs[0]);
        if (NPY_DTYPE(original_descrs[1]) == &PyArray_BoolDType) {
            loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);
        }
        else {
            loop_descrs[1] = loop_descrs[0];
        }
        Py_INCREF(loop_descrs[1]);
    }
    else {
        return -1;
    }
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

    return ufunc;
}

static int
add_wrapping_loop(const char *ufunc_name, PyArray_DTypeMeta **dtypes,
                  PyArray_DTypeMeta **wrapped_dtypes)
{
    PyObject *ufunc = get_ufunc(ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }
    int res = PyUFunc_AddWrappingLoop(ufunc, dtypes, wrapped_dtypes,
                                      &translate_given_descrs,
                                      &translate_loop_descrs);
    return res;
}

int
init_ufuncs(PyArray_DTypeMeta *MetadataDType)
{
    PyArray_DTypeMeta *binary_orig_dtypes[3] = {MetadataDType, MetadataDType,
                                                MetadataDType};
    PyArray_DTypeMeta *binary_wrapped_dtypes[3] = {
            &PyArray_DoubleDType, &PyArray_DoubleDType, &PyArray_DoubleDType};
    if (add_wrapping_loop("multiply", binary_orig_dtypes,
                          binary_wrapped_dtypes) == -1) {
        goto error;
    }

    PyArray_DTypeMeta *unary_boolean_dtypes[2] = {MetadataDType,
                                                  &PyArray_BoolDType};
    PyArray_DTypeMeta *unary_boolean_wrapped_dtypes[2] = {&PyArray_DoubleDType,
                                                          &PyArray_BoolDType};
    if (add_wrapping_loop("isnan", unary_boolean_dtypes,
                          unary_boolean_wrapped_dtypes) == -1) {
        goto error;
    }

    return 0;
error:
    return -1;
}
