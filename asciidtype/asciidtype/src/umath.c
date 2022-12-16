#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL asciidtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "dtype.h"
#include "string.h"
#include "umath.h"

static int
ascii_add_strided_loop(PyArrayMethod_Context *context, char *const data[],
                       npy_intp const dimensions[], npy_intp const strides[],
                       NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in1_size = ((ASCIIDTypeObject *)descrs[0])->size;
    long in2_size = ((ASCIIDTypeObject *)descrs[1])->size;

    npy_intp N = dimensions[0];
    char *in1 = data[0], *in2 = data[1], *out = data[2];
    npy_intp in1_stride = strides[0], in2_stride = strides[1],
             out_stride = strides[2];

    while (N--) {
        strncpy(out, in1, in1_size);
        strncpy(out + in1_size, in2, in2_size);
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    return 0;
}

static NPY_CASTING
ascii_add_resolve_descriptors(PyObject *NPY_UNUSED(self),
                              PyArray_DTypeMeta *dtypes[],
                              PyArray_Descr *given_descrs[],
                              PyArray_Descr *loop_descrs[],
                              npy_intp *NPY_UNUSED(view_offset))
{
    long op1_size = ((ASCIIDTypeObject *)given_descrs[0])->size;
    long op2_size = ((ASCIIDTypeObject *)given_descrs[1])->size;
    long out_size = op1_size + op2_size;

    /* the input descriptors can be used as-is */
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    /* create new DType instance to hold the output */
    loop_descrs[2] = (PyArray_Descr *)new_asciidtype_instance(out_size);
    if (loop_descrs[2] == NULL) {
        return -1;
    }

    return NPY_SAFE_CASTING;
}

int
init_add_ufunc(PyObject *numpy)
{
    PyObject *add = PyObject_GetAttrString(numpy, "add");
    if (add == NULL) {
        return -1;
    }

    /*
     * Initialize spec for addition
     */
    static PyArray_DTypeMeta *add_dtypes[3] = {&ASCIIDType, &ASCIIDType,
                                               &ASCIIDType};

    static PyType_Slot add_slots[] = {
            {NPY_METH_resolve_descriptors, &ascii_add_resolve_descriptors},
            {NPY_METH_strided_loop, &ascii_add_strided_loop},
            {0, NULL}};

    PyArrayMethod_Spec AddSpec = {
            .name = "ascii_add",
            .nin = 2,
            .nout = 1,
            .dtypes = add_dtypes,
            .slots = add_slots,
            .flags = 0,
            .casting = NPY_SAFE_CASTING,
    };

    /* register ufunc */
    if (PyUFunc_AddLoopFromSpec(add, &AddSpec) < 0) {
        Py_DECREF(add);
        return -1;
    }
    Py_DECREF(add);
    return 0;
}

static int
ascii_equal_strided_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[],
                         NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in1_size = ((ASCIIDTypeObject *)descrs[0])->size;
    long in2_size = ((ASCIIDTypeObject *)descrs[1])->size;

    npy_intp N = dimensions[0];
    char *in1 = data[0], *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0], in2_stride = strides[1],
             out_stride = strides[2];

    while (N--) {
        *out = (npy_bool)1;
        char *_in1 = in1;
        char *_in2 = in2;
        npy_bool *_out = out;
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
        if (in1_size > in2_size) {
            if (_in1[in2_size] != '\0') {
                *_out = (npy_bool)0;
                continue;
            }
            if (strncmp(_in1, _in2, in2_size) != 0) {
                *_out = (npy_bool)0;
            }
        }
        else {
            if (in2_size > in1_size) {
                if (_in2[in1_size] != '\0') {
                    *_out = (npy_bool)0;
                    continue;
                }
            }
            if (strncmp(_in1, _in2, in1_size) != 0) {
                *_out = (npy_bool)0;
            }
        }
    }

    return 0;
}

static NPY_CASTING
ascii_equal_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                PyArray_DTypeMeta *dtypes[],
                                PyArray_Descr *given_descrs[],
                                PyArray_Descr *loop_descrs[],
                                npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);  // cannot fail

    return NPY_SAFE_CASTING;
}

static char *equal_name = "ascii_equal";

int
init_equal_ufunc(PyObject *numpy)
{
    PyObject *equal = PyObject_GetAttrString(numpy, "equal");
    if (equal == NULL) {
        return -1;
    }

    /*
     * Initialize spec for equality
     */
    PyArray_DTypeMeta **eq_dtypes = malloc(3 * sizeof(PyArray_DTypeMeta *));
    eq_dtypes[0] = &ASCIIDType;
    eq_dtypes[1] = &ASCIIDType;
    eq_dtypes[2] = &PyArray_BoolDType;

    static PyType_Slot eq_slots[] = {
            {NPY_METH_resolve_descriptors, &ascii_equal_resolve_descriptors},
            {NPY_METH_strided_loop, &ascii_equal_strided_loop},
            {0, NULL}};

    PyArrayMethod_Spec *EqualSpec = malloc(sizeof(PyArrayMethod_Spec));

    EqualSpec->name = equal_name;
    EqualSpec->nin = 2;
    EqualSpec->nout = 1;
    EqualSpec->casting = NPY_SAFE_CASTING;
    EqualSpec->flags = 0;
    EqualSpec->dtypes = eq_dtypes;
    EqualSpec->slots = eq_slots;

    if (PyUFunc_AddLoopFromSpec(equal, EqualSpec) < 0) {
        Py_DECREF(equal);
        free(eq_dtypes);
        free(EqualSpec);
        return -1;
    }

    Py_DECREF(equal);
    free(eq_dtypes);
    free(EqualSpec);
    return 0;
}

int
init_ufuncs(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    if (init_add_ufunc(numpy) < 0) {
        goto error;
    }

    if (init_equal_ufunc(numpy) < 0) {
        goto error;
    }

    Py_DECREF(numpy);
    return 0;

error:
    Py_DECREF(numpy);
    return -1;
}
