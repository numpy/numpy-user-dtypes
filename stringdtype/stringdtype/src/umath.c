#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
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
string_equal_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char **in1 = (char **)data[0];
    char **in2 = (char **)data[1];
    npy_bool *out = (npy_bool *)data[2];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp in1_stride = strides[0] / context->descriptors[0]->elsize;
    npy_intp in2_stride = strides[1] / context->descriptors[1]->elsize;
    npy_intp out_stride = strides[2];

    while (N--) {
        if (strcmp(*in1, *in2) == 0) {
            *out = (npy_bool)1;
        }
        else {
            *out = (npy_bool)0;
        }
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    return 0;
}

static NPY_CASTING
string_equal_resolve_descriptors(PyObject *NPY_UNUSED(self),
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

static char *equal_name = "string_equal";

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
    eq_dtypes[0] = &StringDType;
    eq_dtypes[1] = &StringDType;
    eq_dtypes[2] = &PyArray_BoolDType;

    static PyType_Slot eq_slots[] = {
            {NPY_METH_resolve_descriptors, &string_equal_resolve_descriptors},
            {NPY_METH_strided_loop, &string_equal_strided_loop},
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

    if (init_equal_ufunc(numpy) < 0) {
        goto error;
    }

    Py_DECREF(numpy);
    return 0;

error:
    Py_DECREF(numpy);
    return -1;
}
