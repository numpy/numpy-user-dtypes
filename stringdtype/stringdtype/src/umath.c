#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "dtype.h"
#include "static_string.h"
#include "string.h"
#include "umath.h"

static int
string_equal_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    ss **in1 = (ss **)data[0];
    ss **in2 = (ss **)data[1];
    npy_bool *out = (npy_bool *)data[2];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp in1_stride = strides[0] / context->descriptors[0]->elsize;
    npy_intp in2_stride = strides[1] / context->descriptors[1]->elsize;
    npy_intp out_stride = strides[2];

    while (N--) {
        size_t len1 = (*in1)->len;
        size_t len2 = (*in2)->len;
        size_t maxlen;

        if (len1 > len2) {
            maxlen = len1;
        }
        else {
            maxlen = len2;
        }

        if (strncmp((*in1)->buf, (*in2)->buf, maxlen) == 0) {
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

/*
 * Copied from NumPy, because NumPy doesn't always use it :)
 */
static int
default_ufunc_promoter(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                       PyArray_DTypeMeta *signature[],
                       PyArray_DTypeMeta *new_op_dtypes[])
{
    /* If nin < 2 promotion is a no-op, so it should not be registered */
    assert(ufunc->nin > 1);
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1); /* must be reduction */
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[0] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[1] = op_dtypes[1];
        Py_INCREF(op_dtypes[1]);
        new_op_dtypes[2] = op_dtypes[1];
        return 0;
    }
    PyArray_DTypeMeta *common = NULL;
    /*
     * If a signature is used and homogeneous in its outputs use that
     * (Could/should likely be rather applied to inputs also, although outs
     * only could have some advantage and input dtypes are rarely enforced.)
     */
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        if (signature[i] != NULL) {
            if (common == NULL) {
                Py_INCREF(signature[i]);
                common = signature[i];
            }
            else if (common != signature[i]) {
                Py_CLEAR(common); /* Not homogeneous, unset common */
                break;
            }
        }
    }
    /* Otherwise, use the common DType of all input operands */
    if (common == NULL) {
        common = PyArray_PromoteDTypeSequence(ufunc->nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear(); /* Do not propagate normal promotion errors */
            }
            return -1;
        }
    }

    for (int i = 0; i < ufunc->nargs; i++) {
        PyArray_DTypeMeta *tmp = common;
        if (signature[i]) {
            tmp = signature[i]; /* never replace a fixed one. */
        }
        Py_INCREF(tmp);
        new_op_dtypes[i] = tmp;
    }
    for (int i = ufunc->nin; i < ufunc->nargs; i++) {
        Py_XINCREF(op_dtypes[i]);
        new_op_dtypes[i] = op_dtypes[i];
    }

    Py_DECREF(common);
    return 0;
}

int
init_equal_ufunc(PyObject *numpy)
{
    PyObject *equal = PyObject_GetAttrString(numpy, "equal");
    if (equal == NULL) {
        return -1;
    }

    /*
     *  Initialize spec for equality
     */
    PyArray_DTypeMeta *eq_dtypes[3] = {&StringDType, &StringDType,
                                       &PyArray_BoolDType};

    static PyType_Slot eq_slots[] = {
            {NPY_METH_resolve_descriptors, &string_equal_resolve_descriptors},
            {NPY_METH_strided_loop, &string_equal_strided_loop},
            {0, NULL}};

    PyArrayMethod_Spec EqualSpec = {
            .name = "string_equal",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = 0,
            .dtypes = eq_dtypes,
            .slots = eq_slots,
    };

    if (PyUFunc_AddLoopFromSpec(equal, &EqualSpec) < 0) {
        Py_DECREF(equal);
        return -1;
    }

    /*
     *  Add promoter to ufunc, ensures operations that mix StringDType and
     *  UnicodeDType cast the unicode argument to string.
     */

    PyObject *DTypes[] = {
            PyTuple_Pack(3, &StringDType, &PyArray_UnicodeDType,
                         &PyArray_BoolDType),
            PyTuple_Pack(3, &PyArray_UnicodeDType, &StringDType,
                         &PyArray_BoolDType),
    };

    if ((DTypes[0] == NULL) || (DTypes[1] == NULL)) {
        Py_DECREF(equal);
        return -1;
    }

    PyObject *promoter_capsule = PyCapsule_New((void *)&default_ufunc_promoter,
                                               "numpy._ufunc_promoter", NULL);

    for (int i = 0; i < 2; i++) {
        if (PyUFunc_AddPromoter(equal, DTypes[i], promoter_capsule) < 0) {
            Py_DECREF(promoter_capsule);
            Py_DECREF(DTypes[0]);
            Py_DECREF(DTypes[1]);
            Py_DECREF(equal);
            return -1;
        }
    }

    Py_DECREF(promoter_capsule);
    Py_DECREF(DTypes[0]);
    Py_DECREF(DTypes[1]);
    Py_DECREF(equal);

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
