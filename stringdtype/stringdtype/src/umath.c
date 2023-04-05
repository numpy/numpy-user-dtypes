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
minmax_resolve_descriptors(struct PyArrayMethodObject_tag *NPY_UNUSED(method),
                           PyArray_DTypeMeta *NPY_UNUSED(dtypes[]),
                           PyArray_Descr *given_descrs[],
                           PyArray_Descr *loop_descrs[],
                           npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    StringDTypeObject *new = new_stringdtype_instance();
    if (new == NULL) {
        return -1;
    }
    loop_descrs[2] = (PyArray_Descr *)new;

    return NPY_NO_CASTING;
}

static int
maximum_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                     char *const data[], npy_intp const dimensions[],
                     npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    while (N--) {
        if (compare(in1, in2, NULL) > 0) {
            // Only copy *out* to *in1* if they point to different locations;
            // for *arr.max()* they point to the same address.
            if (in1 != out) {
                ssfree((ss *)out);
                if (ssdup((ss *)in1, (ss *)out) < 0) {
                    return -1;
                }
            }
        }
        else {
            ssfree((ss *)out);
            if (ssdup((ss *)in2, (ss *)out) < 0) {
                return -1;
            }
        }
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    return 0;
}

static int
minimum_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                     char *const data[], npy_intp const dimensions[],
                     npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    char *out = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    while (N--) {
        if (compare(in1, in2, NULL) < 0) {
            if (in1 != out) {
                ssfree((ss *)out);
                if (ssdup((ss *)in1, (ss *)out) < 0) {
                    return -1;
                }
            }
        }
        else {
            ssfree((ss *)out);
            if (ssdup((ss *)in2, (ss *)out) < 0) {
                return -1;
            }
        }
        in1 += in1_stride;
        in2 += in2_stride;
        out += out_stride;
    }

    return 0;
}

static int
string_equal_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                          char *const data[], npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in1 = data[0];
    char *in2 = data[1];
    npy_bool *out = (npy_bool *)data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    ss *s1 = NULL, *s2 = NULL;

    while (N--) {
        load_string(in1, &s1);
        load_string(in2, &s2);

        if (s1->len == s2->len && strncmp(s1->buf, s2->buf, s1->len) == 0) {
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
string_equal_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]), PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);  // cannot fail

    return NPY_NO_CASTING;
}

static int
string_isnan_strided_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                          char *const data[], npy_intp const dimensions[],
                          npy_intp const strides[],
                          NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    npy_bool *out = (npy_bool *)data[1];
    npy_intp out_stride = strides[1];

    while (N--) {
        // we could represent missing data with a null pointer, but
        // should isnan return True in that case?
        *out = (npy_bool)0;

        out += out_stride;
    }

    return 0;
}

static NPY_CASTING
string_isnan_resolve_descriptors(
        struct PyArrayMethodObject_tag *NPY_UNUSED(method),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[]), PyArray_Descr *given_descrs[],
        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);  // cannot fail

    return NPY_NO_CASTING;
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
init_ufunc(PyObject *numpy, const char *ufunc_name, PyArray_DTypeMeta **dtypes,
           resolve_descriptors_function *resolve_func,
           PyArrayMethod_StridedLoop *loop_func, const char *loop_name,
           int nin, int nout, NPY_CASTING casting, NPY_ARRAYMETHOD_FLAGS flags)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyType_Slot slots[] = {{NPY_METH_resolve_descriptors, resolve_func},
                           {NPY_METH_strided_loop, loop_func},
                           {0, NULL}};

    PyArrayMethod_Spec spec = {
            .name = loop_name,
            .nin = nin,
            .nout = nout,
            .casting = casting,
            .flags = flags,
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &spec) < 0) {
        Py_DECREF(ufunc);
        return -1;
    }

    Py_DECREF(ufunc);
    return 0;
}

int
add_promoter(PyObject *numpy, const char *ufunc_name,
             PyArray_DTypeMeta **dtypes)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyObject *DType_tuple = PyTuple_Pack(3, dtypes[0], dtypes[1], dtypes[2]);
    if (DType_tuple == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }

    PyObject *promoter_capsule = PyCapsule_New((void *)&default_ufunc_promoter,
                                               "numpy._ufunc_promoter", NULL);

    if (PyUFunc_AddPromoter(ufunc, DType_tuple, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DType_tuple);
        Py_DECREF(ufunc);
        return -1;
    }

    Py_DECREF(promoter_capsule);
    Py_DECREF(DType_tuple);
    Py_DECREF(ufunc);

    return 0;
}

int
init_ufuncs(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *eq_dtypes[] = {&StringDType, &StringDType,
                                      &PyArray_BoolDType};

    if (init_ufunc(numpy, "equal", eq_dtypes,
                   &string_equal_resolve_descriptors,
                   &string_equal_strided_loop, "string_equal", 2, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    PyArray_DTypeMeta *promoter_dtypes[2][3] = {
            {&StringDType, &PyArray_UnicodeDType, &PyArray_BoolDType},
            {&PyArray_UnicodeDType, &StringDType, &PyArray_BoolDType},
    };

    if (add_promoter(numpy, "equal", promoter_dtypes[0]) < 0) {
        goto error;
    }

    if (add_promoter(numpy, "equal", promoter_dtypes[1]) < 0) {
        goto error;
    }

    PyArray_DTypeMeta *isnan_dtypes[] = {&StringDType, &PyArray_BoolDType};

    if (init_ufunc(numpy, "isnan", isnan_dtypes,
                   &string_isnan_resolve_descriptors,
                   &string_isnan_strided_loop, "string_isnan", 1, 1,
                   NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    PyArray_DTypeMeta *minmax_dtypes[] = {&StringDType, &StringDType,
                                          &StringDType};
    if (init_ufunc(numpy, "maximum", minmax_dtypes,
                   &minmax_resolve_descriptors, &maximum_strided_loop,
                   "string_maximum", 2, 1, NPY_NO_CASTING, 0) < 0) {
        goto error;
    }
    if (init_ufunc(numpy, "minimum", minmax_dtypes,
                   &minmax_resolve_descriptors, &minimum_strided_loop,
                   "string_minimum", 2, 1, NPY_NO_CASTING, 0) < 0) {
        goto error;
    }

    Py_DECREF(numpy);
    return 0;

error:
    Py_DECREF(numpy);
    return -1;
}
