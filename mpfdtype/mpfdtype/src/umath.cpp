
#include "scalar.h"
#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL MPFDType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
    #include <Python.h>

    #include "numpy/arrayobject.h"
    #include "numpy/ndarraytypes.h"
    #include "numpy/ufuncobject.h"

    #include "numpy/dtype_api.h"
}

#include <algorithm>

#include "dtype.h"
#include "umath.h"

#include "ops.hpp"


template <unary_op_def unary_op>
int
generic_unary_op_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    mpfr_prec_t prec1 = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec2 = ((MPFDTypeObject *)context->descriptors[1])->precision;

    mpfr_ptr in, out;

    while (N--) {
        mpf_load(in, in_ptr, prec1);
        mpf_load(out, out_ptr, prec2);

        // TODO: Should maybe do something with the result?
        unary_op(in, out);
        mpf_store(out_ptr, out);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}


/*
 * General promotion for binary ops.  We always use the bigger precision
 * for the result.
 * This effectively means the result has full precisio, which is not normally
 * guaranteed.
 */
static NPY_CASTING
unary_op_resolve_descriptors(PyObject *self,
        PyArray_DTypeMeta *dtypes[], MPFDTypeObject *given_descrs[],
        MPFDTypeObject *loop_descrs[], npy_intp *unused)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
        return NPY_NO_CASTING;
    }
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    if (given_descrs[1]->precision == loop_descrs[2]->precision) {
        return NPY_NO_CASTING;
    }
    else if (given_descrs[1]->precision < loop_descrs[2]->precision) {
        return NPY_SAME_KIND_CASTING;
    }
    else {
        return NPY_SAFE_CASTING;
    }
}


template <unary_op_def unary_op>
int
create_unary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    /*
     * The initializing "wrap up" code from the slides (plus one error check)
     */
    PyArray_DTypeMeta *dtypes[2] = {
       &MPFDType, &MPFDType};

    PyType_Slot slots[] = {
       {NPY_METH_resolve_descriptors,
            (void *)&unary_op_resolve_descriptors},
       {NPY_METH_strided_loop,
            (void *)&generic_unary_op_strided_loop<unary_op>},
       {0, NULL}
    };

    PyArrayMethod_Spec Spec = {
        .name = "mpf_unary_op",
        .nin = 1,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = (NPY_ARRAYMETHOD_FLAGS)0,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    return 0;
}


int init_unary_ops(PyObject *numpy)
{
    if (create_unary_ufunc<negative>(numpy, "negative") < 0) {
        return -1;
    }
    if (create_unary_ufunc<positive>(numpy, "positive") < 0) {
        return -1;
    }
    if (create_unary_ufunc<absolute>(numpy, "absolute") < 0) {
        return -1;
    }
    if (create_unary_ufunc<rint>(numpy, "rint") < 0) {
        return -1;
    }
    if (create_unary_ufunc<trunc>(numpy, "trunc") < 0) {
        return -1;
    }
    if (create_unary_ufunc<floor>(numpy, "floor") < 0) {
        return -1;
    }
    if (create_unary_ufunc<ceil>(numpy, "ceil") < 0) {
        return -1;
    }
    if (create_unary_ufunc<sqrt>(numpy, "sqrt") < 0) {
        return -1;
    }
    if (create_unary_ufunc<square>(numpy, "square") < 0) {
        return -1;
    }
    if (create_unary_ufunc<log>(numpy, "log") < 0) {
        return -1;
    }
    if (create_unary_ufunc<log2>(numpy, "log2") < 0) {
        return -1;
    }
    if (create_unary_ufunc<log10>(numpy, "log10") < 0) {
        return -1;
    }
    if (create_unary_ufunc<log1p>(numpy, "log1p") < 0) {
        return -1;
    }
    if (create_unary_ufunc<exp>(numpy, "exp") < 0) {
        return -1;
    }
    if (create_unary_ufunc<exp>(numpy, "exp2") < 0) {
        return -1;
    }
    if (create_unary_ufunc<expm1>(numpy, "expm1") < 0) {
        return -1;
    }
    if (create_unary_ufunc<arcsin>(numpy, "arcsin") < 0) {
        return -1;
    }
    if (create_unary_ufunc<arccos>(numpy, "arccos") < 0) {
        return -1;
    }
    if (create_unary_ufunc<arctan>(numpy, "arctan") < 0) {
        return -1;
    }
    return 0;
}


/*
 * Binary functions
 */

template <binop_def binop>
int
generic_binop_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    mpfr_prec_t prec1 = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec2 = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec3 = ((MPFDTypeObject *)context->descriptors[0])->precision;

    mpfr_ptr in1, in2, out;

    while (N--) {
        mpf_load(in1, in1_ptr, prec1);
        mpf_load(in2, in2_ptr, prec2);
        mpf_load(out, out_ptr, prec3);

        // TODO: Should maybe do something with the result?
        binop(out, in1, in2);

        mpf_store(out_ptr, out);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}


/*
 * General promotion for binary ops.  We always use the bigger precision
 * for the result.
 * This effectively means the result has full precisio, which is not normally
 * guaranteed.
 */
static NPY_CASTING
binary_op_resolve_descriptors(PyObject *self,
        PyArray_DTypeMeta *dtypes[], MPFDTypeObject *given_descrs[],
        MPFDTypeObject *loop_descrs[], npy_intp *unused)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    mpfr_prec_t res_prec = std::max(
            given_descrs[0]->precision, given_descrs[1]->precision);

    if (given_descrs[2] == NULL) {
        if (given_descrs[0]->precision >= given_descrs[1]->precision) {
            Py_INCREF(given_descrs[0]);
            loop_descrs[2] = given_descrs[0];
        }
        else {
            Py_INCREF(given_descrs[1]);
            loop_descrs[2] = given_descrs[1];
        }
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    if (res_prec == loop_descrs[2]->precision) {
        return NPY_NO_CASTING;
    }
    else if (res_prec < loop_descrs[2]->precision) {
        return NPY_SAME_KIND_CASTING;
    }
    else {
        return NPY_SAFE_CASTING;
    }
}


/*
 * Copied from NumPy, because NumPy doesn't always use it :)
 */
static int
default_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    /* If nin < 2 promotion is a no-op, so it should not be registered */
    assert(ufunc->nin > 1);
    if (op_dtypes[0] == NULL) {
        assert(ufunc->nin == 2 && ufunc->nout == 1);  /* must be reduction */
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
                Py_CLEAR(common);  /* Not homogeneous, unset common */
                break;
            }
        }
    }
    /* Otherwise, use the common DType of all input operands */
    if (common == NULL) {
        common = PyArray_PromoteDTypeSequence(ufunc->nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();  /* Do not propagate normal promotion errors */
            }
            return -1;
        }
    }

    for (int i = 0; i < ufunc->nargs; i++) {
        PyArray_DTypeMeta *tmp = common;
        if (signature[i]) {
            tmp = signature[i];  /* never replace a fixed one. */
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


template <binop_def binop>
int
create_binary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    /*
     * The initializing "wrap up" code from the slides (plus one error check)
     */
    PyArray_DTypeMeta *dtypes[3] = {
       &MPFDType, &MPFDType, &MPFDType};

    PyType_Slot slots[] = {
       {NPY_METH_resolve_descriptors,
            (void *)&binary_op_resolve_descriptors},
       {NPY_METH_strided_loop,
            (void *)&generic_binop_strided_loop<binop>},
       {0, NULL}
    };

    PyArrayMethod_Spec Spec = {
        .name = "mpf_binop",
        .nin = 2,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = (NPY_ARRAYMETHOD_FLAGS)0,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    /*
     * This might interfere with NumPy at this time.
     */
    PyObject *promoter_capsule = PyCapsule_New(
            (void *)&default_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(
            3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArrayDescr_Type);
    if (DTypes  == 0) {
        Py_DECREF(promoter_capsule);
        return -1;
    }

    if (PyUFunc_AddPromoter(ufunc, DTypes, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DTypes);
        return -1;
    }
    Py_DECREF(promoter_capsule);
    Py_DECREF(DTypes);

    return 0;
}




int init_binary_ops(PyObject *numpy)
{
    if (create_binary_ufunc<add>(numpy, "add") < 0) {
        return -1;
    }
    if (create_binary_ufunc<sub>(numpy, "subtract") < 0) {
        return -1;
    }
    if (create_binary_ufunc<mul>(numpy, "multiply") < 0) {
        return -1;
    }
    if (create_binary_ufunc<div>(numpy, "divide") < 0) {
        return -1;
    }
    if (create_binary_ufunc<hypot>(numpy, "hypot") < 0) {
        return -1;
    }
    if (create_binary_ufunc<pow>(numpy, "power") < 0) {
        return -1;
    }
    if (create_binary_ufunc<arctan2>(numpy, "arctan2") < 0) {
        return -1;
    }
    if (create_binary_ufunc<nextafter>(numpy, "nextafter") < 0) {
        return -1;
    }
    return 0;
}


/*
 * Comparisons
 */

template <cmp_def comp>
int
generic_comp_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    mpfr_prec_t prec1 = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec2 = ((MPFDTypeObject *)context->descriptors[0])->precision;

    mpfr_ptr in1, in2;

    while (N--) {
        mpf_load(in1, in1_ptr, prec1);
        mpf_load(in2, in2_ptr, prec2);

        // TODO: Should maybe do something with the result?
        *((npy_bool *)out_ptr) = comp(in1, in2);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}


/* In theory simpler, but reuse default one (except for forcing object) */
NPY_NO_EXPORT int
comparison_ufunc_promoter(PyUFuncObject *ufunc,
        PyArray_DTypeMeta *op_dtypes[], PyArray_DTypeMeta *signature[],
        PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_DTypeMeta *new_signature[NPY_MAXARGS];

    memcpy(new_signature, signature, 3 * sizeof(PyArray_DTypeMeta *));
    new_signature[2] = NULL;
    int res = default_ufunc_promoter(ufunc, op_dtypes, new_signature, new_op_dtypes);
    if (res < 0) {
        return -1;
    }
    Py_XSETREF(new_op_dtypes[2], &PyArray_BoolDType);
    return 0;
}


template <cmp_def comp>
int
create_comparison_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    /*
     * The initializing "wrap up" code from the slides (plus one error check)
     */
    PyArray_DTypeMeta *dtypes[3] = {
       &MPFDType, &MPFDType, &PyArray_BoolDType};

    PyType_Slot slots[] = {
       {NPY_METH_strided_loop,
            (void *)&generic_comp_strided_loop<comp>},
       {0, NULL}
    };

    PyArrayMethod_Spec Spec = {
        .name = "mpf_comp",
        .nin = 2,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = (NPY_ARRAYMETHOD_FLAGS)0,
        .dtypes = dtypes,
        .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    /*
     * This might interfere with NumPy at this time.
     */
    PyObject *promoter_capsule = PyCapsule_New(
            (void *)&comparison_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(
            3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArray_BoolDType);
    if (DTypes  == 0) {
        Py_DECREF(promoter_capsule);
        return -1;
    }

    if (PyUFunc_AddPromoter(ufunc, DTypes, promoter_capsule) < 0) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(DTypes);
        return -1;
    }
    Py_DECREF(promoter_capsule);
    Py_DECREF(DTypes);

    return 0;
}


int
init_comps(PyObject *numpy)
{
    if (create_comparison_ufunc<mpf_equal>(numpy, "equal") < 0) {
        return -1;
    }
    if (create_comparison_ufunc<mpf_notequal>(numpy, "not_equal") < 0) {
        return -1;
    }
    if (create_comparison_ufunc<mpf_less>(numpy, "less") < 0) {
        return -1;
    }
    if (create_comparison_ufunc<mpf_lessequal>(numpy, "less_equal") < 0) {
        return -1;
    }
    if (create_comparison_ufunc<mpf_greater>(numpy, "greater") < 0) {
        return -1;
    }
    if (create_comparison_ufunc<mpf_greaterequal>(numpy, "greater_equal") < 0) {
        return -1;
    }

    return 0;
}


/*
 * Function that adds our multiply loop to NumPy's multiply ufunc.
 */
int
init_mpf_umath(void)
{
    /*
     * Get the multiply ufunc:
     */
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (numpy == NULL) {
        return -1;
    }

    if (init_unary_ops(numpy) < 0) {
        goto err;
    }
    if (init_binary_ops(numpy) < 0) {
        goto err;
    }
    if (init_comps(numpy) < 0) {
        goto err;
    }

    Py_DECREF(numpy);
    return 0;

  err:
    Py_DECREF(numpy);
    return -1;
}

