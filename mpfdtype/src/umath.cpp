
#include "scalar.h"
#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY

extern "C" {
    #include <Python.h>

    #include "numpy/arrayobject.h"
    #include "numpy/ndarraytypes.h"
    #include "numpy/ufuncobject.h"

    #include "numpy/experimental_dtype_api.h"
}

#include <algorithm>

#include "dtype.h"
#include "umath.h"


typedef int unary_op_def(mpfr_t, mpfr_t);
typedef int binop_def(mpfr_t, mpfr_t, mpfr_t);


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

    while (N--) {
        mpf_field *in = ensure_mpf_init((mpf_field *)in_ptr, prec1);
        mpf_field *out = ensure_mpf_init((mpf_field *)out_ptr, prec2);

        // TODO: Should maybe do something with the result?
        unary_op(out->x, in->x);

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

int negative(mpfr_t op, mpfr_t out)
{
    return mpfr_neg(out, op, MPFR_RNDN);
}

int positive(mpfr_t op, mpfr_t out)
{
    if (out == op) {
        return 0;
    }
    return mpfr_set(out, op, MPFR_RNDN);
}

int sqrt(mpfr_t op, mpfr_t out)
{
    return mpfr_sqrt(out, op, MPFR_RNDN);
}

int absolute(mpfr_t op, mpfr_t out)
{
    return mpfr_abs(out, op, MPFR_RNDN);
}

int log(mpfr_t op, mpfr_t out)
{
    return mpfr_log(out, op, MPFR_RNDN);
}

int log2(mpfr_t op, mpfr_t out)
{
    return mpfr_log2(out, op, MPFR_RNDN);
}

int log10(mpfr_t op, mpfr_t out)
{
    return mpfr_log10(out, op, MPFR_RNDN);
}

int log1p(mpfr_t op, mpfr_t out)
{
    return mpfr_log1p(out, op, MPFR_RNDN);
}

int exp(mpfr_t op, mpfr_t out)
{
    return mpfr_exp(out, op, MPFR_RNDN);
}

int exp2(mpfr_t op, mpfr_t out)
{
    return mpfr_exp2(out, op, MPFR_RNDN);
}

int expm1(mpfr_t op, mpfr_t out)
{
    return mpfr_expm1(out, op, MPFR_RNDN);
}

int sin(mpfr_t op, mpfr_t out)
{
    return mpfr_sin(out, op, MPFR_RNDN);
}

int cos(mpfr_t op, mpfr_t out)
{
    return mpfr_cos(out, op, MPFR_RNDN);
}

int tan(mpfr_t op, mpfr_t out)
{
    return mpfr_tan(out, op, MPFR_RNDN);
}

int arcsin(mpfr_t op, mpfr_t out)
{
    return mpfr_asin(out, op, MPFR_RNDN);
}

int arccos(mpfr_t op, mpfr_t out)
{
    return mpfr_acos(out, op, MPFR_RNDN);
}

int arctan(mpfr_t op, mpfr_t out)
{
    return mpfr_tan(out, op, MPFR_RNDN);
}


int init_unary_ops(PyObject *numpy)
{
    if (create_unary_ufunc<negative>(numpy, "negative") < 0) {
        return -1;
    }
    if (create_unary_ufunc<positive>(numpy, "positive") < 0) {
        return -1;
    }
    if (create_unary_ufunc<sqrt>(numpy, "sqrt") < 0) {
        return -1;
    }
    if (create_unary_ufunc<absolute>(numpy, "absolute") < 0) {
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

    while (N--) {
        mpf_field *in1 = ensure_mpf_init((mpf_field *)in1_ptr, prec1);
        mpf_field *in2 = ensure_mpf_init((mpf_field *)in2_ptr, prec2);
        mpf_field *out = ensure_mpf_init((mpf_field *)out_ptr, prec3);

        // TODO: Should maybe do something with the result?
        binop(out->x, in1->x, in2->x);

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

    return 0;
}

/*
 * Not sure how to use templates without these helpers?
 */
static int
add(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_add(out, op1, op2, MPFR_RNDN);
}

static int
sub(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_sub(out, op1, op2, MPFR_RNDN);
}

static int
mul(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_mul(out, op1, op2, MPFR_RNDN);
}

static int
div(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_div(out, op1, op2, MPFR_RNDN);
}

static int
hypot(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_hypot(out, op1, op2, MPFR_RNDN);
}

static int
pow(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_pow(out, op1, op2, MPFR_RNDN);
}

static int
arctan2(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    return mpfr_atan2(out, op1, op2, MPFR_RNDN);
}

static int
nextafter(mpfr_t out, mpfr_t op1, mpfr_t op2)
{
    /*
     * Not ideal at all, but we should operate on the input, not output prec.
     * Plus, we actually know if this is the case or not, so could at least
     * short-cut (or special case both paths).
     */
    mpfr_prec_t prec = mpfr_get_prec(op1);
    if (prec == mpfr_get_prec(out)) {
        mpfr_set(out, op1, MPFR_RNDN);
        mpfr_nexttoward(out, op2);
        return 0;
    }
    mpfr_t tmp;
    mpfr_init2(tmp, prec);  // TODO: This could fail, mabye manual?
    mpfr_set(tmp, op1, MPFR_RNDN);
    mpfr_nexttoward(tmp, op2);
    int res = mpfr_set(out, tmp, MPFR_RNDN);
    mpfr_clear(tmp);

    return res;
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
    if (init_binary_ops(numpy) < 0) {
        goto err;
    }

    return 0;

  err:
    Py_DECREF(numpy);
    return -1;
}

