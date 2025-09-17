#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>
#include <cstdio>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

#include "numpy/dtype_api.h"
}
#include "../quad_common.h"
#include "../scalar.h"
#include "../dtype.h"
#include "../ops.hpp"

static NPY_CASTING
quad_unary_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                  PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                  npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)loop_descrs[1];

    if (descr_in->backend != descr_out->backend) {
        return NPY_UNSAFE_CASTING;
    }

    return NPY_NO_CASTING;
}

template <unary_op_quad_def sleef_op, unary_op_longdouble_def longdouble_op>
int
quad_generic_unary_op_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                             npy_intp const dimensions[], npy_intp const strides[],
                                             NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in, out;
    while (N--) {
        memcpy(&in, in_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            out.sleef_value = sleef_op(&in.sleef_value);
        }
        else {
            out.longdouble_value = longdouble_op(&in.longdouble_value);
        }
        memcpy(out_ptr, &out, elem_size);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <unary_op_quad_def sleef_op, unary_op_longdouble_def longdouble_op>
int
quad_generic_unary_op_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                           npy_intp const dimensions[], npy_intp const strides[],
                                           NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    while (N--) {
        if (backend == BACKEND_SLEEF) {
            *(Sleef_quad *)out_ptr = sleef_op((Sleef_quad *)in_ptr);
        }
        else {
            *(long double *)out_ptr = longdouble_op((long double *)in_ptr);
        }
        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <unary_op_quad_def sleef_op, unary_op_longdouble_def longdouble_op>
int
create_quad_unary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[2] = {&QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_unary_op_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_unary_op_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_unary_op_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_unary_op",
            .nin = 1,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    return 0;
}

int
init_quad_unary_ops(PyObject *numpy)
{
    if (create_quad_unary_ufunc<quad_negative, ld_negative>(numpy, "negative") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_positive, ld_positive>(numpy, "positive") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_absolute, ld_absolute>(numpy, "absolute") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_sign, ld_sign>(numpy, "sign") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_rint, ld_rint>(numpy, "rint") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_trunc, ld_trunc>(numpy, "trunc") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_floor, ld_floor>(numpy, "floor") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_ceil, ld_ceil>(numpy, "ceil") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_sqrt, ld_sqrt>(numpy, "sqrt") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_square, ld_square>(numpy, "square") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_reciprocal, ld_reciprocal>(numpy, "reciprocal") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log, ld_log>(numpy, "log") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log2, ld_log2>(numpy, "log2") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log10, ld_log10>(numpy, "log10") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log1p, ld_log1p>(numpy, "log1p") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_exp, ld_exp>(numpy, "exp") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_exp2, ld_exp2>(numpy, "exp2") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_sin, ld_sin>(numpy, "sin") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_cos, ld_cos>(numpy, "cos") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_tan, ld_tan>(numpy, "tan") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_asin, ld_asin>(numpy, "arcsin") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_acos, ld_acos>(numpy, "arccos") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_atan, ld_atan>(numpy, "arctan") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_sinh, ld_sinh>(numpy, "sinh") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_cosh, ld_cosh>(numpy, "cosh") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_tanh, ld_tanh>(numpy, "tanh") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_asinh, ld_asinh>(numpy, "arcsinh") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_acosh, ld_acosh>(numpy, "arccosh") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_atanh, ld_atanh>(numpy, "arctanh") < 0) {
        return -1;
    }
    return 0;
}