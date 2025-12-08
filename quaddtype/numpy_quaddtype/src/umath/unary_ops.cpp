#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
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

// Logical NOT - returns bool instead of QuadPrecision
static NPY_CASTING
quad_unary_logical_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                         PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                         npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // Output is always bool
    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);
    if (!loop_descrs[1]) {
        return (NPY_CASTING)-1;
    }

    return NPY_NO_CASTING;
}

template <unary_logical_quad_def sleef_op, unary_logical_longdouble_def longdouble_op>
int
quad_logical_not_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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

    quad_value in;
    while (N--) {
        memcpy(&in, in_ptr, elem_size);
        npy_bool result;
        
        if (backend == BACKEND_SLEEF) {
            result = sleef_op(&in.sleef_value);
        }
        else {
            result = longdouble_op(&in.longdouble_value);
        }
        
        memcpy(out_ptr, &result, sizeof(npy_bool));

        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <unary_logical_quad_def sleef_op, unary_logical_longdouble_def longdouble_op>
int
quad_logical_not_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
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
        npy_bool result;
        
        if (backend == BACKEND_SLEEF) {
            result = sleef_op((Sleef_quad *)in_ptr);
        }
        else {
            result = longdouble_op((long double *)in_ptr);
        }
        
        *(npy_bool *)out_ptr = result;

        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <unary_logical_quad_def sleef_op, unary_logical_longdouble_def longdouble_op>
int
create_quad_logical_not_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[2] = {&QuadPrecDType, &PyArray_BoolDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_unary_logical_op_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_logical_not_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_logical_not_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_logical_not",
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

// Resolver for unary ufuncs with 2 outputs (like modf)
static NPY_CASTING
quad_unary_op_2out_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                       PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                       npy_intp *NPY_UNUSED(view_offset))
{
    // Input descriptor
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // Output descriptors (2 outputs)
    for (int i = 1; i < 3; i++) {
        if (given_descrs[i] == NULL) {
            Py_INCREF(given_descrs[0]);
            loop_descrs[i] = given_descrs[0];
        }
        else {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = given_descrs[i];
        }
    }

    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_out1 = (QuadPrecDTypeObject *)loop_descrs[1];
    QuadPrecDTypeObject *descr_out2 = (QuadPrecDTypeObject *)loop_descrs[2];

    if (descr_in->backend != descr_out1->backend || descr_in->backend != descr_out2->backend) {
        return NPY_UNSAFE_CASTING;
    }

    return NPY_NO_CASTING;
}

// Strided loop for unary ops with 2 outputs (unaligned)
template <unary_op_2out_quad_def sleef_op, unary_op_2out_longdouble_def longdouble_op>
int
quad_generic_unary_op_2out_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                                  npy_intp const dimensions[], npy_intp const strides[],
                                                  NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out1_ptr = data[1];
    char *out2_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp out1_stride = strides[1];
    npy_intp out2_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in, out1, out2;
    while (N--) {
        memcpy(&in, in_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            sleef_op(&in.sleef_value, &out1.sleef_value, &out2.sleef_value);
        }
        else {
            longdouble_op(&in.longdouble_value, &out1.longdouble_value, &out2.longdouble_value);
        }
        memcpy(out1_ptr, &out1, elem_size);
        memcpy(out2_ptr, &out2, elem_size);

        in_ptr += in_stride;
        out1_ptr += out1_stride;
        out2_ptr += out2_stride;
    }
    return 0;
}

// Strided loop for unary ops with 2 outputs (aligned)
template <unary_op_2out_quad_def sleef_op, unary_op_2out_longdouble_def longdouble_op>
int
quad_generic_unary_op_2out_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                               npy_intp const dimensions[], npy_intp const strides[],
                                               NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out1_ptr = data[1];
    char *out2_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp out1_stride = strides[1];
    npy_intp out2_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    while (N--) {
        if (backend == BACKEND_SLEEF) {
            sleef_op((Sleef_quad *)in_ptr, (Sleef_quad *)out1_ptr, (Sleef_quad *)out2_ptr);
        }
        else {
            longdouble_op((long double *)in_ptr, (long double *)out1_ptr, (long double *)out2_ptr);
        }
        in_ptr += in_stride;
        out1_ptr += out1_stride;
        out2_ptr += out2_stride;
    }
    return 0;
}

// Create unary ufunc with 2 outputs
template <unary_op_2out_quad_def sleef_op, unary_op_2out_longdouble_def longdouble_op>
int
create_quad_unary_2out_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    // 1 input, 2 outputs
    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_unary_op_2out_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_unary_op_2out_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_unary_op_2out_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_unary_2out",
            .nin = 1,
            .nout = 2,
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

// Frexp-specific resolver: QuadPrecDType -> (QuadPrecDType, int32)
static NPY_CASTING
quad_frexp_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                               PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                               npy_intp *NPY_UNUSED(view_offset))
{
    // Input descriptor
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // Output 1: mantissa (same dtype as input)
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // Output 2: exponent (int32)
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = PyArray_DescrFromType(NPY_INT32);
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_out1 = (QuadPrecDTypeObject *)loop_descrs[1];

    if (descr_in->backend != descr_out1->backend) {
        return NPY_UNSAFE_CASTING;
    }

    return NPY_NO_CASTING;
}

// Strided loop for frexp (unaligned)
template <frexp_op_quad_def sleef_op, frexp_op_longdouble_def longdouble_op>
int
quad_frexp_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                  npy_intp const dimensions[], npy_intp const strides[],
                                  NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_mantissa_ptr = data[1];
    char *out_exp_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp out_mantissa_stride = strides[1];
    npy_intp out_exp_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in, out_mantissa;
    int out_exp;
    
    while (N--) {
        memcpy(&in, in_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            out_mantissa.sleef_value = sleef_op(&in.sleef_value, &out_exp);
        }
        else {
            out_mantissa.longdouble_value = longdouble_op(&in.longdouble_value, &out_exp);
        }
        memcpy(out_mantissa_ptr, &out_mantissa, elem_size);
        memcpy(out_exp_ptr, &out_exp, sizeof(int));

        in_ptr += in_stride;
        out_mantissa_ptr += out_mantissa_stride;
        out_exp_ptr += out_exp_stride;
    }
    return 0;
}

// Strided loop for frexp (aligned)
template <frexp_op_quad_def sleef_op, frexp_op_longdouble_def longdouble_op>
int
quad_frexp_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                               npy_intp const dimensions[], npy_intp const strides[],
                               NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_mantissa_ptr = data[1];
    char *out_exp_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp out_mantissa_stride = strides[1];
    npy_intp out_exp_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    int out_exp;
    
    while (N--) {
        if (backend == BACKEND_SLEEF) {
            Sleef_quad mantissa = sleef_op((Sleef_quad *)in_ptr, &out_exp);
            memcpy(out_mantissa_ptr, &mantissa, sizeof(Sleef_quad));
        }
        else {
            long double mantissa = longdouble_op((long double *)in_ptr, &out_exp);
            memcpy(out_mantissa_ptr, &mantissa, sizeof(long double));
        }
        memcpy(out_exp_ptr, &out_exp, sizeof(int));

        in_ptr += in_stride;
        out_mantissa_ptr += out_mantissa_stride;
        out_exp_ptr += out_exp_stride;
    }
    return 0;
}


template <frexp_op_quad_def sleef_op, frexp_op_longdouble_def longdouble_op>
int
create_quad_frexp_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    // 1 input (QuadPrecDType), 2 outputs (QuadPrecDType, int32)
    PyArray_DTypeMeta *dtypes[3] = {
        &QuadPrecDType,
        &QuadPrecDType,
        &PyArray_Int32DType
    };

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_frexp_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_frexp_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_frexp_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_frexp",
            .nin = 1,
            .nout = 2,
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
    // fabs is simialr to absolute, just not handles complex values (we neither)
    // registering the absolute here
    if (create_quad_unary_ufunc<quad_absolute, ld_absolute>(numpy, "fabs") < 0) {
        return -1;
    }
    // conjugate is a no-op for real numbers (returns the value unchanged)
    if (create_quad_unary_ufunc<quad_conjugate, ld_conjugate>(numpy, "conjugate") < 0) {
        return -1;
    }
    // conj is an alias for conjugate, no need to register
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
    if (create_quad_unary_ufunc<quad_cbrt, ld_cbrt>(numpy, "cbrt") < 0) {
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
    if (create_quad_unary_ufunc<quad_expm1, ld_expm1>(numpy, "expm1") < 0) {
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
    if (create_quad_unary_ufunc<quad_degrees, ld_degrees>(numpy, "degrees") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_degrees, ld_degrees>(numpy, "rad2deg") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_radians, ld_radians>(numpy, "radians") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_radians, ld_radians>(numpy, "deg2rad") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_spacing, ld_spacing>(numpy, "spacing") < 0) {
        return -1;
    }

    // Logical operations (unary: not)
    if (create_quad_logical_not_ufunc<quad_logical_not, ld_logical_not>(numpy, "logical_not") < 0) {
        return -1;
    }
    
    // Unary operations with 2 outputs
    if (create_quad_unary_2out_ufunc<quad_modf, ld_modf>(numpy, "modf") < 0) {
        return -1;
    }
    
    // Frexp: special case with (QuadPrecDType, int32) outputs
    if (create_quad_frexp_ufunc<quad_frexp, ld_frexp>(numpy, "frexp") < 0) {
        return -1;
    }
    
    return 0;
}