#include "scalar.h"

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
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

#include "dtype.h"
#include "umath.h"
#include "ops.hpp"

template <unary_op_def unary_op>
int
quad_generic_unary_op_strided_loop(PyArrayMethod_Context *context, char *const data[],
                                   npy_intp const dimensions[], npy_intp const strides[],
                                   NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        unary_op((Sleef_quad *)in_ptr, (Sleef_quad *)out_ptr);
        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

static NPY_CASTING
quad_unary_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *dtypes[],
                                  QuadPrecDTypeObject *given_descrs[],
                                  QuadPrecDTypeObject *loop_descrs[], npy_intp *unused)
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

    return NPY_NO_CASTING;  // Quad precision is always the same precision
}

template <unary_op_def unary_op>
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
            {NPY_METH_strided_loop, (void *)&quad_generic_unary_op_strided_loop<unary_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_unary_op",
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

int
init_quad_unary_ops(PyObject *numpy)
{
    if (create_quad_unary_ufunc<quad_negative>(numpy, "negative") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_absolute>(numpy, "absolute") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_rint>(numpy, "rint") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_trunc>(numpy, "trunc") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_floor>(numpy, "floor") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_ceil>(numpy, "ceil") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_sqrt>(numpy, "sqrt") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_square>(numpy, "square") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log>(numpy, "log") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log2>(numpy, "log2") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log10>(numpy, "log10") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_log1p>(numpy, "log1p") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_exp>(numpy, "exp") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_exp2>(numpy, "exp2") < 0) {
        return -1;
    }
    return 0;
}

// Binary ufuncs

template <binop_def binop>
int
quad_generic_binop_strided_loop(PyArrayMethod_Context *context, char *const data[],
                                npy_intp const dimensions[], npy_intp const strides[],
                                NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    while (N--) {
        binop((Sleef_quad *)out_ptr, (Sleef_quad *)in1_ptr, (Sleef_quad *)in2_ptr);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

static NPY_CASTING
quad_binary_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *dtypes[],
                                   QuadPrecDTypeObject *given_descrs[],
                                   QuadPrecDTypeObject *loop_descrs[], npy_intp *unused)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    if (given_descrs[2] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[2] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    return NPY_NO_CASTING;  // Quad precision is always the same precision
}

// todo: skipping the promoter for now, since same type operation will be requried

template <binop_def binop>
int
create_quad_binary_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_binary_op_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_generic_binop_strided_loop<binop>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_binop",
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

int
init_quad_binary_ops(PyObject *numpy)
{
    if (create_quad_binary_ufunc<quad_add>(numpy, "add") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_sub>(numpy, "subtract") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_mul>(numpy, "multiply") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_div>(numpy, "divide") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_pow>(numpy, "power") < 0) {
        return -1;
    }
    return 0;
}

// comparison functions

template <cmp_def comp>
int
quad_generic_comp_strided_loop(PyArrayMethod_Context *context, char *const data[],
                               npy_intp const dimensions[], npy_intp const strides[],
                               NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    while (N--) {
        *((npy_bool *)out_ptr) = comp((Sleef_quad *)in1_ptr, (Sleef_quad *)in2_ptr);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <cmp_def comp>
int
create_quad_comparison_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &PyArray_BoolDType};

    PyType_Slot slots[] = {{NPY_METH_strided_loop, (void *)&quad_generic_comp_strided_loop<comp>},
                           {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_comp",
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

int
init_quad_comps(PyObject *numpy)
{
    if (create_quad_comparison_ufunc<quad_equal>(numpy, "equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_notequal>(numpy, "not_equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_less>(numpy, "less") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_lessequal>(numpy, "less_equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_greater>(numpy, "greater") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_greaterequal>(numpy, "greater_equal") < 0) {
        return -1;
    }

    return 0;
}

int
init_quad_umath(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy)
        return -1;

    if (init_quad_unary_ops(numpy) < 0) {
        goto err;
    }

    if (init_quad_binary_ops(numpy) < 0) {
        goto err;
    }

    if (init_quad_comps(numpy) < 0) {
        goto err;
    }

    Py_DECREF(numpy);
    return 0;

err:
    Py_DECREF(numpy);
    return -1;
}