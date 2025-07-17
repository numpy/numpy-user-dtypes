#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC


#include <Python.h>
#include <cstdio>

#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "../quad_common.h"
#include "../scalar.h"
#include "../dtype.h"
#include "../ops.hpp"
#include "binary_ops.h"
#include "matmul.h"

#include <iostream>

static NPY_CASTING
quad_matmul_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                   PyArray_Descr *const given_descrs[],
                                   PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{

  NPY_CASTING casting = NPY_NO_CASTING;    
  std::cout << "exiting the descriptor";
  return casting;
}

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
int
quad_generic_matmul_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                          npy_intp const dimensions[], npy_intp const strides[],
                                          NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in1, in2, out;
    while (N--) {
        memcpy(&in1, in1_ptr, elem_size);
        memcpy(&in2, in2_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            out.sleef_value = sleef_op(&in1.sleef_value, &in2.sleef_value);
        }
        else {
            out.longdouble_value = longdouble_op(&in1.longdouble_value, &in2.longdouble_value);
        }
        memcpy(out_ptr, &out, elem_size);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
int
quad_generic_matmul_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                        npy_intp const dimensions[], npy_intp const strides[],
                                        NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    while (N--) {
        if (backend == BACKEND_SLEEF) {
            *(Sleef_quad *)out_ptr = sleef_op((Sleef_quad *)in1_ptr, (Sleef_quad *)in2_ptr);
        }
        else {
            *(long double *)out_ptr = longdouble_op((long double *)in1_ptr, (long double *)in2_ptr);
        }

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

int
create_matmul_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_matmul_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_matmul_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_matmul_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_matmul",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_IS_REORDERABLE),
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }
    // my guess we don't need any promoter here as of now, since matmul is quad specific
    return 0;
}


int
init_matmul_ops(PyObject *numpy)
{
    if (create_matmul_ufunc<quad_add>(numpy, "matmul") < 0) {
        return -1;
    }
    return 0;
}

