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

// Forward declarations for frexp operations
static Sleef_quad quad_frexp_mantissa(const Sleef_quad *op, int *exp);
static long double ld_frexp_mantissa(const long double *op, int *exp);

static Sleef_quad
quad_frexp_mantissa(const Sleef_quad *op, int *exp)
{
    return Sleef_frexpq1(*op, exp);
}

static long double
ld_frexp_mantissa(const long double *op, int *exp)
{
    return frexpl(*op, exp);
}

static NPY_CASTING
quad_frexp_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                               PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                               npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // Output 1: QuadPrecDType (mantissa)
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // Output 2: Int32 (exponent)
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = PyArray_DescrFromType(NPY_INT32);
    }
    else {
        Py_INCREF(given_descrs[2]);
        loop_descrs[2] = given_descrs[2];
    }

    return NPY_NO_CASTING;
}

static int
quad_frexp_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                  npy_intp const dimensions[], npy_intp const strides[],
                                  NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *mantissa_ptr = data[1];
    char *exp_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp mantissa_stride = strides[1];
    npy_intp exp_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in, mantissa;
    int exp;
    while (N--) {
        memcpy(&in, in_ptr, elem_size);
        
        if (backend == BACKEND_SLEEF) {
            mantissa.sleef_value = quad_frexp_mantissa(&in.sleef_value, &exp);
        }
        else {
            mantissa.longdouble_value = ld_frexp_mantissa(&in.longdouble_value, &exp);
        }
        
        memcpy(mantissa_ptr, &mantissa, elem_size);
        *(npy_int32 *)exp_ptr = (npy_int32)exp;

        in_ptr += in_stride;
        mantissa_ptr += mantissa_stride;
        exp_ptr += exp_stride;
    }
    return 0;
}

static int
quad_frexp_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                npy_intp const dimensions[], npy_intp const strides[],
                                NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *mantissa_ptr = data[1];
    char *exp_ptr = data[2];
    npy_intp in_stride = strides[0];
    npy_intp mantissa_stride = strides[1];
    npy_intp exp_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    int exp;
    while (N--) {
        if (backend == BACKEND_SLEEF) {
            *(Sleef_quad *)mantissa_ptr = quad_frexp_mantissa((Sleef_quad *)in_ptr, &exp);
        }
        else {
            *(long double *)mantissa_ptr = ld_frexp_mantissa((long double *)in_ptr, &exp);
        }
        
        *(npy_int32 *)exp_ptr = (npy_int32)exp;

        in_ptr += in_stride;
        mantissa_ptr += mantissa_stride;
        exp_ptr += exp_stride;
    }
    return 0;
}

int
create_quad_frexp_ufunc(PyObject *numpy)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, "frexp");
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &PyArray_Int32DType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_frexp_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_frexp_strided_loop_aligned},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_frexp_strided_loop_unaligned},
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

    Py_DECREF(ufunc);
    return 0;
}

int
init_quad_frexp(PyObject *numpy)
{
    if (create_quad_frexp_ufunc(numpy) < 0) {
        return -1;
    }
    return 0;
}