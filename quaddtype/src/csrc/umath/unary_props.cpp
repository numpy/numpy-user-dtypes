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
quad_unary_prop_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                    PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                    npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    loop_descrs[1] = PyArray_DescrFromType(NPY_BOOL);

    return NPY_NO_CASTING;
}

template <unary_prop_quad_def sleef_op, unary_prop_longdouble_def longdouble_op>
int
quad_generic_unary_prop_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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

template <unary_prop_quad_def sleef_op, unary_prop_longdouble_def longdouble_op>
int
quad_generic_unary_prop_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
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

template <unary_prop_quad_def sleef_op, unary_prop_longdouble_def longdouble_op>
int
create_quad_unary_prop_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[2] = {&QuadPrecDType, &PyArray_BoolDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_unary_prop_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_unary_prop_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_unary_prop_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_unary_prop",
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
init_quad_unary_props(PyObject *numpy)
{
    if (create_quad_unary_prop_ufunc<quad_isfinite, ld_isfinite>(numpy, "isfinite") < 0) {
        return -1;
    }
    if (create_quad_unary_prop_ufunc<quad_isinf, ld_isinf>(numpy, "isinf") < 0) {
        return -1;
    }
    if (create_quad_unary_prop_ufunc<quad_isnan, ld_isnan>(numpy, "isnan") < 0) {
        return -1;
    }
    if (create_quad_unary_prop_ufunc<quad_signbit, ld_signbit>(numpy, "signbit") < 0) {
        return -1;
    }
    return 0;
}
