#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
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
#include "umath.h"
#include "../ops.hpp"
#include "promoters.hpp"
#include "binary_ops.h"
#include "comparison_ops.h"

static NPY_CASTING
quad_comparison_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                       PyArray_Descr *const given_descrs[],
                                       PyArray_Descr *loop_descrs[],
                                       npy_intp *NPY_UNUSED(view_offset))
{
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_in2 = (QuadPrecDTypeObject *)given_descrs[1];
    QuadBackendType target_backend;

    // As dealing with different backends then cast to boolean
    NPY_CASTING casting = NPY_NO_CASTING;
    if (descr_in1->backend != descr_in2->backend) {
        target_backend = BACKEND_LONGDOUBLE;
        casting = NPY_SAFE_CASTING;
    }
    else {
        target_backend = descr_in1->backend;
    }

    // Set up input descriptors, casting if necessary
    for (int i = 0; i < 2; i++) {
        if (((QuadPrecDTypeObject *)given_descrs[i])->backend != target_backend) {
            loop_descrs[i] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[i]) {
                return (NPY_CASTING)-1;
            }
        }
        else {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = given_descrs[i];
        }
    }

    // Set up output descriptor
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);
    if (!loop_descrs[2]) {
        return (NPY_CASTING)-1;
    }
    return casting;
}

template <cmp_quad_def sleef_comp, cmp_londouble_def ld_comp>
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

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in1, in2;
    while (N--) {
        memcpy(&in1, in1_ptr, elem_size);
        memcpy(&in2, in2_ptr, elem_size);
        npy_bool result;

        if (backend == BACKEND_SLEEF) {
            result = sleef_comp(&in1.sleef_value, &in2.sleef_value);
        }
        else {
            result = ld_comp(&in1.longdouble_value, &in2.longdouble_value);
        }

        memcpy(out_ptr, &result, sizeof(npy_bool));

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <cmp_quad_def sleef_comp, cmp_londouble_def ld_comp>
int
quad_generic_comp_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
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
        quad_value in1;
        quad_value in2;

        npy_bool result;

        if (backend == BACKEND_SLEEF) {
            in1.sleef_value = *(Sleef_quad *)in1_ptr;
            in2.sleef_value = *(Sleef_quad *)in2_ptr;
            result = sleef_comp(&in1.sleef_value, &in2.sleef_value);
        }
        else {
            in1.longdouble_value = *(long double *)in1_ptr;
            in2.longdouble_value = *(long double *)in2_ptr;
            result = ld_comp(&in1.longdouble_value, &in2.longdouble_value);
        }

        *(npy_bool *)out_ptr = result;

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}
// todo: It'll be better to generate separate templates for aligned and unaligned loops
// Resolve desc and strided loops for logical reduction (Bool, Quad) => Bool
static NPY_CASTING
quad_comparison_reduce_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                        PyArray_Descr *const given_descrs[],
                                        PyArray_Descr *loop_descrs[],
                                        npy_intp *NPY_UNUSED(view_offset))
{
    NPY_CASTING casting = NPY_SAFE_CASTING;
    
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }

    // Set up output descriptor
    loop_descrs[2] = PyArray_DescrFromType(NPY_BOOL);
    if (!loop_descrs[2]) {
        return (NPY_CASTING)-1;
    }
    return casting;
}

template <cmp_quad_def sleef_comp, cmp_londouble_def ld_comp>
int
quad_reduce_comp_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                       npy_intp const dimensions[], npy_intp const strides[],
                                       NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0];  // bool
    char *in2_ptr = data[1];  // quad
    char *out_ptr = data[2];  // bool
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[1];
    QuadBackendType backend = descr->backend;
    while (N--) {
        npy_bool in1 = *(npy_bool *)in1_ptr;
        quad_value in1_quad;
        quad_value in2;

        npy_bool result;

        if (backend == BACKEND_SLEEF) {
            in1_quad.sleef_value = Sleef_cast_from_int64q1(in1);
            in2.sleef_value = *(Sleef_quad *)in2_ptr;
            result = sleef_comp(&in1_quad.sleef_value, &in2.sleef_value);
        }
        else {
            in1_quad.longdouble_value = static_cast<long double>(in1);
            in2.longdouble_value = *(long double *)in2_ptr;
            result = ld_comp(&in1_quad.longdouble_value, &in2.longdouble_value);
        }

        *(npy_bool *)out_ptr = result;

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

template <cmp_quad_def sleef_comp, cmp_londouble_def ld_comp>
int
quad_reduce_comp_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                               npy_intp const dimensions[], npy_intp const strides[],
                               NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0];  // bool
    char *in2_ptr = data[1];  // quad
    char *out_ptr = data[2];  // bool
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[1];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    npy_bool in1;
    quad_value in1_quad, in2;
    while (N--) {
        memcpy(&in1, in1_ptr, sizeof(npy_bool));
        if(backend == BACKEND_SLEEF)
            in1_quad.sleef_value = Sleef_cast_from_int64q1(in1);
        else
            in1_quad.longdouble_value = static_cast<long double>(in1);
        memcpy(&in2, in2_ptr, elem_size);
        npy_bool result;

        if (backend == BACKEND_SLEEF) {
            result = sleef_comp(&in1_quad.sleef_value, &in2.sleef_value);
        }
        else {
            result = ld_comp(&in1_quad.longdouble_value, &in2.longdouble_value);
        }

        memcpy(out_ptr, &result, sizeof(npy_bool));

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}


NPY_NO_EXPORT int
comparison_ufunc_promoter(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                          PyArray_DTypeMeta *signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    PyArray_DTypeMeta *new_signature[NPY_MAXARGS];
    memcpy(new_signature, signature, 3 * sizeof(PyArray_DTypeMeta *));
    new_signature[2] = NULL;
    int res = quad_ufunc_promoter(ufunc, op_dtypes, new_signature, new_op_dtypes);
    if (res < 0) {
        return -1;
    }
    Py_XSETREF(new_op_dtypes[2], &PyArray_BoolDType);
    return 0;
}

template <cmp_quad_def sleef_comp, cmp_londouble_def ld_comp>
int
create_quad_comparison_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &PyArray_BoolDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_comparison_op_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_comp_strided_loop_aligned<sleef_comp, ld_comp>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_comp_strided_loop<sleef_comp, ld_comp>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_comp",
            .nin = 2,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    // registering the reduce methods
    PyArray_DTypeMeta *dtypes_reduce[3] = {&PyArray_BoolDType, &QuadPrecDType, &PyArray_BoolDType};

    PyType_Slot slots_reduce[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_comparison_reduce_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_reduce_comp_strided_loop_aligned<sleef_comp, ld_comp>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_reduce_comp_strided_loop_unaligned<sleef_comp, ld_comp>},
            {0, NULL}};

    PyArrayMethod_Spec Spec_reduce = {
            .name = "quad_comp",
            .nin = 2,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes_reduce,
            .slots = slots_reduce,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec_reduce) < 0) {
        return -1;
    }

    PyObject *promoter_capsule =
            PyCapsule_New((void *)&comparison_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArray_BoolDType);
    if (DTypes == 0) {
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
init_quad_comps(PyObject *numpy)
{
    if (create_quad_comparison_ufunc<quad_equal, ld_equal>(numpy, "equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_notequal, ld_notequal>(numpy, "not_equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_less, ld_less>(numpy, "less") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_lessequal, ld_lessequal>(numpy, "less_equal") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_greater, ld_greater>(numpy, "greater") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_greaterequal, ld_greaterequal>(numpy, "greater_equal") <
        0) {
        return -1;
    }

    // Logical operations (binary: and, or, xor)
    if (create_quad_comparison_ufunc<quad_logical_and, ld_logical_and>(numpy, "logical_and") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_logical_or, ld_logical_or>(numpy, "logical_or") < 0) {
        return -1;
    }
    if (create_quad_comparison_ufunc<quad_logical_xor, ld_logical_xor>(numpy, "logical_xor") < 0) {
        return -1;
    }

    return 0;
}