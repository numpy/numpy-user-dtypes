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
#include "promoters.hpp"
#include "binary_ops.h"

static NPY_CASTING
quad_binary_op_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                   PyArray_Descr *const given_descrs[],
                                   PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_in2 = (QuadPrecDTypeObject *)given_descrs[1];
    QuadBackendType target_backend;

    // Determine target backend and if casting is needed
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
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
        if (!loop_descrs[2]) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)given_descrs[2];
        if (descr_out->backend != target_backend) {
            loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[2]) {
                return (NPY_CASTING)-1;
            }
        }
        else {
            Py_INCREF(given_descrs[2]);
            loop_descrs[2] = given_descrs[2];
        }
    }
    return casting;
}

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
int
quad_generic_binop_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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
quad_generic_binop_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
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



// Resolve descriptors for binary ops with 2 outputs (2 inputs, 2 outputs)
static NPY_CASTING
quad_binary_op_2out_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                        PyArray_Descr *const given_descrs[],
                                        PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_in2 = (QuadPrecDTypeObject *)given_descrs[1];
    QuadBackendType target_backend;

    // Determine target backend and if casting is needed
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

    // Set up output descriptors (2 outputs for divmod)
    for (int i = 2; i < 4; i++) {
        if (given_descrs[i] == NULL) {
            loop_descrs[i] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[i]) {
                return (NPY_CASTING)-1;
            }
        }
        else {
            QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)given_descrs[i];
            if (descr_out->backend != target_backend) {
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
    }
    return casting;
}

// Strided loop for binary ops with 2 outputs (unaligned)
template <binary_op_2out_quad_def sleef_op, binary_op_2out_longdouble_def longdouble_op>
int
quad_generic_binop_2out_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                               npy_intp const dimensions[], npy_intp const strides[],
                                               NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out1_ptr = data[2], *out2_ptr = data[3];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out1_stride = strides[2];
    npy_intp out2_stride = strides[3];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in1, in2, out1, out2;
    while (N--) {
        memcpy(&in1, in1_ptr, elem_size);
        memcpy(&in2, in2_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            sleef_op(&in1.sleef_value, &in2.sleef_value, &out1.sleef_value, &out2.sleef_value);
        }
        else {
            longdouble_op(&in1.longdouble_value, &in2.longdouble_value, 
                         &out1.longdouble_value, &out2.longdouble_value);
        }
        memcpy(out1_ptr, &out1, elem_size);
        memcpy(out2_ptr, &out2, elem_size);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out1_ptr += out1_stride;
        out2_ptr += out2_stride;
    }
    return 0;
}

// Strided loop for binary ops with 2 outputs (aligned)
template <binary_op_2out_quad_def sleef_op, binary_op_2out_longdouble_def longdouble_op>
int
quad_generic_binop_2out_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                             npy_intp const dimensions[], npy_intp const strides[],
                                             NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0], *in2_ptr = data[1];
    char *out1_ptr = data[2], *out2_ptr = data[3];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out1_stride = strides[2];
    npy_intp out2_stride = strides[3];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    while (N--) {
        if (backend == BACKEND_SLEEF) {
            sleef_op((Sleef_quad *)in1_ptr, (Sleef_quad *)in2_ptr, 
                    (Sleef_quad *)out1_ptr, (Sleef_quad *)out2_ptr);
        }
        else {
            longdouble_op((long double *)in1_ptr, (long double *)in2_ptr,
                         (long double *)out1_ptr, (long double *)out2_ptr);
        }

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out1_ptr += out1_stride;
        out2_ptr += out2_stride;
    }
    return 0;
}

// todo: I'll preferrable get all this code duplication in templates later
// resolve descriptors for ldexp (QuadPrecDType, int) -> QuadPrecDType
static NPY_CASTING
quad_ldexp_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                               PyArray_Descr *const given_descrs[],
                               PyArray_Descr *loop_descrs[], npy_intp *NPY_UNUSED(view_offset))
{
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadBackendType target_backend = descr_in1->backend;

    // Input 0: QuadPrecDType
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    // Input 1: Use NPY_INTP (int64 on 64-bit, int32 on 32-bit) to match platform integer size
    // This ensures we can handle the full range of PyArray_PyLongDType without data loss
    loop_descrs[1] = PyArray_DescrFromType(NPY_INTP);

    // Output: QuadPrecDType with same backend as input
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
        if (!loop_descrs[2]) {
            return (NPY_CASTING)-1;
        }
    } else {
        QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)given_descrs[2];
        if (descr_out->backend != target_backend) {
            loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[2]) {
                return (NPY_CASTING)-1;
            }
        } else {
            Py_INCREF(given_descrs[2]);
            loop_descrs[2] = given_descrs[2];
        }
    }
    // Return SAFE_CASTING to allow conversion from other integer types to intp
    return NPY_SAFE_CASTING;
}

// Strided loop for ldexp (unaligned)
template <ldexp_op_quad_def sleef_op, ldexp_op_longdouble_def longdouble_op>
int
quad_ldexp_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                  npy_intp const dimensions[], npy_intp const strides[],
                                  NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0];
    char *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in1, out;
    npy_intp in2_intp;  // Platform-native integer (int64 on 64-bit, int32 on 32-bit)
    while (N--) {
        memcpy(&in1, in1_ptr, elem_size);
        memcpy(&in2_intp, in2_ptr, sizeof(npy_intp));
        
        int exp_value = (int)in2_intp;
        
        if (backend == BACKEND_SLEEF) {
            out.sleef_value = sleef_op(&in1.sleef_value, &exp_value);
        } else {
            out.longdouble_value = longdouble_op(&in1.longdouble_value, &exp_value);
        }
        memcpy(out_ptr, &out, elem_size);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

// Strided loop for ldexp (aligned)
template <ldexp_op_quad_def sleef_op, ldexp_op_longdouble_def longdouble_op>
int
quad_ldexp_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                npy_intp const dimensions[], npy_intp const strides[],
                                NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    char *in1_ptr = data[0]; 
    char *in2_ptr = data[1];
    char *out_ptr = data[2];
    npy_intp in1_stride = strides[0];
    npy_intp in2_stride = strides[1];
    npy_intp out_stride = strides[2];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;

    while (N--) {
        npy_intp exp_intp = *(npy_intp *)in2_ptr;
        int exp_value = (int)exp_intp;
        
        if (backend == BACKEND_SLEEF) {
            *(Sleef_quad *)out_ptr = sleef_op((Sleef_quad *)in1_ptr, &exp_value);
        } else {
            *(long double *)out_ptr = longdouble_op((long double *)in1_ptr, &exp_value);
        }

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

// Create ldexp ufunc
template <ldexp_op_quad_def sleef_op, ldexp_op_longdouble_def longdouble_op>
int
create_quad_ldexp_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &PyArray_PyLongDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_ldexp_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_ldexp_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_ldexp_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_ldexp",
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

    PyObject *promoter_capsule =
            PyCapsule_New((void *)&quad_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArray_PyLongDType, &PyArrayDescr_Type);
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

// Create binary ufunc with 2 outputs (generic for divmod, modf, frexp, etc.)
template <binary_op_2out_quad_def sleef_op, binary_op_2out_longdouble_def longdouble_op>
int
create_quad_binary_2out_ufunc(PyObject *numpy, const char *ufunc_name)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, ufunc_name);
    if (ufunc == NULL) {
        return -1;
    }

    // 2 inputs, 2 outputs
    PyArray_DTypeMeta *dtypes[4] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_binary_op_2out_resolve_descriptors},
            {NPY_METH_strided_loop,
             (void *)&quad_generic_binop_2out_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_binop_2out_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_binop_2out",
            .nin = 2,
            .nout = 2,
            .casting = NPY_NO_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_IS_REORDERABLE),
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        return -1;
    }

    PyObject *promoter_capsule =
            PyCapsule_New((void *)&quad_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(4, &PyArrayDescr_Type, &PyArrayDescr_Type, 
                                      &PyArrayDescr_Type, &PyArrayDescr_Type);
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

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
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
            {NPY_METH_strided_loop,
             (void *)&quad_generic_binop_strided_loop_aligned<sleef_op, longdouble_op>},
            {NPY_METH_unaligned_strided_loop,
             (void *)&quad_generic_binop_strided_loop_unaligned<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_binop",
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

    PyObject *promoter_capsule =
            PyCapsule_New((void *)&quad_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArrayDescr_Type);
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
init_quad_binary_ops(PyObject *numpy)
{
    if (create_quad_binary_ufunc<quad_add, ld_add>(numpy, "add") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_sub, ld_sub>(numpy, "subtract") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_mul, ld_mul>(numpy, "multiply") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_div, ld_div>(numpy, "divide") < 0) {
        return -1;
    }
    // Note: true_divide is an alias to divide in NumPy for floating-point types
    // No need to register separately
    if (create_quad_binary_ufunc<quad_floor_divide, ld_floor_divide>(numpy, "floor_divide") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_pow, ld_pow>(numpy, "power") < 0) {
        return -1;
    }
    // float_power uses the same implementation as power for floating-point types
    // The only difference is that float_power promotes integer inputs to float (quaddtype is already float)
    if (create_quad_binary_ufunc<quad_pow, ld_pow>(numpy, "float_power") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_mod, ld_mod>(numpy, "mod") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_fmod, ld_fmod>(numpy, "fmod") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_minimum, ld_minimum>(numpy, "minimum") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_maximum, ld_maximum>(numpy, "maximum") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_fmin, ld_fmin>(numpy, "fmin") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_fmax, ld_fmax>(numpy, "fmax") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_atan2, ld_atan2>(numpy, "arctan2") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_copysign, ld_copysign>(numpy, "copysign") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_nextafter, ld_nextafter>(numpy, "nextafter") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_logaddexp, ld_logaddexp>(numpy, "logaddexp") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_logaddexp2, ld_logaddexp2>(numpy, "logaddexp2") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_heaviside, ld_heaviside>(numpy, "heaviside") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_hypot, ld_hypot>(numpy, "hypot") < 0) {
        return -1;
    }
    if (create_quad_binary_2out_ufunc<quad_divmod, ld_divmod>(numpy, "divmod") < 0) {
        return -1;
    }
    if (create_quad_ldexp_ufunc<quad_ldexp, ld_ldexp>(numpy, "ldexp") < 0) {
        return -1;
    }
    return 0;
}