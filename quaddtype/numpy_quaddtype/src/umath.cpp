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
#include "quad_common.h"
#include "scalar.h"
#include "dtype.h"
#include "umath.h"
#include "ops.hpp"

// helper debugging function
static const char *
get_dtype_name(PyArray_DTypeMeta *dtype)
{
    if (dtype == &QuadPrecDType) {
        return "QuadPrecDType";
    }
    else if (dtype == &PyArray_BoolDType) {
        return "BoolDType";
    }
    else if (dtype == &PyArray_ByteDType) {
        return "ByteDType";
    }
    else if (dtype == &PyArray_UByteDType) {
        return "UByteDType";
    }
    else if (dtype == &PyArray_ShortDType) {
        return "ShortDType";
    }
    else if (dtype == &PyArray_UShortDType) {
        return "UShortDType";
    }
    else if (dtype == &PyArray_IntDType) {
        return "IntDType";
    }
    else if (dtype == &PyArray_UIntDType) {
        return "UIntDType";
    }
    else if (dtype == &PyArray_LongDType) {
        return "LongDType";
    }
    else if (dtype == &PyArray_ULongDType) {
        return "ULongDType";
    }
    else if (dtype == &PyArray_LongLongDType) {
        return "LongLongDType";
    }
    else if (dtype == &PyArray_ULongLongDType) {
        return "ULongLongDType";
    }
    else if (dtype == &PyArray_FloatDType) {
        return "FloatDType";
    }
    else if (dtype == &PyArray_DoubleDType) {
        return "DoubleDType";
    }
    else if (dtype == &PyArray_LongDoubleDType) {
        return "LongDoubleDType";
    }
    else {
        return "UnknownDType";
    }
}

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
quad_generic_unary_op_strided_loop(PyArrayMethod_Context *context, char *const data[],
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
            sleef_op(&in.sleef_value, &out.sleef_value);
        }
        else {
            longdouble_op(&in.longdouble_value, &out.longdouble_value);
        }
        memcpy(out_ptr, &out, elem_size);

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
             (void *)&quad_generic_unary_op_strided_loop<sleef_op, longdouble_op>},
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
    if (create_quad_unary_ufunc<quad_negative, ld_negative>(numpy, "negative") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_positive, ld_positive>(numpy, "positive") < 0) {
        return -1;
    }
    if (create_quad_unary_ufunc<quad_absolute, ld_absolute>(numpy, "absolute") < 0) {
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
    return 0;
}

// Binary ufuncs

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
    } else {
        target_backend = descr_in1->backend;
    }

    // Set up input descriptors, casting if necessary
    for (int i = 0; i < 2; i++) {
        if (((QuadPrecDTypeObject *)given_descrs[i])->backend != target_backend) {
            loop_descrs[i] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[i]) {
                return (NPY_CASTING)-1;
            }
        } else {
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
    return casting;
}

template <binary_op_quad_def sleef_op, binary_op_longdouble_def longdouble_op>
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

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    quad_value in1, in2, out;
    while (N--) {
        memcpy(&in1, in1_ptr, elem_size);
        memcpy(&in2, in2_ptr, elem_size);
        if (backend == BACKEND_SLEEF) {
            sleef_op(&out.sleef_value, &in1.sleef_value, &in2.sleef_value);
        }
        else {
            longdouble_op(&out.longdouble_value, &in1.longdouble_value, &in2.longdouble_value);
        }
        memcpy(out_ptr, &out, elem_size);

        in1_ptr += in1_stride;
        in2_ptr += in2_stride;
        out_ptr += out_stride;
    }
    return 0;
}

static int
quad_ufunc_promoter(PyUFuncObject *ufunc, PyArray_DTypeMeta *op_dtypes[],
                    PyArray_DTypeMeta *signature[], PyArray_DTypeMeta *new_op_dtypes[])
{
    int nin = ufunc->nin;
    int nargs = ufunc->nargs;
    PyArray_DTypeMeta *common = NULL;
    bool has_quad = false;

    // Handle the special case for reductions
    if (op_dtypes[0] == NULL) {
        assert(nin == 2 && ufunc->nout == 1); /* must be reduction */
        for (int i = 0; i < 3; i++) {
            Py_INCREF(op_dtypes[1]);
            new_op_dtypes[i] = op_dtypes[1];

        }
        return 0;
    }

    // Check if any input or signature is QuadPrecision
    for (int i = 0; i < nin; i++) {

        if (op_dtypes[i] == &QuadPrecDType) {
            has_quad = true;

        }
    }

    if (has_quad) {
        common = &QuadPrecDType;

    }
    else {
        for (int i = nin; i < nargs; i++) {
            if (signature[i] != NULL) {
                if (common == NULL) {
                    Py_INCREF(signature[i]);
                    common = signature[i];

                }
                else if (common != signature[i]) {
                    Py_CLEAR(common);  // Not homogeneous, unset common

                    break;
                }
            }
        }
    }
    // If no common output dtype, use standard promotion for inputs
    if (common == NULL) {
        common = PyArray_PromoteDTypeSequence(nin, op_dtypes);
        if (common == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();  // Do not propagate normal promotion errors
            }

            return -1;
        }

    }

    // Set all new_op_dtypes to the common dtype
    for (int i = 0; i < nargs; i++) {
        if (signature[i]) {
            // If signature is specified for this argument, use it
            Py_INCREF(signature[i]);
            new_op_dtypes[i] = signature[i];
        }
        else {
            // Otherwise, use the common dtype
            Py_INCREF(common);
            new_op_dtypes[i] = common;
        }
    }

    Py_XDECREF(common);

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
             (void *)&quad_generic_binop_strided_loop<sleef_op, longdouble_op>},
            {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_binop",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_IS_REORDERABLE,
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
    if (create_quad_binary_ufunc<quad_pow, ld_pow>(numpy, "power") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_mod, ld_mod>(numpy, "mod") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_minimum, ld_minimum>(numpy, "minimum") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_maximum, ld_maximum>(numpy, "maximum") < 0) {
        return -1;
    }
    if (create_quad_binary_ufunc<quad_atan2, ld_atan2>(numpy, "arctan2") < 0) {
        return -1;
    }
    return 0;
}

// comparison functions

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

    while (N--) {
        if (backend == BACKEND_SLEEF) {
            *((npy_bool *)out_ptr) =
                    sleef_comp((const Sleef_quad *)in1_ptr, (const Sleef_quad *)in2_ptr);
        }
        else {
            *((npy_bool *)out_ptr) =
                    ld_comp((const long double *)in1_ptr, (const long double *)in2_ptr);
        }

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
            {NPY_METH_strided_loop, (void *)&quad_generic_comp_strided_loop<sleef_comp, ld_comp>},
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

    return 0;
}

int
init_quad_umath(void)
{
    PyObject *numpy = PyImport_ImportModule("numpy");
    if (!numpy) {
        PyErr_SetString(PyExc_ImportError, "Failed to import numpy module");
        return -1;
    }

    if (init_quad_unary_ops(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad unary operations");
        goto err;
    }

    if (init_quad_binary_ops(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad binary operations");
        goto err;
    }

    if (init_quad_comps(numpy) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize quad comparison operations");
        goto err;
    }

    Py_DECREF(numpy);
    return 0;

err:
    Py_DECREF(numpy);
    return -1;
}