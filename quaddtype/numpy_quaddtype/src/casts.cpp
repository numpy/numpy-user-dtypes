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
#include "numpy/dtype_api.h"
}
#include "sleef.h"
#include "sleefquad.h"

#include "quad_common.h"
#include "scalar.h"
#include "casts.h"
#include "dtype.h"

#define NUM_CASTS 29  // 14 to_casts + 14 from_casts + 1 quad_to_quad

static NPY_CASTING
quad_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                 PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                 QuadPrecDTypeObject *given_descrs[2],
                                 QuadPrecDTypeObject *loop_descrs[2], npy_intp *view_offset)
{
    if (given_descrs[0]->backend != given_descrs[1]->backend) {
        return NPY_UNSAFE_CASTING;
    }

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

    *view_offset = 0;
    return NPY_NO_CASTING;
}

static int
quad_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)context->descriptors[1];

    if (descr_in->backend != descr_out->backend) {
        PyErr_SetString(PyExc_TypeError,
                        "Cannot convert between different quad-precision backends");
        return -1;
    }

    size_t elem_size =
            (descr_in->backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    while (N--) {
        memcpy(out_ptr, in_ptr, elem_size);
        in_ptr += in_stride;
        out_ptr += out_stride;
    }
    return 0;
}

// Casting from other types to QuadDType

template <typename T>
static inline quad_value
to_quad(T x, QuadBackendType backend);

template <>
inline quad_value
to_quad<npy_bool>(npy_bool x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = x ? Sleef_cast_from_doubleq1(1.0) : Sleef_cast_from_doubleq1(0.0);
    }
    else {
        result.longdouble_value = x ? 1.0L : 0.0L;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_byte>(npy_byte x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_int64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_short>(npy_short x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_int64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_ushort>(npy_ushort x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_uint64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_int>(npy_int x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_int64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_uint>(npy_uint x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_uint64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_long>(npy_long x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_int64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_ulong>(npy_ulong x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_uint64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_longlong>(npy_longlong x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_int64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<npy_ulonglong>(npy_ulonglong x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_uint64q1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}
template <>
inline quad_value
to_quad<float>(float x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_doubleq1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<double>(double x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_doubleq1(x);
    }
    else {
        result.longdouble_value = (long double)x;
    }
    return result;
}

template <>
inline quad_value
to_quad<long double>(long double x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_doubleq1(x);
    }
    else {
        result.longdouble_value = x;
    }
    return result;
}

template <typename T>
static NPY_CASTING
numpy_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                  PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                  npy_intp *view_offset)
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
        if (loop_descrs[1] == nullptr) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    loop_descrs[0] = PyArray_GetDefaultDescr(dtypes[0]);
    return NPY_SAFE_CASTING;
}

template <typename T>
static int
numpy_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                           npy_intp const dimensions[], npy_intp const strides[],
                           void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)context->descriptors[1];
    QuadBackendType backend = descr_out->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    while (N--) {
        T in_val;
        quad_value out_val;

        memcpy(&in_val, in_ptr, sizeof(T));
        out_val = to_quad<T>(in_val, backend);
        memcpy(out_ptr, &out_val, elem_size);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

// Casting from QuadDType to other types

template <typename T>
static inline T
from_quad(quad_value x, QuadBackendType backend);

template <>
inline npy_bool
from_quad<npy_bool>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_int64q1(x.sleef_value) != 0;
    }
    else {
        return x.longdouble_value != 0;
    }
}

template <>
inline npy_byte
from_quad<npy_byte>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_byte)Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_byte)x.longdouble_value;
    }
}

template <>
inline npy_short
from_quad<npy_short>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_short)Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_short)x.longdouble_value;
    }
}

template <>
inline npy_ushort
from_quad<npy_ushort>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ushort)Sleef_cast_to_uint64q1(x.sleef_value);
    }
    else {
        return (npy_ushort)x.longdouble_value;
    }
}

template <>
inline npy_int
from_quad<npy_int>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_int)Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_int)x.longdouble_value;
    }
}

template <>
inline npy_uint
from_quad<npy_uint>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_uint)Sleef_cast_to_uint64q1(x.sleef_value);
    }
    else {
        return (npy_uint)x.longdouble_value;
    }
}

template <>
inline npy_long
from_quad<npy_long>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_long)Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_long)x.longdouble_value;
    }
}

template <>
inline npy_ulong
from_quad<npy_ulong>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ulong)Sleef_cast_to_uint64q1(x.sleef_value);
    }
    else {
        return (npy_ulong)x.longdouble_value;
    }
}

template <>
inline npy_longlong
from_quad<npy_longlong>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_longlong)x.longdouble_value;
    }
}

template <>
inline npy_ulonglong
from_quad<npy_ulonglong>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_uint64q1(x.sleef_value);
    }
    else {
        return (npy_ulonglong)x.longdouble_value;
    }
}

template <>
inline float
from_quad<float>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (float)Sleef_cast_to_doubleq1(x.sleef_value);
    }
    else {
        return (float)x.longdouble_value;
    }
}

template <>
inline double
from_quad<double>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_doubleq1(x.sleef_value);
    }
    else {
        return (double)x.longdouble_value;
    }
}

template <>
inline long double
from_quad<long double>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (long double)Sleef_cast_to_doubleq1(x.sleef_value);
    }
    else {
        return x.longdouble_value;
    }
}

template <typename T>
static NPY_CASTING
quad_to_numpy_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                  PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                  npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    loop_descrs[1] = PyArray_GetDefaultDescr(dtypes[1]);
    return NPY_UNSAFE_CASTING;
}

template <typename T>
static int
quad_to_numpy_strided_loop(PyArrayMethod_Context *context, char *const data[],
                           npy_intp const dimensions[], npy_intp const strides[],
                           void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = quad_descr->backend;

    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    while (N--) {
        quad_value in_val;
        memcpy(&in_val, in_ptr, elem_size);

        T out_val = from_quad<T>(in_val, backend);
        memcpy(out_ptr, &out_val, sizeof(T));

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

static PyArrayMethod_Spec *specs[NUM_CASTS + 1];  // +1 for NULL terminator
static size_t spec_count = 0;

void
add_spec(PyArrayMethod_Spec *spec)
{
    if (spec_count < NUM_CASTS) {
        specs[spec_count++] = spec;
    }
}

// functions to add casts
template <typename T>
void
add_cast_from(PyArray_DTypeMeta *to)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{&QuadPrecDType, to};

    PyType_Slot *slots = new PyType_Slot[3]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_numpy_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&quad_to_numpy_strided_loop<T>},
            {0, nullptr}};

    PyArrayMethod_Spec *spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_NumPy",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)0,
            .dtypes = dtypes,
            .slots = slots,
    };
    add_spec(spec);
}

template <typename T>
void
add_cast_to(PyArray_DTypeMeta *from)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{from, &QuadPrecDType};

    PyType_Slot *slots = new PyType_Slot[3]{
            {NPY_METH_resolve_descriptors, (void *)&numpy_to_quad_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&numpy_to_quad_strided_loop<T>},
            {0, nullptr}};

    PyArrayMethod_Spec *spec = new PyArrayMethod_Spec{
            .name = "cast_NumPy_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)0,
            .dtypes = dtypes,
            .slots = slots,
    };

    add_spec(spec);
}

PyArrayMethod_Spec **
init_casts_internal(void)
{
    PyArray_DTypeMeta **quad2quad_dtypes =
            new PyArray_DTypeMeta *[2]{&QuadPrecDType, &QuadPrecDType};
    PyType_Slot *quad2quad_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_to_quad_strided_loop},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_quad_strided_loop},
            {0, nullptr}};

    PyArrayMethod_Spec *quad2quad_spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = quad2quad_dtypes,
            .slots = quad2quad_slots,
    };

    add_spec(quad2quad_spec);

    add_cast_to<npy_bool>(&PyArray_BoolDType);
    add_cast_to<npy_byte>(&PyArray_ByteDType);
    add_cast_to<npy_short>(&PyArray_ShortDType);
    add_cast_to<npy_ushort>(&PyArray_UShortDType);
    add_cast_to<npy_int>(&PyArray_IntDType);
    add_cast_to<npy_uint>(&PyArray_UIntDType);
    add_cast_to<npy_long>(&PyArray_LongDType);
    add_cast_to<npy_ulong>(&PyArray_ULongDType);
    add_cast_to<npy_longlong>(&PyArray_LongLongDType);
    add_cast_to<npy_ulonglong>(&PyArray_ULongLongDType);
    add_cast_to<float>(&PyArray_FloatDType);
    add_cast_to<double>(&PyArray_DoubleDType);
    add_cast_to<long double>(&PyArray_LongDoubleDType);

    add_cast_from<npy_bool>(&PyArray_BoolDType);
    add_cast_from<npy_byte>(&PyArray_ByteDType);
    add_cast_from<npy_short>(&PyArray_ShortDType);
    add_cast_from<npy_ushort>(&PyArray_UShortDType);
    add_cast_from<npy_int>(&PyArray_IntDType);
    add_cast_from<npy_uint>(&PyArray_UIntDType);
    add_cast_from<npy_long>(&PyArray_LongDType);
    add_cast_from<npy_ulong>(&PyArray_ULongDType);
    add_cast_from<npy_longlong>(&PyArray_LongLongDType);
    add_cast_from<npy_ulonglong>(&PyArray_ULongLongDType);
    add_cast_from<float>(&PyArray_FloatDType);
    add_cast_from<double>(&PyArray_DoubleDType);
    add_cast_from<long double>(&PyArray_LongDoubleDType);

    specs[spec_count] = nullptr;
    return specs;
}

PyArrayMethod_Spec **
init_casts(void)
{
    try {
        return init_casts_internal();
    }
    catch (int e) {
        PyErr_NoMemory();
        return nullptr;
    }
}

void
free_casts(void)
{
    for (auto cast : specs) {
        if (cast == nullptr) {
            continue;
        }
        delete[] cast->dtypes;
        delete[] cast->slots;
        delete cast;
    }
    spec_count = 0;
}