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
#include <vector>

#include "scalar.h"
#include "casts.h"
#include "dtype.h"

static NPY_CASTING
quad_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                 PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                 QuadPrecDTypeObject *given_descrs[2],
                                 QuadPrecDTypeObject *loop_descrs[2], npy_intp *view_offset)
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

    *view_offset = 0;
    return NPY_SAME_KIND_CASTING;
}

static int
quad_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    while (N--) {
        Sleef_quad *in = (Sleef_quad *)in_ptr;
        Sleef_quad *out = (Sleef_quad *)out_ptr;

        *out = *in;

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

// Casting from other types to QuadDType

template <typename T>
static inline Sleef_quad
to_quad(T x);

template <>
inline Sleef_quad
to_quad<npy_bool>(npy_bool x)
{
    return x ? Sleef_cast_from_doubleq1(1.0) : Sleef_cast_from_doubleq1(0.0);
}
template <>
inline Sleef_quad
to_quad<npy_byte>(npy_byte x)
{
    return Sleef_cast_from_int64q1(x);
}
// template <>
// inline Sleef_quad
// to_quad<npy_ubyte>(npy_ubyte x)
// {
//     return Sleef_cast_from_uint64q1(x);
// }
template <>
inline Sleef_quad
to_quad<npy_short>(npy_short x)
{
    return Sleef_cast_from_int64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_ushort>(npy_ushort x)
{
    return Sleef_cast_from_uint64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_int>(npy_int x)
{
    return Sleef_cast_from_int64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_uint>(npy_uint x)
{
    return Sleef_cast_from_uint64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_long>(npy_long x)
{
    return Sleef_cast_from_int64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_ulong>(npy_ulong x)
{
    return Sleef_cast_from_uint64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_longlong>(npy_longlong x)
{
    return Sleef_cast_from_int64q1(x);
}
template <>
inline Sleef_quad
to_quad<npy_ulonglong>(npy_ulonglong x)
{
    return Sleef_cast_from_uint64q1(x);
}
template <>
inline Sleef_quad
to_quad<float>(float x)
{
    return Sleef_cast_from_doubleq1(x);
}
template <>
inline Sleef_quad
to_quad<double>(double x)
{
    return Sleef_cast_from_doubleq1(x);
}
template <>
inline Sleef_quad
to_quad<long double>(long double x)
{
    return Sleef_cast_from_doubleq1(x);
}

template <typename T>
static NPY_CASTING
numpy_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                  PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                  npy_intp *view_offset)
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_quaddtype_instance();
        if (loop_descrs[1] == nullptr) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    loop_descrs[0] = PyArray_GetDefaultDescr(dtypes[0]);
    *view_offset = 0;
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

    while (N--) {
        T in_val = *(T *)in_ptr;
        Sleef_quad *out_val = (Sleef_quad *)out_ptr;
        *out_val = to_quad<T>(in_val);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

// Casting from QuadDType to other types

template <typename T>
static inline T
from_quad(Sleef_quad x);

template <>
inline npy_bool
from_quad<npy_bool>(Sleef_quad x)
{
    return Sleef_cast_to_int64q1(x) != 0;
}
template <>
inline npy_byte
from_quad<npy_byte>(Sleef_quad x)
{
    return (npy_byte)Sleef_cast_to_int64q1(x);
}
// template <>
// inline npy_ubyte
// from_quad<npy_ubyte>(Sleef_quad x)
// {
//     return (npy_ubyte)Sleef_cast_to_uint64q1(x);
// }
template <>
inline npy_short
from_quad<npy_short>(Sleef_quad x)
{
    return (npy_short)Sleef_cast_to_int64q1(x);
}
template <>
inline npy_ushort
from_quad<npy_ushort>(Sleef_quad x)
{
    return (npy_ushort)Sleef_cast_to_uint64q1(x);
}
template <>
inline npy_int
from_quad<npy_int>(Sleef_quad x)
{
    return (npy_int)Sleef_cast_to_int64q1(x);
}
template <>
inline npy_uint
from_quad<npy_uint>(Sleef_quad x)
{
    return (npy_uint)Sleef_cast_to_uint64q1(x);
}
template <>
inline npy_long
from_quad<npy_long>(Sleef_quad x)
{
    return (npy_long)Sleef_cast_to_int64q1(x);
}
template <>
inline npy_ulong
from_quad<npy_ulong>(Sleef_quad x)
{
    return (npy_ulong)Sleef_cast_to_uint64q1(x);
}
template <>
inline npy_longlong
from_quad<npy_longlong>(Sleef_quad x)
{
    return Sleef_cast_to_int64q1(x);
}
template <>
inline npy_ulonglong
from_quad<npy_ulonglong>(Sleef_quad x)
{
    return Sleef_cast_to_uint64q1(x);
}
template <>
inline float
from_quad<float>(Sleef_quad x)
{
    return Sleef_cast_to_doubleq1(x);
}
template <>
inline double
from_quad<double>(Sleef_quad x)
{
    return Sleef_cast_to_doubleq1(x);
}
template <>
inline long double
from_quad<long double>(Sleef_quad x)
{
    return Sleef_cast_to_doubleq1(x);
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
    *view_offset = 0;
    return NPY_SAME_KIND_CASTING;
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

    while (N--) {
        Sleef_quad in_val = *(Sleef_quad *)in_ptr;
        T *out_val = (T *)out_ptr;
        *out_val = from_quad<T>(in_val);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

static std::vector<PyArrayMethod_Spec *> specs;

// functions to add casts
template <typename T>
void
add_cast_from(PyArray_DTypeMeta *to)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{nullptr, to};

    PyType_Slot *slots = new PyType_Slot[3]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_numpy_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&quad_to_numpy_strided_loop<T>},
            {0, nullptr}};

    specs.push_back(new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_NumPy",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAME_KIND_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)0,
            .dtypes = dtypes,
            .slots = slots,
    });
}

template <typename T>
void
add_cast_to(PyArray_DTypeMeta *from)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{from, nullptr};

    PyType_Slot *slots = new PyType_Slot[3]{
            {NPY_METH_resolve_descriptors, (void *)&numpy_to_quad_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&numpy_to_quad_strided_loop<T>},
            {0, nullptr}};

    specs.push_back(new PyArrayMethod_Spec{
            .name = "cast_NumPy_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = (NPY_ARRAYMETHOD_FLAGS)0,
            .dtypes = dtypes,
            .slots = slots,
    });
}

PyArrayMethod_Spec **
init_casts_internal(void)
{
    PyArray_DTypeMeta **quad2quad_dtypes = new PyArray_DTypeMeta *[2]{nullptr, nullptr};

    specs.push_back(new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAME_KIND_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = quad2quad_dtypes,
            .slots = new PyType_Slot[4]{
                    {NPY_METH_resolve_descriptors, (void *)&quad_to_quad_resolve_descriptors},
                    {NPY_METH_strided_loop, (void *)&quad_to_quad_strided_loop},
                    {NPY_METH_unaligned_strided_loop, (void *)&quad_to_quad_strided_loop},
                    {0, NULL}}});

    add_cast_to<npy_bool>(&PyArray_BoolDType);
    add_cast_to<npy_byte>(&PyArray_ByteDType);
    add_cast_to<npy_ubyte>(&PyArray_UByteDType);
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
    add_cast_from<npy_ubyte>(&PyArray_UByteDType);
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

    specs.push_back(nullptr);
    return specs.data();
}

PyArrayMethod_Spec **
init_casts(void)
{
    try {
        return init_casts_internal();
    }
    catch (const std::exception &e) {
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
        delete cast->dtypes;
        delete cast->slots;
        delete cast;
    }
    specs.clear();
}
