#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"
}
#include <cstring>
#include <cstdlib>
#include "sleef.h"
#include "sleefquad.h"

#include "quad_common.h"
#include "scalar.h"
#include "casts.h"
#include "dtype.h"
#include "utilities.h"
#include "lock.h"
#include "dragon4.h"
#include "ops.hpp"

#define NUM_CASTS 40  // 18 to_casts + 18 from_casts + 1 quad_to_quad + 1 void_to_quad
#define QUAD_STR_WIDTH 50  // 42 is enough for scientific notation float128, just keeping some buffer

static NPY_CASTING
quad_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                 PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                 QuadPrecDTypeObject *given_descrs[2],
                                 QuadPrecDTypeObject *loop_descrs[2], npy_intp *view_offset)
{
    NPY_CASTING casting = NPY_NO_CASTING;

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
        if (given_descrs[0]->backend != given_descrs[1]->backend) {
            casting = NPY_UNSAFE_CASTING;
        }
    }

    *view_offset = 0;
    return casting;
}

static int
quad_to_quad_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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

    // inter-backend casting
    if (descr_in->backend != descr_out->backend) {
        while (N--) {
            quad_value in_val, out_val;
            if (descr_in->backend == BACKEND_SLEEF) {
                memcpy(&in_val.sleef_value, in_ptr, sizeof(Sleef_quad));
                out_val.longdouble_value = Sleef_cast_to_doubleq1(in_val.sleef_value);
            }
            else {
                memcpy(&in_val.longdouble_value, in_ptr, sizeof(long double));
                out_val.sleef_value = Sleef_cast_from_doubleq1(in_val.longdouble_value);
            }
            memcpy(out_ptr, &out_val,
                   (descr_out->backend == BACKEND_SLEEF) ? sizeof(Sleef_quad)
                                                         : sizeof(long double));
            in_ptr += in_stride;
            out_ptr += out_stride;
        }

        return 0;
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

static int
quad_to_quad_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
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

    // inter-backend casting
    if (descr_in->backend != descr_out->backend) {
        if (descr_in->backend == BACKEND_SLEEF) {
            while (N--) {
                Sleef_quad in_val = *(Sleef_quad *)in_ptr;
                *(long double *)out_ptr = Sleef_cast_to_doubleq1(in_val);
                in_ptr += in_stride;
                out_ptr += out_stride;
            }
        }
        else {
            while (N--) {
                long double in_val = *(long double *)in_ptr;
                *(Sleef_quad *)out_ptr = Sleef_cast_from_doubleq1(in_val);
                in_ptr += in_stride;
                out_ptr += out_stride;
            }
        }

        return 0;
    }

    if (descr_in->backend == BACKEND_SLEEF) {
        while (N--) {
            *(Sleef_quad *)out_ptr = *(Sleef_quad *)in_ptr;
            in_ptr += in_stride;
            out_ptr += out_stride;
        }
    }
    else {
        while (N--) {
            *(long double *)out_ptr = *(long double *)in_ptr;
            in_ptr += in_stride;
            out_ptr += out_stride;
        }
    }

    return 0;
}

static NPY_CASTING
void_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                 PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                 npy_intp *view_offset)
{
    PyErr_SetString(PyExc_TypeError, "Void to QuadPrecision cast is not implemented");
    return (NPY_CASTING)-1;
}

static int
void_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          void *NPY_UNUSED(auxdata))
{
    PyErr_SetString(PyExc_RuntimeError, "void_to_quad_strided_loop should not be called");
    return -1;
}

// Unicode/String to QuadDType casting
static NPY_CASTING
unicode_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                    PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                    npy_intp *view_offset)
{
    if (!PyArray_ISNBO(given_descrs[0]->byteorder)) {
        loop_descrs[0] = PyArray_DescrNewByteorder(given_descrs[0], NPY_NATIVE);
        if (loop_descrs[0] == nullptr) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[0]);
        loop_descrs[0] = given_descrs[0];
    }

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
        if (loop_descrs[1] == nullptr) {
            Py_DECREF(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    return NPY_UNSAFE_CASTING;
}

// Helper function: Convert UCS4 string to quad_value
static inline int
unicode_to_quad_convert(const Py_UCS4 *ucs4_str, npy_intp unicode_size_chars,
                       QuadBackendType backend, quad_value *out_val)
{
    PyObject *unicode_obj = PyUnicode_FromKindAndData(PyUnicode_4BYTE_KIND, ucs4_str, unicode_size_chars);
    if (unicode_obj == NULL) {
        return -1;
    }

    const char *utf8_str = PyUnicode_AsUTF8(unicode_obj);
    if (utf8_str == NULL) {
        Py_DECREF(unicode_obj);
        return -1;
    }
    
    char *endptr;
    int err = NumPyOS_ascii_strtoq(utf8_str, backend, out_val, &endptr);
    
    if (err < 0) {
        PyErr_Format(PyExc_ValueError,
                    "could not convert string to QuadPrecision: np.str_('%s')", utf8_str);
        Py_DECREF(unicode_obj);
        return -1;
    }
    
    // Check that we parsed the entire string (skip trailing whitespace)
    while (ascii_isspace(*endptr)) {
        endptr++;
    }
    
    if (*endptr != '\0') {
        PyErr_Format(PyExc_ValueError,
                    "could not convert string to QuadPrecision: np.str_('%s')", utf8_str);
        Py_DECREF(unicode_obj);
        return -1;
    }
    
    Py_DECREF(unicode_obj);
    return 0;
}

template <bool Aligned>
static int
unicode_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                             npy_intp const dimensions[], npy_intp const strides[],
                             void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)descrs[1];
    QuadBackendType backend = descr_out->backend;

    // Unicode strings are stored as UCS4 (4 bytes per character)
    npy_intp unicode_size_chars = descrs[0]->elsize / 4;

    while (N--) {
        Py_UCS4 *ucs4_str = (Py_UCS4 *)in_ptr;
        quad_value out_val;
        
        if (unicode_to_quad_convert(ucs4_str, unicode_size_chars, backend, &out_val) < 0) {
            return -1;
        }

        store_quad<Aligned>(out_ptr, out_val, backend);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// QuadDType to unicode/string
static NPY_CASTING
quad_to_unicode_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                    PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                    npy_intp *view_offset)
{
    npy_intp required_size_chars = QUAD_STR_WIDTH;
    npy_intp required_size_bytes = required_size_chars * 4;  // UCS4 = 4 bytes per char

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        // Create descriptor with required size
        PyArray_Descr *unicode_descr = PyArray_DescrNewFromType(NPY_UNICODE);
        if (unicode_descr == nullptr) {
            Py_DECREF(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }

        unicode_descr->elsize = required_size_bytes;
        loop_descrs[1] = unicode_descr;
    }
    else {
        // Handle non-native byte order by requesting native byte order
        // NumPy will handle the byte swapping automatically
        if (!PyArray_ISNBO(given_descrs[1]->byteorder)) {
            loop_descrs[1] = PyArray_DescrNewByteorder(given_descrs[1], NPY_NATIVE);
            if (loop_descrs[1] == nullptr) {
                Py_DECREF(loop_descrs[0]);
                return (NPY_CASTING)-1;
            }
        }
        else {
            Py_INCREF(given_descrs[1]);
            loop_descrs[1] = given_descrs[1];
        }
    }

    *view_offset = 0;

    // If target descriptor is wide enough, it's a safe cast
    if (loop_descrs[1]->elsize >= required_size_bytes) {
        return NPY_SAFE_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}

// Helper function: Convert quad to string with adaptive notation
static inline PyObject *
quad_to_string_adaptive(Sleef_quad *sleef_val, npy_intp unicode_size_chars)
{
    // Try positional format first to see if it would fit
    PyObject *positional_str = Dragon4_Positional_QuadDType(
            sleef_val, DigitMode_Unique, CutoffMode_TotalLength, SLEEF_QUAD_DECIMAL_DIG, 0, 1,
            TrimMode_LeaveOneZero, 1, 0);

    if (positional_str == NULL) {
        return NULL;
    }

    const char *pos_str = PyUnicode_AsUTF8(positional_str);
    if (pos_str == NULL) {
        Py_DECREF(positional_str);
        return NULL;
    }

    // no need to scan full, only checking if its longer
    npy_intp pos_len = strnlen(pos_str, unicode_size_chars + 1);

    // If positional format fits, use it; otherwise use scientific notation
    if (pos_len <= unicode_size_chars) {
        return positional_str;  // Keep the positional string
    }
    else {
        Py_DECREF(positional_str);
        // Use scientific notation with full precision
        return Dragon4_Scientific_QuadDType(sleef_val, DigitMode_Unique,
                                           SLEEF_QUAD_DECIMAL_DIG, 0, 1,
                                           TrimMode_LeaveOneZero, 1, 2);
    }
}

template <bool Aligned>
static int
quad_to_unicode_loop(PyArrayMethod_Context *context, char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)descrs[0];
    QuadBackendType backend = descr_in->backend;

    npy_intp unicode_size_chars = descrs[1]->elsize / 4;

    while (N--) {
        quad_value in_val = load_quad<Aligned>(in_ptr, backend);

        // Convert to Sleef_quad for Dragon4
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);

        // Get string representation with adaptive notation
        PyObject *py_str = quad_to_string_adaptive(&sleef_val, unicode_size_chars);
        if (py_str == NULL) {
            return -1;
        }

        const char *temp_str = PyUnicode_AsUTF8(py_str);
        if (temp_str == NULL) {
            Py_DECREF(py_str);
            return -1;
        }

        // Convert char string to UCS4 and store in output
        Py_UCS4 *out_ucs4 = (Py_UCS4 *)out_ptr;
        npy_intp str_len = strnlen(temp_str, unicode_size_chars);
        for (npy_intp i = 0; i < str_len; i++) {
            out_ucs4[i] = (Py_UCS4)temp_str[i];
        }
        for (npy_intp i = str_len; i < unicode_size_chars; i++) {
            out_ucs4[i] = 0;
        }

        Py_DECREF(py_str);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// Bytes to QuadDType casting
static NPY_CASTING
bytes_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                   PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                   npy_intp *view_offset)
{
    // Bytes dtype doesn't have byte order concerns like Unicode
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
        if (loop_descrs[1] == nullptr) {
            Py_DECREF(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    return NPY_UNSAFE_CASTING;
}

// Helper function: Convert bytes string to quad_value
static inline int
bytes_to_quad_convert(const char *bytes_str, npy_intp bytes_size,
                      QuadBackendType backend, quad_value *out_val)
{
    // Create a null-terminated copy since bytes might not be null-terminated
    char *temp_str = (char *)malloc(bytes_size + 1);
    if (temp_str == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    
    memcpy(temp_str, bytes_str, bytes_size);
    
    // Find the actual end (null byte or first occurrence)
    npy_intp actual_len = 0;
    while (actual_len < bytes_size && temp_str[actual_len] != '\0') {
        actual_len++;
    }
    temp_str[actual_len] = '\0';
    
    char *endptr;
    int err = NumPyOS_ascii_strtoq(temp_str, backend, out_val, &endptr);
    
    if (err < 0) {
        PyErr_Format(PyExc_ValueError,
                    "could not convert bytes to QuadPrecision: np.bytes_(%s)", temp_str);
        free(temp_str);
        return -1;
    }
    
    while (ascii_isspace(*endptr)) {
        endptr++;
    }
    
    if (*endptr != '\0') {
        PyErr_Format(PyExc_ValueError,
                    "could not convert bytes to QuadPrecision: np.bytes_(%s)", temp_str);
        free(temp_str);
        return -1;
    }
    
    free(temp_str);
    return 0;
}

template <bool Aligned>
static int
bytes_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                           npy_intp const dimensions[], npy_intp const strides[],
                           void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)descrs[1];
    QuadBackendType backend = descr_out->backend;

    npy_intp bytes_size = descrs[0]->elsize;

    while (N--) {
        quad_value out_val;
        
        if (bytes_to_quad_convert(in_ptr, bytes_size, backend, &out_val) < 0) {
            return -1;
        }

        store_quad<Aligned>(out_ptr, out_val, backend);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// QuadDType to bytes
static NPY_CASTING
quad_to_bytes_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                   PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                   npy_intp *view_offset)
{
    npy_intp required_size_bytes = QUAD_STR_WIDTH;

    if (given_descrs[1] == NULL) {
        PyArray_Descr *new_descr = PyArray_DescrNewFromType(NPY_STRING);
        if (new_descr == NULL) {
            return (NPY_CASTING)-1;
        }
        new_descr->elsize = required_size_bytes;
        loop_descrs[1] = new_descr;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    *view_offset = 0;

    // If target descriptor is wide enough, it's a safe cast
    if (loop_descrs[1]->elsize >= required_size_bytes) {
        return NPY_SAFE_CASTING;
    }
    return NPY_SAME_KIND_CASTING;
}

template <bool Aligned>
static int
quad_to_bytes_loop(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)descrs[0];
    QuadBackendType backend = descr_in->backend;

    npy_intp bytes_size = descrs[1]->elsize;

    while (N--) {
        quad_value in_val = load_quad<Aligned>(in_ptr, backend);
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);
        PyObject *py_str = quad_to_string_adaptive(&sleef_val, bytes_size);
        if (py_str == NULL) {
            return -1;
        }
        const char *temp_str = PyUnicode_AsUTF8(py_str);
        if (temp_str == NULL) {
            Py_DECREF(py_str);
            return -1;
        }

        // Copy string to output buffer, padding with nulls
        strncpy(out_ptr, temp_str, bytes_size);

        Py_DECREF(py_str);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    return 0;
}

// StringDType to QuadDType casting
static NPY_CASTING
stringdtype_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_quaddtype_instance(BACKEND_SLEEF);
        if (loop_descrs[1] == nullptr) {
            Py_DECREF(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    // no notion of fix length, so always unsafe
    return NPY_UNSAFE_CASTING;
}

// Note: StringDType elements are always aligned, so Aligned template parameter
// is kept for API consistency but both versions use the same logic
template <bool Aligned>
static int
stringdtype_to_quad_strided_loop(PyArrayMethod_Context *context, char *const data[],
                                 npy_intp const dimensions[], npy_intp const strides[],
                                 void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    PyArray_StringDTypeObject *str_descr = (PyArray_StringDTypeObject *)descrs[0];
    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)descrs[1];
    QuadBackendType backend = descr_out->backend;

    npy_string_allocator *allocator = NpyString_acquire_allocator(str_descr);

    while (N--) {
        const npy_packed_static_string *ps = (npy_packed_static_string *)in_ptr;
        npy_static_string s = {0, NULL};
        int is_null = NpyString_load(allocator, ps, &s);
        
        if (is_null == -1) {
            NpyString_release_allocator(allocator);
            PyErr_SetString(PyExc_MemoryError, "Failed to load string in StringDType to Quad cast");
            return -1;
        }
        else if (is_null) {
            // Handle null string - use the default string if available, otherwise error
            if (str_descr->has_string_na || str_descr->default_string.buf != NULL) {
                s = str_descr->default_string;
            }
            else {
                NpyString_release_allocator(allocator);
                PyErr_SetString(PyExc_ValueError, "Cannot convert null string to QuadPrecision");
                return -1;
            }
        }

        quad_value out_val;
        if (bytes_to_quad_convert(s.buf, s.size, backend, &out_val) < 0) {
            NpyString_release_allocator(allocator);
            return -1;
        }

        store_quad<Aligned>(out_ptr, out_val, backend);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;
}

// QuadDType to StringDType casting
static NPY_CASTING
quad_to_stringdtype_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                        npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        // Default StringDType() already has coerce=True
        loop_descrs[1] = (PyArray_Descr *)PyObject_CallNoArgs(
                (PyObject *)&PyArray_StringDType);
        if (loop_descrs[1] == NULL) {
            Py_DECREF(loop_descrs[0]);
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    return NPY_SAFE_CASTING;
}

// Note: StringDType elements are always aligned, so Aligned template parameter
// is kept for API consistency but both versions use the same logic
template <bool Aligned>
static int
quad_to_stringdtype_strided_loop(PyArrayMethod_Context *context, char *const data[],
                                 npy_intp const dimensions[], npy_intp const strides[],
                                 void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    PyArray_Descr *const *descrs = context->descriptors;
    QuadPrecDTypeObject *descr_in = (QuadPrecDTypeObject *)descrs[0];
    PyArray_StringDTypeObject *str_descr = (PyArray_StringDTypeObject *)descrs[1];
    QuadBackendType backend = descr_in->backend;

    npy_string_allocator *allocator = NpyString_acquire_allocator(str_descr);

    while (N--) {
        quad_value in_val = load_quad<Aligned>(in_ptr, backend);
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);

        // Get string representation with adaptive notation
        // Use a large buffer size to allow for full precision
        PyObject *py_str = quad_to_string_adaptive(&sleef_val, QUAD_STR_WIDTH);
        if (py_str == NULL) {
            NpyString_release_allocator(allocator);
            return -1;
        }

        Py_ssize_t str_size;
        const char *str_buf = PyUnicode_AsUTF8AndSize(py_str, &str_size);
        if (str_buf == NULL) {
            Py_DECREF(py_str);
            NpyString_release_allocator(allocator);
            return -1;
        }

        npy_packed_static_string *out_ps = (npy_packed_static_string *)out_ptr;
        if (NpyString_pack(allocator, out_ps, str_buf, (size_t)str_size) < 0) {
            Py_DECREF(py_str);
            NpyString_release_allocator(allocator);
            PyErr_SetString(PyExc_MemoryError, "Failed to pack string in Quad to StringDType cast");
            return -1;
        }

        Py_DECREF(py_str);

        in_ptr += in_stride;
        out_ptr += out_stride;
    }

    NpyString_release_allocator(allocator);
    return 0;
}

// Tag dispatching to ensure npy_bool/npy_ubyte and npy_half/npy_ushort do not alias in templates
// see e.g. https://stackoverflow.com/q/32522279
struct spec_npy_bool {};
struct spec_npy_half {};

template <typename T>
struct NpyType {
    typedef T TYPE;
};
template <>
struct NpyType<spec_npy_bool> {
    typedef npy_bool TYPE;
};
template <>
struct NpyType<spec_npy_half> {
    typedef npy_half TYPE;
};

// Casting from other types to QuadDType

template <typename T>
static inline quad_value
to_quad(typename NpyType<T>::TYPE x, QuadBackendType backend);

template <>
inline quad_value
to_quad<spec_npy_bool>(npy_bool x, QuadBackendType backend)
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
to_quad<npy_ubyte>(npy_ubyte x, QuadBackendType backend)
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
to_quad<spec_npy_half>(npy_half x, QuadBackendType backend)
{
    quad_value result;
    if (backend == BACKEND_SLEEF) {
        result.sleef_value = Sleef_cast_from_doubleq1(npy_half_to_double(x));
    }
    else {
        result.longdouble_value = (long double)npy_half_to_double(x);
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
    // todo: here it is converting this to SLEEF, losing data and getting 0
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
numpy_to_quad_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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
        typename NpyType<T>::TYPE in_val;
        quad_value out_val;

        memcpy(&in_val, in_ptr, sizeof(typename NpyType<T>::TYPE));
        out_val = to_quad<T>(in_val, backend);
        memcpy(out_ptr, &out_val, elem_size);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

template <typename T>
static int
numpy_to_quad_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                   npy_intp const dimensions[], npy_intp const strides[],
                                   void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)context->descriptors[1];
    QuadBackendType backend = descr_out->backend;

    while (N--) {
        typename NpyType<T>::TYPE in_val = *(typename NpyType<T>::TYPE *)in_ptr;
        quad_value out_val = to_quad<T>(in_val, backend);

        if (backend == BACKEND_SLEEF) {
            *(Sleef_quad *)(out_ptr) = out_val.sleef_value;
        }
        else {
            *(long double *)(out_ptr) = out_val.longdouble_value;
        }

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

// Casting from QuadDType to other types

template <typename T>
static inline typename NpyType<T>::TYPE
from_quad(quad_value x, QuadBackendType backend);

template <>
inline npy_bool
from_quad<spec_npy_bool>(quad_value x, QuadBackendType backend)
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
    // runtime warnings often comes from/to casting of NaN, inf
    // casting is used by ops at several positions leading to warnings
    // fix can be catching the cases and returning corresponding type value without casting
    if (backend == BACKEND_SLEEF) {
        return (npy_byte)Sleef_cast_to_int64q1(x.sleef_value);
    }
    else {
        return (npy_byte)x.longdouble_value;
    }
}

template <>
inline npy_ubyte
from_quad<npy_ubyte>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ubyte)Sleef_cast_to_uint64q1(x.sleef_value);
    }
    else {
        return (npy_ubyte)x.longdouble_value;
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
inline npy_half
from_quad<spec_npy_half>(quad_value x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return npy_double_to_half(Sleef_cast_to_doubleq1(x.sleef_value));
    }
    else {
        return npy_double_to_half((double)x.longdouble_value);
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
quad_to_numpy_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
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

        typename NpyType<T>::TYPE out_val = from_quad<T>(in_val, backend);
        memcpy(out_ptr, &out_val, sizeof(typename NpyType<T>::TYPE));

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

template <typename T>
static int
quad_to_numpy_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                   npy_intp const dimensions[], npy_intp const strides[],
                                   void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = quad_descr->backend;

    while (N--) {
        quad_value in_val;
        if (backend == BACKEND_SLEEF) {
            in_val.sleef_value = *(Sleef_quad *)in_ptr;
        }
        else {
            in_val.longdouble_value = *(long double *)in_ptr;
        }

        typename NpyType<T>::TYPE out_val = from_quad<T>(in_val, backend);
        *(typename NpyType<T>::TYPE *)(out_ptr) = out_val;

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
    else {
        delete[] spec->dtypes;
        delete[] spec->slots;
        delete spec;
    }
}

// functions to add casts
template <typename T>
void
add_cast_from(PyArray_DTypeMeta *to)
{
    PyArray_DTypeMeta **dtypes = new PyArray_DTypeMeta *[2]{&QuadPrecDType, to};

    PyType_Slot *slots = new PyType_Slot[]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_numpy_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&quad_to_numpy_strided_loop_aligned<T>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_numpy_strided_loop_unaligned<T>},
            {0, nullptr}};

    PyArrayMethod_Spec *spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_NumPy",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
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

    PyType_Slot *slots = new PyType_Slot[]{
            {NPY_METH_resolve_descriptors, (void *)&numpy_to_quad_resolve_descriptors<T>},
            {NPY_METH_strided_loop, (void *)&numpy_to_quad_strided_loop_aligned<T>},
            {NPY_METH_unaligned_strided_loop, (void *)&numpy_to_quad_strided_loop_unaligned<T>},
            {0, nullptr}};

    PyArrayMethod_Spec *spec = new PyArrayMethod_Spec{
            .name = "cast_NumPy_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    add_spec(spec);
}

PyArrayMethod_Spec **
init_casts_internal(void)
{
    PyArray_DTypeMeta **quad2quad_dtypes = new PyArray_DTypeMeta *[2]{nullptr, nullptr};
    PyType_Slot *quad2quad_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_to_quad_strided_loop_aligned},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_quad_strided_loop_unaligned},
            {0, nullptr}};

    PyArrayMethod_Spec *quad2quad_spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,  // since SLEEF -> ld might lose precision
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = quad2quad_dtypes,
            .slots = quad2quad_slots,
    };

    add_spec(quad2quad_spec);

    PyArray_DTypeMeta **void_dtypes =
            new PyArray_DTypeMeta *[2]{&PyArray_VoidDType, &QuadPrecDType};
    PyType_Slot *void_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&void_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&void_to_quad_strided_loop},
            {NPY_METH_unaligned_strided_loop, (void *)&void_to_quad_strided_loop},
            {0, nullptr}};

    PyArrayMethod_Spec *void_spec = new PyArrayMethod_Spec{
            .name = "cast_Void_to_QuadPrec_ERROR",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = void_dtypes,
            .slots = void_slots,
    };
    add_spec(void_spec);

    add_cast_to<spec_npy_bool>(&PyArray_BoolDType);
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
    add_cast_to<spec_npy_half>(&PyArray_HalfDType);
    add_cast_to<float>(&PyArray_FloatDType);
    add_cast_to<double>(&PyArray_DoubleDType);
    add_cast_to<long double>(&PyArray_LongDoubleDType);

    add_cast_from<spec_npy_bool>(&PyArray_BoolDType);
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
    add_cast_from<spec_npy_half>(&PyArray_HalfDType);
    add_cast_from<float>(&PyArray_FloatDType);
    add_cast_from<double>(&PyArray_DoubleDType);
    add_cast_from<long double>(&PyArray_LongDoubleDType);

    // Unicode/String to QuadPrecision cast
    PyArray_DTypeMeta **unicode_to_quad_dtypes = new PyArray_DTypeMeta *[2]{&PyArray_UnicodeDType, &QuadPrecDType};
    PyType_Slot *unicode_to_quad_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&unicode_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&unicode_to_quad_strided_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&unicode_to_quad_strided_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *unicode_to_quad_spec = new PyArrayMethod_Spec{
            .name = "cast_Unicode_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = unicode_to_quad_dtypes,
            .slots = unicode_to_quad_slots,
    };
    add_spec(unicode_to_quad_spec);

    // QuadPrecision to Unicode
    PyArray_DTypeMeta **quad_to_unicode_dtypes = new PyArray_DTypeMeta *[2]{&QuadPrecDType, &PyArray_UnicodeDType};
    PyType_Slot *quad_to_unicode_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_unicode_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_to_unicode_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_unicode_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *quad_to_unicode_spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_Unicode",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = quad_to_unicode_dtypes,
            .slots = quad_to_unicode_slots,
    };
    add_spec(quad_to_unicode_spec);

    // Bytes to QuadPrecision cast
    PyArray_DTypeMeta **bytes_to_quad_dtypes = new PyArray_DTypeMeta *[2]{&PyArray_BytesDType, &QuadPrecDType};
    PyType_Slot *bytes_to_quad_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&bytes_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&bytes_to_quad_strided_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&bytes_to_quad_strided_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *bytes_to_quad_spec = new PyArrayMethod_Spec{
            .name = "cast_Bytes_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = bytes_to_quad_dtypes,
            .slots = bytes_to_quad_slots,
    };
    add_spec(bytes_to_quad_spec);

    // QuadPrecision to Bytes
    PyArray_DTypeMeta **quad_to_bytes_dtypes = new PyArray_DTypeMeta *[2]{&QuadPrecDType, &PyArray_BytesDType};
    PyType_Slot *quad_to_bytes_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_bytes_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_to_bytes_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_bytes_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *quad_to_bytes_spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_Bytes",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = quad_to_bytes_dtypes,
            .slots = quad_to_bytes_slots,
    };
    add_spec(quad_to_bytes_spec);

    // StringDType to QuadPrecision cast
    PyArray_DTypeMeta **stringdtype_to_quad_dtypes = new PyArray_DTypeMeta *[2]{&PyArray_StringDType, &QuadPrecDType};
    PyType_Slot *stringdtype_to_quad_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&stringdtype_to_quad_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&stringdtype_to_quad_strided_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&stringdtype_to_quad_strided_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *stringdtype_to_quad_spec = new PyArrayMethod_Spec{
            .name = "cast_StringDType_to_QuadPrec",
            .nin = 1,
            .nout = 1,
            .casting = NPY_UNSAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = stringdtype_to_quad_dtypes,
            .slots = stringdtype_to_quad_slots,
    };
    add_spec(stringdtype_to_quad_spec);

    // QuadPrecision to StringDType cast
    PyArray_DTypeMeta **quad_to_stringdtype_dtypes = new PyArray_DTypeMeta *[2]{&QuadPrecDType, &PyArray_StringDType};
    PyType_Slot *quad_to_stringdtype_slots = new PyType_Slot[4]{
            {NPY_METH_resolve_descriptors, (void *)&quad_to_stringdtype_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_to_stringdtype_strided_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_stringdtype_strided_loop<false>},
            {0, nullptr}};

    PyArrayMethod_Spec *quad_to_stringdtype_spec = new PyArrayMethod_Spec{
            .name = "cast_QuadPrec_to_StringDType",
            .nin = 1,
            .nout = 1,
            .casting = NPY_SAFE_CASTING,
            .flags = static_cast<NPY_ARRAYMETHOD_FLAGS>(NPY_METH_SUPPORTS_UNALIGNED | NPY_METH_REQUIRES_PYAPI),
            .dtypes = quad_to_stringdtype_dtypes,
            .slots = quad_to_stringdtype_slots,
    };
    add_spec(quad_to_stringdtype_spec);

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
    for (size_t i = 0; i < spec_count; i++) {
        if (specs[i]) {
            delete[] specs[i]->dtypes;
            delete[] specs[i]->slots;
            delete specs[i];
            specs[i] = nullptr;
        }
    }
    spec_count = 0;
}
