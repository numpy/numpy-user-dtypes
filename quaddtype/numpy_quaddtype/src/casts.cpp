#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
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
#include <type_traits>
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
#include "constants.hpp"

#define NUM_CASTS 40  // 18 to_casts + 18 from_casts + 1 quad_to_quad + 1 void_to_quad
#define QUAD_STR_WIDTH 50  // 42 is enough for scientific notation float128, just keeping some buffer

// forward declarations
static inline const char *
quad_to_string_adaptive_cstr(Sleef_quad *sleef_val, npy_intp unicode_size_chars);

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
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    Py_INCREF(given_descrs[1]);
    loop_descrs[1] = given_descrs[1];

    if (given_descrs[0]->backend != given_descrs[1]->backend) {
        // Different backends require actual conversion, no view possible
        *view_offset = NPY_MIN_INTP;
        if (given_descrs[0]->backend == BACKEND_SLEEF) {
            // SLEEF -> long double may lose precision
            return static_cast<NPY_CASTING>(NPY_SAME_KIND_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
        }
        // long double -> SLEEF preserves value exactly
        return static_cast<NPY_CASTING>(NPY_SAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
    }

    *view_offset = 0;
    return static_cast<NPY_CASTING>(NPY_NO_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
}

// Helper function for quad-to-quad same_value check (inter-backend)
// NOTE: the inter-backend uses `double` as intermediate,
// so only values that can be exactly represented in double can pass same_value check
static inline int
quad_to_quad_same_value_check(const quad_value *in_val, QuadBackendType backend_in,
                              const quad_value *out_val, QuadBackendType backend_out)
{
    // Convert output back to input backend for comparison
    quad_value roundtrip;
    
    if (backend_in == BACKEND_SLEEF) {
        // Input was SLEEF, output is longdouble
        // Convert longdouble back to SLEEF for comparison
        long double ld = out_val->longdouble_value;
        if (std::isnan(ld)) {
            // Preserve sign of NaN
            roundtrip.sleef_value = (!ld_signbit(&ld)) ? QUAD_PRECISION_NAN : QUAD_PRECISION_NEG_NAN;
        }
        else if (std::isinf(ld)) {
            roundtrip.sleef_value = (ld > 0) ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
        }
        else {
            Sleef_quad temp = Sleef_cast_from_doubleq1(static_cast<double>(ld));
            memcpy(&roundtrip.sleef_value, &temp, sizeof(Sleef_quad));
        }
        
        // Compare in SLEEF domain && signbit preserved
        bool is_sign_preserved = (quad_signbit(&in_val->sleef_value) == quad_signbit(&roundtrip.sleef_value));
        if(quad_isnan(&in_val->sleef_value) && quad_isnan(&roundtrip.sleef_value) && is_sign_preserved)
            return 1;  // Both NaN
        if (Sleef_icmpeqq1(in_val->sleef_value, roundtrip.sleef_value) && is_sign_preserved)
            return 1;  // Equal
    }
    else {
        // Input was longdouble, output is SLEEF
        // Convert SLEEF back to longdouble for comparison
        roundtrip.longdouble_value = static_cast<long double>(cast_sleef_to_double(out_val->sleef_value));
        
        // Compare in longdouble domain && signbit preserved
        bool is_sign_preserved = (ld_signbit(&in_val->longdouble_value) == ld_signbit(&roundtrip.longdouble_value));
        if ((std::isnan(in_val->longdouble_value) && std::isnan(roundtrip.longdouble_value)) && is_sign_preserved)
            return 1;
        if ((in_val->longdouble_value == roundtrip.longdouble_value) && is_sign_preserved)
            return 1;
    }
    
    // Values don't match
    Sleef_quad sleef_val = quad_to_sleef_quad(in_val, backend_in);
    const char *val_str = quad_to_string_adaptive_cstr(&sleef_val, QUAD_STR_WIDTH);
    if (val_str != NULL) {
        PyErr_Format(PyExc_ValueError,
                     "QuadPrecision value '%s' cannot be represented exactly in target backend",
                     val_str);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "QuadPrecision value cannot be represented exactly in target backend");
    }
    return -1;
}

template <bool Aligned>
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
    QuadBackendType backend_in = descr_in->backend;
    QuadBackendType backend_out = descr_out->backend;
    int same_value_casting = ((context->flags & NPY_SAME_VALUE_CONTEXT_FLAG) == NPY_SAME_VALUE_CONTEXT_FLAG);

    // inter-backend casting
    if (backend_in != backend_out) {
        while (N--) {
            quad_value in_val;
            load_quad<Aligned>(in_ptr, backend_in, &in_val);
            quad_value out_val;
            if (backend_in == BACKEND_SLEEF) 
            {
              out_val.longdouble_value = static_cast<long double>(cast_sleef_to_double(in_val.sleef_value));
            }
            else 
            {
              long double ld = in_val.longdouble_value;
              if (std::isnan(ld)) {
                  out_val.sleef_value = (!ld_signbit(&ld)) ? QUAD_PRECISION_NAN : QUAD_PRECISION_NEG_NAN;
              }
              else if (std::isinf(ld)) {
                  out_val.sleef_value = (ld > 0) ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
              }
              else 
              {
                  // to prevent compiler optimizations, ABI handling issues with __float128 on x86-64 machines
                  // won't be expensive as for fixed size compiler can optimize memcpy with movq
                  Sleef_quad temp = Sleef_cast_from_doubleq1(static_cast<double>(ld));
                  std::memcpy(&out_val.sleef_value, &temp, sizeof(Sleef_quad));
              }
            }
            
            // check same_value for inter-backend casts
            if(same_value_casting)
            {
              int ret = quad_to_quad_same_value_check(&in_val, backend_in, &out_val, backend_out);
              if (ret < 0) {
                  return -1;
              }
            }

            store_quad<Aligned>(out_ptr, &out_val, backend_out);
            in_ptr += in_stride;
            out_ptr += out_stride;
        }
        return 0;
    }

    // same backend: direct copy
    // same_value casting not needed here as values are identical
    while(N--) {
        quad_value val;
        load_quad<Aligned>(in_ptr, backend_in, &val);
        store_quad<Aligned>(out_ptr, &val, backend_out);
        in_ptr += in_stride;
        out_ptr += out_stride;
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

    return static_cast<NPY_CASTING>(NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
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

        store_quad<Aligned>(out_ptr, &out_val, backend);

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
        return static_cast<NPY_CASTING>(NPY_SAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
    }
    return static_cast<NPY_CASTING>(NPY_SAME_KIND_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
}

static inline const char *
quad_to_string_adaptive_cstr(Sleef_quad *sleef_val, npy_intp unicode_size_chars)
{
    // Try positional format first to see if it would fit
    const char* positional_str = Dragon4_Positional_QuadDType_CStr(
            sleef_val, DigitMode_Unique, CutoffMode_TotalLength, SLEEF_QUAD_DECIMAL_DIG, 0, 1,
            TrimMode_LeaveOneZero, 1, 0);

    if (positional_str == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Float formatting failed");
        return NULL;
    }

    // no need to scan full, only checking if its longer
    npy_intp pos_len = strnlen(positional_str, unicode_size_chars + 1);

    // If positional format fits, use it; otherwise use scientific notation
    if (pos_len <= unicode_size_chars) {
        return positional_str;  // Keep the positional string
    }

    // Use scientific notation with full precision
    const char *scientific_str = Dragon4_Scientific_QuadDType_CStr(sleef_val, DigitMode_Unique,
                                        SLEEF_QUAD_DECIMAL_DIG, 0, 1,
                                        TrimMode_LeaveOneZero, 1, 2);
    if (scientific_str == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Float formatting failed");
        return NULL;
    }
    return scientific_str;

}

static inline int
quad_to_string_same_value_check(const quad_value *in_val, const char *str_buf, npy_intp str_len,
                                 QuadBackendType backend)
{
    char *truncated_str = (char *)malloc(str_len + 1);
    if (truncated_str == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    memcpy(truncated_str, str_buf, str_len);
    truncated_str[str_len] = '\0';
    
    // Parse the truncated string back to quad
    quad_value roundtrip;
    char *endptr;
    
    int err = NumPyOS_ascii_strtoq(truncated_str, backend, &roundtrip, &endptr);
    if (err < 0) {
        PyErr_Format(PyExc_ValueError,
                     "QuadPrecision value cannot be represented exactly: string '%s' failed to parse back",
                     truncated_str);
        free(truncated_str);
        return -1;
    }
    free(truncated_str);
    
    // Compare original and roundtripped values along with signbit
    if (backend == BACKEND_SLEEF) {
        bool is_sign_preserved = (quad_signbit(&in_val->sleef_value) == quad_signbit(&roundtrip.sleef_value));
        if(quad_isnan(&in_val->sleef_value) && quad_isnan(&roundtrip.sleef_value) && is_sign_preserved)
            return 1;
        if (Sleef_icmpeqq1(in_val->sleef_value, roundtrip.sleef_value) && is_sign_preserved)
            return 1;
    }
    else {
        bool is_sign_preserved = (ld_signbit(&in_val->longdouble_value) == ld_signbit(&roundtrip.longdouble_value));
        if ((std::isnan(in_val->longdouble_value) && std::isnan(roundtrip.longdouble_value)) && is_sign_preserved)
            return 1;
        if ((in_val->longdouble_value == roundtrip.longdouble_value) && is_sign_preserved)
            return 1;
    }
    
    // Values don't match - the string width is too narrow for exact representation
    Sleef_quad sleef_val = quad_to_sleef_quad(in_val, backend);
    const char *val_str = quad_to_string_adaptive_cstr(&sleef_val, QUAD_STR_WIDTH);
    if (val_str != NULL) {
        PyErr_Format(PyExc_ValueError,
                     "QuadPrecision value '%s' cannot be represented exactly in target string dtype "
                     "(string width too narrow or precision loss occurred)",
                     val_str);
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "QuadPrecision value cannot be represented exactly in target string dtype "
                        "(string width too narrow or precision loss occurred)");
    }
    return -1;
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
    int same_value_casting = ((context->flags & NPY_SAME_VALUE_CONTEXT_FLAG) == NPY_SAME_VALUE_CONTEXT_FLAG);

    while (N--) {
        quad_value in_val;
        load_quad<Aligned>(in_ptr, backend, &in_val);

        // Convert to Sleef_quad for Dragon4
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);

        const char *temp_str = quad_to_string_adaptive_cstr(&sleef_val, unicode_size_chars);
        if (temp_str == NULL) {
            return -1;
        }

        npy_intp str_len = strnlen(temp_str, unicode_size_chars);

        // Perform same_value check if requested
        if (same_value_casting) {
            if (quad_to_string_same_value_check(&in_val, temp_str, str_len, backend) < 0) {
                return -1;
            }
        }

        // Convert char string to UCS4 and store in output
        Py_UCS4 *out_ucs4 = (Py_UCS4 *)out_ptr;
        for (npy_intp i = 0; i < str_len; i++) {
            out_ucs4[i] = (Py_UCS4)temp_str[i];
        }
        for (npy_intp i = str_len; i < unicode_size_chars; i++) {
            out_ucs4[i] = 0;
        }

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

    return static_cast<NPY_CASTING>(NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
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

        store_quad<Aligned>(out_ptr, &out_val, backend);

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
        return static_cast<NPY_CASTING>(NPY_SAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
    }
    return static_cast<NPY_CASTING>(NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
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
    int same_value_casting = ((context->flags & NPY_SAME_VALUE_CONTEXT_FLAG) == NPY_SAME_VALUE_CONTEXT_FLAG);

    while (N--) {
        quad_value in_val;
        load_quad<Aligned>(in_ptr, backend, &in_val);
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);

        const char *temp_str = quad_to_string_adaptive_cstr(&sleef_val, bytes_size);
        if (temp_str == NULL) {
            return -1;
        }

        npy_intp str_len = strnlen(temp_str, bytes_size);

        // Perform same_value check if requested
        if (same_value_casting) {
            if (quad_to_string_same_value_check(&in_val, temp_str, str_len, backend) < 0) {
                return -1;
            }
        }

        // Copy string to output buffer, padding with nulls
        strncpy(out_ptr, temp_str, bytes_size);

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

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return static_cast<NPY_CASTING>(NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
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

        store_quad<Aligned>(out_ptr, &out_val, backend);

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
    if (given_descrs[1] == NULL) {
        // Default StringDType() already has coerce=True
        loop_descrs[1] = (PyArray_Descr *)PyObject_CallNoArgs(
                (PyObject *)&PyArray_StringDType);
        if (loop_descrs[1] == NULL) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return static_cast<NPY_CASTING>(NPY_SAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
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
    int same_value_casting = ((context->flags & NPY_SAME_VALUE_CONTEXT_FLAG) == NPY_SAME_VALUE_CONTEXT_FLAG);

    npy_string_allocator *allocator = NpyString_acquire_allocator(str_descr);

    while (N--) {
        quad_value in_val;
        load_quad<Aligned>(in_ptr, backend, &in_val);
        Sleef_quad sleef_val = quad_to_sleef_quad(&in_val, backend);

        // Get string representation with adaptive notation
        // Use a large buffer size to allow for full precision
        const char *str_buf = quad_to_string_adaptive_cstr(&sleef_val, QUAD_STR_WIDTH);
        if (str_buf == NULL) {
            NpyString_release_allocator(allocator);
            return -1;
        }

        Py_ssize_t str_size = strnlen(str_buf, QUAD_STR_WIDTH);

        // Perform same_value check if requested
        if (same_value_casting) {
            if (quad_to_string_same_value_check(&in_val, str_buf, str_size, backend) < 0) {
                NpyString_release_allocator(allocator);
                return -1;
            }
        }

        npy_packed_static_string *out_ps = (npy_packed_static_string *)out_ptr;
        if (NpyString_pack(allocator, out_ps, str_buf, (size_t)str_size) < 0) {
            NpyString_release_allocator(allocator);
            PyErr_SetString(PyExc_MemoryError, "Failed to pack string in Quad to StringDType cast");
            return -1;
        }

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
    if (backend == BACKEND_SLEEF) 
    {
      if (std::isnan(x)) {
          result.sleef_value = std::signbit(x) ? QUAD_PRECISION_NEG_NAN : QUAD_PRECISION_NAN;
      }
      else if (std::isinf(x)) {
          result.sleef_value = (x > 0) ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
      }
      else {
          Sleef_quad temp = Sleef_cast_from_doubleq1(static_cast<double>(x));
          std::memcpy(&result.sleef_value, &temp, sizeof(Sleef_quad));
      }
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
    if (backend == BACKEND_SLEEF) 
    {
      if (std::isnan(x)) {
          result.sleef_value = std::signbit(x) ? QUAD_PRECISION_NEG_NAN : QUAD_PRECISION_NAN;
      }
      else if (std::isinf(x)) {
          result.sleef_value = (x > 0) ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
      }
      else {
          Sleef_quad temp = Sleef_cast_from_doubleq1(x);
          std::memcpy(&result.sleef_value, &temp, sizeof(Sleef_quad));
      }
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
    if (backend == BACKEND_SLEEF) 
    {
      if (std::isnan(x)) {
          result.sleef_value = std::signbit(x) ? QUAD_PRECISION_NEG_NAN : QUAD_PRECISION_NAN;
      }
      else if (std::isinf(x)) {
          result.sleef_value = (x > 0) ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
      }
      else {
          Sleef_quad temp = Sleef_cast_from_doubleq1(static_cast<double>(x));
          std::memcpy(&result.sleef_value, &temp, sizeof(Sleef_quad));
      }
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
    // since QUAD precision is the highest precision, we can always cast to it
    return static_cast<NPY_CASTING>(NPY_SAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
}

template <bool Aligned, typename T>
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

    while (N--) {
        typename NpyType<T>::TYPE in_val = load<Aligned, typename NpyType<T>::TYPE>(in_ptr);
        quad_value out_val = to_quad<T>(in_val, backend);
        store_quad<Aligned>(out_ptr, &out_val, backend);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}

// Casting from QuadDType to other types

template <typename T>
static inline typename NpyType<T>::TYPE
from_quad(const quad_value *x, QuadBackendType backend);

template <>
inline npy_bool
from_quad<spec_npy_bool>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_int64q1(x->sleef_value) != 0;
    }
    else {
        return x->longdouble_value != 0;
    }
}

template <>
inline npy_byte
from_quad<npy_byte>(const quad_value *x, QuadBackendType backend)
{
    // runtime warnings often comes from/to casting of NaN, inf
    // casting is used by ops at several positions leading to warnings
    // fix can be catching the cases and returning corresponding type value without casting
    if (backend == BACKEND_SLEEF) {
        return (npy_byte)Sleef_cast_to_int64q1(x->sleef_value);
    }
    else {
        return (npy_byte)x->longdouble_value;
    }
}

template <>
inline npy_ubyte
from_quad<npy_ubyte>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ubyte)Sleef_cast_to_uint64q1(x->sleef_value);
    }
    else {
        return (npy_ubyte)x->longdouble_value;
    }
}

template <>
inline npy_short
from_quad<npy_short>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_short)Sleef_cast_to_int64q1(x->sleef_value);
    }
    else {
        return (npy_short)x->longdouble_value;
    }
}

template <>
inline npy_ushort
from_quad<npy_ushort>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ushort)Sleef_cast_to_uint64q1(x->sleef_value);
    }
    else {
        return (npy_ushort)x->longdouble_value;
    }
}

template <>
inline npy_int
from_quad<npy_int>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_int)Sleef_cast_to_int64q1(x->sleef_value);
    }
    else {
        return (npy_int)x->longdouble_value;
    }
}

template <>
inline npy_uint
from_quad<npy_uint>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_uint)Sleef_cast_to_uint64q1(x->sleef_value);
    }
    else {
        return (npy_uint)x->longdouble_value;
    }
}

template <>
inline npy_long
from_quad<npy_long>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_long)Sleef_cast_to_int64q1(x->sleef_value);
    }
    else {
        return (npy_long)x->longdouble_value;
    }
}

template <>
inline npy_ulong
from_quad<npy_ulong>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (npy_ulong)Sleef_cast_to_uint64q1(x->sleef_value);
    }
    else {
        return (npy_ulong)x->longdouble_value;
    }
}

template <>
inline npy_longlong
from_quad<npy_longlong>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_int64q1(x->sleef_value);
    }
    else {
        return (npy_longlong)x->longdouble_value;
    }
}

template <>
inline npy_ulonglong
from_quad<npy_ulonglong>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return Sleef_cast_to_uint64q1(x->sleef_value);
    }
    else {
        return (npy_ulonglong)x->longdouble_value;
    }
}

template <>
inline npy_half
from_quad<spec_npy_half>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        double d = cast_sleef_to_double(x->sleef_value);
        return npy_double_to_half(d);
    }
    else {
        return npy_double_to_half((double)x->longdouble_value);
    }
}

template <>
inline float
from_quad<float>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (float)cast_sleef_to_double(x->sleef_value);
    }
    else {
        return (float)x->longdouble_value;
    }
}

template <>
inline double
from_quad<double>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return cast_sleef_to_double(x->sleef_value);
    }
    else {
        return (double)x->longdouble_value;
    }
}

template <>
inline long double
from_quad<long double>(const quad_value *x, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return (long double)cast_sleef_to_double(x->sleef_value);
    }
    else {
        return x->longdouble_value;
    }
}

template <typename T>
static inline int quad_to_numpy_same_value_check(const quad_value *x, QuadBackendType backend, typename NpyType<T>::TYPE *y)
{
    *y = from_quad<T>(x, backend);
    quad_value roundtrip = to_quad<T>(*y, backend);
    if(backend == BACKEND_SLEEF) 
    {
        bool is_sign_preserved = (quad_signbit(&x->sleef_value) == quad_signbit(&roundtrip.sleef_value));
        // check if input is NaN and roundtrip is NaN with same sign
        if(quad_isnan(&x->sleef_value) && quad_isnan(&roundtrip.sleef_value) && is_sign_preserved)
            return 1;
        if(Sleef_icmpeqq1(x->sleef_value, roundtrip.sleef_value) && is_sign_preserved)
            return 1;
    }
    else 
    {
        bool is_sign_preserved = (ld_signbit(&x->longdouble_value) == ld_signbit(&roundtrip.longdouble_value));
        if((std::isnan(x->longdouble_value) && std::isnan(roundtrip.longdouble_value)) && is_sign_preserved)
            return 1;
        if((x->longdouble_value == roundtrip.longdouble_value) && is_sign_preserved)
            return 1;

    }
    Sleef_quad sleef_val = quad_to_sleef_quad(x, backend);
    const char *val_str = quad_to_string_adaptive_cstr(&sleef_val, QUAD_STR_WIDTH);
    if (val_str != NULL) {
        PyErr_Format(PyExc_ValueError, 
                     "QuadPrecision value '%s' cannot be represented exactly in the target dtype",
                     val_str);
    }
    else {
        PyErr_SetString(PyExc_ValueError, 
                        "QuadPrecision value cannot be represented exactly in the target dtype");
    }
    return -1;
}

// Type trait to check if a type is a floating-point type for casting purposes
template <typename T>
struct is_float_type : std::false_type {};

template <>
struct is_float_type<spec_npy_half> : std::true_type {};
template <>
struct is_float_type<float> : std::true_type {};
template <>
struct is_float_type<double> : std::true_type {};
template <>
struct is_float_type<long double> : std::true_type {};

template <typename T>
static NPY_CASTING
quad_to_numpy_resolve_descriptors(PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *dtypes[2],
                                  PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
                                  npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    loop_descrs[1] = PyArray_GetDefaultDescr(dtypes[1]);
    // For floating-point types: same_kind casting (precision loss but same kind)
    if constexpr (is_float_type<T>::value) {
        return static_cast<NPY_CASTING>(NPY_SAME_KIND_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
    } else {
        // For integer/bool types: unsafe casting (cross-kind conversion)
        return static_cast<NPY_CASTING>(NPY_UNSAFE_CASTING | NPY_SAME_VALUE_CASTING_FLAG);
    }
}

template <bool Aligned, typename T>
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
    int same_value_casting = ((context->flags & NPY_SAME_VALUE_CONTEXT_FLAG) == NPY_SAME_VALUE_CONTEXT_FLAG);

    if (same_value_casting) {
        while (N--) {
            quad_value in_val;
            load_quad<Aligned>(in_ptr, backend, &in_val);
            typename NpyType<T>::TYPE out_val;
            int ret = quad_to_numpy_same_value_check<T>(&in_val, backend, &out_val);
            if(ret < 0)
                return -1;
            store<Aligned, typename NpyType<T>::TYPE>(out_ptr, out_val);

            in_ptr += strides[0];
            out_ptr += strides[1];
        }
        return 0;
    }
    while (N--) {
        quad_value in_val;
        load_quad<Aligned>(in_ptr, backend, &in_val);
        typename NpyType<T>::TYPE out_val = from_quad<T>(&in_val, backend);
        store<Aligned, typename NpyType<T>::TYPE>(out_ptr, out_val);

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
            {NPY_METH_strided_loop, (void *)&quad_to_numpy_strided_loop<true, T>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_numpy_strided_loop<false, T>},
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
            {NPY_METH_strided_loop, (void *)&numpy_to_quad_strided_loop<true, T>},
            {NPY_METH_unaligned_strided_loop, (void *)&numpy_to_quad_strided_loop<false, T>},
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
            {NPY_METH_strided_loop, (void *)&quad_to_quad_strided_loop<true>},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_to_quad_strided_loop<false>},
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
