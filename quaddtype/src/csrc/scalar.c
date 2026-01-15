#include <Python.h>
#include <sleef.h>
#include <sleefquad.h>
#include <stdlib.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "scalar.h"
#include "scalar_ops.h"
#include "dragon4.h"
#include "dtype.h"
#include "lock.h"
#include "utilities.h"
#include "constants.hpp"
#include "pythoncapi_compat.h"


QuadPrecisionObject *
QuadPrecision_raw_new(QuadBackendType backend)
{
    QuadPrecisionObject *new;
    new = PyObject_New(QuadPrecisionObject, &QuadPrecision_Type);

    if (!new)
        return NULL;
    new->backend = backend;
    if (backend == BACKEND_SLEEF) {
        new->value.sleef_value = Sleef_cast_from_doubleq1(0.0);
    }
    else {
        new->value.longdouble_value = 0.0L;
    }
    return new;
}

static QuadPrecisionObject *
quad_from_py_int(PyObject *py_int, QuadBackendType backend, QuadPrecisionObject *self_to_cleanup)
{
    int overflow = 0;
    long long lval = PyLong_AsLongLongAndOverflow(py_int, &overflow);
    
    if (overflow != 0) {
        // Integer is too large, convert to string and recursively call QuadPrecision_from_object
        PyObject *str_obj = PyObject_Str(py_int);
        if (str_obj == NULL) {
            if (self_to_cleanup) {
                Py_DECREF(self_to_cleanup);
            }
            return NULL;
        }
        
        QuadPrecisionObject *result = QuadPrecision_from_object(str_obj, backend);
        Py_DECREF(str_obj);
        if (self_to_cleanup) {
            Py_DECREF(self_to_cleanup);  // discard the default one
        }
        return result;
    }
    else if (lval == -1 && PyErr_Occurred()) {
        if (self_to_cleanup) {
            Py_DECREF(self_to_cleanup);
        }
        return NULL;
    }
   
    // No overflow, use the integer value directly
    QuadPrecisionObject *self = self_to_cleanup ? self_to_cleanup : QuadPrecision_raw_new(backend);
    if (!self) {
        return NULL;
    }
    
    if (backend == BACKEND_SLEEF) {
        self->value.sleef_value = Sleef_cast_from_int64q1(lval);
    }
    else {
        self->value.longdouble_value = (long double)lval;
    }
    return self;
    
}

QuadPrecisionObject *
QuadPrecision_from_object(PyObject *value, QuadBackendType backend)
{   
    // Handle numpy scalars (np.int32, np.float32, etc.) before arrays
    // We need to check this before PySequence_Check because some numpy scalars are sequences
    if (PyArray_CheckScalar(value)) {
        QuadPrecisionObject *self = QuadPrecision_raw_new(backend);
        if (!self)
            return NULL;
        
        // Try as floating point first
        if (PyArray_IsScalar(value, Floating)) {
            PyObject *py_float = PyNumber_Float(value);
            if (py_float == NULL) {
                Py_DECREF(self);
                return NULL;
            }
            double dval = PyFloat_AsDouble(py_float);
            Py_DECREF(py_float);
            if (backend == BACKEND_SLEEF) {
                self->value.sleef_value = Sleef_cast_from_doubleq1(dval);
            }
            else {
                self->value.longdouble_value = (long double)dval;
            }
            return self;
        }
        // Try as integer
        else if (PyArray_IsScalar(value, Integer)) {
            PyObject *py_int = PyNumber_Long(value);
            if (py_int == NULL) {
                Py_DECREF(self);
                return NULL;
            }
            
            QuadPrecisionObject *result = quad_from_py_int(py_int, backend, self);
            Py_DECREF(py_int);
            return result;
        }
        // Try as boolean
        else if (PyArray_IsScalar(value, Bool)) {
            PyObject *py_int = PyNumber_Long(value);
            if (py_int == NULL) {
                Py_DECREF(self);
                return NULL;
            }
            
            // Booleans are always 0 or 1, so no overflow check needed
            long long lval = PyLong_AsLongLong(py_int);
            Py_DECREF(py_int);
            
            if (lval == -1 && PyErr_Occurred()) {
                Py_DECREF(self);
                return NULL;
            }
            
            if (backend == BACKEND_SLEEF) {
                self->value.sleef_value = Sleef_cast_from_int64q1(lval);
            }
            else {
                self->value.longdouble_value = (long double)lval;
            }
            return self;
        }
        // For other scalar types, fall through to error handling
        Py_DECREF(self);
    }
    
    // this checks arrays and sequences (array, tuple)
    // rejects strings; they're parsed below
    if (PyArray_Check(value) || (PySequence_Check(value) && !PyUnicode_Check(value) && !PyBytes_Check(value))) 
    {
        QuadPrecDTypeObject *dtype_descr = new_quaddtype_instance(backend);
        if (dtype_descr == NULL) {
            return NULL;
        }
        
        // steals reference to the descriptor
        PyObject *result = PyArray_FromAny(
            value,
            (PyArray_Descr *)dtype_descr,
            0,
            0,
            NPY_ARRAY_ENSUREARRAY, // this should handle the casting if possible
            NULL
        );
        
        // PyArray_FromAny steals the reference to dtype_descr, so no need to DECREF
        return (QuadPrecisionObject *)result;
    }

    QuadPrecisionObject *self = QuadPrecision_raw_new(backend);
    if (!self)
        return NULL;

    if (PyFloat_Check(value)) {
        double dval = PyFloat_AsDouble(value);
        if (backend == BACKEND_SLEEF) {
            self->value.sleef_value = Sleef_cast_from_doubleq1(dval);
        }
        else {
            self->value.longdouble_value = (long double)dval;
        }
    }
    else if (PyUnicode_Check(value)) {
        const char *s = PyUnicode_AsUTF8(value);
        char *endptr = NULL;
        int err = NumPyOS_ascii_strtoq(s, backend, &self->value, &endptr);
        if (err < 0) {
            PyErr_SetString(PyExc_ValueError, "Unable to parse string to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
        // Skip trailing whitespace (matches Python's float() behavior)
        while (ascii_isspace(*endptr)) {
            endptr++;
        }
        if (*endptr != '\0') {
            PyErr_SetString(PyExc_ValueError, "Unable to parse string to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
    }
    else if (PyBytes_Check(value)) {
        const char *s = PyBytes_AsString(value);
        if (s == NULL) {
            Py_DECREF(self);
            return NULL;
        }
        char *endptr = NULL;
        int err = NumPyOS_ascii_strtoq(s, backend, &self->value, &endptr);
        if (err < 0) {
            PyErr_SetString(PyExc_ValueError, "Unable to parse bytes to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
        // Skip trailing whitespace (matches Python's float() behavior)
        while (ascii_isspace(*endptr)) {
            endptr++;
        }
        if (*endptr != '\0') {
            PyErr_SetString(PyExc_ValueError, "Unable to parse bytes to QuadPrecision");
            Py_DECREF(self);
            return NULL;
        }
    }
    else if (PyLong_Check(value)) {
        return quad_from_py_int(value, backend, self);
    }
    else if (Py_TYPE(value) == &QuadPrecision_Type) {
        Py_DECREF(self);  // discard the default one
        QuadPrecisionObject *quad_obj = (QuadPrecisionObject *)value;
        // create a new one with the same backend
        QuadPrecisionObject *self = QuadPrecision_raw_new(quad_obj->backend);
        if (quad_obj->backend == BACKEND_SLEEF) {
            self->value.sleef_value = quad_obj->value.sleef_value;
        }
        else {
            self->value.longdouble_value = quad_obj->value.longdouble_value;
        }

        return self;
    }
    else {
        PyObject *type_str = PyObject_Str((PyObject *)Py_TYPE(value));
        if (type_str != NULL) {
            const char *type_cstr = PyUnicode_AsUTF8(type_str);
            if (type_cstr != NULL) {
                PyErr_Format(PyExc_TypeError,
                             "QuadPrecision value must be a quad, float, int, string, bytes, array or sequence, but got %s "
                             "instead",
                             type_cstr);
            }
            else {
                PyErr_SetString(
                        PyExc_TypeError,
                        "QuadPrecision value must be a quad, float, int, string, bytes, array or sequence, but got an "
                        "unknown type instead");
            }
            Py_DECREF(type_str);
        }
        else {
            PyErr_SetString(PyExc_TypeError,
                            "QuadPrecision value must be a quad, float, int, string, bytes, array or sequence, but got an "
                            "unknown type instead");
        }
        Py_DECREF(self);
        return NULL;
    }

    return self;
}

static PyObject *
QuadPrecision_new(PyTypeObject *cls, PyObject *args, PyObject *kwargs)
{
    PyObject *value;
    const char *backend_str = "sleef";
    static char *kwlist[] = {"value", "backend", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", kwlist, &value, &backend_str)) {
        return NULL;
    }

    QuadBackendType backend = BACKEND_SLEEF;
    if (strcmp(backend_str, "longdouble") == 0) {
        backend = BACKEND_LONGDOUBLE;
    }
    else if (strcmp(backend_str, "sleef") != 0) {
        PyErr_SetString(PyExc_ValueError, "Invalid backend. Use 'sleef' or 'longdouble'.");
        return NULL;
    }

    return (PyObject *)QuadPrecision_from_object(value, backend);
}

static PyObject *
QuadPrecision_str_dragon4(QuadPrecisionObject *self)
{
    Dragon4_Options opt = {.scientific = 0,
                           .digit_mode = DigitMode_Unique,
                           .cutoff_mode = CutoffMode_TotalLength,
                           .precision = SLEEF_QUAD_DECIMAL_DIG,
                           .sign = 1,
                           .trim_mode = TrimMode_LeaveOneZero,
                           .digits_left = 1,
                           .digits_right = 0};

    if (self->backend == BACKEND_SLEEF) {
        return Dragon4_Positional_QuadDType(
                &self->value.sleef_value, opt.digit_mode, opt.cutoff_mode, opt.precision,
                opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left, opt.digits_right);
    }
    else {
        Sleef_quad sleef_val = Sleef_cast_from_doubleq1(self->value.longdouble_value);
        return Dragon4_Positional_QuadDType(&sleef_val, opt.digit_mode, opt.cutoff_mode,
                                            opt.precision, opt.min_digits, opt.sign, opt.trim_mode,
                                            opt.digits_left, opt.digits_right);
    }
}

static PyObject *
QuadPrecision_str(QuadPrecisionObject *self)
{
    char buffer[128];
    if (self->backend == BACKEND_SLEEF) {
        Sleef_snprintf(buffer, sizeof(buffer), "%.*Qe", SLEEF_QUAD_DIG, self->value.sleef_value);
    }
    else {
        snprintf(buffer, sizeof(buffer), "%.35Le", self->value.longdouble_value);
    }
    return PyUnicode_FromString(buffer);
}

static PyObject *
QuadPrecision_repr_dragon4(QuadPrecisionObject *self)
{
    Dragon4_Options opt = {.scientific = 1,
                           .digit_mode = DigitMode_Unique,
                           .cutoff_mode = CutoffMode_TotalLength,
                           .precision = SLEEF_QUAD_DECIMAL_DIG,
                           .sign = 1,
                           .trim_mode = TrimMode_LeaveOneZero,
                           .digits_left = 1,
                           .exp_digits = 3};

    PyObject *str;
    if (self->backend == BACKEND_SLEEF) {
        str = Dragon4_Scientific_QuadDType(&self->value.sleef_value, opt.digit_mode, opt.precision,
                                           opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left,
                                           opt.exp_digits);
    }
    else {
        Sleef_quad sleef_val = Sleef_cast_from_doubleq1(self->value.longdouble_value);
        str = Dragon4_Scientific_QuadDType(&sleef_val, opt.digit_mode, opt.precision,
                                           opt.min_digits, opt.sign, opt.trim_mode, opt.digits_left,
                                           opt.exp_digits);
    }

    if (str == NULL) {
        return NULL;
    }

    const char *backend_str = (self->backend == BACKEND_SLEEF) ? "sleef" : "longdouble";
    PyObject *res = PyUnicode_FromFormat("QuadPrecision('%S', backend='%s')", str, backend_str);
    Py_DECREF(str);
    return res;
}

static void
QuadPrecision_dealloc(QuadPrecisionObject *self)
{
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int
QuadPrecision_getbuffer(QuadPrecisionObject *self, Py_buffer *view, int flags)
{
    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "QuadPrecision scalar is not writable");
        return -1;
    }

    size_t elem_size = (self->backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    view->obj = (PyObject *)self;
    Py_INCREF(self);
    view->buf = &self->value;
    view->len = elem_size;
    view->readonly = 1;
    view->itemsize = elem_size;
    view->format = NULL;  // No format string for now
    view->ndim = 0;
    view->shape = NULL;
    view->strides = NULL;
    view->suboffsets = NULL;
    view->internal = NULL;

    return 0;
}

static PyBufferProcs QuadPrecision_as_buffer = {
    .bf_getbuffer = (getbufferproc)QuadPrecision_getbuffer,
    .bf_releasebuffer = NULL,
};

static PyObject *
QuadPrecision_get_real(QuadPrecisionObject *self, void *closure)
{
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
QuadPrecision_get_imag(QuadPrecisionObject *self, void *closure)
{
    // For real floating-point types, the imaginary part is always 0
    return (PyObject *)QuadPrecision_raw_new(self->backend);
}

// Method implementations for float compatibility
static PyObject *
QuadPrecision_is_integer(QuadPrecisionObject *self, PyObject *Py_UNUSED(ignored))
{
    Sleef_quad value;
    
    if (self->backend == BACKEND_SLEEF) {
        value = self->value.sleef_value;
    }
    else {
        // lets also tackle ld from sleef functions as well
        value = Sleef_cast_from_doubleq1((double)self->value.longdouble_value);
    }
    
    if(Sleef_iunordq1(value, value)) {
      Py_RETURN_FALSE;
    }
    
    // Check if value is finite (not inf or nan)
    Sleef_quad abs_value = Sleef_fabsq1(value);
    Sleef_quad pos_inf = sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 16384);
    int32_t is_finite = Sleef_icmpltq1(abs_value, pos_inf);
    
    if (!is_finite) {
        Py_RETURN_FALSE;
    }
    
    // Check if value equals its truncated version
    Sleef_quad truncated = Sleef_truncq1(value);
    int32_t is_equal = Sleef_icmpeqq1(value, truncated);
    
    if (is_equal) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

PyObject* quad_to_pylong(Sleef_quad value)
{
    char buffer[128];

    // Sleef_snprintf call is thread-unsafe
    LOCK_SLEEF;
    // Format as integer (%.0Qf gives integer with no decimal places)
    // Q modifier means pass Sleef_quad by value
    int written = Sleef_snprintf(buffer, sizeof(buffer), "%.0Qf", value);
    UNLOCK_SLEEF;
    if (written < 0 || written >= sizeof(buffer)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert quad to string");
        return NULL;
    }
    
    // Already raises ValueError and returns NULL on failure
    PyObject *result = PyLong_FromString(buffer, NULL, 10);
    return result;
}

// inspired by the CPython implementation
// https://github.com/python/cpython/blob/ac1ffd77858b62d169a08040c08aa5de26e145ac/Objects/floatobject.c#L1503C1-L1572C2
static PyObject *
QuadPrecision_as_integer_ratio(QuadPrecisionObject *self, PyObject *Py_UNUSED(ignored))
{

    Sleef_quad value;
    Sleef_quad pos_inf = sleef_q(+0x1000000000000LL, 0x0000000000000000ULL, 16384);
    const int FLOAT128_PRECISION = 113;
    
    if (self->backend == BACKEND_SLEEF) {
        value = self->value.sleef_value;
    }
    else {
        // lets also tackle ld from sleef functions as well
        value = Sleef_cast_from_doubleq1((double)self->value.longdouble_value);
    }
    
    if(Sleef_iunordq1(value, value)) {
      PyErr_SetString(PyExc_ValueError, "Cannot convert NaN to integer ratio");
      return NULL;
    }
    if(Sleef_icmpgeq1(Sleef_fabsq1(value), pos_inf)) {
      PyErr_SetString(PyExc_OverflowError, "Cannot convert infinite value to integer ratio");
      return NULL;
    }

    // Sleef_value == float_part * 2**exponent exactly
    int exponent;
    Sleef_quad mantissa = Sleef_frexpq1(value, &exponent); // within [0.5, 1.0)

    /*
    CPython loops for 300 (some huge number) to make sure 
    float_part gets converted to the floor(float_part) i.e. near integer as
    
    for (i=0; i<300 && float_part != floor(float_part) ; i++) {
        float_part *= 2.0;
        exponent--;
    }

    It seems highly inefficient from performance perspective, maybe they pick 300 for future-proof
    or If FLT_RADIX != 2, the 300 steps may leave a tiny fractional part

    Another way can be doing as:
    ```
    mantissa = ldexpq(mantissa, FLOAT128_PRECISION);
    exponent -= FLOAT128_PRECISION;
    ```
    This should work but give non-simplified, huge integers (although they also come down to same representation)
    We can also do gcd to find simplified values, but it'll add more O(log(N))
    For the sake of simplicity and fixed 128-bit nature, we will loop till 113 only
    */

    for (int i = 0; i < FLOAT128_PRECISION && !Sleef_icmpeqq1(mantissa, Sleef_floorq1(mantissa)); i++) {
        mantissa = Sleef_mulq1_u05(mantissa, Sleef_cast_from_doubleq1(2.0));
        exponent--;
    }

    // numerator and denominators can't fit in int
    // convert items to PyLongObject from string instead
    PyObject *py_exp = PyLong_FromLongLong(Py_ABS(exponent));
    if(py_exp == NULL)
    {
        return NULL;
    }
    
    PyObject *numerator = quad_to_pylong(mantissa);
    if(numerator == NULL)
    {
        Py_DECREF(py_exp);  
        return NULL;
    }
    PyObject *denominator = PyLong_FromLong(1);
    if (denominator == NULL) {
        Py_DECREF(py_exp);
        Py_DECREF(numerator);
        return NULL;
    }

    // fold in 2**exponent
    if(exponent > 0)
    {
        PyObject *new_num = PyNumber_Lshift(numerator, py_exp);
        Py_DECREF(numerator);
        if(new_num == NULL)
        {
            Py_DECREF(denominator);
            Py_DECREF(py_exp);
            return NULL;
        }
        numerator = new_num;
    }
    else
    {
        PyObject *new_denom = PyNumber_Lshift(denominator, py_exp);
        Py_DECREF(denominator);
        if(new_denom == NULL)
        {
            Py_DECREF(numerator);
            Py_DECREF(py_exp);
            return NULL;
        }
        denominator = new_denom;
    }

    Py_DECREF(py_exp);
    return PyTuple_Pack(2, numerator, denominator);
}

static PyMethodDef QuadPrecision_methods[] = {
    {"is_integer", (PyCFunction)QuadPrecision_is_integer, METH_NOARGS,
     "Return True if the value is an integer."},
    {"as_integer_ratio", (PyCFunction)QuadPrecision_as_integer_ratio, METH_NOARGS,
     "Return a pair of integers whose ratio is exactly equal to the original value."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyObject *
QuadPrecision_get_dtype(QuadPrecisionObject *self, void *NPY_UNUSED(closure))
{
    QuadPrecDTypeObject *dtype = new_quaddtype_instance(self->backend);
    return (PyObject *)dtype;
}

static PyGetSetDef QuadPrecision_getset[] = {
    {"dtype", (getter)QuadPrecision_get_dtype, NULL, "Data type descriptor for this scalar", NULL},
    {"real", (getter)QuadPrecision_get_real, NULL, "Real part of the scalar", NULL},
    {"imag", (getter)QuadPrecision_get_imag, NULL, "Imaginary part of the scalar (always 0 for real types)", NULL},
    {NULL}  /* Sentinel */
};

/*
 * Hash function for QuadPrecision scalars.
 * 
 * This implements the same algorithm as CPython's _Py_HashDouble, adapted for
 * 128-bit floating point. The algorithm computes a hash based
 * on the reduction of the value modulo the prime P = 2**PYHASH_BITS - 1.
 * https://github.com/python/cpython/blob/20b69aac0d19a5e5358362410d9710887762f0e7/Python/pyhash.c#L87
 * 
 * Key invariant: hash(x) == hash(y) whenever x and y are numerically equal,
 * even if x and y have different types. This ensures that:
 *   hash(QuadPrecision(1.0)) == hash(1.0) == hash(1)
 * 
 * The algorithm:
 * 1. Handle special cases: inf returns PyHASH_INF, nan uses pointer hash
 * 2. Extract mantissa m in [0.5, 1.0) and exponent e via frexp(v) = m * 2^e
 * 3. Process mantissa 28 bits at a time, accumulating into hash value x
 * 4. Adjust for exponent using bit rotation (since 2^PyHASH_BITS ≡ 1 mod P)
 * 5. Apply sign and handle the special case of -1 -> -2
 */

static Py_hash_t
QuadPrecision_hash(QuadPrecisionObject *self)
{
    Sleef_quad value;
    int sign = 1;
    
    if (self->backend == BACKEND_SLEEF) {
        value = self->value.sleef_value;
    }
    else {
        value = Sleef_cast_from_doubleq1((double)self->value.longdouble_value);
    }
    
    // Check for NaN - use pointer hash (each NaN instance gets unique hash)
    // This prevents hash table catastrophic pileups from NaN instances
    if (Sleef_iunordq1(value, value)) {
        return Py_HashPointer((void *)self);
    }
    
    if (Sleef_icmpeqq1(value, QUAD_PRECISION_INF)) {
        return PyHASH_INF;
    }
    if (Sleef_icmpeqq1(value, QUAD_PRECISION_NINF)) {
        return -PyHASH_INF;
    }
    
    // Handle sign
    Sleef_quad zero = Sleef_cast_from_int64q1(0);
    if (Sleef_icmpltq1(value, zero)) {
        sign = -1;
        value = Sleef_negq1(value);
    }
    
    // Get mantissa and exponent: value = m * 2^e, where 0.5 <= m < 1.0
    int exponent;
    Sleef_quad mantissa = Sleef_frexpq1(value, &exponent);
    
    // Process 28 bits at a time (same as CPython's _Py_HashDouble)
    // This works well for both binary and hexadecimal floating point
    Py_uhash_t x = 0;
    // 2^28 = 268435456 - exactly representable in double, so cast is safe
    Sleef_quad multiplier = Sleef_cast_from_int64q1(1LL << 28);
    
    // Continue until mantissa becomes zero (all bits processed)
    while (Sleef_icmpneq1(mantissa, zero)) {
        // Rotate x left by 28 bits within PyHASH_MODULUS
        x = ((x << 28) & PyHASH_MODULUS) | (x >> (PyHASH_BITS - 28));
        
        // Scale mantissa by 2^28
        mantissa = Sleef_mulq1_u05(mantissa, multiplier);
        exponent -= 28;
        
        // Extract integer part
        Sleef_quad int_part = Sleef_truncq1(mantissa);
        Py_uhash_t y = (Py_uhash_t)Sleef_cast_to_int64q1(int_part);
        
        // Remove integer part from mantissa (keep fractional part)
        mantissa = Sleef_subq1_u05(mantissa, int_part);
        
        // Accumulate
        x += y;
        if (x >= PyHASH_MODULUS) {
            x -= PyHASH_MODULUS;
        }
    }
    
    // Adjust for exponent: reduce e modulo PyHASH_BITS
    // For negative exponents: PyHASH_BITS - 1 - ((-1 - e) % PyHASH_BITS)
    int e = exponent >= 0 
            ? exponent % PyHASH_BITS 
            : PyHASH_BITS - 1 - ((-1 - exponent) % PyHASH_BITS);
    
    // Rotate x left by e bits
    x = ((x << e) & PyHASH_MODULUS) | (x >> (PyHASH_BITS - e));
    
    // Apply sign
    x = x * sign;
    
    // -1 is reserved for errors, so use -2 instead
    if (x == (Py_uhash_t)-1) {
        x = (Py_uhash_t)-2;
    }
    
    return (Py_hash_t)x;
}

PyTypeObject QuadPrecision_Type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "numpy_quaddtype.QuadPrecision",
        .tp_basicsize = sizeof(QuadPrecisionObject),
        .tp_itemsize = 0,
        .tp_new = QuadPrecision_new,
        .tp_dealloc = (destructor)QuadPrecision_dealloc,
        .tp_repr = (reprfunc)QuadPrecision_repr_dragon4,
        .tp_str = (reprfunc)QuadPrecision_str_dragon4,
        .tp_hash = (hashfunc)QuadPrecision_hash,
        .tp_as_number = &quad_as_scalar,
        .tp_as_buffer = &QuadPrecision_as_buffer,
        .tp_richcompare = (richcmpfunc)quad_richcompare,
        .tp_methods = QuadPrecision_methods,
        .tp_getset = QuadPrecision_getset,
};

int
init_quadprecision_scalar(void)
{
    QuadPrecision_Type.tp_base = &PyFloatingArrType_Type;
    return PyType_Ready(&QuadPrecision_Type);
}