#ifndef _QUADDTYPE_COMMON_H
#define _QUADDTYPE_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sleef.h>
#include <sleefquad.h>

typedef enum {
    BACKEND_INVALID = -1,
    BACKEND_SLEEF,
    BACKEND_LONGDOUBLE
} QuadBackendType;

typedef union {
    Sleef_quad sleef_value;
    long double longdouble_value;
} quad_value;


// For IEEE 754 binary128 (quad precision), we need 36 decimal digits 
// to guarantee round-trip conversion (string -> parse -> equals original value)
// Formula: ceil(1 + MANT_DIG * log10(2)) = ceil(1 + 113 * 0.30103) = 36
// src: https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
#define SLEEF_QUAD_DECIMAL_DIG 36

#ifdef __cplusplus
}
#endif

#endif