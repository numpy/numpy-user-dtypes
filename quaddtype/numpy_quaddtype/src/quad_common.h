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

#ifdef __cplusplus
}
#endif

#endif