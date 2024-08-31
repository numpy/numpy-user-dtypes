#ifndef _QUADDTYPE_COMMON_H
#define _QUADDTYPE_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BACKEND_SLEEF = 0,
    BACKEND_LONGDOUBLE
} QuadBackendType;

#ifdef __cplusplus
}
#endif

#endif