#ifndef QUAD_UTILITIES_H
#define QUAD_UTILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "quad_common.h"
#include <sleef.h>
#include <sleefquad.h>
#include <stdbool.h>

int ascii_isspace(int c);
int ascii_isalpha(char c);
int ascii_isdigit(char c);
int ascii_isalnum(char c);
int ascii_tolower(int c);
int ascii_strncasecmp(const char *s1, const char *s2, size_t n);

// Locale-independent ASCII string to quad parser (inspired by NumPyOS_ascii_strtold)
int NumPyOS_ascii_strtoq(const char *s, QuadBackendType backend, quad_value *out_value, char **endptr);


// Helper function: Convert quad_value to Sleef_quad for Dragon4
Sleef_quad
quad_to_sleef_quad(const quad_value *in_val, QuadBackendType backend);

#ifdef __cplusplus
}

#include <cstring>

// Load a value from memory, handling alignment
template <bool Aligned, typename T>
static inline T
load(const char *ptr)
{
    if constexpr (Aligned) {
        return *(const T *)ptr;
    }
    else {
        T val;
        std::memcpy(&val, ptr, sizeof(T));
        return val;
    }
}

// Store a value to memory, handling alignment
template <bool Aligned, typename T>
static inline void
store(char *ptr, const T &val)
{
    if constexpr (Aligned) {
        *(T *)ptr = val;
    }
    else {
        std::memcpy(ptr, &val, sizeof(T));
    }
}

// Load quad_value from memory based on backend and alignment
template <bool Aligned>
static inline void
load_quad(const char *ptr, QuadBackendType backend, quad_value *out)
{
    if (backend == BACKEND_SLEEF) {
        out->sleef_value = load<Aligned, Sleef_quad>(ptr);
    }
    else {
        out->longdouble_value = load<Aligned, long double>(ptr);
    }
}

// Store quad_value to memory based on backend and alignment
template <bool Aligned>
static inline void
store_quad(char *ptr, const quad_value *val, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        store<Aligned, Sleef_quad>(ptr, val->sleef_value);
    }
    else {
        store<Aligned, long double>(ptr, val->longdouble_value);
    }
}

#endif  // __cplusplus

#endif  // QUAD_UTILITIES_H