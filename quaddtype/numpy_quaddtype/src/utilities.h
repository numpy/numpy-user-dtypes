#ifndef QUAD_UTILITIES_H
#define QUAD_UTILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

#include "quad_common.h"
#include <sleef.h>
#include <sleefquad.h>
#include <stdbool.h>

int cstring_to_quad(const char *str, QuadBackendType backend, quad_value *out_value, char **endptr, bool require_full_parse);
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
#endif

#endif