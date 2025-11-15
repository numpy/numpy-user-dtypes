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

// Helper function: Convert quad_value to Sleef_quad for Dragon4
Sleef_quad
quad_to_sleef_quad(const quad_value *in_val, QuadBackendType backend);

#ifdef __cplusplus
}
#endif

#endif