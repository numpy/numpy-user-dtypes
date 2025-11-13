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

#ifdef __cplusplus
}
#endif

#endif