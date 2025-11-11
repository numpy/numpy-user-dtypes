#ifndef QUAD_UTILITIES_H
#define QUAD_UTILITIES_H

#include "quad_common.h"
#include <sleef.h>
#include <sleefquad.h>

void cstring_to_quad(const char *str, QuadBackendType backend, quad_value *out_value, char **endptr);

#endif