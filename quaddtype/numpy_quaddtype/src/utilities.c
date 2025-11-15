#include "utilities.h"
#include <stdlib.h>

int cstring_to_quad(const char *str, QuadBackendType backend, quad_value *out_value, 
char **endptr, bool require_full_parse)
{
  if(backend == BACKEND_SLEEF) {
    out_value->sleef_value = Sleef_strtoq(str, endptr);
  } else {
    out_value->longdouble_value = strtold(str, endptr);
  }
  if(*endptr == str) 
    return -1; // parse error - nothing was parsed
  
  // If full parse is required
  if(require_full_parse && **endptr != '\0')
    return -1; // parse error - characters remain to be converted
  
  return 0; // success
}

// Helper function: Convert quad_value to Sleef_quad for Dragon4
Sleef_quad
quad_to_sleef_quad(const quad_value *in_val, QuadBackendType backend)
{
    if (backend == BACKEND_SLEEF) {
        return in_val->sleef_value;
    }
    else {
        return Sleef_cast_from_doubleq1(in_val->longdouble_value);
    }
}