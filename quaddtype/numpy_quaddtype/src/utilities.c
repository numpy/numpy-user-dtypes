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