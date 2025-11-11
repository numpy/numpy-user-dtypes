#include "utilities.h"
#include <stdlib.h>

void cstring_to_quad(const char *str, QuadBackendType backend, quad_value *out_value, char **endptr)
{
  if(backend == BACKEND_SLEEF)
  {
    out_value->sleef_value = Sleef_strtoq(str, endptr);
  }
  else
  {
    out_value->longdouble_value = strtold(str, endptr);
  }
}