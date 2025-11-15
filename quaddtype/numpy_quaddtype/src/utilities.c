#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "utilities.h"
#include "constants.hpp"

// Locale-independent ASCII character classification helpers
static int
ascii_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

static int
ascii_isalpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static int
ascii_isdigit(char c)
{
    return (c >= '0' && c <= '9');
}

static int
ascii_isalnum(char c)
{
    return ascii_isdigit(c) || ascii_isalpha(c);
}

static int
ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
        return c + ('a' - 'A');
    }
    return c;
}

// inspired from NumPyOS_ascii_strncasecmp
static int
ascii_strncasecmp(const char *s1, const char *s2, size_t n)
{
    while (n > 0 && *s1 != '\0' && *s2 != '\0') {
        int c1 = ascii_tolower((unsigned char)*s1);
        int c2 = ascii_tolower((unsigned char)*s2);
        int diff = c1 - c2;
        
        if (diff != 0) {
            return diff;
        }

        s1++;
        s2++;
        n--;
    }

    if(n > 0) {
        return *s1 - *s2;
    }
    return 0;
}

/*
 * NumPyOS_ascii_strtoq:
 *
 * Locale-independent string to quad-precision parser.
 * Inspired by NumPyOS_ascii_strtold from NumPy.
 *
 * This function:
 * - Skips leading whitespace
 * - Recognizes inf/nan case-insensitively with optional signs and payloads
 * - Delegates to cstring_to_quad for numeric parsing
 *
 * Returns:
 *   0 on success
 *  -1 on parse error
 */
int
NumPyOS_ascii_strtoq(const char *s, QuadBackendType backend, quad_value *out_value, char **endptr)
{
    const char *p;
    int sign;
    
    // skip leading whitespace
    while (ascii_isspace(*s)) {
        s++;
    }
    
    p = s;
    sign = 1;
    if (*p == '-') {
        sign = -1;
        ++p;
    }
    else if (*p == '+') {
        ++p;
    }

    // Check for inf/infinity (case-insensitive)
    if (ascii_strncasecmp(p, "inf", 3) == 0) {
        p += 3;
        if (ascii_strncasecmp(p, "inity", 5) == 0) {
            p += 5;
        }

        // Set infinity values with sign applied
        if (backend == BACKEND_SLEEF) {
            out_value->sleef_value = sign > 0 ? QUAD_PRECISION_INF : QUAD_PRECISION_NINF;
        }
        else {
            out_value->longdouble_value = sign > 0 ? strtold("inf", NULL) : strtold("-inf", NULL);
        }
        
        if (endptr) {
            *endptr = (char *)p;
        }
        return 0;
    }

    // Check for nan (case-insensitive) with optional payload
    if (ascii_strncasecmp(p, "nan", 3) == 0) {
        p += 3;

        // Skip optional (payload)
        if (*p == '(') {
            ++p;
            while (ascii_isalnum(*p) || *p == '_') {
                ++p;
            }
            if (*p == ')') {
                ++p;
            }
        }

        // Set NaN value (sign is ignored for NaN)
        if (backend == BACKEND_SLEEF) {
            out_value->sleef_value = QUAD_PRECISION_NAN;
        }
        else {
            out_value->longdouble_value = nanl("");
        }
        
        if (endptr) {
            *endptr = (char *)p;
        }
        return 0;
    }
    
    // For numeric values, delegate to cstring_to_quad
    // Pass the original string position (after whitespace, includes sign if present)
    return cstring_to_quad(s, backend, out_value, endptr, false);
}

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