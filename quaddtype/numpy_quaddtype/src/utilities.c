#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "utilities.h"
#include "constants.hpp"

int
ascii_isspace(int c)
{
    return c == ' ' || c == '\f' || c == '\n' || c == '\r' || c == '\t' || c == '\v';
}

int
ascii_isalpha(char c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

int
ascii_isdigit(char c)
{
    return (c >= '0' && c <= '9');
}

int
ascii_isalnum(char c)
{
    return ascii_isdigit(c) || ascii_isalpha(c);
}

int
ascii_tolower(int c)
{
    if (c >= 'A' && c <= 'Z') {
        return c + ('a' - 'A');
    }
    return c;
}

// inspired from NumPyOS_ascii_strncasecmp
int
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

        // Set NaN value with sign preserved
        if (backend == BACKEND_SLEEF) {
            Sleef_quad nan_val = QUAD_PRECISION_NAN;
            // Apply sign to NaN (negative NaN has sign bit set)
            if (sign < 0) {
                nan_val = Sleef_negq1(nan_val);
            }
            out_value->sleef_value = nan_val;
        }
        else {
            out_value->longdouble_value = sign < 0 ? -nanl("") : nanl("");
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
    // SLEEF 4.0's Sleef_strtoq doesn't properly set endptr to indicate
    // where parsing stopped. It always sets endptr to the end of the string.
    // We need to manually validate and track the parse position.
    
    const char *p = str;
    
    // Skip leading whitespace
    while (ascii_isspace(*p)) {
      p++;
    }
    
    // Track start of number (after whitespace)
    const char *num_start = p;
    
    // Handle optional sign
    if (*p == '+' || *p == '-') {
      p++;
    }
    
    // Must have at least one digit or decimal point followed by digit
    int has_digits = 0;
    int has_decimal = 0;
    
    // Parse integer part
    while (ascii_isdigit(*p)) {
      has_digits = 1;
      p++;
    }
    
    // Parse decimal point and fractional part
    if (*p == '.') {
      has_decimal = 1;
      p++;
      while (ascii_isdigit(*p)) {
        has_digits = 1;
        p++;
      }
    }
    
    // Must have at least one digit somewhere
    if (!has_digits) {
      if (endptr) *endptr = (char *)str;
      return -1;
    }
    
    // Parse optional exponent
    if (*p == 'e' || *p == 'E') {
      const char *exp_start = p;
      p++;
      
      // Optional sign in exponent
      if (*p == '+' || *p == '-') {
        p++;
      }
      
      // Must have at least one digit in exponent
      if (!ascii_isdigit(*p)) {
        // Invalid exponent, backtrack
        p = exp_start;
      } else {
        while (ascii_isdigit(*p)) {
          p++;
        }
      }
    }
    
    // Now p points to where valid parsing ends
    // SLEEF 4.0's Sleef_strtoq has a bug where it doesn't properly stop at whitespace
    // or other delimiters. We need to create a null-terminated substring.
    size_t len = p - str;
    char *temp = (char *)malloc(len + 1);
    if (!temp) {
      if (endptr) *endptr = (char *)str;
      return -1;
    }
    memcpy(temp, str, len);
    temp[len] = '\0';
    
    // Call Sleef_strtoq with the bounded string
    char *sleef_endptr;
    out_value->sleef_value = Sleef_strtoq(temp, &sleef_endptr);
    free(temp);
    
    // Set endptr to our calculated position
    if (endptr) {
      *endptr = (char *)p;
    }
    
  } else {
    out_value->longdouble_value = strtold(str, endptr);
  }
  
  if(endptr && *endptr == str) 
    return -1; // parse error - nothing was parsed
  
  // If full parse is required
  if(require_full_parse && endptr && **endptr != '\0')
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