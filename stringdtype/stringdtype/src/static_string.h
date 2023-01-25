#ifndef _NPY_STATIC_STRING_H
#define _NPY_STATIC_STRING_H

#include "stdlib.h"
#include "string.h"

typedef struct ss {
    size_t len;
    char buf[];
} ss;

// allocates a new ss string of length len, filling with the contents of init
ss *
ssnewlen(const char *init, size_t len);

// returns a new heap-allocated copy of input string *s*
ss *
ssdup(const ss *s);

// returns a new, empty string of length len
// does not do any initialization, the caller must
// initialize and null-terminate the string
ss *
ssnewempty(size_t len);

#endif /*_NPY_STATIC_STRING_H */
