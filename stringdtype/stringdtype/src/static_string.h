#ifndef _NPY_STATIC_STRING_H
#define _NPY_STATIC_STRING_H

#include "stdlib.h"
#include "string.h"

typedef struct ss {
    size_t len;
    char *buf;
} ss;

extern const ss EMPTY_STRING;

// Allocates a new buffer for *to_init*, filling with the copied contents of
// *init* and sets *to_init->len* to *len*. Returns -1 if malloc fails and -2
// if *to_init* is not empty. Returns 0 on success.
int
ssnewlen(const char *init, size_t len, ss *to_init);

// Sets len to 0 and if str->buf is not already NULL, frees it and sets it to
// NULL. Cannot fail.
void
ssfree(ss *str);

// copies the contents out *in* into *out*. Allocates a new string buffer for
// *out*. Returns -1 if malloc fails and -2 if *out* is not empty. Returns 0 on
// success.
int
ssdup(const ss *in, ss *out);

// Allocates a new string buffer for *out* with enough capacity to store
// *num_bytes* of text. The actual allocation is num_bytes + 1 bytes, to
// account for the null terminator. Does not do any initialization, the caller
// must initialize and null-terminate the string buffer. Returns -1 if malloc
// fails and -2 if *out* is not empty. Returns 0 on success.
int
ssnewemptylen(size_t num_bytes, ss *out);

// Determine if *in* corresponds to a NULL ss struct (e.g. len is zero and buf
// is NULL. Returns 1 if this is the case and zero otherwise. Cannot fail.
int
ss_isnull(const ss *in);

// Compare two strings. Has the same sematics as strcmp passed null-terminated
// C strings with the content of *s1* and *s2*.
int
sscmp(const ss *s1, const ss *s2);

#endif /*_NPY_STATIC_STRING_H */
