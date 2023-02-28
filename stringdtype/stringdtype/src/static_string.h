#ifndef _NPY_STATIC_STRING_H
#define _NPY_STATIC_STRING_H

#include "stdlib.h"
#include "string.h"

typedef struct ss {
    size_t len;
    char *buf;
} ss;

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
ssdup(ss *in, ss *out);

// Allocates a new string buffer for *out* with enough capacity to store
// *num_bytes* of text. The actual allocation is num_bytes + 1 bytes, to
// account for the null terminator. Does not do any initialization, the caller
// must initialize and null-terminate the string buffer. Returns -1 if malloc
// fails and -2 if *out* is not empty. Returns 0 on success.
int
ssnewemptylen(size_t num_bytes, ss *out);

// Interpret the contents of buffer *data* as an ss struct and set *out* to
// that struct. If *data* is NULL, set *out* to point to a statically
// allocated, empty SS struct. Since this function may set *out* to point to
// statically allocated data, do not ever free memory owned by an output of
// this function. That means this function is most useful for read-only
// applications.
void
load_string(char *data, ss **out);

#endif /*_NPY_STATIC_STRING_H */
