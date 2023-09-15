#ifndef _NPY_STATIC_STRING_H
#define _NPY_STATIC_STRING_H

#include "stdlib.h"
#include "string.h"

#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN

// the high byte in vstring.size is resolved for flags
// SSSS SSSF

typedef struct _npy_static_string_t {
    char *buf;
    size_t size;
} _npy_static_string_t;

typedef struct _short_string_buffer {
    char buf[sizeof(_npy_static_string_t) - 1];
    unsigned char flags_and_size;
} _short_string_buffer;

#elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN

// the high byte in vstring.size is resolved for flags
// FSSS SSSS

typedef struct _npy_static_string_t {
    size_t size;
    char *buf;
} _npy_static_string_t;

typedef struct _short_string_buffer {
    unsigned char flags_and_size;
    char buf[sizeof(npy_static_string_t) - 1];
} _short_string_buffer;

#endif

typedef union _npy_static_string_u {
    _npy_static_string_t vstring;
    _short_string_buffer direct_buffer;
} _npy_static_string_u;

typedef struct npy_static_string {
    _npy_static_string_u base;
} npy_static_string;

// room for two more flags with values 0x20 and 0x10
#define NPY_STRING_MISSING 0x80  // 1000 0000
#define NPY_STRING_SHORT 0x40    // 0100 0000

// short string sizes fit in a 4-bit integer
#define NPY_SHORT_STRING_SIZE_MASK 0x0F  // 0000 1111
#define NPY_SHORT_STRING_MAX_SIZE \
    (sizeof(npy_static_string) - 1)  // 15 or 7 depending on arch

// one byte in size is reserved for flags and small string optimization
#define MAX_STRING_SIZE (1 << (sizeof(size_t) - 1)) - 1

// represents the empty string and can be passed safely to npy_static_string
// API functions
extern const npy_static_string NPY_EMPTY_STRING;
// represents a sentinel value, *CANNOT* be passed safely to npy_static_string
// API functions, use npy_string_isnull to check if a value is null before
// working with it.
extern const npy_static_string NPY_NULL_STRING;

// Allocates a new buffer for *to_init*, which must be set to NULL before
// calling this function, filling the newly allocated buffer with the copied
// contents of the first *size* entries in *init*, which must be valid and
// initialized beforehand. Calling npy_string_free on *to_init* before calling
// this function on an existing string is sufficient to initialize it. Returns
// -1 if malloc fails and -2 if the internal buffer in *to_init* is not NULL
// to indicate a programming error. Returns 0 on success.
int
npy_string_newsize(const char *init, size_t size, npy_static_string *to_init);

// Sets len to 0 and if the internal buffer is not already NULL, frees it if
// it is allocated on the heap and sets it to NULL. Cannot fail.
void
npy_string_free(npy_static_string *str);

// Copies the contents of *in* into *out*. Allocates a new string buffer for
// *out* and assumes that *out* is uninitialized. Returns -1 if malloc fails
// and -2 if *out* is not initialized. npy_string_free *must* be called before
// this is called if *in* points to an existing string. Returns 0 on success.
int
npy_string_dup(const npy_static_string *in, npy_static_string *out);

// Allocates a new string buffer for *out* with enough capacity to store
// *size* bytes of text. Does not do any initialization, the caller must
// initialize the string buffer. Returns -1 if malloc fails and -2 if *out* is
// not NULL. Returns 0 on success.
int
npy_string_newemptysize(size_t size, npy_static_string *out);

// Determine if *in* corresponds to a NULL npy_static_string struct (e.g. len
// is zero and buf is NULL. Returns 1 if this is the case and zero otherwise.
// Cannot fail.
int
npy_string_isnull(const npy_static_string *in);

// Compare two strings. Has the same sematics as if strcmp were passed
// null-terminated C strings with the content of *s1* and *s2*.
int
npy_string_cmp(const npy_static_string *s1, const npy_static_string *s2);

// Returns the *size* of *s*
size_t
npy_string_size(const npy_static_string *s);

// Returns the string *buf* of *s*. This is not a null-terminated buffer.
char *
npy_string_buf(const npy_static_string *s);

#endif /*_NPY_STATIC_STRING_H */
