#ifndef _NPY_STATIC_STRING_H
#define _NPY_STATIC_STRING_H

#include "stdlib.h"

typedef struct npy_packed_static_string {
    char packed_buffer[sizeof(char *) + sizeof(size_t)];
} npy_packed_static_string;

typedef struct npy_static_string {
    size_t size;
    const char *buf;
} npy_static_string;

// Represents the empty string. The unpacked string can be passed safely to
// npy_static_string API functions.
extern const npy_packed_static_string *NPY_EMPTY_STRING;
// Represents a sentinel value, use npy_string_isnull or the return value of
// npy_load_string to check if a value is null before working with the unpacked
// representation.
extern const npy_packed_static_string *NPY_NULL_STRING;

// one byte in size is reserved for flags and small string optimization
#define NPY_MAX_STRING_SIZE (1 << (sizeof(size_t) - 1)) - 1

// Handles heap allocations for static strings.
typedef struct npy_string_allocator npy_string_allocator;

// Typedefs for allocator functions
typedef void *(*npy_string_malloc_func)(size_t size);
typedef void (*npy_string_free_func)(void *ptr);

// Use these functions to create and destroy string allocators, normally
// users won't use this directly and will use an allocator already
// attached to a dtype instance
npy_string_allocator *
npy_string_new_allocator(npy_string_malloc_func m, npy_string_free_func f);
void
npy_string_free_allocator(npy_string_allocator *allocator);

// Allocates a new buffer for *to_init*, which must be set to NULL before
// calling this function, filling the newly allocated buffer with the copied
// contents of the first *size* entries in *init*, which must be valid and
// initialized beforehand. Calling npy_string_free on *to_init* before calling
// this function on an existing string or copying the contents of
// NPY_EMPTY_STRING into *to_init* is sufficient to initialize it. Does not
// check if *to_init* is NULL or if the internal buffer is non-NULL, undefined
// behavior or memory leaks are possible if this function is passed a pointer
// to a an unintialized struct, a NULL pointer, or an existing heap-allocated
// string.  Returns -1 if allocating the string would exceed the maximum
// allowed string size or exhaust available memory. Returns 0 on success.
int
npy_string_newsize(const char *init, size_t size,
                   npy_packed_static_string *to_init,
                   npy_string_allocator *allocator);

// Zeroes out the packed string and frees any heap allocated data. Cannot
// fail.
void
npy_string_free(npy_packed_static_string *str,
                npy_string_allocator *allocator);

// Copies the contents of *in* into *out*. Allocates a new string buffer for
// *out*, npy_string_free *must* be called before this is called if *out*
// points to an existing string. Returns -1 if malloc fails. Returns 0 on
// success.
int
npy_string_dup(const npy_packed_static_string *in,
               npy_packed_static_string *out, npy_string_allocator *allocator);

// Allocates a new string buffer for *out* with enough capacity to store
// *size* bytes of text. Does not do any initialization, the caller must
// initialize the string buffer after this function returns. Calling
// npy_string_free on *to_init* before calling this function on an existing
// string or initializing a new string with the contents of NPY_EMPTY_STRING
// is sufficient to initialize it. Does not check if *to_init* has already
// been initialized or if the internal buffer is non-NULL, undefined behavior
// or memory leaks are possible if this function is passed a NULL pointer or
// an existing heap-allocated string.  Returns 0 on success. Returns -1 if
// allocating the string would exceed the maximum allowed string size or
// exhaust available memory. Returns 0 on success.
int
npy_string_newemptysize(size_t size, npy_packed_static_string *out,
                        npy_string_allocator *allocator);

// Determine if *in* corresponds to a NULL npy_static_string struct. Returns 1
// if this is the case and zero otherwise.  Cannot fail.
int
npy_string_isnull(const npy_packed_static_string *in);

// Compare two strings. Has the same semantics as if strcmp were passed
// null-terminated C strings with the content of *s1* and *s2*.
int
npy_string_cmp(const npy_static_string *s1, const npy_static_string *s2);

// Extract the packed contents of *packed_string* into *unpacked_string*.  A
// useful pattern is to define a stack-allocated npy_static_string instance
// initialized to {0, NULL} and pass a pointer to that string to unpack the
// contents of a packed string. The *unpacked_string* is a read-only view onto
// the *packed_string* data and should not be used to modify the string
// data. If *packed_string* is the null string, sets *unpacked_string* to the
// NULL pointer. Returns 1 if *packed_string* is the null string and 0
// otherwise so this function can be used to simultaneously unpack a string
// and determine if it is a null string.
int
npy_load_string(const npy_packed_static_string *packed_string,
                npy_static_string *unpacked_string);

// Returns the size of the string data in the packed string. Useful in
// situations where only the string size is needed and determing if this is a
// null or unpacking the string is unnecessary.
size_t
npy_string_size(const npy_packed_static_string *packed_string);

#endif /*_NPY_STATIC_STRING_H */
