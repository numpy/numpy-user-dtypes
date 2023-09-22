#include "Python.h"

#include "static_string.h"

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

// room for two more flags with values 0x20 and 0x10
#define NPY_STRING_MISSING 0x80  // 1000 0000
#define NPY_STRING_SHORT 0x40    // 0100 0000

// short string sizes fit in a 4-bit integer
#define NPY_SHORT_STRING_SIZE_MASK 0x0F  // 0000 1111
#define NPY_SHORT_STRING_MAX_SIZE \
    (sizeof(npy_static_string) - 1)  // 15 or 7 depending on arch

// one byte in size is reserved for flags and small string optimization
#define NPY_MAX_STRING_SIZE (1 << (sizeof(size_t) - 1)) - 1

// Since this has no flags set, technically this is a heap-allocated string
// with size zero. Practically, that doesn't matter because we always do size
// checks before accessing heap data, but that may be confusing. The nice part
// of this choice is a calloc'd array buffer (e.g. from np.empty) is filled
// with empty elements for free
const _npy_static_string_u empty_string_u = {
        .direct_buffer = {.flags_and_size = 0, .buf = {0}}};
const npy_packed_static_string *NPY_EMPTY_STRING =
        (npy_packed_static_string *)&empty_string_u;
// zero-filled, but with the NULL flag set to distinguish from empty string
const _npy_static_string_u null_string_u = {
        .direct_buffer = {.flags_and_size = NPY_STRING_MISSING, .buf = {0}}};
const npy_packed_static_string *NPY_NULL_STRING =
        (npy_packed_static_string *)&null_string_u;

int
is_short_string(const npy_packed_static_string *s)
{
    unsigned char high_byte =
            ((_npy_static_string_u *)s)->direct_buffer.flags_and_size;
    return (high_byte & NPY_STRING_SHORT) == NPY_STRING_SHORT;
}

int
npy_string_isnull(const npy_packed_static_string *s)
{
    unsigned char high_byte =
            ((_npy_static_string_u *)s)->direct_buffer.flags_and_size;
    return (high_byte & NPY_STRING_MISSING) == NPY_STRING_MISSING;
}

int
is_not_a_vstring(const npy_packed_static_string *s)
{
    return is_short_string(s) || npy_string_isnull(s);
}

int
npy_load_string(const npy_packed_static_string *packed_string,
                npy_static_string *unpacked_string)
{
    if (npy_string_isnull(packed_string)) {
        unpacked_string->size = 0;
        unpacked_string->buf = NULL;
        return 1;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        unsigned char high_byte = string_u->direct_buffer.flags_and_size;
        unpacked_string->size = high_byte & NPY_SHORT_STRING_SIZE_MASK;
        unpacked_string->buf = string_u->direct_buffer.buf;
    }

    else {
        unpacked_string->size = string_u->vstring.size;
        unpacked_string->buf = string_u->vstring.buf;
    }

    return 0;
}

int
npy_string_newsize(const char *init, size_t size,
                   npy_packed_static_string *to_init)
{
    if (size == 0) {
        *to_init = *NPY_EMPTY_STRING;
        return 0;
    }

    _npy_static_string_u *to_init_u = ((_npy_static_string_u *)to_init);

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        char *ret_buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

        if (ret_buf == NULL) {
            return -1;
        }

        to_init_u->vstring.size = size;

        memcpy(ret_buf, init, size);

        to_init_u->vstring.buf = ret_buf;
    }
    else {
        // size can be no longer than 7 or 15, depending on CPU architecture
        // in either case, the size data is in at most the least significant 4
        // bits of the byte so it's safe to | with one of 0x10, 0x20, 0x40, or
        // 0x80.
        to_init_u->direct_buffer.flags_and_size = NPY_STRING_SHORT | size;
        memcpy(&(to_init_u->direct_buffer.buf), init, size);
    }

    return 0;
}

int
npy_string_newemptysize(size_t size, npy_packed_static_string *out)
{
    if (size == 0) {
        *out = *NPY_EMPTY_STRING;
        return 0;
    }

    _npy_static_string_u *out_u = (_npy_static_string_u *)out;

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        char *buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

        if (buf == NULL) {
            return -1;
        }

        out_u->vstring.buf = buf;
        out_u->vstring.size = size;
    }
    else {
        out_u->direct_buffer.flags_and_size = NPY_STRING_SHORT | size;
    }

    return 0;
}

void
npy_string_free(npy_packed_static_string *str)
{
    if (is_not_a_vstring(str)) {
        // zero out
        memcpy(str, NPY_EMPTY_STRING, sizeof(npy_packed_static_string));
    }
    else {
        _npy_static_string_u *str_u = (_npy_static_string_u *)str;
        if (str_u->vstring.size != 0) {
            PyMem_RawFree(str_u->vstring.buf);
        }
        str_u->vstring.buf = NULL;
        str_u->vstring.size = 0;
    }
}

int
npy_string_dup(const npy_packed_static_string *in,
               npy_packed_static_string *out)
{
    if (npy_string_isnull(in)) {
        *out = *NPY_NULL_STRING;
        return 0;
    }
    if (is_short_string(in)) {
        memcpy(out, in, sizeof(npy_packed_static_string));
        return 0;
    }

    _npy_static_string_u *in_u = (_npy_static_string_u *)in;

    return npy_string_newsize(in_u->vstring.buf, in_u->vstring.size, out);
}

int
npy_string_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    size_t minsize = s1->size < s2->size ? s1->size : s2->size;

    int cmp = strncmp(s1->buf, s2->buf, minsize);

    if (cmp == 0) {
        if (s1->size > minsize) {
            return 1;
        }
        if (s2->size > minsize) {
            return -1;
        }
    }

    return cmp;
}

size_t
npy_string_size(const npy_packed_static_string *packed_string)
{
    if (npy_string_isnull(packed_string)) {
        return 0;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        return string_u->direct_buffer.flags_and_size &
               NPY_SHORT_STRING_SIZE_MASK;
    }

    return string_u->vstring.size;
}
