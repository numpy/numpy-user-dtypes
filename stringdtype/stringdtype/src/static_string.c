#include "Python.h"

#include "static_string.h"

// Since this has no flags set, technically this is a heap-allocated string
// with size zero practically, that doesn't matter because we always do size
// checks before accessing heap data, but that may be confusing. The nice part
// of this choice is a calloc'd array buffer (e.g. from np.empty) is filled
// with empty elements for free
const npy_static_string NPY_EMPTY_STRING = {
        .base = {.direct_buffer = {.flags_and_size = 0, .buf = {0}}}};
// zero-filled, but with the NULL flag set to distinguish from empty string
const npy_static_string NPY_NULL_STRING = {
        .base = {.direct_buffer = {.flags_and_size = NPY_STRING_MISSING,
                                   .buf = {0}}}};

int
is_short_string(const npy_static_string *s)
{
    unsigned char high_byte = s->base.direct_buffer.flags_and_size;
    return (high_byte & NPY_STRING_SHORT) == NPY_STRING_SHORT;
}

int
npy_string_isnull(const npy_static_string *s)
{
    unsigned char high_byte = s->base.direct_buffer.flags_and_size;
    return (high_byte & NPY_STRING_MISSING) == NPY_STRING_MISSING;
}

int
is_not_a_vstring(const npy_static_string *s)
{
    return is_short_string(s) || npy_string_isnull(s);
}

int
npy_string_newsize(const char *init, size_t size, npy_static_string *to_init)
{
    if (to_init == NULL || npy_string_size(to_init) != 0 ||
        size > MAX_STRING_SIZE) {
        return -2;
    }

    if (size == 0) {
        *to_init = NPY_EMPTY_STRING;
        return 0;
    }

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        char *ret_buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

        if (ret_buf == NULL) {
            return -1;
        }

        to_init->base.vstring.size = size;

        memcpy(ret_buf, init, size);

        to_init->base.vstring.buf = ret_buf;
    }
    else {
        // size can be no longer than 7 or 15, depending on CPU architecture
        // in either case, the size data is in at most the least significant 4
        // bits of the byte so it's safe to | with one of 0x10, 0x20, 0x40, or
        // 0x80.
        to_init->base.direct_buffer.flags_and_size = NPY_STRING_SHORT | size;
        memcpy(&(to_init->base.direct_buffer.buf), init, size);
    }

    return 0;
}

int
npy_string_newemptysize(size_t size, npy_static_string *out)
{
    if (out == NULL || npy_string_size(out) != 0 || size > MAX_STRING_SIZE) {
        return -2;
    }

    if (size == 0) {
        *out = NPY_EMPTY_STRING;
        return 0;
    }

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        char *buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

        if (buf == NULL) {
            return -1;
        }

        out->base.vstring.buf = buf;
        out->base.vstring.size = size;
    }
    else {
        out->base.direct_buffer.flags_and_size = NPY_STRING_SHORT | size;
    }

    return 0;
}

void
npy_string_free(npy_static_string *str)
{
    if (is_not_a_vstring(str)) {
        // zero out
        memcpy(str, &NPY_EMPTY_STRING, sizeof(npy_static_string));
    }
    else {
        if (str->base.vstring.size != 0) {
            PyMem_RawFree(str->base.vstring.buf);
        }
        str->base.vstring.buf = NULL;
        str->base.vstring.size = 0;
    }
}

int
npy_string_dup(const npy_static_string *in, npy_static_string *out)
{
    if (npy_string_isnull(in)) {
        *out = NPY_NULL_STRING;
        return 0;
    }

    return npy_string_newsize(npy_string_buf(in), npy_string_size(in), out);
}

int
npy_string_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    size_t s1_size = npy_string_size(s1);
    size_t s2_size = npy_string_size(s2);

    char *s1_buf = npy_string_buf(s1);
    char *s2_buf = npy_string_buf(s2);

    size_t minsize = s1_size < s2_size ? s1_size : s2_size;

    int cmp = strncmp(s1_buf, s2_buf, minsize);

    if (cmp == 0) {
        if (s1_size > minsize) {
            return 1;
        }
        if (s2_size > minsize) {
            return -1;
        }
    }

    return cmp;
}

size_t
npy_string_size(const npy_static_string *s)
{
    if (is_short_string(s)) {
        unsigned char high_byte = s->base.direct_buffer.flags_and_size;
        return high_byte & NPY_SHORT_STRING_SIZE_MASK;
    }
    return s->base.vstring.size;
}

char *
npy_string_buf(const npy_static_string *s)
{
    if (is_short_string(s)) {
        // the cast drops const, is there a better way?
        return (char *)&s->base.direct_buffer.buf[0];
    }
    return s->base.vstring.buf;
}
