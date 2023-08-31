#include "Python.h"

#include "static_string.h"

// defined this way so EMPTY_STRING has an in-memory representation that is
// distinct from a zero-filled struct, allowing us to use a NULL_STRING
// to represent a sentinel value
const npy_static_string EMPTY_STRING = {0, "\0"};
const npy_static_string NULL_STRING = {0, NULL};

int
npy_string_newlen(const char *init, size_t len, npy_static_string *to_init)
{
    if ((to_init == NULL) || (to_init->buf != NULL) || (to_init->len != 0)) {
        return -2;
    }

    if (len == 0) {
        *to_init = EMPTY_STRING;
        return 0;
    }

    char *ret_buf = (char *)PyMem_RawMalloc(sizeof(char) * len);

    if (ret_buf == NULL) {
        return -1;
    }

    to_init->len = len;

    memcpy(ret_buf, init, len);

    to_init->buf = ret_buf;

    return 0;
}

void
npy_string_free(npy_static_string *str)
{
    if (str->buf != NULL && str->buf != EMPTY_STRING.buf) {
        PyMem_RawFree(str->buf);
        str->buf = NULL;
    }
    str->len = 0;
}

int
npy_string_dup(const npy_static_string *in, npy_static_string *out)
{
    if (npy_string_isnull(in)) {
        out->len = 0;
        out->buf = NULL;
        return 0;
    }
    else {
        return npy_string_newlen(in->buf, in->len, out);
    }
}

int
npy_string_newemptylen(size_t num_bytes, npy_static_string *out)
{
    if (out->len != 0 || out->buf != NULL) {
        return -2;
    }

    out->len = num_bytes;

    if (num_bytes == 0) {
        *out = EMPTY_STRING;
        return 0;
    }

    char *buf = (char *)PyMem_RawMalloc(sizeof(char) * num_bytes);

    if (buf == NULL) {
        return -1;
    }

    out->buf = buf;

    return 0;
}

int
npy_string_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    size_t minlen = s1->len < s2->len ? s1->len : s2->len;

    int cmp = strncmp(s1->buf, s2->buf, minlen);

    if (cmp == 0) {
        if (s1->len > minlen) {
            return 1;
        }
        if (s2->len > minlen) {
            return -1;
        }
    }

    return cmp;
}

int
npy_string_isnull(const npy_static_string *in)
{
    if (in->len == 0 && in->buf == NULL) {
        return 1;
    }
    return 0;
}
