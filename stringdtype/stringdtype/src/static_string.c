#include "Python.h"

#include "static_string.h"

// defined this way so EMPTY_STRING has an in-memory representation that is
// distinct from a zero-filled struct, allowing us to use a NULL_STRING
// to represent a sentinel value
const npy_static_string EMPTY_STRING = {0, "\0"};
const npy_static_string NULL_STRING = {0, NULL};

int
npy_string_newsize(const char *init, size_t size, npy_static_string *to_init)
{
    if ((to_init == NULL) || (to_init->buf != NULL) ||
        (npy_string_size(to_init) != 0)) {
        return -2;
    }

    if (size == 0) {
        *to_init = EMPTY_STRING;
        return 0;
    }

    char *ret_buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

    if (ret_buf == NULL) {
        return -1;
    }

    to_init->size = size;

    memcpy(ret_buf, init, size);

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
    str->size = 0;
}

int
npy_string_dup(const npy_static_string *in, npy_static_string *out)
{
    if (npy_string_isnull(in)) {
        out->size = 0;
        out->buf = NULL;
        return 0;
    }
    else {
        return npy_string_newsize(in->buf, in->size, out);
    }
}

int
npy_string_newemptysize(size_t size, npy_static_string *out)
{
    if (out->size != 0 || out->buf != NULL) {
        return -2;
    }

    out->size = size;

    if (size == 0) {
        *out = EMPTY_STRING;
        return 0;
    }

    char *buf = (char *)PyMem_RawMalloc(sizeof(char) * size);

    if (buf == NULL) {
        return -1;
    }

    out->buf = buf;

    return 0;
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

int
npy_string_isnull(const npy_static_string *in)
{
    if (in->size == 0 && in->buf == NULL) {
        return 1;
    }
    return 0;
}

size_t
npy_string_size(const npy_static_string *s)
{
    return s->size;
}

char *
npy_string_buf(const npy_static_string *s)
{
    return s->buf;
}

int
npy_string_size_and_buf(const npy_static_string *s, size_t *size, char **buf)
{
    *size = s->size;
    *buf = s->buf;

    return 0;
}
