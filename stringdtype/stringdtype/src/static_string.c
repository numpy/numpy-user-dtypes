#include "static_string.h"

// defined this way so it has an in-memory representation that is distinct
// from NULL, allowing us to use NULL to represent a sentinel value
const ss EMPTY_STRING = {0, "\0"};

int
ssnewlen(const char *init, size_t len, ss *to_init)
{
    if ((to_init == NULL) || (to_init->buf != NULL) || (to_init->len != 0)) {
        return -2;
    }

    if (len == 0) {
        *to_init = EMPTY_STRING;
        return 0;
    }

    char *ret_buf = (char *)malloc(sizeof(char) * len);

    if (ret_buf == NULL) {
        return -1;
    }

    to_init->len = len;

    memcpy(ret_buf, init, len);

    to_init->buf = ret_buf;

    return 0;
}

void
ssfree(ss *str)
{
    if (str->buf != NULL && str->buf != EMPTY_STRING.buf) {
        free(str->buf);
        str->buf = NULL;
    }
    str->len = 0;
}

int
ssdup(const ss *in, ss *out)
{
    if (ss_isnull(in)) {
        out->len = 0;
        out->buf = NULL;
        return 0;
    }
    else {
        return ssnewlen(in->buf, in->len, out);
    }
}

int
ssnewemptylen(size_t num_bytes, ss *out)
{
    if (out->len != 0 || out->buf != NULL) {
        return -2;
    }

    out->len = num_bytes;

    if (num_bytes == 0) {
        *out = EMPTY_STRING;
        return 0;
    }

    char *buf = (char *)malloc(sizeof(char) * num_bytes);

    if (buf == NULL) {
        return -1;
    }

    out->buf = buf;

    return 0;
}

// same semantics as strcmp
int
sscmp(const ss *s1, const ss *s2)
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
ss_isnull(const ss *in)
{
    if (in->len == 0 && in->buf == NULL) {
        return 1;
    }
    return 0;
}
