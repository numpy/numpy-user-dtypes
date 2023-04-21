#include "static_string.h"

static ss EMPTY = {0, ""};

int
ssnewlen(const char *init, size_t len, ss *to_init)
{
    if ((to_init == NULL) || (to_init->buf != NULL) || (to_init->len != 0)) {
        return -2;
    }

    if (len == 0) {
        to_init->len = 0;
        to_init->buf = EMPTY.buf;
    }

    // one extra byte for null terminator
    char *ret_buf = (char *)malloc(sizeof(char) * (len + 1));

    if (ret_buf == NULL) {
        return -1;
    }

    to_init->len = len;

    if (len > 0) {
        memcpy(ret_buf, init, len);
    }

    ret_buf[len] = '\0';

    to_init->buf = ret_buf;

    return 0;
}

void
ssfree(ss *str)
{
    if (str->buf != NULL) {
        if (str->buf != EMPTY.buf) {
            free(str->buf);
        }
        str->buf = NULL;
    }
    str->len = 0;
}

int
ssdup(ss *in, ss *out)
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

    char *buf = (char *)malloc(sizeof(char) * (num_bytes + 1));

    if (buf == NULL) {
        return -1;
    }

    out->buf = buf;
    out->len = num_bytes;

    return 0;
}

int
ss_isnull(ss *in)
{
    if (in->len == 0 && in->buf == NULL) {
        return 1;
    }
    return 0;
}
