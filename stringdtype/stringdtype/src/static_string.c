#include "static_string.h"

// allocates a new ss string of length len, filling with the contents of init
ss *
ssnewlen(const char *init, size_t len)
{
    // one extra byte for null terminator
    ss *ret = (ss *)malloc(sizeof(ss) + sizeof(char) * (len + 1));

    if (ret == NULL) {
        return NULL;
    }

    ret->len = len;

    if (len > 0) {
        memcpy(ret->buf, init, len);
    }

    ret->buf[len] = '\0';

    return ret;
}

// returns a new heap-allocated copy of input string *s*
ss *
ssdup(const ss *s)
{
    return ssnewlen(s->buf, s->len);
}

// returns a new, empty string of length len
// does not do any initialization, the caller must
// initialize and null-terminate the string
ss *
ssnewempty(size_t len)
{
    ss *ret = (ss *)malloc(sizeof(ss) + sizeof(char) * (len + 1));
    ret->len = len;
    return ret;
}
