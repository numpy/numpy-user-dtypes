#include "casts.h"

#include "dtype.h"
#include "static_string.h"

void
gil_error(PyObject *type, const char *msg)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    PyErr_SetString(type, msg);
    PyGILState_Release(gstate);
}

static NPY_CASTING
string_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *view_offset)
{
    if (given_descrs[1] == NULL) {
        StringDTypeObject *new = new_stringdtype_instance();
        if (new == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = (PyArray_Descr *)new;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    *view_offset = 0;

    return NPY_NO_CASTING;
}

static int
string_to_string(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    ss **in = (ss **)data[0];
    ss **out = (ss **)data[1];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp in_stride = strides[0] / context->descriptors[0]->elsize;
    npy_intp out_stride = strides[1] / context->descriptors[1]->elsize;

    while (N--) {
        out[0] = ssdup(in[0]);
        if (out[0] == NULL) {
            gil_error(PyExc_MemoryError, "ssdup failed");
            return -1;
        }
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyArray_DTypeMeta *s2s_dtypes[2] = {NULL, NULL};

static PyType_Slot s2s_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_string},
        {NPY_METH_unaligned_strided_loop, &string_to_string},
        {0, NULL}};

PyArrayMethod_Spec StringToStringCastSpec = {
        .name = "cast_StringDType_to_StringDType",
        .nin = 1,
        .nout = 1,
        .casting = NPY_NO_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = s2s_dtypes,
        .slots = s2s_slots,
};

static NPY_CASTING
unicode_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                      PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                      PyArray_Descr *given_descrs[2],
                                      PyArray_Descr *loop_descrs[2],
                                      npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        StringDTypeObject *new = new_stringdtype_instance();
        if (new == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = (PyArray_Descr *)new;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_SAFE_CASTING;
}

// Find the number of bytes, *utf8_bytes*, needed to store the string
// represented by *codepoints* in UTF-8. The array of *codepoints* is
// *max_length* long, but may be padded with null codepoints. *num_codepoints*
// is the number of codepoints that are not trailing null codepoints. Returns
// 0 on success and -1 when an invalid code point is found.
static int
utf8_size(Py_UCS4 *codepoints, long max_length, size_t *num_codepoints,
          size_t *utf8_bytes)
{
    size_t ucs4len = max_length;

    while (ucs4len > 0 && codepoints[ucs4len - 1] == 0) {
        ucs4len--;
    }
    // ucs4len is now the number of codepoints that aren't trailing nulls.

    size_t num_bytes = 0;

    for (int i = 0; i < ucs4len; i++) {
        Py_UCS4 code = codepoints[i];

        if (code <= 0x7F) {
            num_bytes += 1;
        }
        else if (code <= 0x07FF) {
            num_bytes += 2;
        }
        else if (code <= 0xFFFF) {
            if ((code >= 0xD800) && (code <= 0xDFFF)) {
                // surrogates are invalid UCS4 code points
                return -1;
            }
            num_bytes += 3;
        }
        else if (code <= 0x10FFFF) {
            num_bytes += 4;
        }
        else {
            // codepoint is outside the valid unicode range
            return -1;
        }
    }

    *num_codepoints = ucs4len;
    *utf8_bytes = num_bytes;

    return 0;
}

// Converts UCS4 code point *code* to 4-byte character array *c*. Assumes *c*
// is a zero-filled 4 byte array and *code* is a valid codepoint and does not
// do any error checking! Returns the number of bytes in the UTF-8 character.
static size_t
ucs4_code_to_utf8_char(const Py_UCS4 code, char *c)
{
    if (code <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        c[0] = (char)code;
        return 1;
    }
    else if (code <= 0x07FF) {
        // 00000yyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xC0 | (code >> 6));
        c[1] = (0x80 | (code & 0x3F));
        return 2;
    }
    else if (code <= 0xFFFF) {
        // xxxxyyyy yyzzzzzz -> 110yyyyy 10zzzzzz
        c[0] = (0xe0 | (code >> 12));
        c[1] = (0x80 | ((code >> 6) & 0x3f));
        c[2] = (0x80 | (code & 0x3f));
        return 3;
    }
    else {
        // 00wwwxx xxxxyyyy yyzzzzzz -> 11110www 10xxxxxx 10yyyyyy 10zzzzzz
        c[0] = (0xf0 | (code >> 18));
        c[1] = (0x80 | ((code >> 12) & 0x3f));
        c[2] = (0x80 | ((code >> 6) & 0x3f));
        c[3] = (0x80 | (code & 0x3f));
        return 4;
    }
}

static int
unicode_to_string(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long max_in_size = (descrs[0]->elsize) / 4;

    npy_intp N = dimensions[0];
    Py_UCS4 *in = (Py_UCS4 *)data[0];
    ss **out = (ss **)data[1];

    // 4 bytes per UCS4 character
    npy_intp in_stride = strides[0] / 4;
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp out_stride = strides[1] / context->descriptors[1]->elsize;

    while (N--) {
        size_t out_num_bytes = 0;
        size_t num_codepoints = 0;
        if (utf8_size(in, max_in_size, &num_codepoints, &out_num_bytes) ==
            -1) {
            gil_error(PyExc_TypeError, "Invalid unicode code point found");
            return -1;
        }
        ss *out_ss = ssnewempty(out_num_bytes);
        if (out_ss == NULL) {
            gil_error(PyExc_MemoryError, "ssnewempty failed");
        }
        char *out_buf = out_ss->buf;
        for (int i = 0; i < num_codepoints; i++) {
            // get code point
            Py_UCS4 code = in[i];

            // will be filled with UTF-8 bytes
            char utf8_c[4] = {0};

            // we already checked for invalid code points above,
            // so no need to do error checking here
            size_t num_bytes = ucs4_code_to_utf8_char(code, utf8_c);

            // copy utf8_c into out_buf
            strncpy(out_buf, utf8_c, num_bytes);

            // increment out_buf by the size of the character
            out_buf += num_bytes;
        }

        // reset out_buf to the beginning of the string
        out_buf -= out_num_bytes;

        // pad string with null character
        out_buf[out_num_bytes] = '\0';

        // set out to the address of the beginning of the string
        out[0] = out_ss;

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot u2s_slots[] = {
        {NPY_METH_resolve_descriptors, &unicode_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, &unicode_to_string},
        {0, NULL}};

static char *u2s_name = "cast_Unicode_to_StringDType";

static NPY_CASTING
string_to_unicode_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                      PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                      PyArray_Descr *given_descrs[2],
                                      PyArray_Descr *loop_descrs[2],
                                      npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        // currently there's no way to determine the correct output
        // size, so set an error and bail
        PyErr_SetString(
                PyExc_TypeError,
                "Casting from StringDType to a fixed-width dtype with an "
                "unspecified size is not currently supported, specify "
                "an explicit size for the output dtype instead.");
        return (NPY_CASTING)-1;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

// Given UTF-8 bytes in *c*, sets *code* to the corresponding unicode
// codepoint for the next character, returning the size of the character in
// bytes. Does not do any validation or error checking: assumes *c* is valid
// utf-8
static size_t
utf8_char_to_ucs4_code(unsigned char *c, Py_UCS4 *code)
{
    if (c[0] <= 0x7F) {
        // 0zzzzzzz -> 0zzzzzzz
        *code = (Py_UCS4)(c[0]);
        return 1;
    }
    else if (c[0] <= 0xDF) {
        // 110yyyyy 10zzzzzz -> 00000yyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 6) + c[1]) - ((0xC0 << 6) + 0x80));
        return 2;
    }
    else if (c[0] <= 0xEF) {
        // 1110xxxx 10yyyyyy 10zzzzzz -> xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 12) + (c[1] << 6) + c[2]) -
                          ((0xE0 << 12) + (0x80 << 6) + 0x80));
        return 3;
    }
    else {
        // 11110www 10xxxxxx 10yyyyyy 10zzzzzz -> 000wwwxx xxxxyyyy yyzzzzzz
        *code = (Py_UCS4)(((c[0] << 18) + (c[1] << 12) + (c[2] << 6) + c[3]) -
                          ((0xF0 << 18) + (0x80 << 12) + (0x80 << 6) + 0x80));
        return 4;
    }
}

static int
string_to_unicode(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    ss **in = (ss **)data[0];
    Py_UCS4 *out = (Py_UCS4 *)data[1];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp in_stride = strides[0] / context->descriptors[0]->elsize;
    // 4 bytes per UCS4 character
    npy_intp out_stride = strides[1] / 4;
    // max number of 4 byte UCS4 characters that can fit in the output
    long max_out_size = (context->descriptors[1]->elsize) / 4;

    while (N--) {
        unsigned char *this_string = (unsigned char *)((*in)->buf);
        size_t n_bytes = (*in)->len;
        size_t tot_n_bytes = 0;

        for (int i = 0; i < max_out_size; i++) {
            Py_UCS4 code;

            // get code point for character this_string is currently pointing
            // too
            size_t num_bytes = utf8_char_to_ucs4_code(this_string, &code);

            // move to next character
            this_string += num_bytes;
            tot_n_bytes += num_bytes;

            // set output codepoint
            out[i] = code;

            // stop if we've exhausted the input string
            if (tot_n_bytes >= n_bytes) {
                break;
            }
        }
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot s2u_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_unicode_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_unicode},
        {0, NULL}};

static char *s2u_name = "cast_StringDType_to_Unicode";

PyArrayMethod_Spec **
get_casts(void)
{
    PyArray_DTypeMeta **u2s_dtypes = malloc(2 * sizeof(PyArray_DTypeMeta *));
    u2s_dtypes[0] = &PyArray_UnicodeDType;
    u2s_dtypes[1] = NULL;

    PyArrayMethod_Spec *UnicodeToStringCastSpec =
            malloc(sizeof(PyArrayMethod_Spec));

    UnicodeToStringCastSpec->name = u2s_name;
    UnicodeToStringCastSpec->nin = 1;
    UnicodeToStringCastSpec->nout = 1;
    UnicodeToStringCastSpec->casting = NPY_SAFE_CASTING;
    UnicodeToStringCastSpec->flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    UnicodeToStringCastSpec->dtypes = u2s_dtypes;
    UnicodeToStringCastSpec->slots = u2s_slots;

    PyArray_DTypeMeta **s2u_dtypes = malloc(2 * sizeof(PyArray_DTypeMeta *));
    s2u_dtypes[0] = NULL;
    s2u_dtypes[1] = &PyArray_UnicodeDType;

    PyArrayMethod_Spec *StringToUnicodeCastSpec =
            malloc(sizeof(PyArrayMethod_Spec));

    StringToUnicodeCastSpec->name = s2u_name;
    StringToUnicodeCastSpec->nin = 1;
    StringToUnicodeCastSpec->nout = 1;
    StringToUnicodeCastSpec->casting = NPY_SAFE_CASTING;
    StringToUnicodeCastSpec->flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    StringToUnicodeCastSpec->dtypes = s2u_dtypes;
    StringToUnicodeCastSpec->slots = s2u_slots;

    PyArrayMethod_Spec **casts = malloc(4 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &StringToStringCastSpec;
    casts[1] = UnicodeToStringCastSpec;
    casts[2] = StringToUnicodeCastSpec;
    casts[3] = NULL;

    return casts;
}
