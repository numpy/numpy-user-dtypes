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

// string to string

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
string_to_string(PyArrayMethod_Context *NPY_UNUSED(context),
                 char *const data[], npy_intp const dimensions[],
                 npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    ss *s = NULL, *os = NULL;

    while (N--) {
        load_string(in, &s);
        os = (ss *)out;
        ssfree(os);
        if (ssdup(s, os) < 0) {
            gil_error(PyExc_MemoryError, "ssdup failed");
            return -1;
        }

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot s2s_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_string},
        {NPY_METH_unaligned_strided_loop, &string_to_string},
        {0, NULL}};

static char *s2s_name = "cast_StringDType_to_StringDType";

// unicode to string

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
utf8_size(const Py_UCS4 *codepoints, long max_length, size_t *num_codepoints,
          size_t *utf8_bytes)
{
    size_t ucs4len = max_length;

    while (ucs4len > 0 && codepoints[ucs4len - 1] == 0) {
        ucs4len--;
    }
    // ucs4len is now the number of codepoints that aren't trailing nulls.

    size_t num_bytes = 0;

    for (size_t i = 0; i < ucs4len; i++) {
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
    char *out = data[1];

    // 4 bytes per UCS4 character
    npy_intp in_stride = strides[0] / 4;
    npy_intp out_stride = strides[1];

    while (N--) {
        size_t out_num_bytes = 0;
        size_t num_codepoints = 0;
        if (utf8_size(in, max_in_size, &num_codepoints, &out_num_bytes) ==
            -1) {
            gil_error(PyExc_TypeError, "Invalid unicode code point found");
            return -1;
        }
        ss *out_ss = (ss *)out;
        if (ssnewemptylen(out_num_bytes, out_ss) < 0) {
            gil_error(PyExc_MemoryError, "ssnewemptylen failed");
            return -1;
        }
        char *out_buf = out_ss->buf;
        for (size_t i = 0; i < num_codepoints; i++) {
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

// string to unicode

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
size_t
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
    char *in = data[0];
    Py_UCS4 *out = (Py_UCS4 *)data[1];
    npy_intp in_stride = strides[0];
    // 4 bytes per UCS4 character
    npy_intp out_stride = strides[1] / 4;
    // max number of 4 byte UCS4 characters that can fit in the output
    long max_out_size = (context->descriptors[1]->elsize) / 4;

    ss *s = NULL;

    while (N--) {
        load_string(in, &s);
        unsigned char *this_string = (unsigned char *)(s->buf);
        size_t n_bytes = s->len;
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

// string to bool

static NPY_CASTING
string_to_bool_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        loop_descrs[1] = PyArray_DescrNewFromType(NPY_BOOL);
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

static int
string_to_bool(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    ss *s = NULL;

    while (N--) {
        load_string(in, &s);
        if (s->len == 0) {
            *out = (npy_bool)0;
        }
        else {
            *out = (npy_bool)1;
        }

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot s2b_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_bool_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_bool},
        {0, NULL}};

static char *s2b_name = "cast_StringDType_to_Bool";

PyArrayMethod_Spec *
get_cast_spec(const char *name, NPY_CASTING casting,
              NPY_ARRAYMETHOD_FLAGS flags, PyArray_DTypeMeta **dtypes,
              PyType_Slot *slots)
{
    PyArrayMethod_Spec *ret = malloc(sizeof(PyArrayMethod_Spec));

    ret->name = name;
    ret->nin = 1;
    ret->nout = 1;
    ret->casting = casting;
    ret->flags = flags;
    ret->dtypes = dtypes;
    ret->slots = slots;

    return ret;
}

PyArray_DTypeMeta **
get_dtypes(PyArray_DTypeMeta *dt1, PyArray_DTypeMeta *dt2)
{
    PyArray_DTypeMeta **ret = malloc(2 * sizeof(PyArray_DTypeMeta *));

    ret[0] = dt1;
    ret[1] = dt2;

    return ret;
}

PyArrayMethod_Spec **
get_casts(void)
{
    PyArray_DTypeMeta **s2s_dtypes = get_dtypes(NULL, NULL);

    PyArrayMethod_Spec *StringToStringCastSpec =
            get_cast_spec(s2s_name, NPY_NO_CASTING,
                          NPY_METH_SUPPORTS_UNALIGNED, s2s_dtypes, s2s_slots);

    PyArray_DTypeMeta **u2s_dtypes = get_dtypes(&PyArray_UnicodeDType, NULL);

    PyArrayMethod_Spec *UnicodeToStringCastSpec = get_cast_spec(
            u2s_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            u2s_dtypes, u2s_slots);

    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(NULL, &PyArray_UnicodeDType);

    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
            s2u_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2u_dtypes, s2u_slots);

    PyArray_DTypeMeta **s2b_dtypes = get_dtypes(NULL, &PyArray_BoolDType);

    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
            s2b_name, NPY_UNSAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2b_dtypes, s2b_slots);

    PyArrayMethod_Spec **casts = malloc(5 * sizeof(PyArrayMethod_Spec *));
    casts[0] = StringToStringCastSpec;
    casts[1] = UnicodeToStringCastSpec;
    casts[2] = StringToUnicodeCastSpec;
    casts[3] = StringToBoolCastSpec;
    casts[4] = NULL;

    return casts;
}
