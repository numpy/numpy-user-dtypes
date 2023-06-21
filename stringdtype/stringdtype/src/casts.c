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
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
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

    ss *s = NULL;
    ss *os = NULL;

    while (N--) {
        s = (ss *)in;
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
static char *p2p_name = "cast_PandasStringDType_to_PandasStringDType";
static char *s2p_name = "cast_StringDType_to_PandasStringDType";
static char *p2s_name = "cast_PandasStringDType_to_StringDType";

// unicode to string

static NPY_CASTING
unicode_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                      PyArray_DTypeMeta *dtypes[2],
                                      PyArray_Descr *given_descrs[2],
                                      PyArray_Descr *loop_descrs[2],
                                      npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyArray_Descr *new = (PyArray_Descr *)new_stringdtype_instance(
                (PyTypeObject *)dtypes[1]);
        if (new == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = new;
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
        ssfree(out_ss);
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
        s = (ss *)in;
        unsigned char *this_string = NULL;
        size_t n_bytes;
        if (ss_isnull(s)) {
            // lossy but not much else we can do
            this_string = (unsigned char *)"NA";
            n_bytes = 3;
        }
        else {
            this_string = (unsigned char *)(s->buf);
            n_bytes = s->len;
        }
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
string_to_bool(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
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
        s = (ss *)in;
        if (ss_isnull(s)) {
            // numpy treats NaN as truthy, following python
            *out = (npy_bool)1;
        }
        else if (s->len == 0) {
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

// bool to string

static NPY_CASTING
bool_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *dtypes[2],
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyArray_Descr *new = (PyArray_Descr *)new_stringdtype_instance(
                (PyTypeObject *)dtypes[1]);
        if (new == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = new;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_SAFE_CASTING;
}

static int
bool_to_string(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        ss *out_ss = (ss *)out;
        ssfree(out_ss);
        if ((npy_bool)(*in) == 1) {
            if (ssnewlen("True", 4, out_ss) < 0) {
                gil_error(PyExc_MemoryError, "ssnewlen failed");
                return -1;
            }
        }
        else if ((npy_bool)(*in) == 0) {
            if (ssnewlen("False", 5, out_ss) < 0) {
                gil_error(PyExc_MemoryError, "ssnewlen failed");
                return -1;
            }
        }
        else {
            gil_error(PyExc_RuntimeError,
                      "invalid value encountered in bool to string cast");
            return -1;
        }
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot b2s_slots[] = {
        {NPY_METH_resolve_descriptors, &bool_to_string_resolve_descriptors},
        {NPY_METH_strided_loop, &bool_to_string},
        {0, NULL}};

static char *b2s_name = "cast_Bool_to_StringDType";

static PyObject *
string_to_pylong(char *in)
{
    ss *s = (ss *)in;
    if (ss_isnull(s)) {
        PyErr_SetString(
                PyExc_ValueError,
                "Arrays with missing data cannot be converted to integers");
        return NULL;
    }
    // interpret as an integer in base 10
    return PyLong_FromString(s->buf, NULL, 10);
}

static npy_longlong
string_to_uint(char *in, npy_ulonglong *value)
{
    PyObject *pylong_value = string_to_pylong(in);
    *value = PyLong_AsUnsignedLongLong(pylong_value);
    if (*value == (unsigned long long)-1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static npy_longlong
string_to_int(char *in, npy_longlong *value)
{
    PyObject *pylong_value = string_to_pylong(in);
    *value = PyLong_AsLongLong(pylong_value);
    if (*value == -1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static int
pylong_to_string(PyObject *pylong_val, char *out)
{
    if (pylong_val == NULL) {
        return -1;
    }
    PyObject *pystr_val = PyObject_Str(pylong_val);
    Py_DECREF(pylong_val);
    if (pystr_val == NULL) {
        return -1;
    }
    Py_ssize_t length;
    const char *cstr_val = PyUnicode_AsUTF8AndSize(pystr_val, &length);
    if (cstr_val == NULL) {
        return -1;
    }
    ss *out_ss = (ss *)out;
    ssfree(out_ss);
    if (ssnewlen(cstr_val, length, out_ss) < 0) {
        PyErr_SetString(PyExc_MemoryError, "ssnewlen failed");
        Py_DECREF(pystr_val);
        return -1;
    }
    // implicitly deallocates cstr_val as well
    Py_DECREF(pystr_val);
    return 0;
}

static int
int_to_string(long long in, char *out)
{
    PyObject *pylong_val = PyLong_FromLongLong(in);
    return pylong_to_string(pylong_val, out);
}

static int
uint_to_string(unsigned long long in, char *out)
{
    PyObject *pylong_val = PyLong_FromUnsignedLongLong(in);
    return pylong_to_string(pylong_val, out);
}

static NPY_CASTING
int_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                  PyArray_DTypeMeta *dtypes[2],
                                  PyArray_Descr *given_descrs[2],
                                  PyArray_Descr *loop_descrs[2],
                                  npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyArray_Descr *new = (PyArray_Descr *)new_stringdtype_instance(
                (PyTypeObject *)dtypes[1]);
        if (new == NULL) {
            return (NPY_CASTING)-1;
        }
        loop_descrs[1] = new;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    return NPY_UNSAFE_CASTING;
}

#define STRING_TO_INT(typename, typekind, shortname, numpy_tag, printf_code, \
                      npy_longtype)                                          \
    static NPY_CASTING string_to_##typename##_resolve_descriptors(           \
            PyObject *NPY_UNUSED(self),                                      \
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                        \
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],   \
            npy_intp *NPY_UNUSED(view_offset))                               \
    {                                                                        \
        if (given_descrs[1] == NULL) {                                       \
            loop_descrs[1] = PyArray_DescrNewFromType(numpy_tag);            \
        }                                                                    \
        else {                                                               \
            Py_INCREF(given_descrs[1]);                                      \
            loop_descrs[1] = given_descrs[1];                                \
        }                                                                    \
                                                                             \
        Py_INCREF(given_descrs[0]);                                          \
        loop_descrs[0] = given_descrs[0];                                    \
                                                                             \
        return NPY_UNSAFE_CASTING;                                           \
    }                                                                        \
                                                                             \
    static int string_to_##typename(                                         \
            PyArrayMethod_Context * NPY_UNUSED(context), char *const data[], \
            npy_intp const dimensions[], npy_intp const strides[],           \
            NpyAuxData *NPY_UNUSED(auxdata))                                 \
    {                                                                        \
        npy_intp N = dimensions[0];                                          \
        char *in = data[0];                                                  \
        npy_##typename *out = (npy_##typename *)data[1];                     \
                                                                             \
        npy_intp in_stride = strides[0];                                     \
        npy_intp out_stride = strides[1] / sizeof(npy_##typename);           \
                                                                             \
        while (N--) {                                                        \
            npy_longtype value;                                              \
            if (string_to_##typekind(in, &value) != 0) {                     \
                return -1;                                                   \
            }                                                                \
            *out = (npy_##typename)value;                                    \
            if (*out != value) {                                             \
                /* out of bounds, raise error following NEP 50 behavior */   \
                PyErr_Format(PyExc_OverflowError,                            \
                             "Integer %" #printf_code                        \
                             " is out of bounds "                            \
                             "for " #typename,                               \
                             value);                                         \
                return -1;                                                   \
            }                                                                \
            in += in_stride;                                                 \
            out += out_stride;                                               \
        }                                                                    \
                                                                             \
        return 0;                                                            \
    }                                                                        \
                                                                             \
    static PyType_Slot s2##shortname##_slots[] = {                           \
            {NPY_METH_resolve_descriptors,                                   \
             &string_to_##typename##_resolve_descriptors},                   \
            {NPY_METH_strided_loop, &string_to_##typename},                  \
            {0, NULL}};                                                      \
                                                                             \
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;

#define INT_TO_STRING(typename, typekind, shortname, longtype)              \
    static int typename##_to_string(                                        \
            PyArrayMethod_Context *NPY_UNUSED(context), char *const data[], \
            npy_intp const dimensions[], npy_intp const strides[],          \
            NpyAuxData *NPY_UNUSED(auxdata))                                \
    {                                                                       \
        npy_intp N = dimensions[0];                                         \
        npy_##typename *in = (npy_##typename *)data[0];                     \
        char *out = data[1];                                                \
                                                                            \
        npy_intp in_stride = strides[0] / sizeof(npy_##typename);           \
        npy_intp out_stride = strides[1];                                   \
                                                                            \
        while (N--) {                                                       \
            if (typekind##_to_string((longtype)*in, out) != 0) {            \
                return -1;                                                  \
            }                                                               \
            in += in_stride;                                                \
            out += out_stride;                                              \
        }                                                                   \
                                                                            \
        return 0;                                                           \
    }                                                                       \
                                                                            \
    static PyType_Slot shortname##2s_slots [] = {                           \
            {NPY_METH_resolve_descriptors,                                  \
             &int_to_string_resolve_descriptors},                           \
            {NPY_METH_strided_loop, &typename##_to_string},                 \
            {0, NULL}};                                                     \
                                                                            \
    static char *shortname##2s_name = "cast_" #typename "_to_StringDType";

STRING_TO_INT(int8, int, i8, NPY_INT8, lli, npy_longlong)
INT_TO_STRING(int8, int, i8, long long)

STRING_TO_INT(int16, int, i16, NPY_INT16, lli, npy_longlong)
INT_TO_STRING(int16, int, i16, long long)

STRING_TO_INT(int32, int, i32, NPY_INT32, lli, npy_longlong)
INT_TO_STRING(int32, int, i32, long long)

STRING_TO_INT(int64, int, i64, NPY_INT64, lli, npy_longlong)
INT_TO_STRING(int64, int, i64, long long)

STRING_TO_INT(uint8, uint, ui8, NPY_UINT8, llu, npy_ulonglong)
INT_TO_STRING(uint8, uint, ui8, unsigned long long)

STRING_TO_INT(uint16, uint, ui16, NPY_UINT16, llu, npy_ulonglong)
INT_TO_STRING(uint16, uint, ui16, unsigned long long)

STRING_TO_INT(uint32, uint, ui32, NPY_UINT32, llu, npy_ulonglong)
INT_TO_STRING(uint32, uint, ui32, unsigned long long)

STRING_TO_INT(uint64, uint, ui64, NPY_UINT64, llu, npy_ulonglong)
INT_TO_STRING(uint64, uint, ui64, unsigned long long)

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
get_casts(PyArray_DTypeMeta *this, PyArray_DTypeMeta *other)
{
    char *t2t_name = NULL;

    if (this == (PyArray_DTypeMeta *)&StringDType) {
        t2t_name = s2s_name;
    }
    else {
        t2t_name = p2p_name;
    }

    PyArray_DTypeMeta **t2t_dtypes = get_dtypes(this, this);

    PyArrayMethod_Spec *ThisToThisCastSpec =
            get_cast_spec(t2t_name, NPY_NO_CASTING,
                          NPY_METH_SUPPORTS_UNALIGNED, t2t_dtypes, s2s_slots);

    PyArrayMethod_Spec *ThisToOtherCastSpec = NULL;
    PyArrayMethod_Spec *OtherToThisCastSpec = NULL;

    int is_pandas = (this == (PyArray_DTypeMeta *)&PandasStringDType);

    int num_casts = 21;

    if (is_pandas) {
        num_casts += 2;

        PyArray_DTypeMeta **t2o_dtypes = get_dtypes(this, other);

        ThisToOtherCastSpec = get_cast_spec(p2s_name, NPY_NO_CASTING,
                                            NPY_METH_SUPPORTS_UNALIGNED,
                                            t2o_dtypes, s2s_slots);

        PyArray_DTypeMeta **o2t_dtypes = get_dtypes(other, this);

        OtherToThisCastSpec = get_cast_spec(s2p_name, NPY_NO_CASTING,
                                            NPY_METH_SUPPORTS_UNALIGNED,
                                            o2t_dtypes, s2s_slots);
    }

    PyArray_DTypeMeta **u2s_dtypes = get_dtypes(&PyArray_UnicodeDType, this);

    PyArrayMethod_Spec *UnicodeToStringCastSpec = get_cast_spec(
            u2s_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            u2s_dtypes, u2s_slots);

    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(this, &PyArray_UnicodeDType);

    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
            s2u_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2u_dtypes, s2u_slots);

    PyArray_DTypeMeta **s2b_dtypes = get_dtypes(this, &PyArray_BoolDType);

    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
            s2b_name, NPY_UNSAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2b_dtypes, s2b_slots);

    PyArray_DTypeMeta **b2s_dtypes = get_dtypes(&PyArray_BoolDType, this);

    PyArrayMethod_Spec *BoolToStringCastSpec = get_cast_spec(
            b2s_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            b2s_dtypes, b2s_slots);

    PyArray_DTypeMeta **s2i8_dtypes = get_dtypes(this, &PyArray_Int8DType);

    PyArrayMethod_Spec *StringToInt8CastSpec =
            get_cast_spec(s2i8_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, s2i8_dtypes, s2i8_slots);

    PyArray_DTypeMeta **i82s_dtypes = get_dtypes(&PyArray_Int8DType, this);

    PyArrayMethod_Spec *Int8ToStringCastSpec =
            get_cast_spec(i82s_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, i82s_dtypes, i82s_slots);

    PyArray_DTypeMeta **s2ui8_dtypes = get_dtypes(this, &PyArray_UInt8DType);

    PyArrayMethod_Spec *StringToUInt8CastSpec =
            get_cast_spec(s2ui8_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, s2ui8_dtypes, s2ui8_slots);

    PyArray_DTypeMeta **ui82s_dtypes = get_dtypes(&PyArray_UInt8DType, this);

    PyArrayMethod_Spec *UInt8ToStringCastSpec =
            get_cast_spec(ui82s_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, ui82s_dtypes, ui82s_slots);

    PyArray_DTypeMeta **s2i16_dtypes = get_dtypes(this, &PyArray_Int16DType);

    PyArrayMethod_Spec *StringToInt16CastSpec =
            get_cast_spec(s2i16_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, s2i16_dtypes, s2i16_slots);

    PyArray_DTypeMeta **i162s_dtypes = get_dtypes(&PyArray_Int16DType, this);

    PyArrayMethod_Spec *Int16ToStringCastSpec =
            get_cast_spec(i162s_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, i162s_dtypes, i162s_slots);

    PyArray_DTypeMeta **s2ui16_dtypes = get_dtypes(this, &PyArray_UInt16DType);

    PyArrayMethod_Spec *StringToUInt16CastSpec = get_cast_spec(
            s2ui16_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            s2ui16_dtypes, s2ui16_slots);

    PyArray_DTypeMeta **ui162s_dtypes = get_dtypes(&PyArray_UInt16DType, this);

    PyArrayMethod_Spec *UInt16ToStringCastSpec = get_cast_spec(
            ui162s_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            ui162s_dtypes, ui162s_slots);

    PyArray_DTypeMeta **s2i32_dtypes = get_dtypes(this, &PyArray_Int32DType);

    PyArrayMethod_Spec *StringToInt32CastSpec =
            get_cast_spec(s2i32_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, s2i32_dtypes, s2i32_slots);

    PyArray_DTypeMeta **i322s_dtypes = get_dtypes(&PyArray_Int32DType, this);

    PyArrayMethod_Spec *Int32ToStringCastSpec =
            get_cast_spec(i322s_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, i322s_dtypes, i322s_slots);

    PyArray_DTypeMeta **s2ui32_dtypes = get_dtypes(this, &PyArray_UInt32DType);

    PyArrayMethod_Spec *StringToUInt32CastSpec = get_cast_spec(
            s2ui32_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            s2ui32_dtypes, s2ui32_slots);

    PyArray_DTypeMeta **ui322s_dtypes = get_dtypes(&PyArray_UInt32DType, this);

    PyArrayMethod_Spec *UInt32ToStringCastSpec = get_cast_spec(
            ui322s_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            ui322s_dtypes, ui322s_slots);

    PyArray_DTypeMeta **s2i64_dtypes = get_dtypes(this, &PyArray_Int64DType);

    PyArrayMethod_Spec *StringToInt64CastSpec =
            get_cast_spec(s2i64_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, s2i64_dtypes, s2i64_slots);

    PyArray_DTypeMeta **i642s_dtypes = get_dtypes(&PyArray_Int64DType, this);

    PyArrayMethod_Spec *Int64ToStringCastSpec =
            get_cast_spec(i642s_name, NPY_UNSAFE_CASTING,
                          NPY_METH_REQUIRES_PYAPI, i642s_dtypes, i642s_slots);

    PyArray_DTypeMeta **s2ui64_dtypes = get_dtypes(this, &PyArray_UInt64DType);

    PyArrayMethod_Spec *StringToUInt64CastSpec = get_cast_spec(
            s2ui64_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            s2ui64_dtypes, s2ui64_slots);

    PyArray_DTypeMeta **ui642s_dtypes = get_dtypes(&PyArray_UInt64DType, this);

    PyArrayMethod_Spec *UInt64ToStringCastSpec = get_cast_spec(
            ui642s_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI,
            ui642s_dtypes, ui642s_slots);

    PyArrayMethod_Spec **casts = NULL;

    casts = malloc((num_casts + 1) * sizeof(PyArrayMethod_Spec *));

    casts[0] = ThisToThisCastSpec;
    casts[1] = UnicodeToStringCastSpec;
    casts[2] = StringToUnicodeCastSpec;
    casts[3] = StringToBoolCastSpec;
    casts[4] = BoolToStringCastSpec;
    casts[5] = StringToInt8CastSpec;
    casts[6] = Int8ToStringCastSpec;
    casts[7] = StringToInt16CastSpec;
    casts[8] = Int16ToStringCastSpec;
    casts[9] = StringToInt32CastSpec;
    casts[10] = Int32ToStringCastSpec;
    casts[11] = StringToInt64CastSpec;
    casts[12] = Int64ToStringCastSpec;
    casts[13] = StringToUInt8CastSpec;
    casts[14] = UInt8ToStringCastSpec;
    casts[15] = StringToUInt16CastSpec;
    casts[16] = UInt16ToStringCastSpec;
    casts[17] = StringToUInt32CastSpec;
    casts[18] = UInt32ToStringCastSpec;
    casts[19] = StringToUInt64CastSpec;
    casts[20] = UInt64ToStringCastSpec;
    if (is_pandas) {
        casts[21] = ThisToOtherCastSpec;
        casts[22] = OtherToThisCastSpec;
        casts[23] = NULL;
    }
    else {
        casts[21] = NULL;
    }

    assert(casts[num_casts] == NULL);

    return casts;
}
