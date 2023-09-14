#include "casts.h"

#include "dtype.h"
#include "static_string.h"

#define ANY_TO_STRING_RESOLVE_DESCRIPTORS(safety)                          \
    static NPY_CASTING any_to_string_##safety##_resolve_descriptors(       \
            PyObject *NPY_UNUSED(self),                                    \
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                      \
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2], \
            npy_intp *NPY_UNUSED(view_offset))                             \
    {                                                                      \
        if (given_descrs[1] == NULL) {                                     \
            PyArray_Descr *new =                                           \
                    (PyArray_Descr *)new_stringdtype_instance(NULL, 1);    \
            if (new == NULL) {                                             \
                return (NPY_CASTING)-1;                                    \
            }                                                              \
            loop_descrs[1] = new;                                          \
        }                                                                  \
        else {                                                             \
            Py_INCREF(given_descrs[1]);                                    \
            loop_descrs[1] = given_descrs[1];                              \
        }                                                                  \
                                                                           \
        Py_INCREF(given_descrs[0]);                                        \
        loop_descrs[0] = given_descrs[0];                                  \
                                                                           \
        return NPY_##safety##_CASTING;                                     \
    }

ANY_TO_STRING_RESOLVE_DESCRIPTORS(SAFE)
ANY_TO_STRING_RESOLVE_DESCRIPTORS(UNSAFE)

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

    const npy_static_string *s = NULL;
    npy_static_string *os = NULL;

    while (N--) {
        s = (npy_static_string *)in;
        os = (npy_static_string *)out;
        if (in != out) {
            npy_string_free(os);
            if (npy_string_dup(s, os) < 0) {
                gil_error(PyExc_MemoryError, "npy_string_dup failed");
                return -1;
            }
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
        npy_static_string *out_ss = (npy_static_string *)out;
        npy_string_free(out_ss);
        if (npy_string_newemptysize(out_num_bytes, out_ss) < 0) {
            gil_error(PyExc_MemoryError, "npy_string_newemptysize failed");
            return -1;
        }
        char *out_buf = npy_string_buf(out_ss);
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

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot u2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAFE_resolve_descriptors},
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
utf8_char_to_ucs4_code(unsigned char *c, size_t len, Py_UCS4 *code)
{
    if (len == 0) {
        *code = (Py_UCS4)0;
        return 0;
    }
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
    StringDTypeObject *descr = (StringDTypeObject *)context->descriptors[0];
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    npy_static_string default_string = descr->default_string;
    npy_intp N = dimensions[0];
    char *in = data[0];
    Py_UCS4 *out = (Py_UCS4 *)data[1];
    npy_intp in_stride = strides[0];
    // 4 bytes per UCS4 character
    npy_intp out_stride = strides[1] / 4;
    // max number of 4 byte UCS4 characters that can fit in the output
    long max_out_size = (context->descriptors[1]->elsize) / 4;

    const npy_static_string *s = NULL;

    while (N--) {
        s = (npy_static_string *)in;
        unsigned char *this_string = NULL;
        size_t n_bytes;
        const npy_static_string *name = NULL;
        if (npy_string_isnull(s)) {
            if (has_null && !has_string_na) {
                // lossy but not much else we can do
                name = &descr->na_name;
            }
            else {
                name = &default_string;
            }
        }
        else {
            name = s;
        }

        this_string = (unsigned char *)npy_string_buf(name);
        n_bytes = npy_string_size(name);
        size_t tot_n_bytes = 0;

        for (int i = 0; i < max_out_size; i++) {
            Py_UCS4 code;

            // code point for character this_string is currently pointing at
            size_t num_bytes =
                    utf8_char_to_ucs4_code(this_string, n_bytes, &code);

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
    StringDTypeObject *descr = (StringDTypeObject *)context->descriptors[0];
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    npy_static_string default_string = descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    const npy_static_string *s = NULL;

    while (N--) {
        s = (npy_static_string *)in;
        if (npy_string_isnull(s)) {
            if (has_null && !has_string_na) {
                // numpy treats NaN as truthy, following python
                *out = (npy_bool)1;
            }
            else {
                *out = (npy_bool)(npy_string_size(&default_string) == 0);
            }
        }
        else if (npy_string_size(s) == 0) {
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
        npy_static_string *out_ss = (npy_static_string *)out;
        npy_string_free(out_ss);
        if ((npy_bool)(*in) == 1) {
            if (npy_string_newsize("True", 4, out_ss) < 0) {
                gil_error(PyExc_MemoryError, "npy_string_newsize failed");
                return -1;
            }
        }
        else if ((npy_bool)(*in) == 0) {
            if (npy_string_newsize("False", 5, out_ss) < 0) {
                gil_error(PyExc_MemoryError, "npy_string_newsize failed");
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

static PyType_Slot b2s_slots[] = {{NPY_METH_resolve_descriptors,
                                   &any_to_string_SAFE_resolve_descriptors},
                                  {NPY_METH_strided_loop, &bool_to_string},
                                  {0, NULL}};

static char *b2s_name = "cast_Bool_to_StringDType";

// casts between string and (u)int dtypes

static PyObject *
string_to_pylong(char *in, int hasnull)
{
    const npy_static_string *s = (npy_static_string *)in;
    if (npy_string_isnull(s)) {
        if (hasnull) {
            PyErr_SetString(PyExc_ValueError,
                            "Arrays with missing data cannot be converted to "
                            "integers");
            return NULL;
        }
        s = &NPY_EMPTY_STRING;
    }
    PyObject *val_obj =
            PyUnicode_FromStringAndSize(npy_string_buf(s), npy_string_size(s));
    if (val_obj == NULL) {
        return NULL;
    }
    // interpret as an integer in base 10
    PyObject *pylong_value = PyLong_FromUnicodeObject(val_obj, 10);
    Py_DECREF(val_obj);
    return pylong_value;
}

static npy_longlong
string_to_uint(char *in, npy_ulonglong *value, int hasnull)
{
    PyObject *pylong_value = string_to_pylong(in, hasnull);
    if (pylong_value == NULL) {
        return -1;
    }
    *value = PyLong_AsUnsignedLongLong(pylong_value);
    if (*value == (unsigned long long)-1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static npy_longlong
string_to_int(char *in, npy_longlong *value, int hasnull)
{
    PyObject *pylong_value = string_to_pylong(in, hasnull);
    if (pylong_value == NULL) {
        return -1;
    }
    *value = PyLong_AsLongLong(pylong_value);
    if (*value == -1 && PyErr_Occurred()) {
        Py_DECREF(pylong_value);
        return -1;
    }
    Py_DECREF(pylong_value);
    return 0;
}

static int
pyobj_to_string(PyObject *obj, char *out)
{
    if (obj == NULL) {
        return -1;
    }
    PyObject *pystr_val = PyObject_Str(obj);
    Py_DECREF(obj);
    if (pystr_val == NULL) {
        return -1;
    }
    Py_ssize_t length;
    const char *cstr_val = PyUnicode_AsUTF8AndSize(pystr_val, &length);
    if (cstr_val == NULL) {
        return -1;
    }
    npy_static_string *out_ss = (npy_static_string *)out;
    npy_string_free(out_ss);
    if (npy_string_newsize(cstr_val, length, out_ss) < 0) {
        PyErr_SetString(PyExc_MemoryError, "npy_string_newsize failed");
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
    return pyobj_to_string(pylong_val, out);
}

static int
uint_to_string(unsigned long long in, char *out)
{
    PyObject *pylong_val = PyLong_FromUnsignedLongLong(in);
    return pyobj_to_string(pylong_val, out);
}

#define STRING_INT_CASTS(typename, typekind, shortname, numpy_tag,            \
                         printf_code, npy_longtype, longtype)                 \
    static NPY_CASTING string_to_##typename##_resolve_descriptors(            \
            PyObject *NPY_UNUSED(self),                                       \
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                         \
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],    \
            npy_intp *NPY_UNUSED(view_offset))                                \
    {                                                                         \
        if (given_descrs[1] == NULL) {                                        \
            loop_descrs[1] = PyArray_DescrNewFromType(numpy_tag);             \
        }                                                                     \
        else {                                                                \
            Py_INCREF(given_descrs[1]);                                       \
            loop_descrs[1] = given_descrs[1];                                 \
        }                                                                     \
                                                                              \
        Py_INCREF(given_descrs[0]);                                           \
        loop_descrs[0] = given_descrs[0];                                     \
                                                                              \
        return NPY_UNSAFE_CASTING;                                            \
    }                                                                         \
                                                                              \
    static int string_to_##                                                   \
            typename(PyArrayMethod_Context * context, char *const data[],     \
                     npy_intp const dimensions[], npy_intp const strides[],   \
                     NpyAuxData *NPY_UNUSED(auxdata))                         \
    {                                                                         \
        int hasnull =                                                         \
                (((StringDTypeObject *)context->descriptors[0])->na_object != \
                 NULL);                                                       \
                                                                              \
        npy_intp N = dimensions[0];                                           \
        char *in = data[0];                                                   \
        npy_##typename *out = (npy_##typename *)data[1];                      \
                                                                              \
        npy_intp in_stride = strides[0];                                      \
        npy_intp out_stride = strides[1] / sizeof(npy_##typename);            \
                                                                              \
        while (N--) {                                                         \
            npy_longtype value;                                               \
            if (string_to_##typekind(in, &value, hasnull) != 0) {             \
                return -1;                                                    \
            }                                                                 \
            *out = (npy_##typename)value;                                     \
            if (*out != value) {                                              \
                /* out of bounds, raise error following NEP 50 behavior */    \
                PyErr_Format(PyExc_OverflowError,                             \
                             "Integer %" #printf_code                         \
                             " is out of bounds "                             \
                             "for " #typename,                                \
                             value);                                          \
                return -1;                                                    \
            }                                                                 \
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        return 0;                                                             \
    }                                                                         \
                                                                              \
    static PyType_Slot s2##shortname##_slots[] = {                            \
            {NPY_METH_resolve_descriptors,                                    \
             &string_to_##typename##_resolve_descriptors},                    \
            {NPY_METH_strided_loop, &string_to_##typename},                   \
            {0, NULL}};                                                       \
                                                                              \
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;     \
                                                                              \
    static int typename##_to_string(                                          \
            PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],   \
            npy_intp const dimensions[], npy_intp const strides[],            \
            NpyAuxData *NPY_UNUSED(auxdata))                                  \
    {                                                                         \
        npy_intp N = dimensions[0];                                           \
        npy_##typename *in = (npy_##typename *)data[0];                       \
        char *out = data[1];                                                  \
                                                                              \
        npy_intp in_stride = strides[0] / sizeof(npy_##typename);             \
        npy_intp out_stride = strides[1];                                     \
                                                                              \
        while (N--) {                                                         \
            if (typekind##_to_string((longtype)*in, out) != 0) {              \
                return -1;                                                    \
            }                                                                 \
                                                                              \
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        return 0;                                                             \
    }                                                                         \
                                                                              \
    static PyType_Slot shortname##2s_slots [] = {                             \
            {NPY_METH_resolve_descriptors,                                    \
             &any_to_string_UNSAFE_resolve_descriptors},                      \
            {NPY_METH_strided_loop, &typename##_to_string},                   \
            {0, NULL}};                                                       \
                                                                              \
    static char *shortname##2s_name = "cast_" #typename "_to_StringDType";

#define DTYPES_AND_CAST_SPEC(shortname, typename)                            \
    PyArray_DTypeMeta **s2##shortname##_dtypes = get_dtypes(                 \
            (PyArray_DTypeMeta *)&StringDType, &PyArray_##typename##DType);  \
                                                                             \
    PyArrayMethod_Spec *StringTo##typename##CastSpec =                       \
            get_cast_spec(s2##shortname##_name, NPY_UNSAFE_CASTING,          \
                          NPY_METH_REQUIRES_PYAPI, s2##shortname##_dtypes,   \
                          s2##shortname##_slots);                            \
                                                                             \
    PyArray_DTypeMeta **shortname##2s_dtypes = get_dtypes(                   \
            &PyArray_##typename##DType, (PyArray_DTypeMeta *)&StringDType);  \
                                                                             \
    PyArrayMethod_Spec *typename##ToStringCastSpec = get_cast_spec(          \
            shortname##2s_name, NPY_UNSAFE_CASTING, NPY_METH_REQUIRES_PYAPI, \
            shortname##2s_dtypes, shortname##2s_slots);

STRING_INT_CASTS(int8, int, i8, NPY_INT8, lli, npy_longlong, long long)
STRING_INT_CASTS(int16, int, i16, NPY_INT16, lli, npy_longlong, long long)
STRING_INT_CASTS(int32, int, i32, NPY_INT32, lli, npy_longlong, long long)
STRING_INT_CASTS(int64, int, i64, NPY_INT64, lli, npy_longlong, long long)

STRING_INT_CASTS(uint8, uint, u8, NPY_UINT8, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint16, uint, u16, NPY_UINT16, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint32, uint, u32, NPY_UINT32, llu, npy_ulonglong,
                 unsigned long long)
STRING_INT_CASTS(uint64, uint, u64, NPY_UINT64, llu, npy_ulonglong,
                 unsigned long long)

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
// byte doesn't have a bitsized alias
STRING_INT_CASTS(byte, int, byte, NPY_BYTE, lli, npy_longlong, long long)
STRING_INT_CASTS(ubyte, uint, ubyte, NPY_UBYTE, llu, npy_ulonglong,
                 unsigned long long)
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
// short doesn't have a bitsized alias
STRING_INT_CASTS(short, int, short, NPY_SHORT, lli, npy_longlong, long long)
STRING_INT_CASTS(ushort, uint, ushort, NPY_USHORT, llu, npy_ulonglong,
                 unsigned long long)
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
// int doesn't have a bitsized alias
STRING_INT_CASTS(int, int, int, NPY_INT, lli, npy_longlong, long long)
STRING_INT_CASTS(uint, uint, uint, NPY_UINT, llu, npy_longlong, long long)
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
// long long doesn't have a bitsized alias
STRING_INT_CASTS(longlong, int, longlong, NPY_LONGLONG, lli, npy_longlong,
                 long long)
STRING_INT_CASTS(ulonglong, uint, ulonglong, NPY_ULONGLONG, llu, npy_ulonglong,
                 unsigned long long)
#endif

static PyObject *
string_to_pyfloat(char *in, int hasnull)
{
    const npy_static_string *s = (npy_static_string *)in;
    if (npy_string_isnull(s)) {
        if (hasnull) {
            PyErr_SetString(PyExc_ValueError,
                            "Arrays with missing data cannot be converted to "
                            "integers");
            return NULL;
        }
        s = &NPY_EMPTY_STRING;
    }
    PyObject *val_obj =
            PyUnicode_FromStringAndSize(npy_string_buf(s), npy_string_size(s));
    if (val_obj == NULL) {
        return NULL;
    }
    PyObject *pyfloat_value = PyFloat_FromString(val_obj);
    Py_DECREF(val_obj);
    return pyfloat_value;
}

#define STRING_TO_FLOAT_CAST(typename, shortname, isinf_name,                 \
                             double_to_float)                                 \
    static int string_to_##                                                   \
            typename(PyArrayMethod_Context * context, char *const data[],     \
                     npy_intp const dimensions[], npy_intp const strides[],   \
                     NpyAuxData *NPY_UNUSED(auxdata))                         \
    {                                                                         \
        int hasnull =                                                         \
                (((StringDTypeObject *)context->descriptors[0])->na_object != \
                 NULL);                                                       \
                                                                              \
        npy_intp N = dimensions[0];                                           \
        char *in = data[0];                                                   \
        npy_##typename *out = (npy_##typename *)data[1];                      \
                                                                              \
        npy_intp in_stride = strides[0];                                      \
        npy_intp out_stride = strides[1] / sizeof(npy_##typename);            \
                                                                              \
        while (N--) {                                                         \
            PyObject *pyfloat_value = string_to_pyfloat(in, hasnull);         \
            if (pyfloat_value == NULL) {                                      \
                return -1;                                                    \
            }                                                                 \
            double dval = PyFloat_AS_DOUBLE(pyfloat_value);                   \
            npy_##typename fval = (double_to_float)(dval);                    \
                                                                              \
            if (NPY_UNLIKELY(isinf_name(fval) && !(npy_isinf(dval)))) {       \
                if (PyUFunc_GiveFloatingpointErrors("cast",                   \
                                                    NPY_FPE_OVERFLOW) < 0) {  \
                    return -1;                                                \
                }                                                             \
            }                                                                 \
                                                                              \
            *out = fval;                                                      \
                                                                              \
            in += in_stride;                                                  \
            out += out_stride;                                                \
        }                                                                     \
                                                                              \
        return 0;                                                             \
    }                                                                         \
                                                                              \
    static PyType_Slot s2##shortname##_slots[] = {                            \
            {NPY_METH_resolve_descriptors,                                    \
             &string_to_##typename##_resolve_descriptors},                    \
            {NPY_METH_strided_loop, &string_to_##typename},                   \
            {0, NULL}};                                                       \
                                                                              \
    static char *s2##shortname##_name = "cast_StringDType_to_" #typename;

#define STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(typename, npy_typename)        \
    static NPY_CASTING string_to_##typename##_resolve_descriptors(         \
            PyObject *NPY_UNUSED(self),                                    \
            PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),                      \
            PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2], \
            npy_intp *NPY_UNUSED(view_offset))                             \
    {                                                                      \
        if (given_descrs[1] == NULL) {                                     \
            loop_descrs[1] = PyArray_DescrNewFromType(NPY_##npy_typename); \
        }                                                                  \
        else {                                                             \
            Py_INCREF(given_descrs[1]);                                    \
            loop_descrs[1] = given_descrs[1];                              \
        }                                                                  \
                                                                           \
        Py_INCREF(given_descrs[0]);                                        \
        loop_descrs[0] = given_descrs[0];                                  \
                                                                           \
        return NPY_UNSAFE_CASTING;                                         \
    }

#define FLOAT_TO_STRING_CAST(typename, shortname, float_to_double)        \
    static int typename##_to_string(                                      \
            PyArrayMethod_Context *context, char *const data[],           \
            npy_intp const dimensions[], npy_intp const strides[],        \
            NpyAuxData *NPY_UNUSED(auxdata))                              \
    {                                                                     \
        npy_intp N = dimensions[0];                                       \
        npy_##typename *in = (npy_##typename *)data[0];                   \
        char *out = data[1];                                              \
        PyArray_Descr *float_descr = context->descriptors[0];             \
                                                                          \
        npy_intp in_stride = strides[0] / sizeof(npy_##typename);         \
        npy_intp out_stride = strides[1];                                 \
                                                                          \
        while (N--) {                                                     \
            PyObject *scalar_val = PyArray_Scalar(in, float_descr, NULL); \
            if (pyobj_to_string(scalar_val, out) == -1) {                 \
                return -1;                                                \
            }                                                             \
                                                                          \
            in += in_stride;                                              \
            out += out_stride;                                            \
        }                                                                 \
                                                                          \
        return 0;                                                         \
    }                                                                     \
                                                                          \
    static PyType_Slot shortname##2s_slots [] = {                         \
            {NPY_METH_resolve_descriptors,                                \
             &any_to_string_UNSAFE_resolve_descriptors},                  \
            {NPY_METH_strided_loop, &typename##_to_string},               \
            {0, NULL}};                                                   \
                                                                          \
    static char *shortname##2s_name = "cast_" #typename "_to_StringDType";

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float64, DOUBLE)

static int
string_to_float64(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    int hasnull = (((StringDTypeObject *)context->descriptors[0])->na_object !=
                   NULL);
    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_float64 *out = (npy_float64 *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_float64);

    while (N--) {
        PyObject *pyfloat_value = string_to_pyfloat(in, hasnull);
        if (pyfloat_value == NULL) {
            return -1;
        }
        *out = (npy_float64)PyFloat_AS_DOUBLE(pyfloat_value);
        Py_DECREF(pyfloat_value);

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot s2f64_slots[] = {
        {NPY_METH_resolve_descriptors, &string_to_float64_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_float64},
        {0, NULL}};

static char *s2f64_name = "cast_StringDType_to_float64";

FLOAT_TO_STRING_CAST(float64, f64, double)

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float32, FLOAT)
STRING_TO_FLOAT_CAST(float32, f32, npy_isinf, npy_float32)
FLOAT_TO_STRING_CAST(float32, f32, double)

STRING_TO_FLOAT_RESOLVE_DESCRIPTORS(float16, HALF)
STRING_TO_FLOAT_CAST(float16, f16, npy_half_isinf, npy_double_to_half)
FLOAT_TO_STRING_CAST(float16, f16, npy_half_to_double)

// string to datetime

static NPY_CASTING
string_to_datetime_resolve_descriptors(
        PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *NPY_UNUSED(view_offset))
{
    if (given_descrs[1] == NULL) {
        PyErr_SetString(PyExc_TypeError,
                        "Casting from StringDType to datetimes without a unit "
                        "is not currently supported");
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

static int
string_to_datetime(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    StringDTypeObject *descr = (StringDTypeObject *)context->descriptors[0];
    int has_null = descr->na_object != NULL;
    int has_string_na = descr->has_string_na;
    npy_static_string default_string = descr->default_string;

    npy_intp N = dimensions[0];
    char *in = data[0];
    npy_datetime *out = (npy_datetime *)data[1];

    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1] / sizeof(npy_datetime);

    const npy_static_string *s = NULL;
    npy_datetimestruct dts;
    NPY_DATETIMEUNIT in_unit = -1;
    PyArray_DatetimeMetaData in_meta = {0, 1};
    npy_bool out_special;

    PyArray_Descr *dt_descr = context->descriptors[1];
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);

    while (N--) {
        s = (npy_static_string *)in;
        if (npy_string_isnull(s)) {
            if (has_null && !has_string_na) {
                *out = NPY_DATETIME_NAT;
                goto next_step;
            }
            s = &default_string;
        }
        if (NpyDatetime_ParseISO8601Datetime(
                    (const char *)npy_string_buf(s), npy_string_size(s),
                    in_unit, NPY_UNSAFE_CASTING, &dts, &in_meta.base,
                    &out_special) < 0) {
            return -1;
        }
        if (NpyDatetime_ConvertDatetimeStructToDatetime64(dt_meta, &dts, out) <
            0) {
            return -1;
        }

    next_step:
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot s2dt_slots[] = {
        {NPY_METH_resolve_descriptors,
         &string_to_datetime_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_datetime},
        {0, NULL}};

static char *s2dt_name = "cast_StringDType_to_Datetime";

// datetime to string

static int
datetime_to_string(PyArrayMethod_Context *context, char *const data[],
                   npy_intp const dimensions[], npy_intp const strides[],
                   NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    npy_datetime *in = (npy_datetime *)data[0];
    char *out = data[1];

    npy_intp in_stride = strides[0] / sizeof(npy_datetime);
    npy_intp out_stride = strides[1];

    npy_datetimestruct dts;
    PyArray_Descr *dt_descr = context->descriptors[0];
    PyArray_DatetimeMetaData *dt_meta =
            &(((PyArray_DatetimeDTypeMetaData *)dt_descr->c_metadata)->meta);
    // buffer passed to numpy to build datetime string
    char datetime_buf[NPY_DATETIME_MAX_ISO8601_STRLEN];

    while (N--) {
        npy_static_string *out_ss = (npy_static_string *)out;
        npy_string_free(out_ss);
        if (*in == NPY_DATETIME_NAT) {
            /* convert to NA */
            out_ss = NULL;
        }
        else {
            if (NpyDatetime_ConvertDatetime64ToDatetimeStruct(dt_meta, *in,
                                                              &dts) < 0) {
                return -1;
            }

            // zero out buffer
            memset(datetime_buf, 0, NPY_DATETIME_MAX_ISO8601_STRLEN);

            if (NpyDatetime_MakeISO8601Datetime(
                        &dts, datetime_buf, NPY_DATETIME_MAX_ISO8601_STRLEN, 0,
                        0, dt_meta->base, -1, NPY_UNSAFE_CASTING) < 0) {
                return -1;
            }

            if (npy_string_newsize(datetime_buf, strlen(datetime_buf),
                                   out_ss) < 0) {
                PyErr_SetString(PyExc_MemoryError,
                                "npy_string_newsize failed");
                return -1;
            }
        }

        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot dt2s_slots[] = {
        {NPY_METH_resolve_descriptors,
         &any_to_string_UNSAFE_resolve_descriptors},
        {NPY_METH_strided_loop, &datetime_to_string},
        {0, NULL}};

static char *dt2s_name = "cast_Datetime_to_StringDType";

// TODO: longdouble
//        punting on this one because numpy's C routines for handling
//        longdouble are not public (specifically NumPyOS_ascii_strtold)
//        also this type is kinda niche and is not needed by pandas
//
//       cfloat, cdouble, and clongdouble
//        not hard to do in principle but not needed by pandas.

PyArrayMethod_Spec *
get_cast_spec(const char *name, NPY_CASTING casting,
              NPY_ARRAYMETHOD_FLAGS flags, PyArray_DTypeMeta **dtypes,
              PyType_Slot *slots)
{
    PyArrayMethod_Spec *ret = PyMem_Malloc(sizeof(PyArrayMethod_Spec));

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
    PyArray_DTypeMeta **ret = PyMem_Malloc(2 * sizeof(PyArray_DTypeMeta *));

    ret[0] = dt1;
    ret[1] = dt2;

    return ret;
}

PyArrayMethod_Spec **
get_casts()
{
    char *t2t_name = s2s_name;

    PyArray_DTypeMeta **t2t_dtypes =
            get_dtypes((PyArray_DTypeMeta *)&StringDType,
                       (PyArray_DTypeMeta *)&StringDType);

    PyArrayMethod_Spec *ThisToThisCastSpec =
            get_cast_spec(t2t_name, NPY_NO_CASTING,
                          NPY_METH_SUPPORTS_UNALIGNED, t2t_dtypes, s2s_slots);

    int num_casts = 29;

#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    num_casts += 4;
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    num_casts += 4;
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    num_casts += 4;
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    num_casts += 4;
#endif

    PyArray_DTypeMeta **u2s_dtypes = get_dtypes(
            &PyArray_UnicodeDType, (PyArray_DTypeMeta *)&StringDType);

    PyArrayMethod_Spec *UnicodeToStringCastSpec = get_cast_spec(
            u2s_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            u2s_dtypes, u2s_slots);

    PyArray_DTypeMeta **s2u_dtypes = get_dtypes(
            (PyArray_DTypeMeta *)&StringDType, &PyArray_UnicodeDType);

    PyArrayMethod_Spec *StringToUnicodeCastSpec = get_cast_spec(
            s2u_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2u_dtypes, s2u_slots);

    PyArray_DTypeMeta **s2b_dtypes =
            get_dtypes((PyArray_DTypeMeta *)&StringDType, &PyArray_BoolDType);

    PyArrayMethod_Spec *StringToBoolCastSpec = get_cast_spec(
            s2b_name, NPY_UNSAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            s2b_dtypes, s2b_slots);

    PyArray_DTypeMeta **b2s_dtypes =
            get_dtypes(&PyArray_BoolDType, (PyArray_DTypeMeta *)&StringDType);

    PyArrayMethod_Spec *BoolToStringCastSpec = get_cast_spec(
            b2s_name, NPY_SAFE_CASTING, NPY_METH_NO_FLOATINGPOINT_ERRORS,
            b2s_dtypes, b2s_slots);

    DTYPES_AND_CAST_SPEC(i8, Int8)
    DTYPES_AND_CAST_SPEC(i16, Int16)
    DTYPES_AND_CAST_SPEC(i32, Int32)
    DTYPES_AND_CAST_SPEC(i64, Int64)
    DTYPES_AND_CAST_SPEC(u8, UInt8)
    DTYPES_AND_CAST_SPEC(u16, UInt16)
    DTYPES_AND_CAST_SPEC(u32, UInt32)
    DTYPES_AND_CAST_SPEC(u64, UInt64)
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    DTYPES_AND_CAST_SPEC(byte, Byte)
    DTYPES_AND_CAST_SPEC(ubyte, UByte)
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    DTYPES_AND_CAST_SPEC(short, Short)
    DTYPES_AND_CAST_SPEC(ushort, UShort)
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(int, Int)
    DTYPES_AND_CAST_SPEC(uint, UInt)
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    DTYPES_AND_CAST_SPEC(longlong, LongLong)
    DTYPES_AND_CAST_SPEC(ulonglong, ULongLong)
#endif

    DTYPES_AND_CAST_SPEC(f64, Double)
    DTYPES_AND_CAST_SPEC(f32, Float)
    DTYPES_AND_CAST_SPEC(f16, Half)

    PyArray_DTypeMeta **s2dt_dtypes = get_dtypes(
            (PyArray_DTypeMeta *)&StringDType, &PyArray_DatetimeDType);

    PyArrayMethod_Spec *StringToDatetimeCastSpec = get_cast_spec(
            s2dt_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            s2dt_dtypes, s2dt_slots);

    PyArray_DTypeMeta **dt2s_dtypes = get_dtypes(
            &PyArray_DatetimeDType, (PyArray_DTypeMeta *)&StringDType);

    PyArrayMethod_Spec *DatetimeToStringCastSpec = get_cast_spec(
            dt2s_name, NPY_UNSAFE_CASTING,
            NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI,
            dt2s_dtypes, dt2s_slots);

    PyArrayMethod_Spec **casts =
            PyMem_Malloc((num_casts + 1) * sizeof(PyArrayMethod_Spec *));

    int cast_i = 0;

    casts[cast_i++] = ThisToThisCastSpec;
    casts[cast_i++] = UnicodeToStringCastSpec;
    casts[cast_i++] = StringToUnicodeCastSpec;
    casts[cast_i++] = StringToBoolCastSpec;
    casts[cast_i++] = BoolToStringCastSpec;
    casts[cast_i++] = StringToInt8CastSpec;
    casts[cast_i++] = Int8ToStringCastSpec;
    casts[cast_i++] = StringToInt16CastSpec;
    casts[cast_i++] = Int16ToStringCastSpec;
    casts[cast_i++] = StringToInt32CastSpec;
    casts[cast_i++] = Int32ToStringCastSpec;
    casts[cast_i++] = StringToInt64CastSpec;
    casts[cast_i++] = Int64ToStringCastSpec;
    casts[cast_i++] = StringToUInt8CastSpec;
    casts[cast_i++] = UInt8ToStringCastSpec;
    casts[cast_i++] = StringToUInt16CastSpec;
    casts[cast_i++] = UInt16ToStringCastSpec;
    casts[cast_i++] = StringToUInt32CastSpec;
    casts[cast_i++] = UInt32ToStringCastSpec;
    casts[cast_i++] = StringToUInt64CastSpec;
    casts[cast_i++] = UInt64ToStringCastSpec;
#if NPY_SIZEOF_BYTE == NPY_SIZEOF_SHORT
    casts[cast_i++] = StringToByteCastSpec;
    casts[cast_i++] = ByteToStringCastSpec;
    casts[cast_i++] = StringToUByteCastSpec;
    casts[cast_i++] = UByteToStringCastSpec;
#endif
#if NPY_SIZEOF_SHORT == NPY_SIZEOF_INT
    casts[cast_i++] = StringToShortCastSpec;
    casts[cast_i++] = ShortToStringCastSpec;
    casts[cast_i++] = StringToUShortCastSpec;
    casts[cast_i++] = UShortToStringCastSpec;
#endif
#if NPY_SIZEOF_INT == NPY_SIZEOF_LONG
    casts[cast_i++] = StringToIntCastSpec;
    casts[cast_i++] = IntToStringCastSpec;
    casts[cast_i++] = StringToUIntCastSpec;
    casts[cast_i++] = UIntToStringCastSpec;
#endif
#if NPY_SIZEOF_LONGLONG == NPY_SIZEOF_LONG
    casts[cast_i++] = StringToLongLongCastSpec;
    casts[cast_i++] = LongLongToStringCastSpec;
    casts[cast_i++] = StringToULongLongCastSpec;
    casts[cast_i++] = ULongLongToStringCastSpec;
#endif
    casts[cast_i++] = StringToDoubleCastSpec;
    casts[cast_i++] = DoubleToStringCastSpec;
    casts[cast_i++] = StringToFloatCastSpec;
    casts[cast_i++] = FloatToStringCastSpec;
    casts[cast_i++] = StringToHalfCastSpec;
    casts[cast_i++] = HalfToStringCastSpec;
    casts[cast_i++] = StringToDatetimeCastSpec;
    casts[cast_i++] = DatetimeToStringCastSpec;
    casts[cast_i++] = NULL;

    assert(casts[num_casts] == NULL);
    assert(cast_i == num_casts + 1);

    return casts;
}
