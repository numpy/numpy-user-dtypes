#include "casts.h"

#include "dtype.h"

static NPY_CASTING
string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                           PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                           PyArray_Descr *given_descrs[2],
                           PyArray_Descr *loop_descrs[2],
                           npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (given_descrs[1] == NULL) {
        loop_descrs[1] = (PyArray_Descr *)new_stringdtype_instance();
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    return NPY_SAFE_CASTING;
}

static int
string_to_string(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char **in = (char **)data[0];
    char **out = (char **)data[1];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp in_stride = strides[0] / context->descriptors[0]->elsize;
    npy_intp out_stride = strides[1] / context->descriptors[1]->elsize;

    while (N--) {
        size_t length = strlen(*in);
        out[0] = (char *)malloc((sizeof(char) * length) + 1);
        strncpy(*out, *in, length + 1);
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyArray_DTypeMeta *s2s_dtypes[2] = {NULL, NULL};

static PyType_Slot s2s_slots[] = {
        {NPY_METH_resolve_descriptors, &string_resolve_descriptors},
        {NPY_METH_strided_loop, &string_to_string},
        {NPY_METH_unaligned_strided_loop, &string_to_string},
        {0, NULL}};

PyArrayMethod_Spec StringToStringCastSpec = {
        .name = "cast_StringDType_to_StringDType",
        .nin = 1,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = s2s_dtypes,
        .slots = s2s_slots,
};

// converts UCS4 code point to 4-byte char* assumes in is a zero-filled 4 byte
// array returns -1 if the code point is not a valid unicode code point, the
// number of bytes in the in the UTF-8 character on success
static int
ucs4_to_utf8_char(const Py_UCS4 code, char *in)
{
    if (code <= 0x7F) {
        // ASCII
        in[0] = (char)code;
        return 1;
    }
    else if (code <= 0x07FF) {
        in[0] = (0xc0 | (code >> 6));
        in[1] = (0x80 | (code & 0x3f));
        return 2;
    }
    else if (code <= 0xFFFF) {
        in[0] = (0xe0 | (code >> 12));
        in[1] = (0x80 | ((code >> 6) & 0x3f));
        in[2] = (0x80 | (code & 0x3f));
        return 3;
    }
    else if (code <= 0x10FFFF) {
        in[0] = (0xf0 | (code >> 18));
        in[1] = (0x80 | ((code >> 12) & 0x3f));
        in[2] = (0x80 | ((code >> 6) & 0x3f));
        in[3] = (0x80 | (code & 0x3f));
        return 4;
    }
    return -1;
}

static int
unicode_to_string(PyArrayMethod_Context *context, char *const data[],
                  npy_intp const dimensions[], npy_intp const strides[],
                  NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in_size = (descrs[0]->elsize) / 4;

    npy_intp N = dimensions[0];
    char *in = data[0];
    char **out = (char **)data[1];
    npy_intp in_stride = strides[0];
    // strides are in bytes but pointer offsets are in pointer widths, so
    // divide by the element size (one pointer width) to get the pointer offset
    npy_intp out_stride = strides[1] / context->descriptors[1]->elsize;

    while (N--) {
        // pessimistically allocate 4 bytes per allowed character
        char *out_buf = calloc(in_size * 4 + 1, sizeof(char));
        size_t out_num_bytes = 0;
        for (int i = 0; i < in_size; i++) {
            // get code point
            Py_UCS4 code = ((Py_UCS4 *)in)[i];

            if (code == 0) {
                break;
            }

            // convert codepoint to UTF8 bytes
            char utf8_c[4] = {0};
            size_t num_bytes = ucs4_to_utf8_char(code, utf8_c);
            out_num_bytes += num_bytes;

            if (num_bytes == -1) {
                // acquire GIL, set error, return
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                PyErr_SetString(PyExc_TypeError,
                                "Invalid unicode code point found");
                PyGILState_Release(gstate);
                return -1;
            }

            // copy utf8_c into out_buf
            strncpy(out_buf, utf8_c, num_bytes);

            // increment out_buf by the size of the character
            out_buf += num_bytes;
        }

        // reset out_buf to the beginning of the string
        out_buf -= out_num_bytes;

        // pad string with null character
        out_buf[out_num_bytes] = '\0';

        // resize out_buf now that we know the real size
        out_buf = realloc(out_buf, out_num_bytes + 1);

        // set out to the address of the beginning of the string
        out[0] = out_buf;

        // increment out and in by strides
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyType_Slot u2s_slots[] = {
        {NPY_METH_resolve_descriptors, &string_resolve_descriptors},
        {NPY_METH_strided_loop, &unicode_to_string},
        {0, NULL}};

static char *u2s_name = "cast_Unicode_to_StringDType";

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

    PyArrayMethod_Spec **casts = malloc(3 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &StringToStringCastSpec;
    casts[1] = UnicodeToStringCastSpec;
    casts[2] = NULL;

    return casts;
}
