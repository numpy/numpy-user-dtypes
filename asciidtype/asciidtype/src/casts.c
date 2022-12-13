#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL asciidtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "casts.h"
#include "dtype.h"

static NPY_CASTING
ascii_to_ascii_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                   PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                   PyArray_Descr *given_descrs[2],
                                   PyArray_Descr *loop_descrs[2],
                                   npy_intp *view_offset)
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    if (given_descrs[1] == NULL) {
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    long in_size = ((ASCIIDTypeObject *)loop_descrs[0])->size;
    long out_size = ((ASCIIDTypeObject *)loop_descrs[1])->size;

    if (in_size == out_size) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }
    else if (in_size > out_size) {
        return NPY_UNSAFE_CASTING;
    }
    return NPY_SAFE_CASTING;
}

static int
ascii_to_ascii(PyArrayMethod_Context *context, char *const data[],
               npy_intp const dimensions[], npy_intp const strides[],
               NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in_size = ((ASCIIDTypeObject *)descrs[0])->size;
    long out_size = ((ASCIIDTypeObject *)descrs[1])->size;
    long copy_size;

    if (out_size > in_size) {
        copy_size = in_size;
    }
    else {
        copy_size = out_size;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        memcpy(out, in, copy_size * sizeof(char));  // NOLINT
        for (int i = copy_size; i < out_size; i++) {
            *(out + i) = '\0';
        }
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static NPY_CASTING
unicode_to_ascii_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    // numpy stores unicode as UCS4 (4 bytes wide), so bitshift
    // by 2 to get the number of ASCII bytes needed
    long in_size = (loop_descrs[0]->elsize) >> 2;
    if (given_descrs[1] == NULL) {
        ASCIIDTypeObject *ascii_descr = new_asciidtype_instance(in_size);
        loop_descrs[1] = (PyArray_Descr *)ascii_descr;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    long out_size = ((ASCIIDTypeObject *)loop_descrs[1])->size;

    if (out_size >= in_size) {
        return NPY_SAFE_CASTING;
    }

    return NPY_UNSAFE_CASTING;
}

static int
ucs4_character_is_ascii(char *buffer)
{
    unsigned char first_char = buffer[0];

    if (first_char > 127) {
        return -1;
    }

    for (int i = 1; i < 4; i++) {
        if (buffer[i] != 0) {
            return -1;
        }
    }

    return 0;
}

static int
unicode_to_ascii(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in_size = (descrs[0]->elsize) / 4;
    long out_size = ((ASCIIDTypeObject *)descrs[1])->size;
    long copy_size;

    if (out_size > in_size) {
        copy_size = in_size;
    }
    else {
        copy_size = out_size;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        // copy input characters, checking that input UCS4
        // characters are all ascii, raising an error otherwise
        for (int i = 0; i < copy_size; i++) {
            if (ucs4_character_is_ascii(in) == -1) {
                PyErr_SetString(
                        PyExc_TypeError,
                        "Can only store ASCII text in a ASCIIDType array.");
                return -1;
            }
            // UCS4 character is ascii, so copy first byte of character
            // into output, ignoring the rest
            *(out + i) = *(in + i * 4);
        }
        // write zeros to remaining ASCII characters (if any)
        for (int i = copy_size; i < out_size; i++) {
            *(out + i) = '\0';
        }
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static int
ascii_to_unicode(PyArrayMethod_Context *context, char *const data[],
                 npy_intp const dimensions[], npy_intp const strides[],
                 NpyAuxData *NPY_UNUSED(auxdata))
{
    PyArray_Descr **descrs = context->descriptors;
    long in_size = ((ASCIIDTypeObject *)descrs[0])->size;
    long out_size = (descrs[1]->elsize) / 4;
    long copy_size;

    if (out_size > in_size) {
        copy_size = in_size;
    }
    else {
        copy_size = out_size;
    }

    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        // copy ASCII input to first byte, fill rest with zeros
        for (int i = 0; i < copy_size; i++) {
            ((Py_UCS4 *)out)[i] = ((Py_UCS1 *)in)[i];
        }
        // fill all remaining UCS4 characters with zeros
        for (int i = copy_size; i < out_size; i++) {
            ((Py_UCS4 *)out)[i] = (Py_UCS1)0;
        }
        in += in_stride;
        out += out_stride;
    }
    return 0;
}

static NPY_CASTING
ascii_to_unicode_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *NPY_UNUSED(view_offset))
{
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];
    long in_size = ((ASCIIDTypeObject *)given_descrs[0])->size;
    if (given_descrs[1] == NULL) {
        PyArray_Descr *unicode_descr = PyArray_DescrNewFromType(NPY_UNICODE);
        // numpy stores unicode as UCS4 (4 bytes wide), so bitshift
        // by 2 to get the number of bytes needed to store the UCS4 charaters
        unicode_descr->elsize = in_size << 2;
        loop_descrs[1] = unicode_descr;
    }
    else {
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }

    long out_size = (loop_descrs[1]->elsize) >> 2;

    if (out_size >= in_size) {
        return NPY_SAFE_CASTING;
    }

    return NPY_UNSAFE_CASTING;
}

static PyArray_DTypeMeta *a2a_dtypes[2] = {NULL, NULL};

static PyType_Slot a2a_slots[] = {
        {NPY_METH_resolve_descriptors, &ascii_to_ascii_resolve_descriptors},
        {NPY_METH_strided_loop, &ascii_to_ascii},
        {NPY_METH_unaligned_strided_loop, &ascii_to_ascii},
        {0, NULL}};

PyArrayMethod_Spec ASCIIToASCIICastSpec = {
        .name = "cast_ASCIIDType_to_ASCIIDType",
        .nin = 1,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .flags = (NPY_METH_NO_FLOATINGPOINT_ERRORS |
                  NPY_METH_SUPPORTS_UNALIGNED),
        .dtypes = a2a_dtypes,
        .slots = a2a_slots,
};

static PyType_Slot u2a_slots[] = {
        {NPY_METH_resolve_descriptors, &unicode_to_ascii_resolve_descriptors},
        {NPY_METH_strided_loop, &unicode_to_ascii},
        {0, NULL}};

static char *u2a_name = "cast_Unicode_to_ASCIIDType";

static PyType_Slot a2u_slots[] = {
        {NPY_METH_resolve_descriptors, &ascii_to_unicode_resolve_descriptors},
        {NPY_METH_strided_loop, &ascii_to_unicode},
        {0, NULL}};

static char *a2u_name = "cast_ASCIIDType_to_Unicode";

PyArrayMethod_Spec **
get_casts(void)
{
    PyArray_DTypeMeta **u2a_dtypes = malloc(2 * sizeof(PyArray_DTypeMeta *));
    u2a_dtypes[0] = &PyArray_UnicodeDType;
    u2a_dtypes[1] = NULL;

    PyArrayMethod_Spec *UnicodeToASCIICastSpec =
            malloc(sizeof(PyArrayMethod_Spec));

    UnicodeToASCIICastSpec->name = u2a_name;
    UnicodeToASCIICastSpec->nin = 1;
    UnicodeToASCIICastSpec->nout = 1;
    UnicodeToASCIICastSpec->casting = NPY_UNSAFE_CASTING,
    UnicodeToASCIICastSpec->flags =
            (NPY_METH_NO_FLOATINGPOINT_ERRORS | NPY_METH_REQUIRES_PYAPI);
    UnicodeToASCIICastSpec->dtypes = u2a_dtypes;
    UnicodeToASCIICastSpec->slots = u2a_slots;

    PyArray_DTypeMeta **a2u_dtypes = malloc(2 * sizeof(PyArray_DTypeMeta *));
    a2u_dtypes[0] = NULL;
    a2u_dtypes[1] = &PyArray_UnicodeDType;

    PyArrayMethod_Spec *ASCIIToUnicodeCastSpec =
            malloc(sizeof(PyArrayMethod_Spec));

    ASCIIToUnicodeCastSpec->name = a2u_name;
    ASCIIToUnicodeCastSpec->nin = 1;
    ASCIIToUnicodeCastSpec->nout = 1;
    ASCIIToUnicodeCastSpec->casting = NPY_UNSAFE_CASTING,
    ASCIIToUnicodeCastSpec->flags = NPY_METH_NO_FLOATINGPOINT_ERRORS;
    ASCIIToUnicodeCastSpec->dtypes = a2u_dtypes;
    ASCIIToUnicodeCastSpec->slots = a2u_slots;

    PyArrayMethod_Spec **casts = malloc(4 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &ASCIIToASCIICastSpec;
    casts[1] = UnicodeToASCIICastSpec;
    casts[2] = ASCIIToUnicodeCastSpec;
    casts[3] = NULL;

    return casts;
}
