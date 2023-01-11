#include "casts.h"

static NPY_CASTING
string_to_string_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                     PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                     PyArray_Descr *given_descrs[2],
                                     PyArray_Descr *loop_descrs[2],
                                     npy_intp *NPY_UNUSED(view_offset))
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
        {NPY_METH_resolve_descriptors, &string_to_string_resolve_descriptors},
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

PyArrayMethod_Spec **
get_casts(void)
{
    PyArrayMethod_Spec **casts = malloc(2 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &StringToStringCastSpec;
    casts[1] = NULL;

    return casts;
}
