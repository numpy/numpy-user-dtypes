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

    if (((ASCIIDTypeObject *)loop_descrs[0])->size ==
        ((ASCIIDTypeObject *)loop_descrs[1])->size) {
        *view_offset = 0;
        return NPY_NO_CASTING;
    }

    return NPY_SAME_KIND_CASTING;
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

static int
ascii_to_ascii_get_loop(PyArrayMethod_Context *context, int aligned,
                        int NPY_UNUSED(move_references),
                        const npy_intp *strides,
                        PyArrayMethod_StridedLoop **out_loop,
                        NpyAuxData **NPY_UNUSED(out_transferdata),
                        NPY_ARRAYMETHOD_FLAGS *flags)
{
    *out_loop = (PyArrayMethod_StridedLoop *)&ascii_to_ascii;

    *flags = 0;
    return 0;
}

static PyArray_DTypeMeta *a2a_dtypes[2] = {NULL, NULL};

static PyType_Slot a2a_slots[] = {
        {NPY_METH_resolve_descriptors, &ascii_to_ascii_resolve_descriptors},
        {_NPY_METH_get_loop, &ascii_to_ascii_get_loop},
        {0, NULL}};

PyArrayMethod_Spec ASCIIToASCIICastSpec = {
        .name = "cast_ASCIIDType_to_ASCIIDType",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .casting = NPY_SAME_KIND_CASTING,
        .dtypes = a2a_dtypes,
        .slots = a2a_slots,
};
