#include "casts.h"

// And now the actual cast code!  Starting with the "resolver" which tells
// us about cast safety.
// Note also the `view_offset`!  It is a way for you to tell NumPy, that this
// cast does not require anything at all, but the cast can simply be done as
// a view.
// For `arr.astype()` it might mean returning a view (eventually, not yet).
// For ufuncs, it already means that they don't have to do a cast at all!
//
// From https://numpy.org/neps/nep-0043-extensible-ufuncs.html#arraymethod:
// resolve_descriptors returns the safety of the operation (casting safety)
static NPY_CASTING
strptr_to_strptr_resolve_descriptors(PyObject *NPY_UNUSED(self),
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

    return NPY_SAFE_CASTING;
}

static int
strptr_to_strptr(PyArrayMethod_Context *NPY_UNUSED(context),
                 char **const data[], npy_intp const dimensions[],
                 npy_intp const strides[], NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char **in = data[0];
    char **out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        strcpy(*out, *in);
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static PyArray_DTypeMeta *a2a_dtypes[2] = {NULL, NULL};

static PyType_Slot a2a_slots[] = {
        {NPY_METH_resolve_descriptors, &strptr_to_strptr_resolve_descriptors},
        {NPY_METH_strided_loop, &strptr_to_strptr},
        {NPY_METH_unaligned_strided_loop, &strptr_to_strptr},
        {0, NULL}};

PyArrayMethod_Spec StrPtrToStrPtrCastSpec = {
        .name = "cast_StrPtrDType_to_StrPtrDType",
        .nin = 1,
        .nout = 1,
        .casting = NPY_UNSAFE_CASTING,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .dtypes = a2a_dtypes,
        .slots = a2a_slots,
};

PyArrayMethod_Spec **
get_casts(void)
{
    PyArrayMethod_Spec **casts = malloc(2 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &StrPtrToStrPtrCastSpec;
    casts[1] = NULL;

    return casts;
}
