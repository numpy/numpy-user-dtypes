#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL quaddtype_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL quaddtype_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#include "dtype.h"
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
quad_to_quad_resolve_descriptors(PyObject *NPY_UNUSED(self),
                                 PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
                                 PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
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

    return NPY_SAME_KIND_CASTING;
}

// Each element is a __float128 element; no casting needed
static int
quad_to_quad_contiguous(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
                        npy_intp const dimensions[], npy_intp const strides[],
                        NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    __float128 *in = (__float128 *)data[0];
    __float128 *out = (__float128 *)data[1];

    while (N--) {
        *out = *in;
        out++;
        in++;
    }

    return 0;
}

// Elements are strided, e.g.
//
// x = np.linspace(40)
// x[::3]
//
// Therefore the stride needs to be used to increment the pointers inside the loop.
static int
quad_to_quad_strided(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
                     npy_intp const dimensions[], npy_intp const strides[],
                     NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(__float128 *)out = *(__float128 *)in;
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

// Arrays are unaligned.
static int
quad_to_quad_unaligned(PyArrayMethod_Context *NPY_UNUSED(context), char *const data[],
                       npy_intp const dimensions[], npy_intp const strides[],
                       NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        memcpy(out, in, sizeof(__float128));  // NOLINT
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

// Returns the low-level C (strided inner-loop) function which
// performs the actual operation. This method may initially be private, users will be
// able to provide a set of optimized inner-loop functions instead:
// * `strided_inner_loop`
// * `contiguous_inner_loop`
// * `unaligned_strided_loop`
// * ...
static int
quad_to_quad_get_loop(PyArrayMethod_Context *context, int aligned, int NPY_UNUSED(move_references),
                      const npy_intp *strides, PyArrayMethod_StridedLoop **out_loop,
                      NpyAuxData **out_transferdata, NPY_ARRAYMETHOD_FLAGS *flags)
{
    int contig = (strides[0] == sizeof(__float128) && strides[1] == sizeof(__float128));

    if (aligned && contig)
        *out_loop = (PyArrayMethod_StridedLoop *)&quad_to_quad_contiguous;
    else if (aligned)
        *out_loop = (PyArrayMethod_StridedLoop *)&quad_to_quad_strided;
    else
        *out_loop = (PyArrayMethod_StridedLoop *)&quad_to_quad_unaligned;

    *flags = 0;
    return 0;
}

/*
 * NumPy currently allows NULL for the own DType/"cls".  For other DTypes
 * we would have to fill it in here:
 */
static PyArray_DTypeMeta *QuadToQuadDtypes[2] = {NULL, NULL};

static PyType_Slot QuadToQuadSlots[] = {
        {NPY_METH_resolve_descriptors, &quad_to_quad_resolve_descriptors},
        {NPY_METH_get_loop, &quad_to_quad_get_loop},
        {0, NULL}};

PyArrayMethod_Spec QuadToQuadCastSpec = {
        .name = "cast_QuadDType_to_QuadDType",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .casting = NPY_SAME_KIND_CASTING,
        .dtypes = QuadToQuadDtypes,
        .slots = QuadToQuadSlots,
};
