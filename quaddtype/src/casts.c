#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "casts.h"
#include "dtype.h"

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
static NPY_CASTING quad_to_quad_resolve_descriptors(
    PyObject* NPY_UNUSED(self),
    PyArray_DTypeMeta* NPY_UNUSED(dtypes[2]),
    PyArray_Descr* given_descrs[2],
    PyArray_Descr* loop_descrs[2],
    npy_intp* view_offset
) {
    return NPY_SAME_KIND_CASTING;
}

// typedef struct
// {
//   NpyAuxData base;
//   double factor;
//   double offset;
// } conv_auxdata;


static int quad_to_quad_contiguous(
    PyArrayMethod_Context *NPY_UNUSED(context),
    char* const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    void* auxdata
) {
    return 0;
}

static int quad_to_quad_strided(
    PyArrayMethod_Context *NPY_UNUSED(context),
    char* const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    void* auxdata
) {
    return 0;
}

static int quad_to_quad_unaligned(
    PyArrayMethod_Context *NPY_UNUSED(context),
    char* const data[],
    npy_intp const dimensions[],
    npy_intp const strides[],
    void* auxdata
) {
    return 0;
}

// Returns the low-level C (strided inner-loop) function which
// performs the actual operation. This method may initially be private, users will be
// able to provide a set of optimized inner-loop functions instead:
// * `strided_inner_loop`
// * `contiguous_inner_loop`
// * `unaligned_strided_loop`
// * ...
static int quad_to_quad_get_loop(
    PyArrayMethod_Context* context,
    int aligned,
    int NPY_UNUSED(move_references),
    const npy_intp* strides,
    PyArrayMethod_StridedLoop** out_loop,
    NpyAuxData** out_transferdata,
    NPY_ARRAYMETHOD_FLAGS* flags
) {
    int contig = (strides[0] == sizeof(__float128) && strides[1] == sizeof(__float128));

    if (aligned && contig) *out_loop = (PyArrayMethod_StridedLoop*)&quad_to_quad_contiguous;
    else if (aligned) *out_loop = (PyArrayMethod_StridedLoop*)&quad_to_quad_strided;
    else *out_loop = (PyArrayMethod_StridedLoop*)&quad_to_quad_unaligned;

    *flags = 0;
    return 0;
}

/*
 * NumPy currently allows NULL for the own DType/"cls".  For other DTypes
 * we would have to fill it in here:
 */
static PyArray_DTypeMeta* QuadToQuadDtypes[2] = { NULL, NULL };

static PyType_Slot QuadToQuadSlots[] = {
    { NPY_METH_resolve_descriptors, &quad_to_quad_resolve_descriptors },
    { _NPY_METH_get_loop, &quad_to_quad_get_loop },
    { 0, NULL }
};


PyArrayMethod_Spec QuadToQuadCastSpec = {
    .name = "cast_QuadDType_to_QuadDType",
    .nin = 1,
    .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,
    .casting = NPY_SAME_KIND_CASTING,
    .dtypes = QuadToQuadDtypes,
    .slots = QuadToQuadSlots,
};
