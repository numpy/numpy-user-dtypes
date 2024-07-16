#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "casts.h"
#include "dtype.h"

/*
 * And now the actual cast code!  Starting with the "resolver" which tells
 * us about cast safety.
 * Note also the `view_offset`!  It is a way for you to tell NumPy, that this
 * cast does not require anything at all, but the cast can simply be done as
 * a view.
 * For `arr.astype()` it might mean returning a view (eventually, not yet).
 * For ufuncs, it already means that they don't have to do a cast at all!
 */
static NPY_CASTING
metadata_to_metadata_resolve_descriptors(
        PyObject *NPY_UNUSED(self), PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2], PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
    PyObject *meta1 = NULL;
    PyObject *meta2 = NULL;
    if (given_descrs[1] == NULL) {
        meta1 = given_descrs[0]->metadata;
        Py_INCREF(given_descrs[0]);
        loop_descrs[1] = given_descrs[0];
    }
    else {
        meta1 = given_descrs[0]->metadata;
        meta2 = given_descrs[1]->metadata;
        Py_INCREF(given_descrs[1]);
        loop_descrs[1] = given_descrs[1];
    }
    Py_INCREF(given_descrs[0]);
    loop_descrs[0] = given_descrs[0];

    if (meta2 != NULL) {
        int comp = PyObject_RichCompareBool(meta1, meta2, Py_EQ);
        if (comp == -1) {
            return -1;
        }
        if (comp == 1) {
            // identical metadata so no need to cast, views are OK
            *view_offset = 0;
            return NPY_NO_CASTING;
        }
    }

    /* Should this use safe casting? */
    return NPY_SAFE_CASTING;
}

static int
metadata_to_float64like_contiguous(PyArrayMethod_Context *NPY_UNUSED(context),
                                   char *const data[],
                                   npy_intp const dimensions[],
                                   npy_intp const NPY_UNUSED(strides[]),
                                   NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    double *in = (double *)data[0];
    double *out = (double *)data[1];

    while (N--) {
        *out = *in;
        out++;
        in++;
    }

    return 0;
}

static int
metadata_to_float64like_strided(PyArrayMethod_Context *NPY_UNUSED(context),
                                char *const data[],
                                npy_intp const dimensions[],
                                npy_intp const strides[],
                                NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        *(double *)out = *(double *)in;
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static int
metadata_to_float64like_unaligned(PyArrayMethod_Context *NPY_UNUSED(context),
                                  char *const data[],
                                  npy_intp const dimensions[],
                                  npy_intp const strides[],
                                  NpyAuxData *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in = data[0];
    char *out = data[1];
    npy_intp in_stride = strides[0];
    npy_intp out_stride = strides[1];

    while (N--) {
        memcpy(out, in, sizeof(double));  // NOLINT
        in += in_stride;
        out += out_stride;
    }

    return 0;
}

static int
metadata_to_metadata_get_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                              int aligned, int NPY_UNUSED(move_references),
                              const npy_intp *strides,
                              PyArrayMethod_StridedLoop **out_loop,
                              NpyAuxData **NPY_UNUSED(out_transferdata),
                              NPY_ARRAYMETHOD_FLAGS *flags)
{
    int contig =
            (strides[0] == sizeof(double) && strides[1] == sizeof(double));

    if (aligned && contig) {
        *out_loop = (PyArrayMethod_StridedLoop
                             *)&metadata_to_float64like_contiguous;
    }
    else if (aligned) {
        *out_loop =
                (PyArrayMethod_StridedLoop *)&metadata_to_float64like_strided;
    }
    else {
        *out_loop = (PyArrayMethod_StridedLoop
                             *)&metadata_to_float64like_unaligned;
    }

    *flags = 0;
    return 0;
}

static int
metadata_to_float64_get_loop(PyArrayMethod_Context *NPY_UNUSED(context),
                             int aligned, int NPY_UNUSED(move_references),
                             const npy_intp *strides,
                             PyArrayMethod_StridedLoop **out_loop,
                             NpyAuxData **NPY_UNUSED(out_transferdata),
                             NPY_ARRAYMETHOD_FLAGS *flags)
{
    int contig =
            (strides[0] == sizeof(double) && strides[1] == sizeof(double));

    if (aligned && contig) {
        *out_loop = (PyArrayMethod_StridedLoop
                             *)&metadata_to_float64like_contiguous;
    }
    else if (aligned) {
        *out_loop =
                (PyArrayMethod_StridedLoop *)&metadata_to_float64like_strided;
    }
    else {
        *out_loop = (PyArrayMethod_StridedLoop
                             *)&metadata_to_float64like_unaligned;
    }

    *flags = 0;
    return 0;
}

/*
 * NumPy currently allows NULL for the own DType/"cls".  For other DTypes
 * we would have to fill it in here:
 */
static PyArray_DTypeMeta *m2m_dtypes[2] = {NULL, NULL};

static PyType_Slot m2m_slots[] = {
        {NPY_METH_resolve_descriptors,
         &metadata_to_metadata_resolve_descriptors},
        {NPY_METH_get_loop, &metadata_to_metadata_get_loop},
        {0, NULL}};

PyArrayMethod_Spec MetadataToMetadataCastSpec = {
        .name = "cast_MetadataDType_to_MetadataDType",
        .nin = 1,
        .nout = 1,
        .flags = NPY_METH_SUPPORTS_UNALIGNED,
        .casting = NPY_SAME_KIND_CASTING,
        .dtypes = m2m_dtypes,
        .slots = m2m_slots,
};

static PyType_Slot m2f_slots[] = {
        {NPY_METH_get_loop, &metadata_to_float64_get_loop}, {0, NULL}};

static char *m2f_name = "cast_MetadataDType_to_Float64";

PyArrayMethod_Spec **
get_casts(void)
{
    PyArray_DTypeMeta **m2f_dtypes = malloc(2 * sizeof(PyArray_DTypeMeta *));
    m2f_dtypes[0] = NULL;
    m2f_dtypes[1] = &PyArray_DoubleDType;

    PyArrayMethod_Spec *MetadataToFloat64CastSpec =
            malloc(sizeof(PyArrayMethod_Spec));
    MetadataToFloat64CastSpec->name = m2f_name;
    MetadataToFloat64CastSpec->nin = 1;
    MetadataToFloat64CastSpec->nout = 1;
    MetadataToFloat64CastSpec->flags = NPY_METH_SUPPORTS_UNALIGNED;
    MetadataToFloat64CastSpec->casting = NPY_SAFE_CASTING;
    MetadataToFloat64CastSpec->dtypes = m2f_dtypes;
    MetadataToFloat64CastSpec->slots = m2f_slots;

    PyArrayMethod_Spec **casts = malloc(3 * sizeof(PyArrayMethod_Spec *));
    casts[0] = &MetadataToMetadataCastSpec;
    casts[1] = MetadataToFloat64CastSpec;
    casts[2] = NULL;

    return casts;
}
