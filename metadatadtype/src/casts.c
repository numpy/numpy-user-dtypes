#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

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
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        PyArray_Descr *given_descrs[2],
        PyArray_Descr *loop_descrs[2],
        npy_intp *view_offset)
{
	PyObject* meta1 = NULL;
	PyObject* meta2 = NULL;
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
    return NPY_SAME_KIND_CASTING;
}


typedef struct {
    NpyAuxData base;
    double factor;
    double offset;
} conv_auxdata;


static void
conv_auxdata_free(conv_auxdata *conv_auxdata)
{
    PyMem_Free(conv_auxdata);
}


static int
metadata_to_metadata_contiguous_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const NPY_UNUSED(strides[]), conv_auxdata *auxdata)
{
    return 0;
}


static int
metadata_to_metadata_strided_no_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    return 0;
}


static int
metadata_to_metadata_contiguous_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const NPY_UNUSED(strides[]), conv_auxdata *auxdata)
{
    return 0;
}


static int
metadata_to_metadata_strided_offset(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    return 0;
}


static int
metadata_to_metadata_unaligned(PyArrayMethod_Context *NPY_UNUSED(context),
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], conv_auxdata *auxdata)
{
    return 0;
}


static int
metadata_to_metadata_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int NPY_UNUSED(move_references), const npy_intp *strides,
        PyArrayMethod_StridedLoop **out_loop, NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags)
{
    conv_auxdata *conv_auxdata = PyMem_Calloc(1, sizeof(conv_auxdata));
	if (conv_auxdata < 0) {
        PyErr_NoMemory();
        return -1;
    }

    conv_auxdata->base.free = (void *)conv_auxdata_free;

    *out_transferdata = (NpyAuxData *)conv_auxdata;

    int contig = (strides[0] == sizeof(double)
                  && strides[1] == sizeof(double));
    int no_offset = conv_auxdata->offset == 0;

    if (aligned && contig && no_offset) {
        *out_loop = (PyArrayMethod_StridedLoop *)&metadata_to_metadata_contiguous_no_offset;
    }
    else if (aligned && contig) {
        *out_loop = (PyArrayMethod_StridedLoop *)&metadata_to_metadata_contiguous_offset;
    }
    else if (aligned && no_offset) {
        *out_loop = (PyArrayMethod_StridedLoop *)&metadata_to_metadata_strided_no_offset;
    }
    else if (aligned) {
        *out_loop = (PyArrayMethod_StridedLoop *)&metadata_to_metadata_strided_offset;
    }
    else {
        *out_loop = (PyArrayMethod_StridedLoop *)&metadata_to_metadata_unaligned;
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
        {NPY_METH_resolve_descriptors, &metadata_to_metadata_resolve_descriptors},
        {_NPY_METH_get_loop, &metadata_to_metadata_get_loop},
        {0, NULL}
};


PyArrayMethod_Spec MetadataToMetadataCastSpec = {
    .name = "cast_MetadataDType_to_MetadataDType",
    .nin = 1,
    .nout = 1,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,
    .casting = NPY_SAME_KIND_CASTING,
    .dtypes = m2m_dtypes,
    .slots = m2m_slots,
};
