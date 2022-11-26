
#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY

extern "C" {
    #include <Python.h>

    #include "numpy/arrayobject.h"
    #include "numpy/ndarraytypes.h"
    #include "numpy/experimental_dtype_api.h"
}

#include "mpfr.h"

#include "casts.h"
#include "dtype.h"


static NPY_CASTING
mpf_to_mpf_resolve_descriptors(
        PyObject *NPY_UNUSED(self),
        PyArray_DTypeMeta *NPY_UNUSED(dtypes[2]),
        MPFDTypeObject *given_descrs[2],
        MPFDTypeObject *loop_descrs[2],
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

    if (loop_descrs[0]->precision == loop_descrs[1]->precision) {
        return NPY_EQUIV_CASTING;
    }
    if (loop_descrs[0]->precision > loop_descrs[1]->precision) {
        return NPY_SAFE_CASTING;
    }

    return NPY_SAME_KIND_CASTING;
}


static int
mpf_to_mof_strided_loop(PyArrayMethod_Context *context,
        char *const data[], npy_intp const dimensions[],
        npy_intp const strides[], void *NPY_UNUSED(auxdata))
{
    npy_intp N = dimensions[0];
    char *in_ptr = data[0];
    char *out_ptr = data[1];

    mpfr_prec_t prec_in = ((MPFDTypeObject *)context->descriptors[0])->precision;
    mpfr_prec_t prec_out = ((MPFDTypeObject *)context->descriptors[1])->precision;

    while (N--) {
        mpf_field *in = (mpf_field *)in_ptr;
        mpf_field *out = (mpf_field *)out_ptr;
        ensure_mpf_init(in, prec_in);
        ensure_mpf_init(out, prec_out);

        mpfr_set(out->x, in->x, MPFR_RNDN);

        in_ptr += strides[0];
        out_ptr += strides[1];
    }
    return 0;
}


/*
 * NumPy currently allows NULL for the own DType/"cls".  For other DTypes
 * we would have to fill it in here:
 */
static PyArray_DTypeMeta *mpf2mpf_dtypes[2] = {NULL, NULL};


static PyType_Slot mpf2mpf_slots[] = {
    {NPY_METH_resolve_descriptors, (void *)&mpf_to_mpf_resolve_descriptors},
    {NPY_METH_strided_loop, (void *)&mpf_to_mof_strided_loop},
    /* We don't actually support unaligned access... */
    {NPY_METH_unaligned_strided_loop, (void *)&mpf_to_mof_strided_loop},
    {0, NULL}
};


PyArrayMethod_Spec MPFToMPFCastSpec = {
    .name = "cast_MPF_to_MPF",
    .nin = 1,
    .nout = 1,
    .casting = NPY_SAME_KIND_CASTING,
    .flags = NPY_METH_SUPPORTS_UNALIGNED,  /* not really ... */
    .dtypes = mpf2mpf_dtypes,
    .slots = mpf2mpf_slots,
};

