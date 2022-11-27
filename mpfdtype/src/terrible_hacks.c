#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "scalar.h"
/*
 * Some terrible hacks to make things work that need to be improved in NumPy.
 */


/*
 * A copy swap function (we never need to swap, but do need the copy).
 */
static void
copyswap_mpf(char *dst, char *src, int swap, PyArrayObject *arr)
{
    PyArray_Descr *descr = PyArray_DESCR(arr);

    memcpy(dst, src, descr->elsize);
    /* Support unaligned access */
    if ((uintptr_t)dst % _Alignof(mpf_field) == 0) {
        mpf_field *mpf = (mpf_field *)dst;
        /* May yet be uninitialized/NULL, in which case do nothing. */
        if (mpfr_custom_get_significand(mpf->x) != NULL) {
            /* Otherwise, move the signficand to the correct new offset: */
            mpfr_custom_move(mpf->x, mpf->significand);
        }
    }
    /* No point in updating signficand, needs to be done on load. */
}


/* Should only be used for sorting, so more complex than necessary, probably */
int compare_mpf(mpf_field *in1, mpf_field *in2, PyArrayObject *ap)
{
    mpfr_prec_t precision = ((MPFDTypeObject *)PyArray_DESCR(ap))->precision;
    in1 = ensure_mpf_init(in1, precision);
    in2 = ensure_mpf_init(in2, precision);

    if (!mpfr_total_order_p(in1->x, in2->x)) {
        return 1;
    }
    if (!mpfr_total_order_p(in2->x, in1->x)) {
        return -1;
    }
    return 0;
}


int
init_terrible_hacks(void) {
    /* Defaults to -1 byt ISNUMBER misfires for it, so use MAX */
    MPFDType.type_num = NPY_MAX_INT;

    /*
     * Add a some ->f slots the terrible way:
     */
    MPFDTypeObject *descr = new_MPFDType_instance(10);
    if (descr == NULL) {
        return -1;
    }
    /* ->f slots are the same for all instances (currently). */
    descr->base.f->copyswap = &copyswap_mpf;
    descr->base.f->compare = &compare_mpf;
    Py_DECREF(descr);

    return 0;
}
