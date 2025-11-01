#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"

#include "mpfr.h"

#include "dtype.h"
#include "scalar.h"
/*
 * Some terrible hacks to make things work that need to be improved in NumPy.
 */


/*
 * A previous verion had a more tricky copyswap, but really we can just
 * copy the itemsize, needed because NumPy still uses it occasionally
 * (for larger itemsizes at least).
 */
static void
copyswap_mpf(char *dst, char *src, int swap, PyArrayObject *ap)
{
    /* Note that it is probably better to only get the descr from `ap` */
    PyArray_Descr *descr = PyArray_DESCR(ap);

    /* copy data and then fix significand (could also do same as cast...) */
    memcpy(dst, src, descr->elsize);
    // TODO: To support unaligned data, only need to do this if it is aligned:
    mpfr_custom_move((mpfr_ptr)dst, ((mpf_field *)dst)->significand);
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
    PyDataType_GetArrFuncs(&descr->base)->copyswap = (PyArray_CopySwapFunc *)&copyswap_mpf;
    Py_DECREF(descr);

    return 0;
}
