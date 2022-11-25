#ifndef _MPRFDTYPE_DTYPE_H
#define _MPRFDTYPE_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mpfr.h"

#include "scalar.h"


typedef struct {
    PyArray_Descr base;
    mpfr_prec_t precision;
} MPFDTypeObject;

extern PyArray_DTypeMeta MPFDType;


/*
 * We can make sure NumPy initializes with NULL, but currently not more
 * which should maybe be added (cleanup may need to deal with NULLs, but
 * other than that we probably could allow init if necessary).
 */
static inline void
ensure_mpf_init(mpf_field *mpf_ptr, mpfr_prec_t precision) {
    if (mpfr_custom_get_significand(mpf_ptr->x) == NULL) {
        /* since we need to init anyway, set it to NAN... */
        mpfr_custom_init_set(
            mpf_ptr->x, MPFR_ZERO_KIND, 0, precision, mpf_ptr->significand);
    }
}


MPFDTypeObject *
new_MPFDType_instance(mpfr_prec_t precision);

int
init_mpf_dtype(void);

#ifdef __cplusplus
}
#endif

#endif  /*_MPRFDTYPE_DTYPE_H*/
