#ifndef _MPRFDTYPE_DTYPE_H
#define _MPRFDTYPE_DTYPE_H

#include "mpfr.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "scalar.h"


typedef struct {
    PyArray_Descr base;
    mpfr_prec_t precision;
} MPFDTypeObject;


/*
 * It would be slightly easier/faster to store the internals, but use the
 * proper public API here:
 */
typedef struct {
    int kind;
    mpfr_exp_t exp;
    mp_limb_t significand[];
} mpf_storage;

extern PyArray_DTypeMeta MPFDType;


/*
 * We currently use this also when init would suffice (to set significand).
 */
static inline void
mpf_load(mpfr_t x, char *data_ptr, mpfr_prec_t precision) {
    mpf_storage *mpf_ptr = (mpf_storage *)data_ptr;
    /* if the kind is 0, reinitialize significand (presumably it never was) */
    if (mpf_ptr->kind == 0) {
        mpfr_custom_init(mpf_ptr->significand, precision);
        mpfr_custom_init_set(x, MPFR_NAN_KIND, 0, precision, mpf_ptr->significand);
    }
    else {
        mpfr_custom_init_set(
            x, mpf_ptr->kind, mpf_ptr->exp, precision, mpf_ptr->significand);
    }
}

/*
 * Signficand is always stored, but write back kind and exp
 */
static inline void
mpf_store(char *data_ptr, mpfr_t x) {
    mpf_storage *mpf_ptr = (mpf_storage *)data_ptr;
    assert(mpfr_custom_get_signficand(x) == mpf_ptr->Signficand);
    mpf_ptr->kind = mpfr_custom_get_kind(x);
    mpf_ptr->exp = mpfr_custom_get_exp(x);
}


MPFDTypeObject *
new_MPFDType_instance(mpfr_prec_t precision);

int
init_mpf_dtype(void);

#ifdef __cplusplus
}
#endif

#endif  /*_MPRFDTYPE_DTYPE_H*/
