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
 * It would be more compat to just store the kind, exponent and signfificand,
 * however.  For in-place operations mpfr needs cannot share the same
 * significand for multiple ops (but can have an op repeat).
 * So not storeing only those (saving 16 bytes of 48 for a 128 bit number)
 * removes the need to worry about this.
 */
static_assert(_Alignof(mpfr_t) >= _Alignof(mp_limb_t),
              "mpfr_t storage not aligned as much as limb_t?!");
typedef mpfr_t mpf_storage;

extern PyArray_DTypeMeta MPFDType;


/*
 * Load into an mpfr_ptr, use a macro which may allow easier changing back
 * to a compact storage scheme.
 */
static inline void
_mpf_load(mpfr_ptr *x, char *data_ptr, mpfr_prec_t precision) {
    x[0] = (mpfr_ptr)data_ptr;
    /*
     * We must ensure the signficand is initialized, but NumPy only ensures
     * everything is NULL'ed.
     */
    if (mpfr_custom_get_significand(x[0]) == NULL) {
        void *signficand = data_ptr + sizeof(mpf_storage);
        mpfr_custom_init(signficand, precision);
        mpfr_custom_init_set(x[0], MPFR_NAN_KIND, 0, precision, signficand);
    }
}
#define mpf_load(x, data_ptr, precision) _mpf_load(&x, data_ptr, precision)



/*
 * Not actually required in the current scheme, but keep for now.
 * (I had a more compat storage scheme at some point.)
 */
static inline void
mpf_store(char *data_ptr, mpfr_t x) {
    assert(data_ptr == (char *)x);
}


MPFDTypeObject *
new_MPFDType_instance(mpfr_prec_t precision);

int
init_mpf_dtype(void);

#ifdef __cplusplus
}
#endif

#endif  /*_MPRFDTYPE_DTYPE_H*/
