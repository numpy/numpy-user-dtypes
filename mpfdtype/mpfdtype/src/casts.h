#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

#ifdef __cplusplus
extern "C" {
#endif

extern PyArrayMethod_Spec MPFToMPFCastSpec;

PyArrayMethod_Spec **
init_casts(void);

void
free_casts(void);

#ifdef __cplusplus
}
#endif

#endif  /* _NPY_CASTS_H */
