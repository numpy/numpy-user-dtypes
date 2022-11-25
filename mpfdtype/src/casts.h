#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Gets the conversion between two units: */
int
get_conversion_factor(
        PyObject *from_unit, PyObject *to_unit, double *factor, double *offset);

extern PyArrayMethod_Spec MPFToMPFCastSpec;

#ifdef __cplusplus
}
#endif

#endif  /* _NPY_CASTS_H */
