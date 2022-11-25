#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

/* Gets the conversion between two units: */
int
get_conversion_factor(
        PyObject *from_unit, PyObject *to_unit, double *factor, double *offset);

extern PyArrayMethod_Spec MPFToMPFCastSpec;


#endif  /* _NPY_CASTS_H */
