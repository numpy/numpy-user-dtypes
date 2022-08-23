#ifndef _NPY_CASTS_H
#define _NPY_CASTS_H

/* Gets the conversion between two units: */
int
get_conversion_factor(
        PyObject *from_unit, PyObject *to_unit, double *factor, double *offset);

extern PyArrayMethod_Spec UnitToUnitCastSpec;
extern PyArrayMethod_Spec DoubleToUnitCastSpec;
extern PyArrayMethod_Spec UnitToDoubleCastSpec;
extern PyArrayMethod_Spec UnitToBoolCastSpec;


/*
 * These are actually defined in `additional_numeric_casts.c.src`.
 *
 * Verbose, but lets do it like this for now (there should be better ways to
 * do this), NumPy could help.  Another thing to consider is allowing
 * registering casts later, so that some of this could be done dynamically.
 */

extern PyArrayMethod_Spec BoolToUnitCastSpec;

extern PyArrayMethod_Spec UnitToUByteCastSpec;
extern PyArrayMethod_Spec UByteToUnitCastSpec;

extern PyArrayMethod_Spec UnitToUShortCastSpec;
extern PyArrayMethod_Spec UShortToUnitCastSpec;

extern PyArrayMethod_Spec UnitToUIntCastSpec;
extern PyArrayMethod_Spec UIntToUnitCastSpec;

extern PyArrayMethod_Spec UnitToULongCastSpec;
extern PyArrayMethod_Spec ULongToUnitCastSpec;

extern PyArrayMethod_Spec UnitToULongLongCastSpec;
extern PyArrayMethod_Spec ULongLongToUnitCastSpec;

extern PyArrayMethod_Spec UnitToByteCastSpec;
extern PyArrayMethod_Spec ByteToUnitCastSpec;

extern PyArrayMethod_Spec UnitToShortCastSpec;
extern PyArrayMethod_Spec ShortToUnitCastSpec;

extern PyArrayMethod_Spec UnitToIntCastSpec;
extern PyArrayMethod_Spec IntToUnitCastSpec;

extern PyArrayMethod_Spec UnitToLongCastSpec;
extern PyArrayMethod_Spec LongToUnitCastSpec;

extern PyArrayMethod_Spec UnitToLongLongCastSpec;
extern PyArrayMethod_Spec LongLongToUnitCastSpec;

extern PyArrayMethod_Spec UnitToFloatCastSpec;
extern PyArrayMethod_Spec FloatToUnitCastSpec;


#endif  /* _NPY_CASTS_H */
