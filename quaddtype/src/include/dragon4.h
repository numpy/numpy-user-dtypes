#ifndef _QUADDTYPE_DRAGON4_H
#define _QUADDTYPE_DRAGON4_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#include "numpy/arrayobject.h"
#include <sleef.h>
#include "quad_common.h"

typedef enum DigitMode
{
    /* Round digits to print shortest uniquely identifiable number. */
    DigitMode_Unique,
    /* Output the digits of the number as if with infinite precision */
    DigitMode_Exact,
} DigitMode;

typedef enum CutoffMode
{
    /* up to cutoffNumber significant digits */
    CutoffMode_TotalLength,
    /* up to cutoffNumber significant digits past the decimal point */
    CutoffMode_FractionLength,
} CutoffMode;

typedef enum TrimMode
{
    TrimMode_None,         /* don't trim zeros, always leave a decimal point */
    TrimMode_LeaveOneZero, /* trim all but the zero before the decimal point */
    TrimMode_Zeros,        /* trim all trailing zeros, leave decimal point */
    TrimMode_DptZeros,     /* trim trailing zeros & trailing decimal point */
} TrimMode;

typedef struct {
    int scientific;
    DigitMode digit_mode;
    CutoffMode cutoff_mode;
    int precision;
    int min_digits;
    int sign;
    TrimMode trim_mode;
    int digits_left;
    int digits_right;
    int exp_digits;
} Dragon4_Options;

PyObject *Dragon4_Positional_QuadDType(Sleef_quad *val, DigitMode digit_mode,
                   CutoffMode cutoff_mode, int precision, int min_digits,
                   int sign, TrimMode trim, int pad_left, int pad_right);

const char *Dragon4_Positional_QuadDType_CStr(Sleef_quad *val, DigitMode digit_mode,
                   CutoffMode cutoff_mode, int precision, int min_digits,
                   int sign, TrimMode trim, int pad_left, int pad_right);

PyObject *Dragon4_Scientific_QuadDType(Sleef_quad *val, DigitMode digit_mode, 
                   int precision, int min_digits, int sign, TrimMode trim, 
                   int pad_left, int exp_digits);

const char *Dragon4_Scientific_QuadDType_CStr(Sleef_quad *val, DigitMode digit_mode, 
                   int precision, int min_digits, int sign, TrimMode trim, 
                   int pad_left, int exp_digits);

PyObject *Dragon4_Positional(PyObject *obj, DigitMode digit_mode, 
                   CutoffMode cutoff_mode, int precision, int min_digits, 
                   int sign, TrimMode trim, int pad_left, int pad_right);

PyObject *Dragon4_Scientific(PyObject *obj, DigitMode digit_mode, int precision,
                   int min_digits, int sign, TrimMode trim, int pad_left,
                   int exp_digits);

#ifdef __cplusplus
}
#endif

#endif /* _QUADDTYPE_DRAGON4_H */