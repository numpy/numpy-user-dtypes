#ifndef _MPF_NUMBERS_H
#define _MPF_NUMBERS_H

#ifdef __cplusplus
extern "C" {
#endif

PyObject *
mpf_richcompare(MPFloatObject *self, PyObject *other, int cmp_op);

extern PyNumberMethods mpf_as_number;

#ifdef __cplusplus
}
#endif

#endif  /* _MPF_NUMBERS_H */
