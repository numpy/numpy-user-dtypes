#ifndef _NPY_SCALAR_H
#define _NPY_SCALAR_H


typedef struct {
    PyObject_HEAD;
    double value;
    PyObject *unit;
} QuantityScalarObject;

int
UnitConverter(PyObject *obj, PyObject **unit);

extern PyTypeObject QuantityScalar_Type;

#endif  /* _NPY_SCALAR_H */
