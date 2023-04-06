#ifndef _STRINGDTYPE_SCALAR_H
#define _STRINGDTYPE_SCALAR_H

#include <Python.h>

extern PyTypeObject StringScalar_Type;

int
init_stringdtype_scalar(void);

typedef struct {
    PyUnicodeObject str;
} StringScalarObject;

#endif /* _STRINGDTYPE_SCALAR_H */
