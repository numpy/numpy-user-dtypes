#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H

typedef struct {
    PyArray_Descr base;
    long size;
} ASCIIDTypeObject;

extern PyArray_DTypeMeta ASCIIDType;
extern PyTypeObject *ASCIIScalar_Type;

ASCIIDTypeObject *
new_asciidtype_instance(long size);

int
init_ascii_dtype(void);

#endif /*_NPY_DTYPE_H*/
