#ifndef _NPY_DTYPE_H
#define _NPY_DTYPE_H


typedef struct {
    PyArray_Descr base;
    PyObject *metadata;
} MetadataDTypeObject;

extern PyArray_DTypeMeta MetadataDType;

MetadataDTypeObject *
new_metadatadtype_instance(PyObject *metadata);

int init_metadata_dtype(void);

#endif  /*_NPY_DTYPE_H*/
