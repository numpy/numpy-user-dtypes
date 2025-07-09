#ifndef _QUADDTYPE_QUADBLAS_INTERFACE_H
#define _QUADDTYPE_QUADBLAS_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>

// Main dot function that handles vector-vector, matrix-vector, and matrix-matrix operations
// Renamed to avoid conflicts with QuadBLAS C interface
PyObject* py_quadblas_dot(PyObject* self, PyObject* args);

// Threading control functions
// Renamed to avoid conflicts with QuadBLAS C interface
PyObject* py_quadblas_set_num_threads(PyObject* self, PyObject* args);
PyObject* py_quadblas_get_num_threads(PyObject* self, PyObject* args);

// Version information
// Renamed to avoid conflicts with QuadBLAS C interface
PyObject* py_quadblas_get_version(PyObject* self, PyObject* args);

#ifdef __cplusplus
}
#endif

#endif // _QUADDTYPE_QUADBLAS_INTERFACE_H