#ifndef QUADBLAS_INTERFACE_H
#define QUADBLAS_INTERFACE_H

#include <stddef.h>
#include <Python.h>
#include "quad_common.h"
#include <sleefquad.h>

#ifdef __cplusplus
extern "C" {
#endif

int
qblas_dot(size_t n, Sleef_quad *x, size_t incx, Sleef_quad *y, size_t incy, Sleef_quad *result);

int
qblas_gemv(char layout, char trans, size_t m, size_t n, Sleef_quad *alpha, Sleef_quad *A,
           size_t lda, Sleef_quad *x, size_t incx, Sleef_quad *beta, Sleef_quad *y, size_t incy);

int
qblas_gemm(char layout, char transa, char transb, size_t m, size_t n, size_t k, Sleef_quad *alpha,
           Sleef_quad *A, size_t lda, Sleef_quad *B, size_t ldb, Sleef_quad *beta, Sleef_quad *C,
           size_t ldc);

int
qblas_supports_backend(QuadBackendType backend);

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args);
PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args);
PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args);

int
_quadblas_set_num_threads(int num_threads);
int
_quadblas_get_num_threads(void);

#ifdef __cplusplus
}
#endif

#endif  // QUADBLAS_INTERFACE_H