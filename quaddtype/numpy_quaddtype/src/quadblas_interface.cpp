#include "quadblas_interface.h"
#include <cstring>
#include <algorithm>

#ifndef DISABLE_QUADBLAS
#include "../QBLAS/include/quadblas/quadblas.hpp"
#endif // DISABLE_QUADBLAS

extern "C" {


#ifndef  DISABLE_QUADBLAS

int
qblas_dot(size_t n, Sleef_quad *x, size_t incx, Sleef_quad *y, size_t incy, Sleef_quad *result)
{
    if (!x || !y || !result || n == 0) {
        return -1;
    }

    try {
        *result = QuadBLAS::dot(n, x, incx, y, incy);
        return 0;
    }
    catch (...) {
        return -1;
    }
}

int
qblas_gemv(char layout, char trans, size_t m, size_t n, Sleef_quad *alpha, Sleef_quad *A,
           size_t lda, Sleef_quad *x, size_t incx, Sleef_quad *beta, Sleef_quad *y, size_t incy)
{
    if (!alpha || !A || !x || !beta || !y || m == 0 || n == 0) {
        return -1;
    }

    try {
        // Convert layout
        QuadBLAS::Layout qblas_layout;
        if (layout == 'R' || layout == 'r') {
            qblas_layout = QuadBLAS::Layout::RowMajor;
        }
        else if (layout == 'C' || layout == 'c') {
            qblas_layout = QuadBLAS::Layout::ColMajor;
        }
        else {
            return -1;  // Invalid layout
        }

        // Handle transpose (swap dimensions for transpose)
        size_t actual_m = m, actual_n = n;
        if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c') {
            std::swap(actual_m, actual_n);
            // For transpose, we need to adjust the layout
            if (qblas_layout == QuadBLAS::Layout::RowMajor) {
                qblas_layout = QuadBLAS::Layout::ColMajor;
            }
            else {
                qblas_layout = QuadBLAS::Layout::RowMajor;
            }
        }

        // Call QBLAS GEMV
        QuadBLAS::gemv(qblas_layout, actual_m, actual_n, *alpha, A, lda, x, incx, *beta, y, incy);

        return 0;
    }
    catch (...) {
        return -1;
    }
}

int
qblas_gemm(char layout, char transa, char transb, size_t m, size_t n, size_t k, Sleef_quad *alpha,
           Sleef_quad *A, size_t lda, Sleef_quad *B, size_t ldb, Sleef_quad *beta, Sleef_quad *C,
           size_t ldc)
{
    if (!alpha || !A || !B || !beta || !C || m == 0 || n == 0 || k == 0) {
        return -1;
    }

    try {
        QuadBLAS::Layout qblas_layout;
        if (layout == 'R' || layout == 'r') {
            qblas_layout = QuadBLAS::Layout::RowMajor;
        }
        else if (layout == 'C' || layout == 'c') {
            qblas_layout = QuadBLAS::Layout::ColMajor;
        }
        else {
            return -1;  // Invalid layout
        }

        // For now, we only support no transpose
        // TODO: Implement transpose support if needed
        if ((transa != 'N' && transa != 'n') || (transb != 'N' && transb != 'n')) {
            return -1;  // Transpose not implemented yet
        }

        QuadBLAS::gemm(qblas_layout, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);

        return 0;
    }
    catch (...) {
        return -1;
    }
}

int
qblas_supports_backend(QuadBackendType backend)
{
    // QBLAS only supports SLEEF backend
    return (backend == BACKEND_SLEEF) ? 1 : 0;
}

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    int num_threads;
    if (!PyArg_ParseTuple(args, "i", &num_threads)) {
        return NULL;
    }

    if (num_threads <= 0) {
        PyErr_SetString(PyExc_ValueError, "Number of threads must be positive");
        return NULL;
    }

    QuadBLAS::set_num_threads(num_threads);
    Py_RETURN_NONE;
}

PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    int num_threads = QuadBLAS::get_num_threads();
    return PyLong_FromLong(num_threads);
}

PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString("QuadBLAS 1.0.0 - High Performance Quad Precision BLAS");
}

int
_quadblas_set_num_threads(int num_threads)
{
    QuadBLAS::set_num_threads(num_threads);
    return 0;
}

int
_quadblas_get_num_threads(void)
{
    int num_threads = QuadBLAS::get_num_threads();
    return num_threads;
}

#else  // DISABLE_QUADBLAS


int
qblas_dot(size_t n, Sleef_quad *x, size_t incx, Sleef_quad *y, size_t incy, Sleef_quad *result)
{
    return -1;  // QBLAS is disabled, dot product not available
}

int
qblas_gemv(char layout, char trans, size_t m, size_t n, Sleef_quad *alpha, Sleef_quad *A,
           size_t lda, Sleef_quad *x, size_t incx, Sleef_quad *beta, Sleef_quad *y, size_t incy)
{
    return -1;  // QBLAS is disabled, GEMV not available
}

int
qblas_gemm(char layout, char transa, char transb, size_t m, size_t n, size_t k, Sleef_quad *alpha,
           Sleef_quad *A, size_t lda, Sleef_quad *B, size_t ldb, Sleef_quad *beta, Sleef_quad *C,
           size_t ldc)
{
    return -1;  // QBLAS is disabled, GEMM not available
}

int
qblas_supports_backend(QuadBackendType backend)
{
    return -1; // QBLAS is disabled, backend support not available
}

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    // raise error
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}

PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    // raise error
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}

PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    // raise error
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return NULL;
}

int
_quadblas_set_num_threads(int num_threads)
{
    // raise error
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return -1;
}

int
_quadblas_get_num_threads(void)
{
    // raise error
    PyErr_SetString(PyExc_NotImplementedError, "QuadBLAS is disabled");
    return -1;
}

#endif // DISABLE_QUADBLAS

}  // extern "C"