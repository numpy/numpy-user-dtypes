#ifndef _QUADDTYPE_MATMUL_H
#define _QUADDTYPE_MATMUL_H

/**
 * Quad Precision Matrix Multiplication for NumPy
 *
 * This module implements matrix multiplication functionality for the QuadPrecDType
 * by registering custom loops with numpy's matmul generalized ufunc.
 *
 * Supports all matmul operation types:
 * - Vector-vector (dot product): (n,) @ (n,) -> scalar
 * - Matrix-vector: (m,n) @ (n,) -> (m,)
 * - Vector-matrix: (n,) @ (n,p) -> (p,)
 * - Matrix-matrix: (m,n) @ (n,p) -> (m,p)
 *
 * Uses naive algorithms optimized for correctness rather than performance.
 * For production use, consider integration with QBLAS optimized routines.
 */

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the matmul operations for the quad precision dtype.
 * This function registers the matmul generalized ufunc with numpy.
 *
 * @param numpy The numpy module object
 * @return 0 on success, -1 on failure
 */
int
init_matmul_ops(PyObject *numpy);

#ifdef __cplusplus
}
#endif

#endif  // _QUADDTYPE_MATMUL_H