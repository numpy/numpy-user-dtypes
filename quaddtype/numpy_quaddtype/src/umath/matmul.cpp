#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>
#include <cstdio>
#include <string.h>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"
}

#include "../quad_common.h"
#include "../scalar.h"
#include "../dtype.h"
#include "../ops.hpp"
#include "matmul.h"
#include "promoters.hpp"
#include "../quadblas_interface.h"

/**
 * Resolve descriptors for matmul operation.
 * Only supports SLEEF backend when QBLAS is enabled.
 */
static NPY_CASTING
quad_matmul_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                npy_intp *NPY_UNUSED(view_offset))
{
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_in2 = (QuadPrecDTypeObject *)given_descrs[1];

    // QBLAS only supports SLEEF backend
    if (descr_in1->backend != BACKEND_SLEEF || descr_in2->backend != BACKEND_SLEEF) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "QBLAS-accelerated matmul only supports SLEEF backend. "
                        "Other backends are not supported with QBLAS.");
        return (NPY_CASTING)-1;
    }

    // Both inputs must use SLEEF backend
    QuadBackendType target_backend = BACKEND_SLEEF;
    NPY_CASTING casting = NPY_NO_CASTING;

    // Set up input descriptors
    for (int i = 0; i < 2; i++) {
        Py_INCREF(given_descrs[i]);
        loop_descrs[i] = given_descrs[i];
    }

    // Set up output descriptor
    if (given_descrs[2] == NULL) {
        loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
        if (!loop_descrs[2]) {
            return (NPY_CASTING)-1;
        }
    }
    else {
        QuadPrecDTypeObject *descr_out = (QuadPrecDTypeObject *)given_descrs[2];
        if (descr_out->backend != target_backend) {
            PyErr_SetString(PyExc_NotImplementedError,
                            "QBLAS-accelerated matmul only supports SLEEF backend for output.");
            return (NPY_CASTING)-1;
        }
        else {
            Py_INCREF(given_descrs[2]);
            loop_descrs[2] = given_descrs[2];
        }
    }
    return casting;
}

/**
 * Determine the type of operation based on input dimensions
 */
enum MatmulOperationType {
    MATMUL_DOT,   // 1D x 1D -> scalar
    MATMUL_GEMV,  // 2D x 1D -> 1D
    MATMUL_GEMM   // 2D x 2D -> 2D
};

static MatmulOperationType
determine_operation_type(npy_intp m, npy_intp n, npy_intp p)
{
    // For matmul signature (m?,n),(n,p?)->(m?,p?):
    // - If m=1 and p=1: vector dot product (1D x 1D)
    // - If p=1: matrix-vector multiplication (2D x 1D)
    // - Otherwise: matrix-matrix multiplication (2D x 2D)

    if (m == 1 && p == 1) {
        return MATMUL_DOT;
    }
    else if (p == 1) {
        return MATMUL_GEMV;
    }
    else {
        return MATMUL_GEMM;
    }
}

/**
 * Matrix multiplication strided loop using QBLAS.
 * Automatically selects the appropriate QBLAS operation based on input dimensions.
 */
static int
quad_matmul_strided_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[], NpyAuxData *auxdata)
{
    // Extract dimensions
    npy_intp N = dimensions[0];  // Batch size, this remains always 1 for matmul afaik
    npy_intp m = dimensions[1];  // Rows of first matrix
    npy_intp n = dimensions[2];  // Cols of first matrix / rows of second matrix
    npy_intp p = dimensions[3];  // Cols of second matrix

    // Extract batch strides
    npy_intp A_stride = strides[0];
    npy_intp B_stride = strides[1];
    npy_intp C_stride = strides[2];

    // Extract core strides for matrix dimensions
    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    // Note: B_col_stride and C_col_stride not needed for row-major QBLAS calls

    // Get backend from descriptor (should be SLEEF only)
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    if (descr->backend != BACKEND_SLEEF) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error: non-SLEEF backend in QBLAS matmul");
        return -1;
    }

    // Determine operation type
    MatmulOperationType op_type = determine_operation_type(m, n, p);

    // Constants for QBLAS
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    // print all information for debugging
    printf("DEBUG: Performing %ld batch operations with dimensions (%ld, %ld, %ld)\n", (long)N,
           (long)m, (long)n, (long)p);
    printf("DEBUG: Strides - A: (%ld, %ld), B: (%ld, %ld), C: (%ld, %ld)\n", (long)A_row_stride,
           (long)A_col_stride, (long)B_row_stride, (long)B_col_stride, (long)C_row_stride,
           (long)C_col_stride);
    printf("DEBUG: Operation type: %d\n", op_type);

    char *A = data[0];
    char *B = data[1];
    char *C = data[2];

    Sleef_quad *A_ptr = (Sleef_quad *)A;
    Sleef_quad *B_ptr = (Sleef_quad *)B;
    Sleef_quad *C_ptr = (Sleef_quad *)C;

    int result = -1;

    switch (op_type) {
        case MATMUL_DOT: {
            // Vector dot product: C = A^T * B (both are vectors)
            // A has shape (1, n), B has shape (n, 1), C has shape (1, 1)

            printf("DEBUG: Using QBLAS dot product for %ld elements\n", (long)n);

            // A is effectively a vector of length n
            // B is effectively a vector of length n
            size_t incx = A_col_stride / sizeof(Sleef_quad);
            size_t incy = B_row_stride / sizeof(Sleef_quad);

            result = qblas_dot(n, A_ptr, incx, B_ptr, incy, C_ptr);
            break;
        }

        case MATMUL_GEMV: {
            // Matrix-vector multiplication: C = A * B
            // A has shape (m, n), B has shape (n, 1), C has shape (m, 1)

            printf("DEBUG: Using QBLAS GEMV for %ldx%ld matrix times %ld vector\n", (long)m,
                   (long)n, (long)n);

            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t incx = B_row_stride / sizeof(Sleef_quad);
            size_t incy = C_row_stride / sizeof(Sleef_quad);

            result =
                    qblas_gemv('R', 'N', m, n, &alpha, A_ptr, lda, B_ptr, incx, &beta, C_ptr, incy);
            break;
        }

        case MATMUL_GEMM: {
            // Matrix-matrix multiplication: C = A * B
            // A has shape (m, n), B has shape (n, p), C has shape (m, p)

            printf("DEBUG: Using QBLAS GEMM for %ldx%ldx%ld matrices\n", (long)m, (long)n, (long)p);

            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t ldb = B_row_stride / sizeof(Sleef_quad);
            size_t ldc = C_row_stride / sizeof(Sleef_quad);

            result = qblas_gemm('R', 'N', 'N', m, p, n, &alpha, A_ptr, lda, B_ptr, ldb, &beta,
                                C_ptr, ldc);
            break;
        }
    }

    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "QBLAS operation failed");
        return -1;
    }

    return 0;
}

/**
 * Register matmul support with QBLAS acceleration
 */
int
init_matmul_ops(PyObject *numpy)
{
    printf("DEBUG: init_matmul_ops - registering QBLAS-accelerated matmul\n");

    // Get the existing matmul ufunc
    PyObject *ufunc = PyObject_GetAttrString(numpy, "matmul");
    if (ufunc == NULL) {
        printf("DEBUG: Failed to get numpy.matmul\n");
        return -1;
    }

    // Setup method specification for QBLAS-accelerated matmul
    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {{NPY_METH_resolve_descriptors, (void *)&quad_matmul_resolve_descriptors},
                           {NPY_METH_strided_loop, (void *)&quad_matmul_strided_loop},
                           {NPY_METH_unaligned_strided_loop, (void *)&quad_matmul_strided_loop},
                           {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_matmul_qblas",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    printf("DEBUG: About to add QBLAS loop to matmul ufunc...\n");

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        printf("DEBUG: Failed to add QBLAS loop to matmul ufunc\n");
        Py_DECREF(ufunc);
        return -1;
    }

    printf("DEBUG: Successfully added QBLAS matmul loop!\n");

    // Add promoter
    PyObject *promoter_capsule =
            PyCapsule_New((void *)&quad_ufunc_promoter, "numpy._ufunc_promoter", NULL);
    if (promoter_capsule == NULL) {
        Py_DECREF(ufunc);
        return -1;
    }

    PyObject *DTypes = PyTuple_Pack(3, &PyArrayDescr_Type, &PyArrayDescr_Type, &PyArrayDescr_Type);
    if (DTypes == NULL) {
        Py_DECREF(promoter_capsule);
        Py_DECREF(ufunc);
        return -1;
    }

    if (PyUFunc_AddPromoter(ufunc, DTypes, promoter_capsule) < 0) {
        printf("DEBUG: Failed to add promoter (continuing anyway)\n");
        PyErr_Clear();  // Don't fail if promoter fails
    }
    else {
        printf("DEBUG: Successfully added promoter\n");
    }

    Py_DECREF(DTypes);
    Py_DECREF(promoter_capsule);
    Py_DECREF(ufunc);

    printf("DEBUG: init_matmul_ops completed successfully with QBLAS acceleration\n");
    return 0;
}