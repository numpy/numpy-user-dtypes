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

/**
 * Resolve descriptors for matmul operation.
 * Follows the same pattern as binary_ops.cpp
 */
static NPY_CASTING
quad_matmul_resolve_descriptors(PyObject *self, PyArray_DTypeMeta *const dtypes[],
                                PyArray_Descr *const given_descrs[], PyArray_Descr *loop_descrs[],
                                npy_intp *NPY_UNUSED(view_offset))
{
    // Follow the exact same pattern as quad_binary_op_resolve_descriptors
    QuadPrecDTypeObject *descr_in1 = (QuadPrecDTypeObject *)given_descrs[0];
    QuadPrecDTypeObject *descr_in2 = (QuadPrecDTypeObject *)given_descrs[1];
    QuadBackendType target_backend;

    // Determine target backend and if casting is needed
    NPY_CASTING casting = NPY_NO_CASTING;
    if (descr_in1->backend != descr_in2->backend) {
        target_backend = BACKEND_LONGDOUBLE;
        casting = NPY_SAFE_CASTING;
    }
    else {
        target_backend = descr_in1->backend;
    }

    // Set up input descriptors, casting if necessary
    for (int i = 0; i < 2; i++) {
        if (((QuadPrecDTypeObject *)given_descrs[i])->backend != target_backend) {
            loop_descrs[i] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[i]) {
                return (NPY_CASTING)-1;
            }
        }
        else {
            Py_INCREF(given_descrs[i]);
            loop_descrs[i] = given_descrs[i];
        }
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
            loop_descrs[2] = (PyArray_Descr *)new_quaddtype_instance(target_backend);
            if (!loop_descrs[2]) {
                return (NPY_CASTING)-1;
            }
        }
        else {
            Py_INCREF(given_descrs[2]);
            loop_descrs[2] = given_descrs[2];
        }
    }
    return casting;
}

/**
 * Matrix multiplication strided loop using NumPy 2.0 API.
 * Implements general matrix multiplication for arbitrary dimensions.
 *
 * For matmul with signature (m?,n),(n,p?)->(m?,p?):
 * - dimensions[0] = N (loop dimension, number of batch operations)
 * - dimensions[1] = m (rows of first matrix)
 * - dimensions[2] = n (cols of first matrix / rows of second matrix)
 * - dimensions[3] = p (cols of second matrix)
 *
 * - strides[0], strides[1], strides[2] = batch strides for A, B, C
 * - strides[3], strides[4] = row stride, col stride for A (m, n)
 * - strides[5], strides[6] = row stride, col stride for B (n, p)
 * - strides[7], strides[8] = row stride, col stride for C (m, p)
 */
static int
quad_matmul_strided_loop(PyArrayMethod_Context *context, char *const data[],
                         npy_intp const dimensions[], npy_intp const strides[], NpyAuxData *auxdata)
{
    // Extract dimensions
    npy_intp N = dimensions[0];  // Number of batch operations
    npy_intp m = dimensions[1];  // Rows of first matrix
    npy_intp n = dimensions[2];  // Cols of first matrix / rows of second matrix
    npy_intp p = dimensions[3];  // Cols of second matrix

    // Extract batch strides
    npy_intp A_batch_stride = strides[0];
    npy_intp B_batch_stride = strides[1];
    npy_intp C_batch_stride = strides[2];

    // Extract core strides for matrix dimensions
    npy_intp A_row_stride = strides[3];  // Stride along m dimension of A
    npy_intp A_col_stride = strides[4];  // Stride along n dimension of A
    npy_intp B_row_stride = strides[5];  // Stride along n dimension of B
    npy_intp B_col_stride = strides[6];  // Stride along p dimension of B
    npy_intp C_row_stride = strides[7];  // Stride along m dimension of C
    npy_intp C_col_stride = strides[8];  // Stride along p dimension of C

    // Get backend from descriptor
    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    // Process each batch
    for (npy_intp batch = 0; batch < N; batch++) {
        char *A_batch = data[0] + batch * A_batch_stride;
        char *B_batch = data[1] + batch * B_batch_stride;
        char *C_batch = data[2] + batch * C_batch_stride;

        // Perform matrix multiplication: C = A @ B
        // C[i,j] = sum_k(A[i,k] * B[k,j])
        for (npy_intp i = 0; i < m; i++) {
            for (npy_intp j = 0; j < p; j++) {
                char *C_ij = C_batch + i * C_row_stride + j * C_col_stride;

                if (backend == BACKEND_SLEEF) {
                    Sleef_quad sum = Sleef_cast_from_doubleq1(0.0);  // Initialize to 0

                    for (npy_intp k = 0; k < n; k++) {
                        char *A_ik = A_batch + i * A_row_stride + k * A_col_stride;
                        char *B_kj = B_batch + k * B_row_stride + j * B_col_stride;

                        Sleef_quad a_val, b_val;
                        memcpy(&a_val, A_ik, sizeof(Sleef_quad));
                        memcpy(&b_val, B_kj, sizeof(Sleef_quad));

                        // sum += A[i,k] * B[k,j]
                        sum = Sleef_addq1_u05(sum, Sleef_mulq1_u05(a_val, b_val));
                    }

                    memcpy(C_ij, &sum, sizeof(Sleef_quad));
                }
                else {
                    // Long double backend
                    long double sum = 0.0L;

                    for (npy_intp k = 0; k < n; k++) {
                        char *A_ik = A_batch + i * A_row_stride + k * A_col_stride;
                        char *B_kj = B_batch + k * B_row_stride + j * B_col_stride;

                        long double a_val, b_val;
                        memcpy(&a_val, A_ik, sizeof(long double));
                        memcpy(&b_val, B_kj, sizeof(long double));

                        sum += a_val * b_val;
                    }

                    memcpy(C_ij, &sum, sizeof(long double));
                }
            }
        }
    }

    return 0;
}

/**
 * Register matmul support following the exact same pattern as binary_ops.cpp
 */
int
init_matmul_ops(PyObject *numpy)
{
    printf("DEBUG: init_matmul_ops - registering matmul using NumPy 2.0 API\n");

    // Get the existing matmul ufunc - same pattern as binary_ops
    PyObject *ufunc = PyObject_GetAttrString(numpy, "matmul");
    if (ufunc == NULL) {
        printf("DEBUG: Failed to get numpy.matmul\n");
        return -1;
    }

    // Use the same pattern as binary_ops.cpp
    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

    PyType_Slot slots[] = {{NPY_METH_resolve_descriptors, (void *)&quad_matmul_resolve_descriptors},
                           {NPY_METH_strided_loop, (void *)&quad_matmul_strided_loop},
                           {NPY_METH_unaligned_strided_loop, (void *)&quad_matmul_strided_loop},
                           {0, NULL}};

    PyArrayMethod_Spec Spec = {
            .name = "quad_matmul",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    printf("DEBUG: About to add loop to matmul ufunc...\n");

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        printf("DEBUG: Failed to add loop to matmul ufunc\n");
        Py_DECREF(ufunc);
        return -1;
    }

    printf("DEBUG: Successfully added matmul loop!\n");

    // Add promoter following binary_ops pattern
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

    printf("DEBUG: init_matmul_ops completed successfully\n");
    return 0;
}