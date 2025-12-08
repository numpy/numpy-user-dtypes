#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_4_API_VERSION
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
                        "Please raise the issue at SwayamInSync/QBLAS for longdouble support");
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
                        "QBLAS-accelerated matmul only supports SLEEF backend. "
                        "Please raise the issue at SwayamInSync/QBLAS for longdouble support");
            return (NPY_CASTING)-1;
        }
        else {
            Py_INCREF(given_descrs[2]);
            loop_descrs[2] = given_descrs[2];
        }
    }
    return casting;
}

enum MatmulOperationType {
    MATMUL_DOT,
    MATMUL_GEMV,
    MATMUL_GEMM
};

static MatmulOperationType
determine_operation_type(npy_intp m, npy_intp n, npy_intp p)
{
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

static int
quad_matmul_strided_loop_aligned(PyArrayMethod_Context *context, char *const data[],
                                 npy_intp const dimensions[], npy_intp const strides[],
                                 NpyAuxData *auxdata)
{
    // Extract dimensions
    npy_intp N = dimensions[0];  // Batch size, this remains always 1 for matmul afaik
    npy_intp m = dimensions[1];  // Rows of first matrix
    npy_intp n = dimensions[2];  // Cols of first matrix / rows of second matrix
    npy_intp p = dimensions[3];  // Cols of second matrix

    // batch strides
    npy_intp A_stride = strides[0];
    npy_intp B_stride = strides[1];
    npy_intp C_stride = strides[2];

    // core strides for matrix dimensions
    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    if (descr->backend != BACKEND_SLEEF) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "QBLAS-accelerated matmul only supports SLEEF backend. "
                        "Please raise the issue at SwayamInSync/QBLAS for longdouble support");
        return -1;
    }

    MatmulOperationType op_type = determine_operation_type(m, n, p);
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    char *A = data[0];
    char *B = data[1];
    char *C = data[2];

    Sleef_quad *A_ptr = (Sleef_quad *)A;
    Sleef_quad *B_ptr = (Sleef_quad *)B;
    Sleef_quad *C_ptr = (Sleef_quad *)C;

    int result = -1;

    switch (op_type) {
        case MATMUL_DOT: {
            size_t incx = A_col_stride / sizeof(Sleef_quad);
            size_t incy = B_row_stride / sizeof(Sleef_quad);

            result = qblas_dot(n, A_ptr, incx, B_ptr, incy, C_ptr);
            break;
        }

        case MATMUL_GEMV: {
            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t incx = B_row_stride / sizeof(Sleef_quad);
            size_t incy = C_row_stride / sizeof(Sleef_quad);

            memset(C_ptr, 0, m * p * sizeof(Sleef_quad));

            result =
                    qblas_gemv('R', 'N', m, n, &alpha, A_ptr, lda, B_ptr, incx, &beta, C_ptr, incy);
            break;
        }

        case MATMUL_GEMM: {
            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t ldb = B_row_stride / sizeof(Sleef_quad);
            size_t ldc_numpy = C_row_stride / sizeof(Sleef_quad);

            memset(C_ptr, 0, m * p * sizeof(Sleef_quad));

            size_t ldc_temp = p;

            result = qblas_gemm('R', 'N', 'N', m, p, n, &alpha, A_ptr, lda, B_ptr, ldb, &beta,
                                C_ptr, ldc_numpy);
            break;
        }
    }

    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "QBLAS operation failed");
        return -1;
    }

    return 0;
}

static int
quad_matmul_strided_loop_unaligned(PyArrayMethod_Context *context, char *const data[],
                                   npy_intp const dimensions[], npy_intp const strides[],
                                   NpyAuxData *auxdata)
{
    // Extract dimensions
    npy_intp N = dimensions[0];  // Batch size, this remains always 1 for matmul afaik
    npy_intp m = dimensions[1];  // Rows of first matrix
    npy_intp n = dimensions[2];  // Cols of first matrix / rows of second matrix
    npy_intp p = dimensions[3];  // Cols of second matrix

    // batch strides
    npy_intp A_stride = strides[0];
    npy_intp B_stride = strides[1];
    npy_intp C_stride = strides[2];

    // core strides for matrix dimensions
    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    if (descr->backend != BACKEND_SLEEF) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "QBLAS-accelerated matmul only supports SLEEF backend. "
                        "Please raise the issue at SwayamInSync/QBLAS for longdouble support");
        return -1;
    }

    MatmulOperationType op_type = determine_operation_type(m, n, p);
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    char *A = data[0];
    char *B = data[1];
    char *C = data[2];

    Sleef_quad *A_ptr = (Sleef_quad *)A;
    Sleef_quad *B_ptr = (Sleef_quad *)B;
    Sleef_quad *C_ptr = (Sleef_quad *)C;

    int result = -1;

    switch (op_type) {
        case MATMUL_DOT: {
            Sleef_quad *temp_A_buffer = new Sleef_quad[n];
            Sleef_quad *temp_B_buffer = new Sleef_quad[n];

            memcpy(temp_A_buffer, A_ptr, n * sizeof(Sleef_quad));
            memcpy(temp_B_buffer, B_ptr, n * sizeof(Sleef_quad));

            size_t incx = 1;
            size_t incy = 1;

            result = qblas_dot(n, temp_A_buffer, incx, temp_B_buffer, incy, C_ptr);

            delete[] temp_A_buffer;
            delete[] temp_B_buffer;
            break;
        }

        case MATMUL_GEMV: {
            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t incx = B_row_stride / sizeof(Sleef_quad);
            size_t incy = C_row_stride / sizeof(Sleef_quad);

            Sleef_quad *temp_A_buffer = new Sleef_quad[m * n];
            Sleef_quad *temp_B_buffer = new Sleef_quad[n * p];
            memcpy(temp_A_buffer, A_ptr, m * n * sizeof(Sleef_quad));
            memcpy(temp_B_buffer, B_ptr, n * p * sizeof(Sleef_quad));
            A_ptr = temp_A_buffer;
            B_ptr = temp_B_buffer;

            // Use temp_C_buffer to avoid unaligned writes
            Sleef_quad *temp_C_buffer = new Sleef_quad[m * p];

            lda = n;
            incx = 1;
            incy = 1;

            memset(temp_C_buffer, 0, m * p * sizeof(Sleef_quad));

            result = qblas_gemv('R', 'N', m, n, &alpha, A_ptr, lda, B_ptr, incx, &beta,
                                temp_C_buffer, incy);
            break;
        }

        case MATMUL_GEMM: {
            size_t lda = A_row_stride / sizeof(Sleef_quad);
            size_t ldb = B_row_stride / sizeof(Sleef_quad);
            size_t ldc_numpy = C_row_stride / sizeof(Sleef_quad);

            Sleef_quad *temp_A_buffer = new Sleef_quad[m * n];
            Sleef_quad *temp_B_buffer = new Sleef_quad[n * p];
            memcpy(temp_A_buffer, A_ptr, m * n * sizeof(Sleef_quad));
            memcpy(temp_B_buffer, B_ptr, n * p * sizeof(Sleef_quad));
            A_ptr = temp_A_buffer;
            B_ptr = temp_B_buffer;

            // since these are now contiguous so,
            lda = n;
            ldb = p;
            size_t ldc_temp = p;

            Sleef_quad *temp_C_buffer = new Sleef_quad[m * p];
            memset(temp_C_buffer, 0, m * p * sizeof(Sleef_quad));

            result = qblas_gemm('R', 'N', 'N', m, p, n, &alpha, A_ptr, lda, B_ptr, ldb, &beta,
                                temp_C_buffer, ldc_temp);

            if (result == 0) {
                memcpy(C_ptr, temp_C_buffer, m * p * sizeof(Sleef_quad));
            }

            delete[] temp_C_buffer;
            delete[] temp_A_buffer;
            delete[] temp_B_buffer;
            break;
        }
    }

    if (result != 0) {
        PyErr_SetString(PyExc_RuntimeError, "QBLAS operation failed");
        return -1;
    }

    return 0;
}

static int
naive_matmul_strided_loop(PyArrayMethod_Context *context, char *const data[],
                          npy_intp const dimensions[], npy_intp const strides[],
                          NpyAuxData *auxdata)
{
    npy_intp N = dimensions[0];
    npy_intp m = dimensions[1];
    npy_intp n = dimensions[2];
    npy_intp p = dimensions[3];

    npy_intp A_batch_stride = strides[0];
    npy_intp B_stride = strides[1];
    npy_intp C_stride = strides[2];

    npy_intp A_row_stride = strides[3];
    npy_intp A_col_stride = strides[4];
    npy_intp B_row_stride = strides[5];
    npy_intp B_col_stride = strides[6];
    npy_intp C_row_stride = strides[7];
    npy_intp C_col_stride = strides[8];

    QuadPrecDTypeObject *descr = (QuadPrecDTypeObject *)context->descriptors[0];
    QuadBackendType backend = descr->backend;
    size_t elem_size = (backend == BACKEND_SLEEF) ? sizeof(Sleef_quad) : sizeof(long double);

    char *A = data[0];
    char *B = data[1];
    char *C = data[2];

    for (npy_intp i = 0; i < m; i++) {
        for (npy_intp j = 0; j < p; j++) {
            char *C_ij = C + i * C_row_stride + j * C_col_stride;

            if (backend == BACKEND_SLEEF) {
                Sleef_quad sum = Sleef_cast_from_doubleq1(0.0);

                for (npy_intp k = 0; k < n; k++) {
                    char *A_ik = A + i * A_row_stride + k * A_col_stride;
                    char *B_kj = B + k * B_row_stride + j * B_col_stride;

                    Sleef_quad a_val, b_val;
                    memcpy(&a_val, A_ik, sizeof(Sleef_quad));
                    memcpy(&b_val, B_kj, sizeof(Sleef_quad));
                    sum = Sleef_fmaq1_u05(a_val, b_val, sum);
                }

                memcpy(C_ij, &sum, sizeof(Sleef_quad));
            }
            else {
                long double sum = 0.0L;

                for (npy_intp k = 0; k < n; k++) {
                    char *A_ik = A + i * A_row_stride + k * A_col_stride;
                    char *B_kj = B + k * B_row_stride + j * B_col_stride;

                    long double a_val, b_val;
                    memcpy(&a_val, A_ik, sizeof(long double));
                    memcpy(&b_val, B_kj, sizeof(long double));

                    sum += a_val * b_val;
                }

                memcpy(C_ij, &sum, sizeof(long double));
            }
        }
    }

    return 0;
}

int
init_matmul_ops(PyObject *numpy)
{
    PyObject *ufunc = PyObject_GetAttrString(numpy, "matmul");
    if (ufunc == NULL) {
        return -1;
    }

    PyArray_DTypeMeta *dtypes[3] = {&QuadPrecDType, &QuadPrecDType, &QuadPrecDType};

#ifndef DISABLE_QUADBLAS

    PyType_Slot slots[] = {
            {NPY_METH_resolve_descriptors, (void *)&quad_matmul_resolve_descriptors},
            {NPY_METH_strided_loop, (void *)&quad_matmul_strided_loop_aligned},
            {NPY_METH_unaligned_strided_loop, (void *)&quad_matmul_strided_loop_unaligned},
            {0, NULL}};
#else
    PyType_Slot slots[] = {{NPY_METH_resolve_descriptors, (void *)&quad_matmul_resolve_descriptors},
                           {NPY_METH_strided_loop, (void *)&naive_matmul_strided_loop},
                           {NPY_METH_unaligned_strided_loop, (void *)&naive_matmul_strided_loop},
                           {0, NULL}};
#endif  // DISABLE_QUADBLAS

    PyArrayMethod_Spec Spec = {
            .name = "quad_matmul_qblas",
            .nin = 2,
            .nout = 1,
            .casting = NPY_NO_CASTING,
            .flags = NPY_METH_SUPPORTS_UNALIGNED,
            .dtypes = dtypes,
            .slots = slots,
    };

    if (PyUFunc_AddLoopFromSpec(ufunc, &Spec) < 0) {
        Py_DECREF(ufunc);
        return -1;
    }

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
        PyErr_Clear();
    }
    else {
    }

    Py_DECREF(DTypes);
    Py_DECREF(promoter_capsule);
    Py_DECREF(ufunc);

    return 0;
}