#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC

extern "C" {
#include <Python.h>
#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"
#include "numpy/dtype_api.h"
}

#include "scalar.h"
#include "dtype.h"
#include "quad_common.h"
#include "quadblas_interface.h"

extern "C" {
#include <sleef.h>
#include <sleefquad.h>
}

#ifndef DISABLE_QUADBLAS
#include "../QBLAS/include/quadblas/quadblas.hpp"
#endif

#ifdef DISABLE_QUADBLAS

static bool
extract_quad_array_info_simple(PyArrayObject *arr, Sleef_quad **data, QuadBackendType *backend)
{
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected numpy array");
        return false;
    }

    PyArray_Descr *descr = PyArray_DESCR(arr);
    if (!PyObject_TypeCheck(descr, (PyTypeObject *)&QuadPrecDType)) {
        PyErr_SetString(PyExc_TypeError, "Array must have QuadPrecDType dtype");
        return false;
    }

    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject *)descr;
    *backend = quad_descr->backend;
    *data = (Sleef_quad *)PyArray_DATA(arr);

    return true;
}

static Sleef_quad *
ensure_sleef_backend_simple(PyArrayObject *arr, QuadBackendType original_backend,
                            Sleef_quad **temp_storage)
{
    if (original_backend == BACKEND_SLEEF) {
        *temp_storage = nullptr;
        return (Sleef_quad *)PyArray_DATA(arr);
    }

    npy_intp size = PyArray_SIZE(arr);
    *temp_storage = (Sleef_quad *)malloc(size * sizeof(Sleef_quad));
    if (!*temp_storage) {
        PyErr_NoMemory();
        return nullptr;
    }

    long double *ld_data = (long double *)PyArray_DATA(arr);
    for (npy_intp i = 0; i < size; i++) {
        (*temp_storage)[i] = Sleef_cast_from_doubleq1((double)ld_data[i]);
    }

    return *temp_storage;
}

// ===============================================================================
// FALLBACK IMPLEMENTATIONS (No QuadBLAS)
// ===============================================================================

static PyObject *
dot_vector_vector_fallback(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
        PyErr_SetString(PyExc_ValueError, "Both inputs must be 1-dimensional arrays");
        return nullptr;
    }

    npy_intp n_a = PyArray_DIM(a, 0);
    npy_intp n_b = PyArray_DIM(b, 0);

    if (n_a != n_b) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same length");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;

    if (!extract_quad_array_info_simple(a, &data_a, &backend_a) ||
        !extract_quad_array_info_simple(b, &data_b, &backend_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend_simple(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend_simple(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        free(temp_a);
        free(temp_b);
        return nullptr;
    }

    // Simple dot product implementation
    Sleef_quad result = Sleef_cast_from_doubleq1(0.0);
    for (npy_intp i = 0; i < n_a; i++) {
        result = Sleef_fmaq1_u05(sleef_a[i], sleef_b[i], result);
    }

    free(temp_a);
    free(temp_b);

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    QuadPrecisionObject *result_obj = QuadPrecision_raw_new(result_backend);
    if (!result_obj) {
        return nullptr;
    }

    if (result_backend == BACKEND_SLEEF) {
        result_obj->value.sleef_value = result;
    }
    else {
        result_obj->value.longdouble_value = (long double)Sleef_cast_to_doubleq1(result);
    }

    return (PyObject *)result_obj;
}

static PyObject *
dot_matrix_vector_fallback(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 1) {
        PyErr_SetString(PyExc_ValueError, "First input must be 2D, second input must be 1D");
        return nullptr;
    }

    npy_intp m = PyArray_DIM(a, 0);
    npy_intp n = PyArray_DIM(a, 1);
    npy_intp n_b = PyArray_DIM(b, 0);

    if (n != n_b) {
        PyErr_SetString(PyExc_ValueError, "Matrix columns must match vector length");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;

    if (!extract_quad_array_info_simple(a, &data_a, &backend_a) ||
        !extract_quad_array_info_simple(b, &data_b, &backend_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend_simple(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend_simple(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        free(temp_a);
        free(temp_b);
        return nullptr;
    }

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    npy_intp result_dims[1] = {m};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        free(temp_a);
        free(temp_b);
        return nullptr;
    }

    PyArrayObject *result =
            (PyArrayObject *)PyArray_Empty(1, result_dims, (PyArray_Descr *)result_dtype, 0);
    if (!result) {
        free(temp_a);
        free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }

    Sleef_quad *result_data = (Sleef_quad *)PyArray_DATA(result);

    // Initialize result to zero
    for (npy_intp i = 0; i < m; i++) {
        result_data[i] = Sleef_cast_from_doubleq1(0.0);
    }

    // Simple matrix-vector multiplication: result[i] = sum(A[i,j] * b[j])
    for (npy_intp i = 0; i < m; i++) {
        Sleef_quad sum = Sleef_cast_from_doubleq1(0.0);
        for (npy_intp j = 0; j < n; j++) {
            // Assume row-major layout: A[i,j] = sleef_a[i*n + j]
            sum = Sleef_fmaq1_u05(sleef_a[i * n + j], sleef_b[j], sum);
        }
        result_data[i] = sum;
    }

    // Convert to longdouble if needed
    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double *)PyArray_DATA(result);
        for (npy_intp i = 0; i < m; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }

    free(temp_a);
    free(temp_b);

    return (PyObject *)result;
}

static PyObject *
dot_matrix_matrix_fallback(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 2) {
        PyErr_SetString(PyExc_ValueError, "Both inputs must be 2-dimensional arrays");
        return nullptr;
    }

    npy_intp m = PyArray_DIM(a, 0);
    npy_intp k = PyArray_DIM(a, 1);
    npy_intp k_b = PyArray_DIM(b, 0);
    npy_intp n = PyArray_DIM(b, 1);

    if (k != k_b) {
        PyErr_SetString(PyExc_ValueError, "Matrix inner dimensions must match");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;

    if (!extract_quad_array_info_simple(a, &data_a, &backend_a) ||
        !extract_quad_array_info_simple(b, &data_b, &backend_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend_simple(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend_simple(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        free(temp_a);
        free(temp_b);
        return nullptr;
    }

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    npy_intp result_dims[2] = {m, n};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        free(temp_a);
        free(temp_b);
        return nullptr;
    }

    PyArrayObject *result =
            (PyArrayObject *)PyArray_Empty(2, result_dims, (PyArray_Descr *)result_dtype, 0);
    if (!result) {
        free(temp_a);
        free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }

    Sleef_quad *result_data = (Sleef_quad *)PyArray_DATA(result);

    // Initialize result matrix to zero
    for (npy_intp i = 0; i < m * n; i++) {
        result_data[i] = Sleef_cast_from_doubleq1(0.0);
    }

    // Simple matrix-matrix multiplication: C[i,j] = sum(A[i,l] * B[l,j])
    for (npy_intp i = 0; i < m; i++) {
        for (npy_intp j = 0; j < n; j++) {
            Sleef_quad sum = Sleef_cast_from_doubleq1(0.0);
            for (npy_intp l = 0; l < k; l++) {
                // Row-major: A[i,l] = sleef_a[i*k + l], B[l,j] = sleef_b[l*n + j]
                sum = Sleef_fmaq1_u05(sleef_a[i * k + l], sleef_b[l * n + j], sum);
            }
            result_data[i * n + j] = sum;
        }
    }

    // Convert to longdouble if needed
    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double *)PyArray_DATA(result);
        for (npy_intp i = 0; i < m * n; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }

    free(temp_a);
    free(temp_b);

    return (PyObject *)result;
}

PyObject *
py_quadblas_dot(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return nullptr;
    }

    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OF(a_obj, NPY_ARRAY_ALIGNED);
    PyArrayObject *b = (PyArrayObject *)PyArray_FROM_OF(b_obj, NPY_ARRAY_ALIGNED);

    if (!a || !b) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Inputs must be convertible to arrays");
        return nullptr;
    }

    PyObject *result = nullptr;

    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);

    if (ndim_a == 1 && ndim_b == 1) {
        result = dot_vector_vector_fallback(a, b);
    }
    else if (ndim_a == 2 && ndim_b == 1) {
        result = dot_matrix_vector_fallback(a, b);
    }
    else if (ndim_a == 2 && ndim_b == 2) {
        result = dot_matrix_matrix_fallback(a, b);
    }
    else if (ndim_a == 1 && ndim_b == 2) {
        PyErr_SetString(PyExc_ValueError,
                        "Vector-Matrix multiplication not supported (use Matrix-Vector instead)");
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "Unsupported array dimensions. Supported: (1D,1D), (2D,1D), (2D,2D)");
    }

    Py_DECREF(a);
    Py_DECREF(b);

    return result;
}

// Dummy implementations for other QuadBLAS functions
PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    // On Windows fallback, just ignore thread setting
    Py_RETURN_NONE;
}

PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    // Return 1 for fallback implementation
    return PyLong_FromLong(1);
}

PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString("QuadBLAS is disabled for MSVC");
}

#else

static QuadBLAS::Layout
get_quadblas_layout(PyArrayObject *arr)
{
    if (PyArray_IS_C_CONTIGUOUS(arr)) {
        return QuadBLAS::Layout::RowMajor;
    }
    else {
        return QuadBLAS::Layout::ColMajor;
    }
}

static bool
extract_quad_array_info(PyArrayObject *arr, Sleef_quad **data, QuadBackendType *backend,
                        QuadBLAS::Layout *layout)
{
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected numpy array");
        return false;
    }

    PyArray_Descr *descr = PyArray_DESCR(arr);
    if (!PyObject_TypeCheck(descr, (PyTypeObject *)&QuadPrecDType)) {
        PyErr_SetString(PyExc_TypeError, "Array must have QuadPrecDType dtype");
        return false;
    }

    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject *)descr;
    *backend = quad_descr->backend;
    *data = (Sleef_quad *)PyArray_DATA(arr);
    *layout = get_quadblas_layout(arr);

    return true;
}

static Sleef_quad *
ensure_sleef_backend(PyArrayObject *arr, QuadBackendType original_backend,
                     Sleef_quad **temp_storage)
{
    if (original_backend == BACKEND_SLEEF) {
        *temp_storage = nullptr;
        return (Sleef_quad *)PyArray_DATA(arr);
    }

    npy_intp size = PyArray_SIZE(arr);
    *temp_storage = QuadBLAS::aligned_alloc<Sleef_quad>(size);
    if (!*temp_storage) {
        PyErr_NoMemory();
        return nullptr;
    }

    long double *ld_data = (long double *)PyArray_DATA(arr);
    for (npy_intp i = 0; i < size; i++) {
        (*temp_storage)[i] = Sleef_cast_from_doubleq1((double)ld_data[i]);
    }

    return *temp_storage;
}

static PyObject *
dot_vector_vector(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 1 || PyArray_NDIM(b) != 1) {
        PyErr_SetString(PyExc_ValueError, "Both inputs must be 1-dimensional arrays");
        return nullptr;
    }

    npy_intp n_a = PyArray_DIM(a, 0);
    npy_intp n_b = PyArray_DIM(b, 0);

    if (n_a != n_b) {
        PyErr_SetString(PyExc_ValueError, "Arrays must have the same length");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;

    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }

    npy_intp stride_a = PyArray_STRIDE(a, 0) / PyArray_ITEMSIZE(a);
    npy_intp stride_b = PyArray_STRIDE(b, 0) / PyArray_ITEMSIZE(b);

    Sleef_quad result = QuadBLAS::dot(n_a, sleef_a, stride_a, sleef_b, stride_b);

    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    QuadPrecisionObject *result_obj = QuadPrecision_raw_new(result_backend);
    if (!result_obj) {
        return nullptr;
    }

    if (result_backend == BACKEND_SLEEF) {
        result_obj->value.sleef_value = result;
    }
    else {
        result_obj->value.longdouble_value = (long double)Sleef_cast_to_doubleq1(result);
    }

    return (PyObject *)result_obj;
}

static PyObject *
dot_matrix_vector(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 1) {
        PyErr_SetString(PyExc_ValueError, "First input must be 2D, second input must be 1D");
        return nullptr;
    }

    npy_intp m = PyArray_DIM(a, 0);
    npy_intp n = PyArray_DIM(a, 1);
    npy_intp n_b = PyArray_DIM(b, 0);

    if (n != n_b) {
        PyErr_SetString(PyExc_ValueError, "Matrix columns must match vector length");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;

    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    npy_intp result_dims[1] = {m};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }

    PyArrayObject *result =
            (PyArrayObject *)PyArray_Empty(1, result_dims, (PyArray_Descr *)result_dtype, 0);
    if (!result) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }

    Sleef_quad *result_data = (Sleef_quad *)PyArray_DATA(result);

    npy_intp lda;
    if (layout_a == QuadBLAS::Layout::RowMajor) {
        lda = n;
    }
    else {
        lda = m;
    }

    npy_intp stride_b = PyArray_STRIDE(b, 0) / PyArray_ITEMSIZE(b);
    npy_intp stride_result = PyArray_STRIDE(result, 0) / PyArray_ITEMSIZE(result);

    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    QuadBLAS::gemv(layout_a, m, n, alpha, sleef_a, lda, sleef_b, stride_b, beta, result_data,
                   stride_result);

    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double *)PyArray_DATA(result);
        for (npy_intp i = 0; i < m; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }

    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);

    return (PyObject *)result;
}

static PyObject *
dot_matrix_matrix(PyArrayObject *a, PyArrayObject *b)
{
    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(b) != 2) {
        PyErr_SetString(PyExc_ValueError, "Both inputs must be 2-dimensional arrays");
        return nullptr;
    }

    npy_intp m = PyArray_DIM(a, 0);
    npy_intp k = PyArray_DIM(a, 1);
    npy_intp k_b = PyArray_DIM(b, 0);
    npy_intp n = PyArray_DIM(b, 1);

    if (k != k_b) {
        PyErr_SetString(PyExc_ValueError, "Matrix inner dimensions must match");
        return nullptr;
    }

    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;

    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }

    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);

    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }

    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }

    npy_intp result_dims[2] = {m, n};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }

    PyArrayObject *result =
            (PyArrayObject *)PyArray_Empty(2, result_dims, (PyArray_Descr *)result_dtype, 0);
    if (!result) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }

    Sleef_quad *result_data = (Sleef_quad *)PyArray_DATA(result);
    for (npy_intp i = 0; i < m * n; i++) {
        result_data[i] = Sleef_cast_from_doubleq1(0.0);
    }

    npy_intp lda, ldb, ldc;

    if (layout_a == QuadBLAS::Layout::RowMajor) {
        lda = k;
    }
    else {
        lda = m;
    }

    if (layout_b == QuadBLAS::Layout::RowMajor) {
        ldb = n;
    }
    else {
        ldb = k;
    }

    QuadBLAS::Layout result_layout = layout_a;
    if (result_layout == QuadBLAS::Layout::RowMajor) {
        ldc = n;
    }
    else {
        ldc = m;
    }

    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);

    QuadBLAS::gemm(result_layout, m, n, k, alpha, sleef_a, lda, sleef_b, ldb, beta, result_data,
                   ldc);

    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double *)PyArray_DATA(result);
        for (npy_intp i = 0; i < m * n; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }

    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);

    return (PyObject *)result;
}

PyObject *
py_quadblas_dot(PyObject *self, PyObject *args)
{
    PyObject *a_obj, *b_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return nullptr;
    }

    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OF(a_obj, NPY_ARRAY_ALIGNED);
    PyArrayObject *b = (PyArrayObject *)PyArray_FROM_OF(b_obj, NPY_ARRAY_ALIGNED);

    if (!a || !b) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Inputs must be convertible to arrays");
        return nullptr;
    }

    PyObject *result = nullptr;

    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);

    if (ndim_a == 1 && ndim_b == 1) {
        result = dot_vector_vector(a, b);
    }
    else if (ndim_a == 2 && ndim_b == 1) {
        result = dot_matrix_vector(a, b);
    }
    else if (ndim_a == 2 && ndim_b == 2) {
        result = dot_matrix_matrix(a, b);
    }
    else if (ndim_a == 1 && ndim_b == 2) {
        PyErr_SetString(PyExc_ValueError,
                        "Vector-Matrix multiplication not supported (use Matrix-Vector instead)");
    }
    else {
        PyErr_SetString(PyExc_ValueError,
                        "Unsupported array dimensions. Supported: (1D,1D), (2D,1D), (2D,2D)");
    }

    Py_DECREF(a);
    Py_DECREF(b);

    return result;
}

PyObject *
py_quadblas_set_num_threads(PyObject *self, PyObject *args)
{
    int num_threads;

    if (!PyArg_ParseTuple(args, "i", &num_threads)) {
        return nullptr;
    }

    if (num_threads < 1) {
        PyErr_SetString(PyExc_ValueError, "Number of threads must be positive");
        return nullptr;
    }

    QuadBLAS::set_num_threads(num_threads);
    Py_RETURN_NONE;
}

PyObject *
py_quadblas_get_num_threads(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(QuadBLAS::get_num_threads());
}

PyObject *
py_quadblas_get_version(PyObject *self, PyObject *args)
{
    return PyUnicode_FromString(QuadBLAS::VERSION);
}

#endif  // DISABLE_QUADBLAS