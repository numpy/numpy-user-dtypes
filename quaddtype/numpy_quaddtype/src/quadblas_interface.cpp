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

// Include QuadBLAS header
#include "../QBLAS/include/quadblas/quadblas.hpp"

// Helper function to get QuadBLAS layout from numpy array
static QuadBLAS::Layout get_quadblas_layout(PyArrayObject *arr) {
    if (PyArray_IS_C_CONTIGUOUS(arr)) {
        return QuadBLAS::Layout::RowMajor;
    } else {
        return QuadBLAS::Layout::ColMajor;
    }
}

// Helper function to extract quad data and backend info from QuadPrecDType array
static bool extract_quad_array_info(PyArrayObject *arr, Sleef_quad **data, 
                                   QuadBackendType *backend, QuadBLAS::Layout *layout) {
    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError, "Expected numpy array");
        return false;
    }
    
    PyArray_Descr *descr = PyArray_DESCR(arr);
    if (!PyObject_TypeCheck(descr, (PyTypeObject*)&QuadPrecDType)) {
        PyErr_SetString(PyExc_TypeError, "Array must have QuadPrecDType dtype");
        return false;
    }
    
    QuadPrecDTypeObject *quad_descr = (QuadPrecDTypeObject*)descr;
    *backend = quad_descr->backend;
    *data = (Sleef_quad*)PyArray_DATA(arr);
    *layout = get_quadblas_layout(arr);
    
    return true;
}

// Helper function to convert between backends if needed
static Sleef_quad* ensure_sleef_backend(PyArrayObject *arr, QuadBackendType original_backend, 
                                       Sleef_quad **temp_storage) {
    if (original_backend == BACKEND_SLEEF) {
        *temp_storage = nullptr;
        return (Sleef_quad*)PyArray_DATA(arr);
    }
    
    // Need to convert from longdouble to sleef
    npy_intp size = PyArray_SIZE(arr);
    *temp_storage = QuadBLAS::aligned_alloc<Sleef_quad>(size);
    if (!*temp_storage) {
        PyErr_NoMemory();
        return nullptr;
    }
    
    long double *ld_data = (long double*)PyArray_DATA(arr);
    for (npy_intp i = 0; i < size; i++) {
        (*temp_storage)[i] = Sleef_cast_from_doubleq1((double)ld_data[i]);
    }
    
    return *temp_storage;
}

// Vector-Vector dot product
static PyObject* dot_vector_vector(PyArrayObject *a, PyArrayObject *b) {
    // Validate dimensions
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
    
    // Extract data and backend info
    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;
    
    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }
    
    // Convert to SLEEF backend if needed (QuadBLAS uses SLEEF internally)
    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);
    
    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }
    
    // Get strides in terms of elements (not bytes)
    npy_intp stride_a = PyArray_STRIDE(a, 0) / PyArray_ITEMSIZE(a);
    npy_intp stride_b = PyArray_STRIDE(b, 0) / PyArray_ITEMSIZE(b);
    
    // Perform dot product using QuadBLAS
    Sleef_quad result = QuadBLAS::dot(n_a, sleef_a, stride_a, sleef_b, stride_b);
    
    // Clean up temporary storage
    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);
    
    // Determine result backend (prefer SLEEF, fall back to common backend)
    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }
    
    // Create result scalar
    QuadPrecisionObject *result_obj = QuadPrecision_raw_new(result_backend);
    if (!result_obj) {
        return nullptr;
    }
    
    if (result_backend == BACKEND_SLEEF) {
        result_obj->value.sleef_value = result;
    } else {
        result_obj->value.longdouble_value = (long double)Sleef_cast_to_doubleq1(result);
    }
    
    return (PyObject*)result_obj;
}

// Matrix-Vector multiplication
static PyObject* dot_matrix_vector(PyArrayObject *a, PyArrayObject *b) {
    // Validate dimensions
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
    
    // Extract data and backend info
    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;
    
    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }
    
    // Convert to SLEEF backend if needed
    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);
    
    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }
    
    // Determine result backend
    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }
    
    // Create result array (1D with length m)
    npy_intp result_dims[1] = {m};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }
    
    PyArrayObject *result = (PyArrayObject*)PyArray_Empty(1, result_dims, 
                                                         (PyArray_Descr*)result_dtype, 0);
    if (!result) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }
    
    Sleef_quad *result_data = (Sleef_quad*)PyArray_DATA(result);
    
    // FIXED: Calculate leading dimensions and strides correctly
    npy_intp lda;
    if (layout_a == QuadBLAS::Layout::RowMajor) {
        lda = n;  // For row-major, leading dimension is number of columns
    } else {
        lda = m;  // For column-major, leading dimension is number of rows
    }
    
    npy_intp stride_b = PyArray_STRIDE(b, 0) / PyArray_ITEMSIZE(b);
    npy_intp stride_result = PyArray_STRIDE(result, 0) / PyArray_ITEMSIZE(result);
    
    // Perform matrix-vector multiplication using QuadBLAS
    // y = 1.0 * A * x + 0.0 * y
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);
    
    QuadBLAS::gemv(layout_a, m, n, alpha, sleef_a, lda, 
                   sleef_b, stride_b, beta, result_data, stride_result);
    
    // Convert result back to longdouble if needed
    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double*)PyArray_DATA(result);
        for (npy_intp i = 0; i < m; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }
    
    // Clean up temporary storage
    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);
    
    return (PyObject*)result;
}

// Matrix-Matrix multiplication
static PyObject* dot_matrix_matrix(PyArrayObject *a, PyArrayObject *b) {
    // Validate dimensions
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
    
    // Extract data and backend info
    Sleef_quad *data_a, *data_b;
    QuadBackendType backend_a, backend_b;
    QuadBLAS::Layout layout_a, layout_b;
    
    if (!extract_quad_array_info(a, &data_a, &backend_a, &layout_a) ||
        !extract_quad_array_info(b, &data_b, &backend_b, &layout_b)) {
        return nullptr;
    }
    
    // Convert to SLEEF backend if needed
    Sleef_quad *temp_a = nullptr, *temp_b = nullptr;
    Sleef_quad *sleef_a = ensure_sleef_backend(a, backend_a, &temp_a);
    Sleef_quad *sleef_b = ensure_sleef_backend(b, backend_b, &temp_b);
    
    if (!sleef_a || !sleef_b) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }
    
    // Determine result backend
    QuadBackendType result_backend = BACKEND_SLEEF;
    if (backend_a == BACKEND_LONGDOUBLE && backend_b == BACKEND_LONGDOUBLE) {
        result_backend = BACKEND_LONGDOUBLE;
    }
    
    // Create result array (2D with shape m x n)
    npy_intp result_dims[2] = {m, n};
    QuadPrecDTypeObject *result_dtype = new_quaddtype_instance(result_backend);
    if (!result_dtype) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        return nullptr;
    }
    
    PyArrayObject *result = (PyArrayObject*)PyArray_Empty(2, result_dims, 
                                                         (PyArray_Descr*)result_dtype, 0);
    if (!result) {
        QuadBLAS::aligned_free(temp_a);
        QuadBLAS::aligned_free(temp_b);
        Py_DECREF(result_dtype);
        return nullptr;
    }
    
    Sleef_quad *result_data = (Sleef_quad*)PyArray_DATA(result);
    
    // FIXED: Calculate leading dimensions correctly
    npy_intp lda, ldb, ldc;
    
    if (layout_a == QuadBLAS::Layout::RowMajor) {
        lda = k;  // For row-major A: leading dimension is number of columns
    } else {
        lda = m;  // For column-major A: leading dimension is number of rows
    }
    
    if (layout_b == QuadBLAS::Layout::RowMajor) {
        ldb = n;  // For row-major B: leading dimension is number of columns
    } else {
        ldb = k;  // For column-major B: leading dimension is number of rows
    }
    
    // Result array layout - assume same as input A
    QuadBLAS::Layout result_layout = layout_a;
    if (result_layout == QuadBLAS::Layout::RowMajor) {
        ldc = n;  // For row-major C: leading dimension is number of columns
    } else {
        ldc = m;  // For column-major C: leading dimension is number of rows
    }
    
    // Perform matrix-matrix multiplication using QuadBLAS
    // C = 1.0 * A * B + 0.0 * C
    Sleef_quad alpha = Sleef_cast_from_doubleq1(1.0);
    Sleef_quad beta = Sleef_cast_from_doubleq1(0.0);
    
    QuadBLAS::gemm(result_layout, m, n, k, alpha, sleef_a, lda, 
                   sleef_b, ldb, beta, result_data, ldc);
    
    // Convert result back to longdouble if needed
    if (result_backend == BACKEND_LONGDOUBLE) {
        long double *ld_result = (long double*)PyArray_DATA(result);
        for (npy_intp i = 0; i < m * n; i++) {
            ld_result[i] = (long double)Sleef_cast_to_doubleq1(result_data[i]);
        }
    }
    
    // Clean up temporary storage
    QuadBLAS::aligned_free(temp_a);
    QuadBLAS::aligned_free(temp_b);
    
    return (PyObject*)result;
}

// Main dot function that dispatches based on input dimensions
PyObject* py_quadblas_dot(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return nullptr;
    }
    
    // Convert to arrays if needed
    PyArrayObject *a = (PyArrayObject*)PyArray_FROM_OF(a_obj, NPY_ARRAY_ALIGNED);
    PyArrayObject *b = (PyArrayObject*)PyArray_FROM_OF(b_obj, NPY_ARRAY_ALIGNED);
    
    if (!a || !b) {
        Py_XDECREF(a);
        Py_XDECREF(b);
        PyErr_SetString(PyExc_TypeError, "Inputs must be convertible to arrays");
        return nullptr;
    }
    
    PyObject *result = nullptr;
    
    // Dispatch based on dimensions
    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);
    
    if (ndim_a == 1 && ndim_b == 1) {
        // Vector-Vector dot product
        result = dot_vector_vector(a, b);
    } else if (ndim_a == 2 && ndim_b == 1) {
        // Matrix-Vector multiplication
        result = dot_matrix_vector(a, b);
    } else if (ndim_a == 2 && ndim_b == 2) {
        // Matrix-Matrix multiplication
        result = dot_matrix_matrix(a, b);
    } else if (ndim_a == 1 && ndim_b == 2) {
        PyErr_SetString(PyExc_ValueError, 
            "Vector-Matrix multiplication not supported (use Matrix-Vector instead)");
    } else {
        PyErr_SetString(PyExc_ValueError, 
            "Unsupported array dimensions. Supported: (1D,1D), (2D,1D), (2D,2D)");
    }
    
    Py_DECREF(a);
    Py_DECREF(b);
    
    return result;
}

// Threading control functions
PyObject* py_quadblas_set_num_threads(PyObject* self, PyObject* args) {
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

PyObject* py_quadblas_get_num_threads(PyObject* self, PyObject* args) {
    return PyLong_FromLong(QuadBLAS::get_num_threads());
}

PyObject* py_quadblas_get_version(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(QuadBLAS::VERSION);
}