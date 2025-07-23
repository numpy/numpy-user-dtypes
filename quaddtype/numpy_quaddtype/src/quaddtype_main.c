#include <Python.h>
#include <sleef.h>
#include <sleefquad.h>
#include <string.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "numpy/ufuncobject.h"

#include "scalar.h"
#include "dtype.h"
#include "umath/umath.h"
#include "quad_common.h"
#include "quadblas_interface.h"
#include "float.h"

static PyObject *
py_is_longdouble_128(PyObject *self, PyObject *args)
{
    if (sizeof(long double) == 16 && LDBL_MANT_DIG == 113 && LDBL_MAX_EXP == 16384) {
        Py_RETURN_TRUE;
    }
    else {
        Py_RETURN_FALSE;
    }
}

static PyObject *
get_sleef_constant(PyObject *self, PyObject *args)
{
    const char *constant_name;
    if (!PyArg_ParseTuple(args, "s", &constant_name)) {
        return NULL;
    }

    QuadPrecisionObject *result = QuadPrecision_raw_new(BACKEND_SLEEF);
    if (result == NULL) {
        return NULL;
    }

    if (strcmp(constant_name, "pi") == 0) {
        result->value.sleef_value = SLEEF_M_PIq;
    }
    else if (strcmp(constant_name, "e") == 0) {
        result->value.sleef_value = SLEEF_M_Eq;
    }
    else if (strcmp(constant_name, "log2e") == 0) {
        result->value.sleef_value = SLEEF_M_LOG2Eq;
    }
    else if (strcmp(constant_name, "log10e") == 0) {
        result->value.sleef_value = SLEEF_M_LOG10Eq;
    }
    else if (strcmp(constant_name, "ln2") == 0) {
        result->value.sleef_value = SLEEF_M_LN2q;
    }
    else if (strcmp(constant_name, "ln10") == 0) {
        result->value.sleef_value = SLEEF_M_LN10q;
    }
    else if (strcmp(constant_name, "max_value") == 0) {
        result->value.sleef_value = SLEEF_QUAD_MAX;
    }
    else if (strcmp(constant_name, "epsilon") == 0) {
        result->value.sleef_value = SLEEF_QUAD_EPSILON;
    }
    else if (strcmp(constant_name, "smallest_normal") == 0) {
        result->value.sleef_value = SLEEF_QUAD_MIN;
    }
    else if (strcmp(constant_name, "smallest_subnormal") == 0) {
        result->value.sleef_value = SLEEF_QUAD_DENORM_MIN;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "Unknown constant name");
        Py_DECREF(result);
        return NULL;
    }

    return (PyObject *)result;
}

static PyMethodDef module_methods[] = {
        {"is_longdouble_128", py_is_longdouble_128, METH_NOARGS, "Check if long double is 128-bit"},
        {"get_sleef_constant", get_sleef_constant, METH_VARARGS, "Get Sleef constant by name"},
        {"set_num_threads", py_quadblas_set_num_threads, METH_VARARGS,
         "Set number of threads for QuadBLAS"},
        {"get_num_threads", py_quadblas_get_num_threads, METH_NOARGS,
         "Get number of threads for QuadBLAS"},
        {"get_quadblas_version", py_quadblas_get_version, METH_NOARGS, "Get QuadBLAS version"},
        {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, .m_name = "_quaddtype_main",
        .m_doc = "Quad (128-bit) floating point Data Type for NumPy with multiple backends",
        .m_size = -1, .m_methods = module_methods};

PyMODINIT_FUNC
PyInit__quaddtype_main(void)
{
    import_array();
    import_umath();
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    if (init_quadprecision_scalar() < 0)
        goto error;

    if (PyModule_AddObject(m, "QuadPrecision", (PyObject *)&QuadPrecision_Type) < 0)
        goto error;

    if (init_quadprec_dtype() < 0)
        goto error;

    if (PyModule_AddObject(m, "QuadPrecDType", (PyObject *)&QuadPrecDType) < 0)
        goto error;

    if (init_quad_umath() < 0) {
        goto error;
    }

    return m;

error:
    Py_XDECREF(m);
    return NULL;
}