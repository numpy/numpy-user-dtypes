#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "numpy/ufuncobject.h"

#include "scalar.h"
#include "dtype.h"
#include "umath.h"
#include "quad_common.h"
#include "float.h"

static PyObject* py_is_longdouble_128(PyObject* self, PyObject* args) {
    if(sizeof(long double) == 16 && 
        LDBL_MANT_DIG == 113 && 
        LDBL_MAX_EXP == 16384) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyMethodDef module_methods[] = {
    {"is_longdouble_128", py_is_longdouble_128, METH_NOARGS, "Check if long double is 128-bit"},
    {NULL, NULL, 0, NULL} 
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "_quaddtype_main",
        .m_doc = "Quad (128-bit) floating point Data Type for NumPy with multiple backends",
        .m_size = -1,
        .m_methods = module_methods
};

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