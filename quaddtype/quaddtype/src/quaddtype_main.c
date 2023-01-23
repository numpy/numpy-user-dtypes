#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL quaddtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "quaddtype_main",
        .m_doc = "Quad (128-bit) floating point experimental numpy dtype",
        .m_size = -1,
};

// Initialize the python module
PyMODINIT_FUNC
PyInit__quaddtype_main(void)
{
    if (_import_array() < 0)
        return NULL;

    // Fail to init if the experimental DType API version 5 isn't supported
    if (import_experimental_dtype_api(6) < 0) {
        PyErr_SetString(PyExc_ImportError,
                        "Error encountered importing the experimental dtype API.");
        return NULL;
    }

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        PyErr_SetString(PyExc_ImportError, "Unable to create the quaddtype_main module.");
        return NULL;
    }

    PyObject *mod = PyImport_ImportModule("quaddtype");
    if (mod == NULL) {
        PyErr_SetString(PyExc_ImportError, "Unable to import the quaddtype module.");
        goto error;
    }
    QuadScalar_Type = (PyTypeObject *)PyObject_GetAttrString(mod, "QuadScalar");
    Py_DECREF(mod);
    if (QuadScalar_Type == NULL) {
        PyErr_SetString(PyExc_AttributeError,
                        "Unable to find QuadScalar attribute in the "
                        "quaddtype_main module.");
        goto error;
    }
    if (init_quad_dtype() < 0) {
        PyErr_SetString(PyExc_AttributeError, "QuadDType failed to initialize.");
        goto error;
    }
    if (PyModule_AddObject(m, "QuadDType", (PyObject *)&QuadDType) < 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to add QuadDType to the quaddtype_main module.");
        goto error;
    }

    if (init_multiply_ufunc() < 0) {
        PyErr_SetString(PyExc_TypeError, "Failed to initialize the quadscalar multiply ufunc.");
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
