#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL quaddtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "dtype.h"
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "quaddtype_main",
    .m_doc = "Quad (128-bit) floating point experimental numpy dtype",
    .m_size = -1,
};

// Initialize the python module
PyMODINIT_FUNC PyInit__quaddtype_main(void) {
    if (_import_array() < 0) return NULL;

    // Fail to init if the experimental DType API version 5 isn't supported
    if (import_experimental_dtype_api(5) < 0) return NULL;

    PyObject* m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject* mod = PyImport_ImportModule("quaddtype");
    if (mod == NULL) goto error;
    QuadScalar_Type = (PyTypeObject*)PyObject_GetAttrString(mod, "QuadScalar");
    Py_DECREF(mod);
    if (QuadScalar_Type == NULL) goto error;
    if (init_quad_dtype() < 0) goto error;
    if (PyModule_AddObject(m, "quaddtype", (PyObject*)&QuadDType) < 0) goto error;

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
