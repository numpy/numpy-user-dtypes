#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL asciidtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "asciidtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__asciidtype_main(void)
{
    if (_import_array() < 0) {
        return NULL;
    }

    if (import_experimental_dtype_api(15) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    PyObject *mod = PyImport_ImportModule("asciidtype");
    if (mod == NULL) {
        goto error;
    }
    ASCIIScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "ASCIIScalar");
    Py_DECREF(mod);

    if (ASCIIScalar_Type == NULL) {
        goto error;
    }

    if (init_ascii_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "ASCIIDType", (PyObject *)&ASCIIDType) < 0) {
        goto error;
    }

    if (init_ufuncs() < 0) {
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
