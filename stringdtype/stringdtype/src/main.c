#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL stringdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
// #include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "stringdtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__main(void)
{
    if (_import_array() < 0) {
        return NULL;
    }
    if (import_experimental_dtype_api(5) < 0) {
        return NULL;
    }

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    PyObject *mod = PyImport_ImportModule("stringdtype");
    if (mod == NULL) {
        goto error;
    }
    StringScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "StringScalar");
    Py_DECREF(mod);

    if (StringScalar_Type == NULL) {
        goto error;
    }

    if (init_string_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "StringDType", (PyObject *)&StringDType) < 0) {
        goto error;
    }

    // if (init_ufuncs() < 0) {
    //     goto error;
    // }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
