#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "metadatadtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__metadatadtype_main(void)
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

    PyObject *mod = PyImport_ImportModule("metadatadtype");
    if (mod == NULL) {
        goto error;
    }
    MetadataScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "MetadataScalar");
    Py_DECREF(mod);

    if (MetadataScalar_Type == NULL) {
        goto error;
    }

    if (init_metadata_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "MetadataDType", (PyObject *)&MetadataDType) <
        0) {
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
