#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#include "umath.h"
#include "dtype.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "metadatadtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__metadatadtype_main(void)
{
    import_array();

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

    if (init_ufuncs(m) == NULL) {
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
