#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unytdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"

#include "dtype.h"
#include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "unytdtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__unytdtype_main(void)
{
    import_array();

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    PyObject *mod = PyImport_ImportModule("unytdtype");
    if (mod == NULL) {
        goto error;
    }
    UnytScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "UnytScalar");
    Py_DECREF(mod);

    if (UnytScalar_Type == NULL) {
        goto error;
    }

    if (init_unyt_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "UnytDType", (PyObject *)&UnytDType) < 0) {
        goto error;
    }

    PyObject *numpy = init_multiply_ufunc();

    if (numpy == NULL) {
        goto error;
    }

    Py_DECREF(numpy);

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
