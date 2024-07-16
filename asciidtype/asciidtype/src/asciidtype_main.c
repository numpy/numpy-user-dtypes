#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL asciidtype_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL asciidtype_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION
#include "numpy/ndarraytypes.h"
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

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
    import_array();
    import_umath();

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

    if (init_ufuncs() == -1) {
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
