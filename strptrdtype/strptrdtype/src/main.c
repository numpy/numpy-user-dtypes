#include <Python.h>

#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
// #include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "strptrdtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__strptrdtype_main(void)
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

    PyObject *mod = PyImport_ImportModule("strptrdtype");
    if (mod == NULL) {
        goto error;
    }
    StrPtrScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "StrPtrScalar");
    Py_DECREF(mod);

    if (StrPtrScalar_Type == NULL) {
        goto error;
    }

    if (init_strptr_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "StrPtrDType", (PyObject *)&StrPtrDType) < 0) {
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
