#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL unitdtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    .m_name = "unitdtype_main",
    .m_size = -1,
};


/* Module initialization function */
PyMODINIT_FUNC PyInit__unitdtype_main(void)
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


    PyObject *mod = PyImport_ImportModule("unitdtype");
    if (mod == NULL) {
        goto error;
    }
    QuantityScalar_Type = PyObject_GetAttrString(mod, "QuantityScalar");
    Py_DECREF(mod);
    if (QuantityScalar_Type == NULL) {
        goto error;
    }

    if (init_unit_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m,
            "UnitDType", (PyObject *)&UnitDType) < 0) {
        goto error;
    }

    if (init_multiply_ufunc() < 0) {
        goto error;
    }

    return m;

  error:
    Py_DECREF(m);
    return NULL;
}
