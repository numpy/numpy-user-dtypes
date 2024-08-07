#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL MPFDType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL MPFDType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/dtype_api.h"

#include "dtype.h"
#include "umath.h"
#include "terrible_hacks.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "mpfdtype_main",
        .m_size = -1,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__mpfdtype_main(void)
{
    import_array();
    import_umath();

    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL) {
        return NULL;
    }

    if (init_mpf_scalar() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "MPFloat", (PyObject *)&MPFloat_Type) < 0) {
        goto error;
    }

    if (init_mpf_dtype() < 0) {
        goto error;
    }

    if (PyModule_AddObject(m, "MPFDType", (PyObject *)&MPFDType) < 0) {
        goto error;
    }

    if (init_mpf_umath() == -1) {
        goto error;
    }

    if (init_terrible_hacks() < 0) {
        goto error;
    }

    return m;

error:
    Py_DECREF(m);
    return NULL;
}
