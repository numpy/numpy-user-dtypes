#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL QuadPrecType_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL QuadPrecType_UFUNC_API
#define NPY_NO_DEPRECATED_API NPY_2_0_API_VERSION
#define NPY_TARGET_VERSION NPY_2_0_API_VERSION

#include "numpy/arrayobject.h"
#include "numpy/dtype_api.h"
#include "numpy/ufuncobject.h"

#include "dtype.h"
#include "umath.h"

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        .m_name = "_quaddtype_main",
        .m_doc = "Quad (128-bit) floating point Data Type for Numpy",
        .m_size = -1,
};

PyMODINIT_FUNC
PyInit__quaddtype_main(void)
{
    import_array();
    import_umath();
    PyObject *m = PyModule_Create(&moduledef);
    if (!m) 
    {
        return NULL;
    }

    if (init_quadprecision_scalar() < 0)
        goto error;
    
    if(PyModule_AddObject(m, "QuadPrecision", (PyObject *)&QuadPrecision_Type) < 0)
        goto error;

    if(init_quadprec_dtype() < 0)
        goto error;

    if(PyModule_AddObject(m, "QuadPrecDType", (PyObject *)&QuadPrecDType) < 0)
        goto error;

    if (init_quad_umath() < 0) {
        goto error;
    }

    return m;
    

error:
    Py_DECREF(m);
    return NULL;
}