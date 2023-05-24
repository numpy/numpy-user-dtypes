#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL metadatadtype_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"

#include "dtype.h"
#include "umath.h"

static int
metadatadtypemodule_modexec(PyObject *m)
{
    metadatadtype_state *state = PyModule_GetState(m);

    if (_import_array() < 0) {
        return -1;
    }
    if (import_experimental_dtype_api(11) < 0) {
        return -1;
    }

    PyObject *mod = PyImport_ImportModule("metadatadtype");
    if (mod == NULL) {
        goto error;
    }
    state->MetadataScalar_Type =
            (PyTypeObject *)PyObject_GetAttrString(mod, "MetadataScalar");
    Py_DECREF(mod);

    if (state->MetadataScalar_Type == NULL) {
        goto error;
    }

    if (init_metadata_dtype(m) < 0) {
        goto error;
    }

    if (PyModule_AddType(m, (PyTypeObject *)state->MetadataDType) < 0) {
        goto error;
    }

    if (init_ufuncs(state->MetadataDType) < 0) {
        goto error;
    }

    return 0;

error:
    Py_DECREF(m);
    return -1;
}

static int
metadatadtypemodule_traverse(PyObject *module, visitproc visit, void *arg)
{
    metadatadtype_state *state = PyModule_GetState(module);
    Py_VISIT(state->MetadataDType);
    Py_VISIT(state->MetadataScalar_Type);
    return 0;
}

static int
metadatadtypemodule_clear(PyObject *module)
{
    metadatadtype_state *state = PyModule_GetState(module);
    Py_CLEAR(state->MetadataDType);
    Py_CLEAR(state->MetadataScalar_Type);
    return 0;
}

static PyModuleDef_Slot metadatadtypemodule_slots[] = {
        {Py_mod_exec, metadatadtypemodule_modexec},
        {0, NULL},
};

struct PyModuleDef metadatadtype_module = {
        PyModuleDef_HEAD_INIT,
        .m_name = "_metadatadtype_main",
        .m_size = sizeof(metadatadtype_state),
        .m_slots = metadatadtypemodule_slots,
        .m_traverse = metadatadtypemodule_traverse,
        .m_clear = metadatadtypemodule_clear,
};

/* Module initialization function */
PyMODINIT_FUNC
PyInit__metadatadtype_main(void)
{
    return PyModuleDef_Init(&metadatadtype_module);
}
