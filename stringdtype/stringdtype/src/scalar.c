#define PY_ARRAY_UNIQUE_SYMBOL STRINGDTYPE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#include "numpy/arrayobject.h"
#include "numpy/experimental_dtype_api.h"
#include "numpy/ndarraytypes.h"

#include "scalar.h"

PyTypeObject StringScalar_Type = {
        PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringdtype.StringScalar",
        .tp_basicsize = sizeof(StringScalarObject),
};

int
init_stringdtype_scalar(void)
{
    StringScalar_Type.tp_base = &PyUnicode_Type;
    if (PyType_Ready(&StringScalar_Type) < 0) {
        return -1;
    }

    return 0;
}
