#ifndef _QUADDTYPE_MATMUL_H
#define _QUADDTYPE_MATMUL_H

#include <Python.h>

#ifdef __cplusplus
extern "C" {
#endif

int
init_matmul_ops(PyObject *numpy);

#ifdef __cplusplus
}
#endif

#endif  // _QUADDTYPE_MATMUL_H