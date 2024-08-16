#ifndef _QUADDTYPE_CASTS_H
#define _QUADDTYPE_CASTS_H

#include <Python.h>
#include "numpy/dtype_api.h"

#ifdef __cplusplus
extern "C" {
#endif

PyArrayMethod_Spec **
init_casts(void);

void
free_casts(void);

#ifdef __cplusplus
}
#endif

#endif