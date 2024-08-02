#ifndef _QUADDTYPE_CASTS_H
#define _QUADDTYPE_CASTS_H

#include<Python.h>
#include "numpy/dtype_api.h"


#ifdef __cplusplus
extern "C" {
#endif

extern PyArrayMethod_Spec QuadtoQuadCastSpec;

PyArrayMethod_Spec ** init_casts(void);

void free_casts(void);

#ifdef __cplusplus
}
#endif

#endif