#ifndef _QUADDTYPE_LOCK_H
#define _QUADDTYPE_LOCK_H

#include <Python.h>

#if PY_VERSION_HEX < 0x30d00b3
extern PyThread_type_lock sleef_lock;
#define LOCK_SLEEF PyThread_acquire_lock(sleef_lock, WAIT_LOCK)
#define UNLOCK_SLEEF PyThread_release_lock(sleef_lock)
#else
extern PyMutex sleef_lock;
#define LOCK_SLEEF PyMutex_Lock(&sleef_lock)
#define UNLOCK_SLEEF PyMutex_Unlock(&sleef_lock)
#endif

void init_sleef_locks(void);

#endif // _QUADDTYPE_LOCK_H