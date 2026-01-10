#include "lock.h"

#if PY_VERSION_HEX < 0x30d00b3
PyThread_type_lock sleef_lock = NULL;
#else
PyMutex sleef_lock = {0};
#endif

void init_sleef_locks(void)
{
#if PY_VERSION_HEX < 0x30d00b3
    sleef_lock = PyThread_allocate_lock();
    if (!sleef_lock) {
        PyErr_NoMemory();
    }
#endif
}