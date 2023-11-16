#include "static_string.h"

#include <stdint.h>
#include <string.h>

#if NPY_BYTE_ORDER == NPY_LITTLE_ENDIAN

// the high byte in vstring.size is reserved for flags
// SSSS SSSF

typedef struct _npy_static_vstring_t {
    size_t offset;
    size_t size_and_flags;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    char buf[sizeof(_npy_static_vstring_t) - 1];
    unsigned char size_and_flags;
} _short_string_buffer;

#elif NPY_BYTE_ORDER == NPY_BIG_ENDIAN

// the high byte in vstring.size is reserved for flags
// FSSS SSSS

typedef struct _npy_static_vstring_t {
    size_t size;
    size_t offset;
} _npy_static_vstring_t;

typedef struct _short_string_buffer {
    unsigned char size_and_flags;
    char buf[sizeof(npy_static_vstring_t) - 1];
} _short_string_buffer;

#endif

typedef union _npy_static_string_u {
    _npy_static_vstring_t vstring;
    _short_string_buffer direct_buffer;
} _npy_static_string_u;

#define NPY_STRING_MISSING 0x80      // 1000 0000
#define NPY_STRING_SHORT 0x40        // 0100 0000
#define NPY_STRING_ARENA_FREED 0x20  // 0010 0000
#define NPY_STRING_ON_HEAP 0x10      // 0001 0000
#define NPY_STRING_MEDIUM 0x08       // 0000 1000
#define NPY_STRING_FLAG_MASK 0xF8    // 1111 1000

// short string sizes fit in a 4-bit integer
#define NPY_SHORT_STRING_SIZE_MASK 0x0F  // 0000 1111
#define NPY_SHORT_STRING_MAX_SIZE \
    (sizeof(npy_static_string) - 1)      // 15 or 7 depending on arch
#define NPY_MEDIUM_STRING_MAX_SIZE 0xFF  // 256

// Since this has no flags set, technically this is a heap-allocated string
// with size zero. Practically, that doesn't matter because we always do size
// checks before accessing heap data, but that may be confusing. The nice part
// of this choice is a calloc'd array buffer (e.g. from np.empty) is filled
// with empty elements for free
const _npy_static_string_u empty_string_u = {
        .direct_buffer = {.size_and_flags = 0, .buf = {0}}};
const npy_packed_static_string *NPY_EMPTY_STRING =
        (npy_packed_static_string *)&empty_string_u;
// zero-filled, but with the NULL flag set to distinguish from empty string
const _npy_static_string_u null_string_u = {
        .direct_buffer = {.size_and_flags = NPY_STRING_MISSING, .buf = {0}}};
const npy_packed_static_string *NPY_NULL_STRING =
        (npy_packed_static_string *)&null_string_u;

#define VSTRING_FLAGS(string) \
    string->direct_buffer.size_and_flags & ~NPY_SHORT_STRING_SIZE_MASK;
#define HIGH_BYTE_MASK ((size_t)0XFF << 8 * (sizeof(size_t) - 1))
#define VSTRING_SIZE(string) (string->vstring.size_and_flags & ~HIGH_BYTE_MASK)

typedef struct npy_string_arena {
    size_t cursor;
    size_t size;
    char *buffer;
} npy_string_arena;

struct npy_string_allocator {
    npy_string_malloc_func malloc;
    npy_string_free_func free;
    npy_string_realloc_func realloc;
    npy_string_arena arena;
};

void
set_vstring_size(_npy_static_string_u *str, size_t size)
{
    unsigned char *flags = &str->direct_buffer.size_and_flags;
    unsigned char current_flags = *flags & ~NPY_SHORT_STRING_SIZE_MASK;
    str->vstring.size_and_flags = size;
    str->direct_buffer.size_and_flags = current_flags;
}

char *
vstring_buffer(npy_string_arena *arena, _npy_static_string_u *string)
{
    char flags = VSTRING_FLAGS(string);
    if (flags & NPY_STRING_ON_HEAP) {
        return (char *)string->vstring.offset;
    }
    if (arena->buffer == NULL) {
        return NULL;
    }
    return (char *)((size_t)arena->buffer + string->vstring.offset);
}

char *
NpyString_arena_malloc(npy_string_arena *arena, npy_string_realloc_func r,
                       size_t size)
{
    // one extra size_t to store the size of the allocation
    size_t string_storage_size;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        string_storage_size = size + sizeof(unsigned char);
    }
    else {
        string_storage_size = size + sizeof(size_t);
    }
    if ((arena->size - arena->cursor) <= string_storage_size) {
        // realloc the buffer so there is enough room
        // first guess is to double the size of the buffer
        size_t newsize;
        if (arena->size == 0) {
            newsize = string_storage_size;
        }
        else if (((2 * arena->size) - arena->cursor) > string_storage_size) {
            newsize = 2 * arena->size;
        }
        else {
            newsize = arena->size + string_storage_size;
        }
        if ((arena->cursor + size) >= newsize) {
            // doubling the current size isn't enough
            newsize = 2 * (arena->cursor + size);
        }
        // passing a NULL buffer to realloc is the same as malloc
        char *newbuf = r(arena->buffer, newsize);
        if (newbuf == NULL) {
            return NULL;
        }
        memset(newbuf + arena->cursor, 0, newsize - arena->cursor);
        arena->buffer = newbuf;
        arena->size = newsize;
    }
    char *ret;
    if (size <= NPY_MEDIUM_STRING_MAX_SIZE) {
        unsigned char *size_loc =
                (unsigned char *)&arena->buffer[arena->cursor];
        *size_loc = size;
        ret = &arena->buffer[arena->cursor + sizeof(char)];
    }
    else {
        char *size_ptr = (char *)&arena->buffer[arena->cursor];
        memcpy(size_ptr, &size, sizeof(size_t));
        ret = &arena->buffer[arena->cursor + sizeof(size_t)];
    }
    arena->cursor += string_storage_size;
    return ret;
}

int
NpyString_arena_free(npy_string_arena *arena, _npy_static_string_u *str)
{
    if (arena->size == 0 && arena->cursor == 0 && arena->buffer == NULL) {
        // empty arena, nothing to do
        return 0;
    }

    char *ptr = vstring_buffer(arena, str);
    if (ptr == NULL) {
        return -1;
    }
    size_t size = VSTRING_SIZE(str);
    uintptr_t buf_start = (uintptr_t)arena->buffer;
    uintptr_t ptr_loc = (uintptr_t)ptr;
    uintptr_t end_loc = ptr_loc + size;
    uintptr_t buf_end = buf_start + arena->size;
    if (ptr_loc < buf_start || ptr_loc > buf_end || end_loc > buf_end) {
        return -1;
    }

    memset(ptr, 0, size);

    return 0;
}

static npy_string_arena NEW_ARENA = {0, 0, NULL};

npy_string_allocator *
NpyString_new_allocator(npy_string_malloc_func m, npy_string_free_func f,
                        npy_string_realloc_func r)
{
    npy_string_allocator *allocator = m(sizeof(npy_string_allocator));
    if (allocator == NULL) {
        return NULL;
    }
    allocator->malloc = m;
    allocator->free = f;
    allocator->realloc = r;
    // arena buffer gets allocated in NpyString_arena_malloc
    allocator->arena = NEW_ARENA;
    return allocator;
}

void
NpyString_free_allocator(npy_string_allocator *allocator)
{
    npy_string_free_func f = allocator->free;

    if (allocator->arena.buffer != NULL) {
        f(allocator->arena.buffer);
    }

    f(allocator);
}

int
is_short_string(const npy_packed_static_string *s)
{
    unsigned char high_byte =
            ((_npy_static_string_u *)s)->direct_buffer.size_and_flags;
    int has_short_flag = (high_byte & NPY_STRING_SHORT);
    int has_on_heap_flag = (high_byte & NPY_STRING_ON_HEAP);
    return has_short_flag && !has_on_heap_flag;
}

int
is_medium_string(const _npy_static_string_u *s)
{
    unsigned char high_byte = s->direct_buffer.size_and_flags;
    int has_short_flag = (high_byte & NPY_STRING_SHORT);
    int has_medium_flag = (high_byte & NPY_STRING_MEDIUM);
    return (!has_short_flag && has_medium_flag);
}

int
NpyString_isnull(const npy_packed_static_string *s)
{
    unsigned char high_byte =
            ((_npy_static_string_u *)s)->direct_buffer.size_and_flags;
    return (high_byte & NPY_STRING_MISSING) == NPY_STRING_MISSING;
}

int
is_not_a_vstring(const npy_packed_static_string *s)
{
    return is_short_string(s) || NpyString_isnull(s);
}

int
is_a_vstring(const npy_packed_static_string *s)
{
    return !is_not_a_vstring(s);
}

int
NpyString_load(npy_string_allocator *allocator,
               const npy_packed_static_string *packed_string,
               npy_static_string *unpacked_string)
{
    if (NpyString_isnull(packed_string)) {
        unpacked_string->size = 0;
        unpacked_string->buf = NULL;
        return 1;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        unsigned char high_byte = string_u->direct_buffer.size_and_flags;
        unpacked_string->size = high_byte & NPY_SHORT_STRING_SIZE_MASK;
        unpacked_string->buf = string_u->direct_buffer.buf;
    }

    else {
        size_t size = VSTRING_SIZE(string_u);
        char *buf = NULL;
        if (size > 0) {
            npy_string_arena *arena = &allocator->arena;
            if (arena == NULL) {
                return -1;
            }
            buf = vstring_buffer(arena, string_u);
            if (buf == NULL) {
                return -1;
            }
        }
        unpacked_string->size = size;
        unpacked_string->buf = buf;
    }

    return 0;
}

char *
heap_or_arena_allocate(npy_string_allocator *allocator,
                       _npy_static_string_u *to_init_u, size_t size,
                       int *on_heap)
{
    unsigned char *flags = &to_init_u->direct_buffer.size_and_flags;
    if (*flags & NPY_STRING_SHORT) {
        // Have to heap allocate since there isn't a preexisting
        // allocation. This leaves the NPY_STRING_SHORT flag set to indicate
        // that there is no room in the arena buffer for strings in this
        // entry
        *flags |= NPY_STRING_ON_HEAP;
        *on_heap = 1;
        return allocator->malloc(sizeof(char) * size);
    }
    npy_string_arena *arena = &allocator->arena;
    if (arena == NULL) {
        return NULL;
    }
    if (*flags & NPY_STRING_ARENA_FREED) {
        // Check if there's room for the new string in the existing
        // allocation. The size is stored one size_t "behind" the beginning of
        // the allocation.
        char *buf = vstring_buffer(arena, to_init_u);
        if (buf == NULL) {
            return NULL;
        }
        size_t alloc_size;
        if (is_medium_string(to_init_u)) {
            // stored in a char so direct access is OK
            alloc_size = (size_t) * (buf - 1);
        }
        else {
            // not necessarily memory-aligned, so need to use memcpy
            size_t *size_loc = (size_t *)((uintptr_t)buf - sizeof(size_t));
            memcpy(&alloc_size, size_loc, sizeof(size_t));
        }
        if (size <= alloc_size) {
            // we have room!
            *flags &= ~NPY_STRING_ARENA_FREED;
            return buf;
        }
        else {
            // No room, resort to a heap allocation.
            *flags = NPY_STRING_ON_HEAP;
            *on_heap = 1;
            return allocator->malloc(sizeof(char) * size);
        }
    }
    // string isn't previously allocated, so add to existing arena allocation
    char *ret = NpyString_arena_malloc(arena, allocator->realloc,
                                       sizeof(char) * size);
    if (size < NPY_MEDIUM_STRING_MAX_SIZE) {
        *flags |= NPY_STRING_MEDIUM;
    }
    return ret;
}

int
heap_or_arena_deallocate(npy_string_allocator *allocator,
                         _npy_static_string_u *str_u)
{
    unsigned char *flags = &str_u->direct_buffer.size_and_flags;
    if (*flags & NPY_STRING_ON_HEAP) {
        // It's a heap string (not in the arena buffer) so it needs to be
        // deallocated with free(). For heap strings the offset is a raw
        // address so this cast is safe.
        allocator->free((char *)str_u->vstring.offset);
        if (*flags & NPY_STRING_SHORT) {
            *flags = 0 | NPY_STRING_SHORT;
        }
        else {
            *flags &= ~NPY_STRING_ON_HEAP;
        }
    }
    else if (VSTRING_SIZE(str_u) != 0) {
        npy_string_arena *arena = &allocator->arena;
        if (arena == NULL) {
            return -1;
        }
        if (NpyString_arena_free(arena, str_u) < 0) {
            return -1;
        }
        if (arena->buffer != NULL) {
            str_u->direct_buffer.size_and_flags |= NPY_STRING_ARENA_FREED;
        }
    }
    return 0;
}

int
NpyString_newsize(const char *init, size_t size,
                  npy_packed_static_string *to_init,
                  npy_string_allocator *allocator)
{
    if (NpyString_newemptysize(size, to_init, allocator) < 0) {
        return -1;
    }

    if (size == 0) {
        return 0;
    }

    _npy_static_string_u *to_init_u = ((_npy_static_string_u *)to_init);

    char *buf = NULL;

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        buf = vstring_buffer(&allocator->arena, to_init_u);
    }
    else {
        buf = to_init_u->direct_buffer.buf;
    }

    memcpy(buf, init, size);

    return 0;
}

int
NpyString_newemptysize(size_t size, npy_packed_static_string *out,
                       npy_string_allocator *allocator)
{
    if (size > NPY_MAX_STRING_SIZE) {
        return -1;
    }

    _npy_static_string_u *out_u = (_npy_static_string_u *)out;

    unsigned char flags =
            out_u->direct_buffer.size_and_flags & ~NPY_SHORT_STRING_SIZE_MASK;

    if (size == 0) {
        *out = *NPY_EMPTY_STRING;
        out_u->direct_buffer.size_and_flags |= flags;
        return 0;
    }

    if (size > NPY_SHORT_STRING_MAX_SIZE) {
        int on_heap = 0;
        char *buf = heap_or_arena_allocate(allocator, out_u, size, &on_heap);

        if (buf == NULL) {
            return -1;
        }

        if (on_heap) {
            out_u->vstring.offset = (size_t)buf;
        }
        else {
            npy_string_arena *arena = &allocator->arena;
            if (arena == NULL) {
                return -1;
            }
            out_u->vstring.offset = (size_t)buf - (size_t)arena->buffer;
        }
        set_vstring_size(out_u, size);
    }
    else {
        // Size can be no larger than 7 or 15, depending on CPU architecture.
        // In either case, the size data is in at most the least significant 4
        // bits of the byte so it's safe to | with one of 0x10, 0x20, 0x40, or
        // 0x80.
        out_u->direct_buffer.size_and_flags = NPY_STRING_SHORT | flags | size;
    }

    return 0;
}

int
NpyString_free(npy_packed_static_string *str, npy_string_allocator *allocator)
{
    _npy_static_string_u *str_u = (_npy_static_string_u *)str;
    if (is_not_a_vstring(str)) {
        // zero out, keeping flags
        unsigned char *flags = &str_u->direct_buffer.size_and_flags;
        unsigned char current_flags = *flags & ~NPY_SHORT_STRING_SIZE_MASK;
        memcpy(str, NPY_EMPTY_STRING, sizeof(npy_packed_static_string));
        *flags |= current_flags;
    }
    else {
        if (VSTRING_SIZE(str_u) == 0) {
            // empty string is a vstring but nothing to deallocate
            return 0;
        }
        if (heap_or_arena_deallocate(allocator, str_u) < 0) {
            return -1;
        }
    }
    return 0;
}

int
NpyString_dup(const npy_packed_static_string *in,
              npy_packed_static_string *out,
              npy_string_allocator *in_allocator,
              npy_string_allocator *out_allocator)
{
    if (NpyString_isnull(in)) {
        *out = *NPY_NULL_STRING;
        return 0;
    }
    if (is_short_string(in)) {
        memcpy(out, in, sizeof(npy_packed_static_string));
        return 0;
    }
    _npy_static_string_u *in_u = (_npy_static_string_u *)in;
    size_t size = VSTRING_SIZE(in_u);
    if (size == 0) {
        _npy_static_string_u *out_u = (_npy_static_string_u *)out;
        unsigned char flags = out_u->direct_buffer.size_and_flags &
                              ~NPY_SHORT_STRING_SIZE_MASK;
        *out = *NPY_EMPTY_STRING;
        out_u->direct_buffer.size_and_flags |= flags;
        return 0;
    }
    char *in_buf = NULL;
    npy_string_arena *arena = &in_allocator->arena;
    if (arena->buffer == NULL) {
        return -1;
    }
    int used_malloc = 0;
    if (in_allocator == out_allocator && is_a_vstring(in)) {
        in_buf = in_allocator->malloc(size);
        memcpy(in_buf, vstring_buffer(arena, in_u), size);
        used_malloc = 1;
    }
    else {
        in_buf = vstring_buffer(arena, in_u);
    }
    int ret =
            NpyString_newsize(in_buf, VSTRING_SIZE(in_u), out, out_allocator);
    if (used_malloc) {
        in_allocator->free(in_buf);
    }
    return ret;
}

int
NpyString_cmp(const npy_static_string *s1, const npy_static_string *s2)
{
    size_t minsize = s1->size < s2->size ? s1->size : s2->size;

    int cmp = 0;

    if (minsize != 0) {
        cmp = strncmp(s1->buf, s2->buf, minsize);
    }

    if (cmp == 0) {
        if (s1->size > minsize) {
            return 1;
        }
        if (s2->size > minsize) {
            return -1;
        }
    }

    return cmp;
}

size_t
NpyString_size(const npy_packed_static_string *packed_string)
{
    if (NpyString_isnull(packed_string)) {
        return 0;
    }

    _npy_static_string_u *string_u = (_npy_static_string_u *)packed_string;

    if (is_short_string(packed_string)) {
        return string_u->direct_buffer.size_and_flags &
               NPY_SHORT_STRING_SIZE_MASK;
    }

    return VSTRING_SIZE(string_u);
}
