#ifndef ROBOTRANSFORMS_TYPES_H
#define ROBOTRANSFORMS_TYPES_H

#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

// Define type mappings based on NumPy version
#if NPY_API_VERSION >= 0x0000000C  // NumPy 1.20.0 and later
#define RT_ARRAY_DOUBLE NPY_DOUBLE
#else
#define RT_ARRAY_DOUBLE PyArray_DOUBLE
#endif

#endif // ROBOTRANSFORMS_TYPES_H
