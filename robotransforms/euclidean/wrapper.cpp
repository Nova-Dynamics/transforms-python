// transformations.c

#include "Python.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include "base.h"


// Vector macros
#define EXTRACT_2_VECTORS(in1, obj1, arr1, n1, in2, obj2, arr2, n2) \
    if ( !PyArg_ParseTuple(args, "OO", &in1, &in2) ) return NULL; \
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1); \
    if ( obj1 == NULL ) { \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        return NULL; \
    } \
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1); \
    if ( obj2 == NULL ) { \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        return NULL; \
    } \
    if ( obj1->dimensions[0] != n1 ) { \
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions"); \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        return NULL; \
    } \
    if ( obj2->dimensions[0] != n2 ) { \
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions"); \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        return NULL; \
    } \
    arr1 = (double *)PyArray_DATA(obj1); \
    arr2 = (double *)PyArray_DATA(obj2);

#define RETURN_2_VECTORS(obj1, obj2) \
    Py_DECREF(obj1); \
    Py_DECREF(obj2); \
    Py_INCREF(Py_None); \
    return Py_None;

#define APPLY_2_VECTORS(name, n1, n2) \
static PyObject *name(PyObject *self, PyObject *args) { \
    PyObject *in1, *in2; \
    PyArrayObject *obj1, *obj2; \
    double *arr1, *arr2; \
    EXTRACT_2_VECTORS(in1, obj1, arr1, n1, in2, obj2, arr2, n2) \
    euclidean::name(arr1,arr2); \
    RETURN_2_VECTORS(obj1, obj2) \
}

#define EXTRACT_3_VECTORS(in1, obj1, arr1, n1, in2, obj2, arr2, n2, in3, obj3, arr3, n3) \
    if ( !PyArg_ParseTuple(args, "OOO", &in1, &in2, &in3) ) return NULL; \
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1); \
    if ( obj1 == NULL ) { \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1); \
    if ( obj2 == NULL ) { \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 1, 1); \
    if ( obj3 == NULL ) { \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    if ( obj1->dimensions[0] != n1 ) { \
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions"); \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    if ( obj2->dimensions[0] != n2 ) { \
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions"); \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    if ( obj3->dimensions[0] != n3 ) { \
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions"); \
        Py_XDECREF(obj1); \
        Py_XDECREF(obj2); \
        Py_XDECREF(obj3); \
        return NULL; \
    } \
    arr1 = (double *)PyArray_DATA(obj1); \
    arr2 = (double *)PyArray_DATA(obj2); \
    arr3 = (double *)PyArray_DATA(obj3);

#define RETURN_3_VECTORS(obj1, obj2, obj3) \
    Py_DECREF(obj1); \
    Py_DECREF(obj2); \
    Py_DECREF(obj3); \
    Py_INCREF(Py_None); \
    return Py_None;

#define APPLY_3_VECTORS(name, n1, n2, n3) \
static PyObject *name(PyObject *self, PyObject *args) { \
    PyObject *in1, *in2, *in3; \
    PyArrayObject *obj1, *obj2, *obj3; \
    double *arr1, *arr2, *arr3; \
    EXTRACT_3_VECTORS(in1, obj1, arr1, n1, in2, obj2, arr2, n2, in3, obj3, arr3, n3) \
    euclidean::name(arr1,arr2,arr3); \
    RETURN_3_VECTORS(obj1, obj2, obj3) \
}


// Conversion
APPLY_2_VECTORS(convert_quat_to_redquat, 4, 3)
APPLY_2_VECTORS(convert_redquat_to_quat, 3, 4)
APPLY_2_VECTORS(convert_quat_to_rotvec, 4, 3)
APPLY_2_VECTORS(convert_rotvec_to_quat, 3, 4)
APPLY_2_VECTORS(convert_lrq_to_lrQ, 6, 7)
APPLY_2_VECTORS(convert_lrQ_to_lrq, 7, 6)
APPLY_2_VECTORS(convert_lrrv_to_lrQ, 6, 7)
APPLY_2_VECTORS(convert_lrQ_to_lrrv, 7, 6)
// Application
APPLY_3_VECTORS(apply_quat, 4, 3, 3)
APPLY_3_VECTORS(apply_redquat, 3, 3, 3)
APPLY_3_VECTORS(apply_rotvec, 3, 3, 3)
APPLY_3_VECTORS(apply_lrQ, 7, 3, 3)
APPLY_3_VECTORS(apply_lrq, 6, 3, 3)
APPLY_3_VECTORS(apply_lrrv, 6, 3, 3)
// Inversion
APPLY_2_VECTORS(invert_quat, 4, 4)
APPLY_2_VECTORS(invert_redquat, 3, 3)
APPLY_2_VECTORS(invert_rotvec, 3, 3)
APPLY_2_VECTORS(invert_lrQ, 7, 7)
APPLY_2_VECTORS(invert_lrq, 6, 6)
APPLY_2_VECTORS(invert_lrrv, 6, 6)
// Composition
APPLY_3_VECTORS(compose_quat, 4, 4, 4)
APPLY_3_VECTORS(compose_redquat, 3, 3, 3)
APPLY_3_VECTORS(compose_rotvec, 3, 3, 3)
APPLY_3_VECTORS(compose_lrQ, 7, 7, 7)
APPLY_3_VECTORS(compose_lrq, 6, 6, 6)
APPLY_3_VECTORS(compose_lrrv, 6, 6, 6)

/*
 * Globals
 */

static PyMethodDef EuclideanMethods[] = {
    {"convert_quat_to_redquat", convert_quat_to_redquat, METH_VARARGS, "TODO"},
    {"convert_redquat_to_quat", convert_redquat_to_quat, METH_VARARGS, "TODO"},
    {"convert_quat_to_rotvec", convert_quat_to_rotvec, METH_VARARGS, "TODO"},
    {"convert_rotvec_to_quat", convert_rotvec_to_quat, METH_VARARGS, "TODO"},
    {"convert_lrq_to_lrQ", convert_lrq_to_lrQ, METH_VARARGS, "TODO"},
    {"convert_lrQ_to_lrq", convert_lrQ_to_lrq, METH_VARARGS, "TODO"},
    {"convert_lrrv_to_lrQ", convert_lrrv_to_lrQ, METH_VARARGS, "TODO"},
    {"convert_lrQ_to_lrrv", convert_lrQ_to_lrrv, METH_VARARGS, "TODO"},
    {"apply_quat", apply_quat, METH_VARARGS, "TODO"},
    {"apply_redquat", apply_redquat, METH_VARARGS, "TODO"},
    {"apply_rotvec", apply_rotvec, METH_VARARGS, "TODO"},
    {"apply_lrQ", apply_lrQ, METH_VARARGS, "TODO"},
    {"apply_lrq", apply_lrq, METH_VARARGS, "TODO"},
    {"apply_lrrv", apply_lrrv, METH_VARARGS, "TODO"},
    {"invert_quat", invert_quat, METH_VARARGS, "TODO"},
    {"invert_redquat", invert_redquat, METH_VARARGS, "TODO"},
    {"invert_rotvec", invert_rotvec, METH_VARARGS, "TODO"},
    {"invert_lrQ", invert_lrQ, METH_VARARGS, "TODO"},
    {"invert_lrq", invert_lrq, METH_VARARGS, "TODO"},
    {"invert_lrrv", invert_lrrv, METH_VARARGS, "TODO"},
    {"compose_quat", compose_quat, METH_VARARGS, "TODO"},
    {"compose_redquat", compose_redquat, METH_VARARGS, "TODO"},
    {"compose_rotvec", compose_rotvec, METH_VARARGS, "TODO"},
    {"compose_lrQ", compose_lrQ, METH_VARARGS, "TODO"},
    {"compose_lrq", compose_lrq, METH_VARARGS, "TODO"},
    {"compose_lrrv", compose_lrrv, METH_VARARGS, "TODO"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef EuclideanModule = {
    PyModuleDef_HEAD_INIT,
    "euclidean_wrapper",
    NULL, // TODO : should be documentation?
    -1,
    EuclideanMethods
};

PyMODINIT_FUNC
PyInit_euclidean_wrapper(void)
{
    import_array();
    return PyModule_Create(&EuclideanModule);
}

