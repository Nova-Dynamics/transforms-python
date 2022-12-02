// transformations.c

#include "Python.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

#include "base.h"

static PyObject *cholesky(PyObject *self, PyObject *args) {
    PyObject *in1, *in2;
    PyArrayObject *obj1, *obj2;
    double *arr1, *arr2;

    if ( !PyArg_ParseTuple(args, "OO", &in1, &in2) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 2, 2);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 2, 2);
    if ( obj2 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    int n = obj1->dimensions[0];
    if ( obj1->dimensions[0] != n || obj1->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    if ( obj2->dimensions[0] != n || obj2->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);

    if ( !utils::cholesky(n, arr1, arr2) ) {
        PyErr_SetString(PyExc_ValueError, "matrix is not semi-positive definite");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *get_sigma_points(PyObject *self, PyObject *args) {
    int _n[1];
    PyObject *in1, *in2, *in3;
    PyArrayObject *obj1, *obj2, *obj3;
    double *arr1, *arr2, *arr3;

    if ( !PyArg_ParseTuple(args, "iOOO", _n, &in1, &in2, &in3) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 2, 2);
    if ( obj2 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 2, 2);
    if ( obj3 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    
    // Unpack the size
    int n = _n[0];
    if ( obj1->dimensions[0] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj2->dimensions[0] != n || obj2->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj3->dimensions[0] != 2*n+1 || obj3->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 4 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);
    arr3 = (double *)PyArray_DATA(obj3);

    if ( !utils::get_sigma_points(n, arr1, arr2, arr3) ) {
        PyErr_SetString(PyExc_ValueError, "cov matrix is not semi-positive definite");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    Py_DECREF(obj3);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *GRV_statistics(PyObject *self, PyObject *args) {
    int _n[1], _L[1];
    PyObject *in1, *in2, *in3;
    PyArrayObject *obj1, *obj2, *obj3;
    double *arr1, *arr2, *arr3;

    if ( !PyArg_ParseTuple(args, "iiOOO", _n, _L, &in1, &in2, &in3) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 2, 2);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1);
    if ( obj2 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 2, 2);
    if ( obj3 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    
    // Unpack the size
    int n = _n[0];
    int L = _L[0];
    if ( obj1->dimensions[0] != 2*L+1 || obj1->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj2->dimensions[0] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 4 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj3->dimensions[0] != n || obj3->dimensions[1] != n ) {
        PyErr_SetString(PyExc_ValueError, "arg 5 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);
    arr3 = (double *)PyArray_DATA(obj3);

    utils::GRV_statistics(n, L, arr1, arr2, arr3);
    
    Py_DECREF(obj1);
    Py_DECREF(obj2);
    Py_DECREF(obj3);

    Py_INCREF(Py_None);
    return Py_None;
}

/*
 * Globals
 */

static PyMethodDef DeadReckonMethods[] = {
    {"cholesky", cholesky, METH_VARARGS, "TODO"},
    {"get_sigma_points", get_sigma_points, METH_VARARGS, "TODO"},
    {"GRV_statistics", GRV_statistics, METH_VARARGS, "TODO"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef DeadReckonModule = {
    PyModuleDef_HEAD_INIT,
    "utils_wrapper",
    NULL, // TODO : should be documentation?
    -1,
    DeadReckonMethods
};

PyMODINIT_FUNC
PyInit_utils_wrapper(void)
{
    import_array();
    return PyModule_Create(&DeadReckonModule);
}

