// transformations.c

#include "Python.h"
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>

#include "base.h"

static PyObject *dr_calculate_d(PyObject *self, PyObject *args) {
    double _vave[1], _vdiff[1];

    if ( !PyArg_ParseTuple(args, "dd", _vave, _vdiff) ) return NULL;

    double result = dead_reckon::dr_calculate_d( _vave[0], _vdiff[0] );

    return Py_BuildValue("d", result);
}

static PyObject *dead_reckon_step(PyObject *self, PyObject *args) {
    double _dl[1], _dr[1], _vave[1], _vdiff[1];
    PyObject *in1, *in2;
    PyArrayObject *obj1, *obj2;
    double *arr1, *arr2;

    if ( !PyArg_ParseTuple(args, "OddddO", &in1, _dl, _dr, _vave, _vdiff,  &in2 ) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1);
    if ( obj2 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }

    if ( obj1->dimensions[0] != 4 ) {
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }
    if ( obj2->dimensions[0] != 3 ) {
        PyErr_SetString(PyExc_ValueError, "arg 6 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);

    dead_reckon::dead_reckon_step( arr1, _dl[0], _dr[0], _vave[0], _vdiff[0], arr2);

    Py_DECREF(obj1);
    Py_DECREF(obj2);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *dead_reckon_step_errors(PyObject *self, PyObject *args) {
    double _dl[1], _dr[1], _vave[1], _vdiff[1], _dl_scale[1];
    PyObject *in1;
    PyArrayObject *obj1;
    double *arr1;

    if ( !PyArg_ParseTuple(args, "dddddO", _dl, _dr, _vave, _vdiff, _dl_scale,  &in1 ) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        return NULL;
    }

    if ( obj1->dimensions[0] != 3 ) {
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions");
        Py_XDECREF(obj1);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);

    dead_reckon::dead_reckon_step_errors( _dl[0], _dr[0], _vave[0], _vdiff[0], _dl_scale[0], arr1);

    Py_DECREF(obj1);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *dead_reckon_apply(PyObject *self, PyObject *args) {
    PyObject *in1, *in2, *in3, *in4, *in5;
    PyArrayObject *obj1, *obj2, *obj3, *obj4, *obj5;
    double *arr1, *arr2, *arr3, *arr4, *arr5;

    if ( !PyArg_ParseTuple(args, "OOOOO", &in1, &in2, &in3, &in4, &in5 ) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1);
    if ( obj1 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1);
    if ( obj2 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 2, 2);
    if ( obj3 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj4 = (PyArrayObject *)PyArray_ContiguousFromObject(in4, PyArray_DOUBLE, 1, 1);
    if ( obj4 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj5 = (PyArrayObject *)PyArray_ContiguousFromObject(in5, PyArray_DOUBLE, 2, 2);
    if ( obj5 == NULL ) {
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    if ( obj1->dimensions[0] != 9 ) {
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj2->dimensions[0] != 7 ) {
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj3->dimensions[0] != 6 || obj3->dimensions[1] != 6 ) {
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj4->dimensions[0] != 7 ) {
        PyErr_SetString(PyExc_ValueError, "arg 4 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj5->dimensions[0] != 6 || obj5->dimensions[1] != 6 ) {
        PyErr_SetString(PyExc_ValueError, "arg 5 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);
    arr3 = (double *)PyArray_DATA(obj3);
    arr4 = (double *)PyArray_DATA(obj4);
    arr5 = (double *)PyArray_DATA(obj5);

    if ( !dead_reckon::dead_reckon_apply( arr1, arr2, arr3, arr4, arr5 ) ) {
        PyErr_SetString(PyExc_ValueError, "matrix was not semi-positive definate");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    Py_DECREF(obj1);
    Py_DECREF(obj2);
    Py_DECREF(obj3);
    Py_DECREF(obj4);
    Py_DECREF(obj5);

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject *dead_reckon_apply_ip(PyObject *self, PyObject *args) {
    PyObject *in1, *in2, *in3;
    PyArrayObject *obj1, *obj2, *obj3;
    double *arr1, *arr2, *arr3;

    if ( !PyArg_ParseTuple(args, "OOO", &in1, &in2, &in3 ) ) return NULL;
    obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 1, 1);
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

    if ( obj1->dimensions[0] != 9 ) {
        PyErr_SetString(PyExc_ValueError, "arg 1 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj2->dimensions[0] != 7 ) {
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }
    if ( obj3->dimensions[0] != 6 || obj3->dimensions[1] != 6 ) {
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        return NULL;
    }

    arr1 = (double *)PyArray_DATA(obj1);
    arr2 = (double *)PyArray_DATA(obj2);
    arr3 = (double *)PyArray_DATA(obj3);

    if ( !dead_reckon::dead_reckon_apply( arr1, arr2, arr3 ) ) {
        PyErr_SetString(PyExc_ValueError, "matrix was not semi-positive definate");
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

static PyObject *_dead_reckon(PyObject *self, PyObject *args) {
    PyObject *in1, *in2, *in3, *in4, *in5;
    PyArrayObject *obj1, *obj2, *obj3, *obj4, *obj5;
    double *arr1, *arr2, *arr3, *arr4, *arr5;
    int _n_steps[1];

    if ( !PyArg_ParseTuple(args, "iOOOOO", _n_steps, &in1, &in2, &in3, &in4, &in5 ) ) return NULL;


    int n_steps = _n_steps[0];
    bool steps_empty = n_steps < 1;

    if ( !steps_empty ) {
        obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 2, 2);
        if ( obj1 == NULL ) {
            Py_XDECREF(obj1);
            Py_XDECREF(obj2);
            Py_XDECREF(obj3);
            Py_XDECREF(obj4);
            Py_XDECREF(obj5);
            return NULL;
        }
    }
    obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1);
    if ( obj2 == NULL ) {
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 2, 2);
    if ( obj3 == NULL ) {
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj4 = (PyArrayObject *)PyArray_ContiguousFromObject(in4, PyArray_DOUBLE, 1, 1);
    if ( obj4 == NULL ) {
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    obj5 = (PyArrayObject *)PyArray_ContiguousFromObject(in5, PyArray_DOUBLE, 2, 2);
    if ( obj5 == NULL ) {
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    if ( !steps_empty && (obj1->dimensions[0] != n_steps || obj1->dimensions[1] != 9) ) {
        PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
        Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj2->dimensions[0] != 7 ) {
        PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj3->dimensions[0] != 6 || obj3->dimensions[1] != 6 ) {
        PyErr_SetString(PyExc_ValueError, "arg 4 has invalid dimensions");
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj4->dimensions[0] != 7 ) {
        PyErr_SetString(PyExc_ValueError, "arg 5 has invalid dimensions");
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }
    if ( obj5->dimensions[0] != 6 || obj5->dimensions[1] != 6 ) {
        PyErr_SetString(PyExc_ValueError, "arg 6 has invalid dimensions");
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    if ( !steps_empty ) {
        arr1 = (double *)PyArray_DATA(obj1);
    } else {
        arr1 = {};
    }
    arr2 = (double *)PyArray_DATA(obj2);
    arr3 = (double *)PyArray_DATA(obj3);
    arr4 = (double *)PyArray_DATA(obj4);
    arr5 = (double *)PyArray_DATA(obj5);

    if ( !dead_reckon::dead_reckon( n_steps, arr1, arr2, arr3, arr4, arr5 ) ) {
        PyErr_SetString(PyExc_ValueError, "matrix was not semi-positive definate");
        if ( !steps_empty ) Py_XDECREF(obj1);
        Py_XDECREF(obj2);
        Py_XDECREF(obj3);
        Py_XDECREF(obj4);
        Py_XDECREF(obj5);
        return NULL;
    }

    if ( !steps_empty ) Py_DECREF(obj1);
    Py_DECREF(obj2);
    Py_DECREF(obj3);
    Py_DECREF(obj4);
    Py_DECREF(obj5);

    Py_INCREF(Py_None);
    return Py_None;
}

// static PyObject *_dead_reckon_ip(PyObject *self, PyObject *args) {
//     PyObject *in1, *in2, *in3;
//     PyArrayObject *obj1, *obj2, *obj3;
//     double *arr1, *arr2, *arr3;
//     int _n_steps[1];
//
//     if ( !PyArg_ParseTuple(args, "iOOO", _n_steps, &in1, &in2, &in3 ) ) return NULL;
//     obj1 = (PyArrayObject *)PyArray_ContiguousFromObject(in1, PyArray_DOUBLE, 2, 2);
//     if ( obj1 == NULL ) {
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//     obj2 = (PyArrayObject *)PyArray_ContiguousFromObject(in2, PyArray_DOUBLE, 1, 1);
//     if ( obj2 == NULL ) {
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//     obj3 = (PyArrayObject *)PyArray_ContiguousFromObject(in3, PyArray_DOUBLE, 2, 2);
//     if ( obj3 == NULL ) {
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//
//     int n_steps = _n_steps[0];
//     if ( obj1->dimensions[0] != n_steps || obj1->dimensions[1] != 9 ) {
//         PyErr_SetString(PyExc_ValueError, "arg 2 has invalid dimensions");
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//     if ( obj2->dimensions[0] != 7 ) {
//         PyErr_SetString(PyExc_ValueError, "arg 3 has invalid dimensions");
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//     if ( obj3->dimensions[0] != 6 || obj3->dimensions[1] != 6 ) {
//         PyErr_SetString(PyExc_ValueError, "arg 4 has invalid dimensions");
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//
//     arr1 = (double *)PyArray_DATA(obj1);
//     arr2 = (double *)PyArray_DATA(obj2);
//     arr3 = (double *)PyArray_DATA(obj3);
//
//     if ( !dead_reckon::dead_reckon( n_steps, arr1, arr2, arr3 ) ) {
//         PyErr_SetString(PyExc_ValueError, "matrix was not semi-positive definate");
//         Py_XDECREF(obj1);
//         Py_XDECREF(obj2);
//         Py_XDECREF(obj3);
//         return NULL;
//     }
//
//     Py_DECREF(obj1);
//     Py_DECREF(obj2);
//     Py_DECREF(obj3);
//
//     Py_INCREF(Py_None);
//     return Py_None;
// }

/*
 * Globals
 */

static PyMethodDef DeadReckonMethods[] = {
    {"dr_calculate_d", dr_calculate_d, METH_VARARGS, "TODO"},
    {"dead_reckon_step", dead_reckon_step, METH_VARARGS, "TODO"},
    {"dead_reckon_step_errors", dead_reckon_step_errors, METH_VARARGS, "TODO"},
    {"dead_reckon_apply", dead_reckon_apply, METH_VARARGS, "TODO"},
    {"dead_reckon_apply_ip", dead_reckon_apply_ip, METH_VARARGS, "TODO"},
    {"dead_reckon", _dead_reckon, METH_VARARGS, "TODO"},
    // {"dead_reckon_ip", _dead_reckon_ip, METH_VARARGS, "TODO"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef DeadReckonModule = {
    PyModuleDef_HEAD_INIT,
    "dead_reckon_wrapper",
    NULL, // TODO : should be documentation?
    -1,
    DeadReckonMethods
};

PyMODINIT_FUNC
PyInit_dead_reckon_wrapper(void)
{
    import_array();
    return PyModule_Create(&DeadReckonModule);
}

