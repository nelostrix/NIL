#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static double *buffer(PyObject *object, Py_ssize_t count) {
    PyObject *sequence = PySequence_Fast(object, "Expected numeric sequence");
    if (!sequence) return NULL;
    if (PySequence_Fast_GET_SIZE(sequence) != count) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "Kernel input shape differs");
        return NULL;
    }
    double *values = PyMem_Malloc((size_t)count * sizeof(double));
    if (!values) { Py_DECREF(sequence); PyErr_NoMemory(); return NULL; }
    for (Py_ssize_t i = 0; i < count; ++i) {
        values[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence, i));
        if (PyErr_Occurred() || !isfinite(values[i])) {
            PyMem_Free(values); Py_DECREF(sequence);
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Kernel values must be finite");
            return NULL;
        }
    }
    Py_DECREF(sequence);
    return values;
}

static PyObject *matmul(PyObject *self, PyObject *args) {
    PyObject *a, *b;
    Py_ssize_t m, k, n;
    if (!PyArg_ParseTuple(args, "OOnnn", &a, &b, &m, &k, &n)) return NULL;
    for (int i = 2; i < 5; ++i) {
        if (!PyLong_CheckExact(PyTuple_GET_ITEM(args, i))) {
            PyErr_SetString(PyExc_ValueError, "Dimensions must be integers"); return NULL;
        }
    }
    /* Division checks precede multiplication, including hostile direct calls. */
    if (m <= 0 || k <= 0 || n <= 0 || m > 100000 || k > 100000 || n > 100000 ||
        m > 100000/k || k > 100000/n || m > 100000/n) {
        PyErr_SetString(PyExc_ValueError, "Kernel dimension or element budget exceeded"); return NULL;
    }
    double *left = buffer(a, m*k);
    if (!left) return NULL;
    double *right = buffer(b, k*n);
    if (!right) { PyMem_Free(left); return NULL; }
    double *output = PyMem_Calloc((size_t)(m*n), sizeof(double));
    if (!output) { PyMem_Free(left); PyMem_Free(right); return PyErr_NoMemory(); }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < m; ++i)
        for (Py_ssize_t p = 0; p < k; ++p)
            for (Py_ssize_t j = 0; j < n; ++j)
                output[i*n+j] += left[i*k+p] * right[p*n+j];
    Py_END_ALLOW_THREADS
    PyMem_Free(left); PyMem_Free(right);
    PyObject *result = PyList_New(m*n);
    if (!result) { PyMem_Free(output); return NULL; }
    for (Py_ssize_t i = 0; i < m*n; ++i) {
        if (!isfinite(output[i])) {
            PyMem_Free(output); Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "Nonfinite kernel output"); return NULL;
        }
        PyObject *value = PyFloat_FromDouble(output[i]);
        if (!value) { PyMem_Free(output); Py_DECREF(result); return NULL; }
        PyList_SET_ITEM(result, i, value);
    }
    PyMem_Free(output);
    return result;
}

static PyMethodDef methods[] = {
    {"matmul", matmul, METH_VARARGS, "Bounded row-major float64 matrix multiplication."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_nil_cpu_kernels", "Explicit NIL-owned CPU kernel, ABI 1.", -1, methods
};
PyMODINIT_FUNC PyInit__nil_cpu_kernels(void) {
    PyObject *result = PyModule_Create(&module);
    if (result && PyModule_AddIntConstant(result, "abi_version", 1) < 0) {
        Py_DECREF(result); return NULL;
    }
    return result;
}
