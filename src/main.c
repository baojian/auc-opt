#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *wrap_test_numpy(PyObject *self, PyObject *args) {
    PyObject *result;
    printf("Test numpy module!\n");
    Py_RETURN_NONE;
}

int main(int argc, char *argv[]) {
    printf("Hello, World!\n");
    return 0;
}


// wrap_opt_auc
static PyMethodDef test_methods[] = { // hello_name
        {"c_test_numpy", (PyCFunction) wrap_test_numpy, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        /* m_name */ /* m_doc */ /* m_size */
        "test_module_3", "This is a module", -1,
        test_methods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
};
#endif


/** Python version 2 for module initialization */
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_libtest_module_3(void) {
    Py_Initialize();
    import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}

#else
initlibtest_module_2() {
    (void) Py_InitModule3("libtest_module_2", test_methods, "some docs.");
    import_array(); // In order to use numpy, you must include this!
}

#endif