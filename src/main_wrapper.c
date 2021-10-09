//
// Created by baojian on 2/20/20.
//
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "algo_spam.h"
#include "algo_spauc.h"

PyObject *get_results(int data_p, AlgoResults *re) {
    PyObject *results = PyTuple_New(4);
    PyObject *wt = PyList_New(data_p);
    PyObject *auc = PyList_New(re->auc_len);
    PyObject *rts = PyList_New(re->auc_len);
    PyObject *epochs = PyList_New(1);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re->wt[i]));
    }
    for (int i = 0; i < re->auc_len; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re->aucs[i]));
        PyList_SetItem(rts, i, PyFloat_FromDouble(re->rts[i]));
    }
    PyList_SetItem(epochs, 0, PyLong_FromLong(re->total_epochs));
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, auc);
    PyTuple_SetItem(results, 2, rts);
    PyTuple_SetItem(results, 3, epochs);
    return results;
}

void init_data(Data *data, PyArrayObject *x_tr_vals, PyArrayObject *x_tr_inds, PyArrayObject *x_tr_poss,
               PyArrayObject *x_tr_lens, PyArrayObject *data_y_tr) {
    data->x_tr_vals = (double *) PyArray_DATA(x_tr_vals);
    data->x_tr_inds = (int *) PyArray_DATA(x_tr_inds);
    data->x_tr_poss = (int *) PyArray_DATA(x_tr_poss);
    data->x_tr_lens = (int *) PyArray_DATA(x_tr_lens);
    data->y_tr = (double *) PyArray_DATA(data_y_tr);
    data->n = (int) PyArray_DIM(data_y_tr, 0);
}

void init_global_paras(GlobalParas *paras, PyArrayObject *global_paras) {
    double *arr_paras = (double *) PyArray_DATA(global_paras);
    //order should be: num_passes, step_len, verbose, record_aucs, stop_eps
    paras->num_passes = (int) arr_paras[0];
    paras->step_len = (int) arr_paras[1];
    paras->verbose = (int) arr_paras[2];
    paras->record_aucs = (int) arr_paras[3];
    paras->stop_eps = arr_paras[4];
}

AlgoResults *make_algo_results(int data_p, int total_num_eval) {
    AlgoResults *re = malloc(sizeof(AlgoResults));
    re->wt = calloc((size_t) data_p, sizeof(double));
    re->wt_prev = calloc((size_t) data_p, sizeof(double));
    re->aucs = calloc((size_t) total_num_eval, sizeof(double));
    re->rts = calloc((size_t) total_num_eval, sizeof(double));
    re->auc_len = 0;
    re->total_epochs = 0;
    re->total_iterations = 0;
    return re;
}

bool free_algo_results(AlgoResults *re) {
    free(re->rts);
    free(re->aucs);
    free(re->wt);
    free(re);
    return true;
}


static PyObject *wrap_algo_opt_auc(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("error!"); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l1_reg, para_l2_reg; //SPAM has three parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!ddd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras,
                          &para_xi, &para_l1_reg, &para_l2_reg)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_spam(data, paras, re, para_xi, para_l1_reg, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}

static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("error!"); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l1_reg, para_l2_reg; //SPAM has three parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!ddd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras,
                          &para_xi, &para_l1_reg, &para_l2_reg)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_spam(data, paras, re, para_xi, para_l1_reg, para_l2_reg);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}


static PyObject *wrap_algo_spauc(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("error!"); }
    PyArrayObject *x_tr_vals, *x_tr_inds, *x_tr_poss, *x_tr_lens, *data_y_tr, *global_paras;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_mu, para_l1, para_l2; //SPAUC has three parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!iiO!ddd",
                          &PyArray_Type, &x_tr_vals, &PyArray_Type, &x_tr_inds, &PyArray_Type, &x_tr_poss,
                          &PyArray_Type, &x_tr_lens, &PyArray_Type, &data_y_tr, &data->is_sparse, &data->p,
                          &PyArray_Type, &global_paras, &para_mu, &para_l1, &para_l2)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_tr_vals, x_tr_inds, x_tr_poss, x_tr_lens, data_y_tr);
    AlgoResults *re = make_algo_results(data->p, (data->n * paras->num_passes) / paras->step_len + 1);
    _algo_spauc(data, paras, re, para_mu, para_l1, para_l2);
    PyObject *results = get_results(data->p, re);
    free(paras), free_algo_results(re), free(data);
    return results;
}


// wrap_algo_solam_sparse
static PyMethodDef list_methods[] = { // hello_name
        {"c_algo_spam",    wrap_algo_spam,    METH_VARARGS, "docs"},
        {"c_algo_opt_auc", wrap_algo_opt_auc, METH_VARARGS, "docs"},
        {"c_algo_spauc",   wrap_algo_spauc,   METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "auc_module",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        list_methods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
};
#endif


/** Python version 2 for module initialization */
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_libauc_module(void) {
    Py_Initialize();
    import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}

#else
// define your own python-2.7 interface here.
#endif

int main() {
    printf("test of main wrapper!\n");
}