//
// Created by --- on ---.
//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "algo_spauc.h"


static inline int _comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &_comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

double _auc_score(const double *true_labels, const double *scores, int len) {
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    _arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    for (int i = 0; i <= len; i++) {
        tpr[i] /= num_posi;
        fpr[i] /= num_nega;
    }
    // AUC score
    double auc = 0.0;
    double delta;
    for (int i = 0; i <= len; i++) {
        delta = (fpr[i] - fpr[i - 1]);
        auc += ((tpr[i - 1] + tpr[i]) / 2.) * delta;
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    return auc;
}

void _evaluate_aucs(Data *data, double *y_pred, AlgoResults *re, double start_time) {
    double t_eval = clock();
    memset(y_pred, 0, sizeof(double) * data->n);
    if (data->is_sparse) {
        for (int q = 0; q < data->n; q++) {
            const int *xt_inds = data->x_tr_inds + data->x_tr_poss[q];
            const double *xt_vals = data->x_tr_vals + data->x_tr_poss[q];
            for (int tt = 0; tt < data->x_tr_lens[q]; tt++)
                y_pred[q] += re->wt[xt_inds[tt]] * xt_vals[tt];
        }
    } else {
        memset(y_pred, 0, sizeof(double) * data->n);
        for (int q = 0; q < data->n; q++) {
            const double *xt_vals = data->x_tr_vals + data->p * q;
            for (int tt = 0; tt < data->p; tt++)
                y_pred[q] += re->wt[tt] * xt_vals[tt];
        }
    }
    re->aucs[re->auc_len] = _auc_score(data->y_tr, y_pred, data->n);
    re->rts[re->auc_len++] = clock() - start_time - (clock() - t_eval);
}

void _algo_spauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_mu,
                 double para_l1,
                 double para_l2) {

    double start_time = clock();
    double *grad_wt = malloc(sizeof(double) * data->p); // gradient
    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n, sizeof(double));
    double a_wt = 0.0, b_wt = 0.0;
    double posi_t = 0.0, nega_t = 0.0;
    double prob_p = 0.0;
    double eta_t;
    for (int tt = 0; tt < paras->num_passes * data->n; tt++) {
        int ind = tt % data->n;
        const double *xt = data->x_tr_vals + ind * data->p;
        eta_t = 2. / (para_mu * (tt + 1.) + 1.0); // current learning rate
        double xtw = 0.0;
        if (data->y_tr[ind] > 0) {
            posi_t++;
            for (int ii = 0; ii < data->p; ii++) {
                xtw += (re->wt[ii] * xt[ii]);
                posi_x[ii] += xt[ii];
            }
            prob_p = (tt * prob_p + 1.) / (tt + 1.0);
            // update a(wt)
            a_wt = 0.0;
            for (int ii = 0; ii < data->p; ii++) {
                a_wt += re->wt[ii] * posi_x[ii];
            }
            a_wt /= posi_t;
        } else {
            nega_t++;
            for (int ii = 0; ii < data->p; ii++) {
                xtw += (re->wt[ii] * xt[ii]);
                nega_x[ii] += xt[ii];
            }
            prob_p = (tt * prob_p) / (tt + 1.0);
            // update b(wt)
            b_wt = 0.0;
            for (int ii = 0; ii < data->p; ii++) {
                b_wt += re->wt[ii] * nega_x[ii];
            }
            b_wt /= nega_t;
        }
        double wei_x, wei_posi, wei_nega;
        if (data->y_tr[ind] > 0) {
            wei_x = 2. * (1. - prob_p) * (xtw - a_wt);
            wei_posi = 2. * (1. - prob_p) * (a_wt - xtw - prob_p * (1. + b_wt - a_wt));
            wei_nega = 2. * prob_p * (1. - prob_p) * (1. + b_wt - a_wt);
        } else {
            wei_x = 2. * prob_p * (xtw - b_wt);
            wei_posi = 2. * prob_p * (1. - prob_p) * (-1. - b_wt + a_wt);
            wei_nega = 2. * prob_p * (b_wt - xtw - (1.0 - prob_p) * (-1. - b_wt + a_wt));
        }
        if (nega_t > 0) {
            wei_nega /= nega_t;
        } else {
            wei_nega = 0.0;
        }
        if (posi_t > 0) {
            wei_posi /= posi_t;
        } else {
            wei_posi = 0.0;
        }
        for (int ii = 0; ii < data->p; ii++) {
            grad_wt[ii] = wei_x * xt[ii] + wei_nega * nega_x[ii] + wei_posi * posi_x[ii];
        }
        // gradient descent
        if (para_l1 > 0.0) {
            double tmp;
            for (int ii = 0; ii < data->p; ii++) {
                tmp = re->wt[ii] - eta_t * grad_wt[ii];
                double tmp_sign = (double) sign(tmp);
                re->wt[ii] = tmp_sign * fmax(0.0, fabs(tmp) - eta_t * para_l1);
            }
        } else {
            for (int ii = 0; ii < data->p; ii++) {
                re->wt[ii] = (re->wt[ii] - eta_t * grad_wt[ii]) / (1.0 + para_l2);
            }
        }
        // evaluate the AUC score
        if ((fmod(tt, paras->step_len) == 1.) && (paras->record_aucs == 1)) {
            _evaluate_aucs(data, y_pred, re, start_time);
        }
        // at the end of each epoch, we check the early stop condition.
        re->total_iterations++;
        if (tt % data->n == 0) {
            re->total_epochs++;
            double norm_wt = 0.0;
            for (int ii = 0; ii < data->p; ii++) {
                norm_wt += re->wt_prev[ii] * re->wt_prev[ii];
            }
            norm_wt = sqrt(norm_wt);
            for (int ii = 0; ii < data->p; ii++) {
                re->wt_prev[ii] -= re->wt[ii];
            }
            double norm_diff = 0.0;
            for (int ii = 0; ii < data->p; ii++) {
                norm_diff += re->wt_prev[ii] * re->wt_prev[ii];
            }
            norm_diff = sqrt(norm_diff);
            if (norm_wt > 0.0 && (norm_diff / norm_wt <= paras->stop_eps)) {
                if (paras->verbose > 0) {
                    printf("early stop at: %d-th epoch where maximal epoch is: %d\n",
                           tt / data->n, paras->num_passes);
                }
                break;
            }
        }
        memcpy(re->wt_prev, re->wt, sizeof(double) * (data->p));
    }
    for (int i = 0; i < re->auc_len; i++) {
        re->rts[i] /= CLOCKS_PER_SEC;
    }
    free(y_pred);
    free(nega_x);
    free(posi_x);
    free(grad_wt);
}


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
        {"c_algo_spauc",   wrap_algo_spauc,   METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spauc_l2",     /* m_name */
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
PyInit_libspauc_l2(void) {
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
