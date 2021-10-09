//
// Created by --- on ---.
//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "algo_spam.h"


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


void _get_posi_nega_x(double *posi_x, double *nega_x, double *posi_t, double *nega_t, double *prob_p, Data *data) {
    if (data->is_sparse) {
        for (int i = 0; i < data->n; i++) {
            const int *xt_inds = data->x_tr_inds + data->x_tr_poss[i];
            const double *xt_vals = data->x_tr_vals + data->x_tr_poss[i];
            if (data->y_tr[i] > 0) {
                (*posi_t)++;
                for (int kk = 0; kk < data->x_tr_lens[i]; kk++)
                    posi_x[xt_inds[kk]] += xt_vals[kk];
            } else {
                (*nega_t)++;
                for (int kk = 0; kk < data->x_tr_lens[i]; kk++)
                    nega_x[xt_inds[kk]] += xt_vals[kk];
            }
        }
    } else {
        for (int i = 0; i < data->n; i++) {
            const double *xt = (data->x_tr_vals + i * data->p);
            if (data->y_tr[i] > 0) {
                (*posi_t)++;
                for (int ii = 0; ii < data->p; ii++) {
                    posi_x[ii] += xt[ii];
                }
            } else {
                (*nega_t)++;
                for (int ii = 0; ii < data->p; ii++) {
                    nega_x[ii] += xt[ii];
                }
            }
        }
    }
    *prob_p = (*posi_t) / (data->n * 1.0);
    for (int i = 0; i < data->p; i++) {
        posi_x[i] = posi_x[i] / (*posi_t);
        nega_x[i] = nega_x[i] / (*nega_t);
    }
}


void _algo_spam(Data *data, GlobalParas *paras, AlgoResults *re,
                double para_xi, double para_l1_reg, double para_l2_reg) {

    double start_time = clock();
    double *grad_wt = malloc(sizeof(double) * data->p); // gradient
    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n, sizeof(double));
    double a_wt;
    double b_wt;
    double alpha_wt;
    double posi_t = 0.0;
    double nega_t = 0.0;
    double prob_p;
    double eta_t;
    _get_posi_nega_x(posi_x, nega_x, &posi_t, &nega_t, &prob_p, data);
    for (int t = 1; t <= paras->num_passes * data->n; t++) {
        eta_t = para_xi / sqrt(t); // current learning rate
        a_wt = 0.0;
        b_wt = 0.0;
        for (int ii = 0; ii < data->p; ii++) {
            a_wt += re->wt[ii] * posi_x[ii];
            b_wt += re->wt[ii] * nega_x[ii];
        }
        alpha_wt = b_wt - a_wt; // alpha(wt)
        const double *xt;
        const int *xt_inds;
        const double *xt_vals;
        double xtw = 0.0, weight;
        if (data->is_sparse) {
            // receive zt=(xt,yt)
            xt_inds = data->x_tr_inds + data->x_tr_poss[(t - 1) % data->n];
            xt_vals = data->x_tr_vals + data->x_tr_poss[(t - 1) % data->n];
            for (int tt = 0; tt < data->x_tr_lens[(t - 1) % data->n]; tt++) {
                xtw += (re->wt[xt_inds[tt]] * xt_vals[tt]);
            }
            weight = data->y_tr[(t - 1) % data->n] > 0 ?
                     2. * (1.0 - prob_p) * (xtw - a_wt) -
                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                     2.0 * prob_p * (xtw - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            // gradient descent
            for (int tt = 0; tt < data->x_tr_lens[(t - 1) % data->n]; tt++) {
                re->wt[xt_inds[tt]] += -eta_t * weight * xt_vals[tt];
            }
        } else {
            xt = data->x_tr_vals + ((t - 1) % data->n) * data->p;
            xtw = 0.0;
            for (int ii = 0; ii < data->p; ii++) {
                xtw += re->wt[ii] * xt[ii];
            }
            weight = data->y_tr[(t - 1) % data->n] > 0 ?
                     2. * (1.0 - prob_p) * (xtw - a_wt) -
                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                     2.0 * prob_p * (xtw - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            // gradient descent
            for (int ii = 0; ii < data->p; ii++) {
                re->wt[ii] -= eta_t * weight * xt[ii];
            }
        }
        if (para_l1_reg <= 0.0 && para_l2_reg > 0.0) {
            // l2-regularization
            for (int ii = 0; ii < data->p; ii++) {
                re->wt[ii] /= (eta_t * para_l2_reg + 1.);
            }
        } else {
            // elastic-net
            double tmp_demon = (eta_t * para_l2_reg + 1.);
            for (int k = 0; k < data->p; k++) {
                double tmp_sign = (double) sign(re->wt[k]) / tmp_demon;
                re->wt[k] = tmp_sign * fmax(0.0, fabs(re->wt[k]) - eta_t * para_l1_reg);
            }
        }
        // evaluate the AUC score
        if ((fmod(t, paras->step_len) == 1.) && (paras->record_aucs == 1)) {
            _evaluate_aucs(data, y_pred, re, start_time);
        }
        // at the end of each epoch, we check the early stop condition.
        re->total_iterations++;
        if (t % data->n == 0) {
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
                           t / data->n, paras->num_passes);
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


// wrap_algo_solam_sparse
static PyMethodDef list_methods[] = { // hello_name
        {"c_algo_spam",    wrap_algo_spam,    METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "spam_l2",     /* m_name */
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
PyInit_libspam_l2(void) {
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
