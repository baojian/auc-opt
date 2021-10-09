//
// Created by --- on ---.
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include <float.h>
#include <stdio.h>
#include <stdbool.h>
#include "algo_opt_auc.h"


typedef struct {
    double val;
    int posi_node;
    int nega_node;
} data_point;

typedef struct {
    double val;
    int index;
} data_pair;

static inline int comp_descend(const void *a, const void *b) {
    return ((data_pair *) b)->val > ((data_pair *) a)->val ? 1 : -1;
}


static inline int comp_ascend(const void *a, const void *b) {
    return ((data_point *) b)->val < ((data_point *) a)->val ? 1 : -1;
}

void arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

// calculate the TPR, FPR, and AUC score.
void roc_curve(const double *true_labels, const double *scores,
               int n, double *tpr, double *fpr, double *auc) {
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < n; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * n);
    arg_sort_descend(scores, sorted_indices, n);
    //Notice, here we assume the score has no -inf
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    for (int i = 0; i < n; i++) { // accumulate sum
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    for (int i = 0; i <= n; i++) {
        tpr[i] /= num_posi;
        fpr[i] /= num_nega;
    }
    //AUC score by using trapezoidal rule
    *auc = 0.0;
    double delta;
    for (int i = 1; i <= n; i++) {
        delta = (fpr[i] - fpr[i - 1]);
        *auc += ((tpr[i - 1] + tpr[i]) / 2.) * delta;
    }
    free(sorted_indices);
}

/**
 * Calculate the AUC score.
 * We assume: 1. True labels are {+1,-1}; 2. scores are real numbers.
 * @param t_labels
 * @param scores
 * @param len
 * @return AUC score.
 */
double roc_auc_score(const double *t_labels,
                     const double *scores, int len) {
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (t_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = t_labels[sorted_indices[i]];
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
    //AUC score by using trapezoidal rule
    double auc = 0.0;
    double delta;
    for (int i = 1; i <= len; i++) {
        delta = (fpr[i] - fpr[i - 1]);
        auc += ((tpr[i - 1] + tpr[i]) / 2.) * delta;
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    return auc;
}

// Give a set of points, the algorithm learn a linear classifier
// so that the AUC score is maximized. This only works for 2 dimension.
int algo_opt_auc(const double *x_tr, const double *y_tr,
                 int n, int d, double para_eps,
                 double *re_wt, double *re_auc, int para_verbose) {
    double start_time = clock();
    if (d != 2) {
        fprintf(stderr, "dimension should be 2 but got %d", d);
    }
    // calculate n_p and n_q
    long t_posi = 0, t_nega = 0;
    for (int i = 0; i < n; i++) {
        t_posi += y_tr[i] > 0 ? 1 : 0;
        t_nega += y_tr[i] < 0 ? 1 : 0;
    }
    // save all positive and negative samples.
    int *posi_inds = malloc(sizeof(int) * t_posi);
    int *nega_inds = malloc(sizeof(int) * t_nega);
    int p_index = 0, n_index = 0;
    for (int i = 0; i < n; i++) {
        if (y_tr[i] > 0) {
            posi_inds[p_index] = i;
            p_index++;
        } else {
            nega_inds[n_index] = i;
            n_index++;
        }
    }
    // considering special case
    if (p_index == 0 || n_index == 0) {
        printf("no positive or negative samples!\n");
        re_wt[0] = 0.0;
        re_wt[1] = 1.0;
        *re_auc = 1.0;
        return 0;
    }
    // at most t_posi * t_nega + 1 different slopes
    data_point *slopes = malloc(sizeof(data_point) * (t_posi * t_nega + 1));
    if (para_verbose > 0) {
        printf("total number of slopes: %ld\n", (t_posi * t_nega + 1));
    }
    long total_slopes = 0;
    double diff[2], min_slope = DBL_MAX, max_slope = DBL_MIN;
    for (int i = 0; i < t_posi; i++) {
        for (int j = 0; j < t_nega; j++) {
            diff[0] = x_tr[d * posi_inds[i]] - x_tr[d * nega_inds[j]];
            diff[1] = x_tr[d * posi_inds[i] + 1] - x_tr[d * nega_inds[j] + 1];
            if (diff[0] != 0.0) {
                double slope = -diff[1] / diff[0];
                slopes[total_slopes].val = slope;
                slopes[total_slopes].posi_node = posi_inds[i];
                slopes[total_slopes].nega_node = nega_inds[j];
                if (para_verbose > 0) {
                    printf("val: %.8f %d %d\n",
                           slope, posi_inds[i], nega_inds[j]);
                }
                min_slope = fmin(min_slope, slope);
                max_slope = fmax(max_slope, slope);
                total_slopes++;
            }
        }
    }
    // the "last" slope is the one that has the slope value
    // 2 + max_slope.
    slopes[total_slopes].val = max_slope + 2.0;
    qsort(slopes, (size_t) total_slopes, sizeof(data_point), &comp_ascend);
    double granularity = 1. / (double) (t_posi * t_nega);
    if (para_verbose > 0) {
        char *output = "total slopes: %ld granularity: %.8f\n";
        printf(output, total_slopes, granularity);
        output = "slopes[0]: %.8f slopes[1]: %.8f slopes[2]: %.8f\n";
        printf(output, slopes[0].val, slopes[1].val, slopes[2].val);
    }
    double prev_eps = 1.0, cur_eps;
    // at the beginning, the initial interesting slope is [min(s) - 1, 1]
    re_wt[0] = min_slope - prev_eps;
    re_wt[1] = 1.0;
    // calculate scores provided by f := re_wt
    double *scores = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        scores[i] = re_wt[0] * x_tr[2 * i] + x_tr[2 * i + 1];
    }
    // calculate AUC score
    double auc = roc_auc_score(y_tr, scores, n);
    if (para_verbose > 0) {
        printf("initial auc: %.8f\n", auc);
    }
    double opt_auc = auc;
    p_index = 0, n_index = 0;
    posi_inds[p_index] = slopes[0].posi_node;
    nega_inds[n_index] = slopes[0].nega_node;
    p_index++;
    n_index++;
    double w_cur[2] = {0.0, 1.0}, w_prev[2] = {0.0, 1.0};
    for (long ii = 0; ii < total_slopes;) {
        double eps = (slopes[ii + 1].val - slopes[ii].val) / 2.;
        if (para_verbose > 0) {
            printf("cur-eps: %.8f\n", eps);
        }
        if (eps <= para_eps) {
            bool p_flag = true, n_flag = true;
            for (int j = 0; j < p_index; j++) {
                if (posi_inds[j] == slopes[ii + 1].posi_node) {
                    p_flag = false;
                    break;
                }
            }
            for (int j = 0; j < n_index; j++) {
                if (nega_inds[j] == slopes[ii + 1].nega_node) {
                    n_flag = false;
                    break;
                }
            }
            if (p_flag) {
                posi_inds[p_index] = slopes[ii + 1].posi_node;
                if (para_verbose > 0) {
                    printf("add: %d ", posi_inds[p_index]);
                }
                p_index++;
            }
            if (n_flag) {
                nega_inds[n_index] = slopes[ii + 1].nega_node;
                n_index++;
            }
            ii++;
        } else {
            cur_eps = eps;
            double c;
            w_cur[0] = slopes[ii].val + cur_eps;
            w_prev[0] = slopes[ii].val - prev_eps;
            int tol_correct_cur = 0, tol_correct_prev = 0;
            for (int i = 0; i < p_index; i++) {
                for (int j = 0; j < n_index; j++) {
                    double p_score = x_tr[2 * posi_inds[i]] * w_cur[0] +
                                     x_tr[2 * posi_inds[i] + 1];
                    double n_score = x_tr[2 * nega_inds[j]] * w_cur[0] +
                                     x_tr[2 * nega_inds[j] + 1];
                    if (p_score > n_score) {
                        tol_correct_cur++;
                    }
                    p_score = x_tr[2 * posi_inds[i]] * w_prev[0] +
                              x_tr[2 * posi_inds[i] + 1];
                    n_score = x_tr[2 * nega_inds[j]] * w_prev[0] +
                              x_tr[2 * nega_inds[j] + 1];
                    if (p_score > n_score) {
                        tol_correct_prev++;
                    }
                }
            }
            ii++;
            c = tol_correct_cur - tol_correct_prev;
            auc += c * granularity;
            // find a better AUC score
            if (opt_auc < auc) {
                opt_auc = auc;
                re_wt[0] = w_cur[0];
                re_wt[1] = 1.0;
            }
            // reverse order
            if (opt_auc < (1. - auc)) {
                opt_auc = 1. - auc;
                re_wt[0] = -w_cur[0];
                re_wt[1] = -1.;
            }
            if (para_verbose > 0) {
                char *output = "c: %.6f auc: %.8f cur_eps: %.8f auc_opt: %.8f ";
                printf(output, c, auc, cur_eps, opt_auc);
                for (int j = 0; j < p_index; j++) {
                    printf(" %d", posi_inds[j]);
                }
                printf("\n");
            }
            prev_eps = cur_eps;
            p_index = 0, n_index = 0;
            posi_inds[p_index] = slopes[ii].posi_node;
            nega_inds[n_index] = slopes[ii].nega_node;
            p_index++;
            n_index++;
        }
    }
    char *output = "opt-auc: %.8f, w: (%.8f. %.8f) run_time: %.8f\n";
    double run_time = (clock() - start_time) / CLOCKS_PER_SEC;
    if (para_verbose > 0) {
        printf(output, opt_auc, re_wt[0], re_wt[1], run_time);
    }
    *re_auc = opt_auc;
    free(slopes);
    free(posi_inds);
    free(nega_inds);
}

static PyObject *wrap_opt_auc(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("some error!"); }
    double start_time = clock();
    PyArrayObject *x_tr, *y_tr;
    double para_eps;
    double re_wt[2], re_auc;
    if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr, &para_eps)) { return NULL; }
    int n = x_tr->dimensions[0];
    int d = x_tr->dimensions[1];
    algo_opt_auc((double *) PyArray_DATA(x_tr),
                 (double *) PyArray_DATA(y_tr),
                 n, d, para_eps, re_wt, &re_auc, 0);
    PyObject *results = PyTuple_New(3);
    PyObject *w = PyList_New(2);
    for (int i = 0; i < 2; i++) {
        PyList_SetItem(w, i, PyFloat_FromDouble(re_wt[i]));
    }
    PyTuple_SetItem(results, 0, w);
    PyTuple_SetItem(results, 1, PyFloat_FromDouble(re_auc));
    PyTuple_SetItem(results, 2, PyFloat_FromDouble(
            (clock() - start_time) / CLOCKS_PER_SEC));
    return results;
}


static PyObject *wrap_roc_auc_score(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("some error!"); }
    PyArrayObject *y_tr, *y_scores;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &y_tr,
                          &PyArray_Type, &y_scores)) { return NULL; }
    int n = y_tr->dimensions[0];
    double auc = roc_auc_score((double *) PyArray_DATA(y_tr),
                               (double *) PyArray_DATA(y_scores), n);
    PyObject *results = PyTuple_New(1);
    PyTuple_SetItem(results, 0, PyFloat_FromDouble(auc));
    return results;
}


static PyObject *wrap_roc_curve(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("some error!"); }
    PyArrayObject *y_tr, *y_scores;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &y_tr,
                          &PyArray_Type, &y_scores)) { return NULL; }
    int n = y_tr->dimensions[0];
    double auc;
    double *fpr = malloc(sizeof(double) * (n + 1));
    double *tpr = malloc(sizeof(double) * (n + 1));
    roc_curve((double *) PyArray_DATA(y_tr),
              (double *) PyArray_DATA(y_scores), n, tpr, fpr, &auc);
    PyObject *results = PyTuple_New(3);
    PyObject *re_fpr = PyList_New(n + 1);
    PyObject *re_tpr = PyList_New(n + 1);
    for (int i = 0; i < n + 1; i++) {
        PyList_SetItem(re_fpr, i, PyFloat_FromDouble(fpr[i]));
        PyList_SetItem(re_tpr, i, PyFloat_FromDouble(tpr[i]));
    }
    PyTuple_SetItem(results, 0, re_fpr);
    PyTuple_SetItem(results, 1, re_tpr);
    PyTuple_SetItem(results, 2, PyFloat_FromDouble(auc));
    free(fpr);
    free(tpr);
    return results;
}


void test_cases_1() {
    /**
     * test some duplicated points
     */
    int n = 10, d = 2;
    double para_eps = 2e-16, re_wt[2], re_auc;
    double x_tr[20] = {0.0, 0.0, 1.0, 1.0, 0.1, 0.3, 0.6, 0.9,
                       0.2, 0.7, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8,
                       0.1, 0.8, 0.1, 0.8};
    double y_tr[10] = {-1, -1., 1., 1., -1., 1., 1., 1., 1., 1.};
    algo_opt_auc(x_tr, y_tr, n, d, para_eps, re_wt, &re_auc, 0);
    printf("optimal auc: %.6f -- w: [ %.8f %.8f]\n",
           re_auc, re_wt[0], re_wt[1]);
}

void test_cases_2() {
    int n = 10, d = 2;
    double para_eps = 2e-16, re_wt[2], re_auc;
    double x_tr[20] = {0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6,
                       0.8, 0.8, 1.0, 1.0, 1.2, 1.2, 1.4, 1.4,
                       0.5, 0.6, 0.7, 0.3};
    double y_tr[10] = {-1, -1., 1., 1., -1., -1., 1., 1., 1., -1.};
    algo_opt_auc(x_tr, y_tr, n, d, para_eps, re_wt, &re_auc, 0);
    printf("optimal auc: %.6f -- w: [ %.8f %.8f]\n",
           re_auc, re_wt[0], re_wt[1]);
}

void test_cases_3() {
    int n = 10, d = 2;
    double para_eps = 2e-16, re_wt[2], re_auc;
    double x_tr[20] = {0.0, 0.0, 1.0, 1.0, 0.1, 0.0, 1.1, 1.0,
                       0.2, 0.0, 1.2, 1.0, 0.3, 0.0, 1.3, 1.0,
                       0.4, 0.0, 1.4, 1.0};
    double y_tr[10] = {-1, 1., 1., 1., -1., -1., 1., 1., 1., 1.};
    algo_opt_auc(x_tr, y_tr, n, d, para_eps, re_wt, &re_auc, 0);
    printf("optimal auc: %.6f -- w: [ %.8f %.8f]\n",
           re_auc, re_wt[0], re_wt[1]);
}

void test_cases_4() {
    int n = 10, d = 2;
    double para_eps = 2e-16, re_wt[2], re_auc;
    double x_tr[20] = {0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                       0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                       1.0, 0.0, 1.0, 1.0};
    double y_tr[10] = {-1, 1., 1., 1., -1., -1., 1., 1., 1., 1.};
    algo_opt_auc(x_tr, y_tr, n, d, para_eps, re_wt, &re_auc, 0);
    printf("optimal auc: %.6f -- w: [ %.8f %.8f]\n",
           re_auc, re_wt[0], re_wt[1]);
}

void test_sort() {
    const int n = 6;
    /** Be careful, using double as keys is not stable. */
    data_point *b = malloc(sizeof(data_point) * 6);
    b[0].val = 2, b[0].posi_node = 6;
    b[1].val = 4, b[1].posi_node = 5;
    b[2].val = 2, b[2].posi_node = 4;
    b[3].val = 0, b[3].posi_node = 3;
    b[4].val = 5, b[4].posi_node = 2;
    b[5].val = 4, b[5].posi_node = 1;
    printf("original array: ");
    for (int i = 0; i < n; i++) {
        printf(" (%.4f,%d) ", b[i].val, b[i].posi_node);
    }
    printf("\nsorted array:   ");
    qsort(b, n, sizeof(data_point), &comp_ascend);
    for (int i = 0; i < n; i++) {
        printf(" (%.4f,%d) ", b[i].val, b[i].posi_node);
    }
    printf("\n");
}

void test_auc() {
    ///
}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        test_sort();
    } else {
        test_cases_1();
        test_cases_2();
        test_cases_3();
        test_cases_4();
    }
    return 0;
}

static PyMethodDef list_methods[] = {
        {"c_opt_auc",   (PyCFunction) wrap_opt_auc,       METH_VARARGS, "docs"},
        {"c_roc_curve", (PyCFunction) wrap_roc_curve,     METH_VARARGS, "docs"},
        {"c_auc_score", (PyCFunction) wrap_roc_auc_score, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

// This is the interface for Python-3.7
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "opt_auc_3",
        "This is a module", -1, list_methods,      /* m_methods */
        NULL, NULL, NULL, NULL,};

PyMODINIT_FUNC PyInit_libopt_auc_3(void) {
    Py_Initialize();
    import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}

#else
// define your own python-2.7 interface here
#endif