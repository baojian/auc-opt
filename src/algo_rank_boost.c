//
// Created by baojian on 2/24/20.
//

#include <Python.h>
#include <numpy/arrayobject.h>
#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include "algo_rank_boost.h"

typedef struct {
    double val;
    int index;
} data_pair;

static inline int comp_descend(const void *a, const void *b) {
    return ((data_pair *) b)->val > ((data_pair *) a)->val ? 1 : -1;
}


void arg_sort_descend(const double *x, int *sorted_indices, long x_len) {
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


void arg_sort_descend_modify(double *x, int *sorted_indices, long x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &comp_descend);
    for (int i = 0; i < x_len; i++) {
        x[i] = w_pairs[i].val;
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

/**
 * A more efficient version of RankBoost for bipartite feedback.
 * It implements RankBoost.B shown in Page 943 of [1].
 *
 * ---
 * [1]  Freund, Yoav, et al. "An efficient boosting algorithm for
 *      combining preferences." Journal of machine learning
 *      research 4.Nov (2003): 933-969.
 * @param x_tr
 * @param y_tr
 * @param n
 * @param p
 * @param para_t
 * @param re_alpha
 * @param re_threshold
 * @param re_rank_feat
 * @return
 */
int algo_rank_boost_debug(const double *x_tr, const double *y_tr,
                          int n, int p, int para_t,
                          double *re_alpha, double *re_theta, int *re_q_def, int *re_i) {
    // obtain the positive/negative training samples.
    int t_posi = 0, t_nega = 0;
    for (int i = 0; i < n; i++) {
        t_posi += y_tr[i] > 0 ? 1 : 0;
        t_nega += y_tr[i] < 0 ? 1 : 0;
    }
    // if it contains only one class, nothing needs to do.
    // or if it has zero iteration, just return.
    if (t_posi == 0 || t_nega == 0 || para_t <= 0) {
        return false;
    }
    double p_gran = 1. / (double) (t_posi);
    double n_gran = 1. / (double) (t_nega);
    int *posi_inds = malloc(sizeof(int) * t_posi);
    int *nega_inds = malloc(sizeof(int) * t_nega);
    t_posi = 0;
    t_nega = 0;
    // initialize v function.
    double *v = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        if (y_tr[i] > 0) {
            posi_inds[t_posi] = i;
            t_posi++;
            v[i] = p_gran;
        } else {
            nega_inds[t_nega] = i;
            t_nega++;
            v[i] = n_gran;
        }
    }
    // getting the sorted features.
    double *sorted_f = malloc(sizeof(double) * (n * p));
    int *sorted_ind = malloc(sizeof(int) * (n * p));
    double *eps_f = malloc(sizeof(double) * p);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            sorted_f[i * n + j] = x_tr[i + j * p];
            printf("%.2f ", sorted_f[i * n + j]);
        }
        printf("\n");
        double *cur_f = sorted_f + i * n;
        int *cur_ind = sorted_ind + i * n;
        arg_sort_descend_modify(cur_f, cur_ind, n);
        for (int j = 0; j < n; j++) {
            printf("%.2f ", cur_f[j]);
        }
        double cur_min_eps = INFINITY;
        for (int j = 0; j < n - 1; j++) {
            cur_min_eps = fmin(cur_min_eps, fabs(cur_f[j + 1] - cur_f[j]));
        }
        eps_f[i] = cur_min_eps;
        for (int j = 0; j < n; j++) {
            cur_f[j] -= eps_f[i] / 2.;
        }
        printf("\n\n");
    }
    // implementing the weak learner by using the third method.
    double v_posi, v_nega;
    double *d = malloc(sizeof(double) * n);
    double *s = malloc(sizeof(double) * n);
    double *pi = malloc(sizeof(double) * n);
    for (int tt = 0; tt < para_t; tt++) {
        // check the Equation (6) in [1]
        v_posi = 0.0, v_nega = 0.0;
        for (int i = 0; i < t_posi; i++) {
            v_posi += v[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            v_nega += v[nega_inds[j]];
        }
        // calculate d in Equation (8) and at the same time
        // calculate pi, in our binary classification case,
        // we can combine Equation (8) and (10), then
        // pi = d * s. shown in Page 945 of [1]
        // at the initial stage, r is the sum of pi.
        for (int i = 0; i < t_posi; i++) {
            d[posi_inds[i]] = v[posi_inds[i]] * v_nega;
            s[posi_inds[i]] = -1.;
            pi[posi_inds[i]] = -d[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            d[nega_inds[j]] = v[nega_inds[j]] * v_posi;
            s[nega_inds[j]] = 1.;
            pi[nega_inds[j]] = d[nega_inds[j]];
        }
        printf("k: %d, vpos: %.2f vneg: %.2f\n", tt, v_posi, v_nega);
        printf("d:  ");
        for (int i = 0; i < n; i++) {
            printf("%.6f ", d[i]);
        }
        printf("\ns:  ");
        for (int i = 0; i < t_posi; i++) {
            printf("%.6f ", -1.);
        }
        for (int j = 0; j < t_nega; j++) {
            printf("%.6f ", 1.);
        }
        printf("\npi: ");
        for (int i = 0; i < n; i++) {
            printf("%.6f ", pi[i]);
        }
        printf("\n");
        // Go to Figure 3 of Page 947 to learn the
        // weak learner: it is uniquely defined by
        // i_opt, q_def_opt, and theta_opt
        // TODO, this can improved by only considering
        // different features.
        double l, r_opt = 0.0, theta_opt = INFINITY;
        int i_opt = 0, q_def_opt = 0;
        double *cand_threshold = malloc(sizeof(double) * (n + 1));
        cand_threshold[0] = INFINITY;
        for (int pp = 0; pp < p; pp++) {
            l = 0.0; // r has been calculated.
            int *cur_ind = sorted_ind + pp * n;
            double *cur_f = sorted_f + pp * n;
            memcpy(cand_threshold + 1, cur_f, sizeof(double) * n);
            printf("fi: ");
            for (int i = 0; i < n; i++) {
                printf(" %.6f", x_tr[pp + cur_ind[i] * p]);
            }
            printf("\n");

            printf("candthreshold: ");
            for (int i = 0; i < n + 1; i++) {
                printf(" %.6f", cand_threshold[i]);
            }
            printf("\n");
            // corresponding to infinity case
            printf("L: %.6f ", l);
            double prev_f = x_tr[pp + cur_ind[0] * p];
            for (int i = 0; i < n - 1; i++) {
                // to handle the same values.
                if (prev_f == x_tr[pp + cur_ind[i + 1] * p]) {
                    prev_f = x_tr[pp + cur_ind[i + 1] * p];
                    l += pi[cur_ind[i]];
                    continue;
                }
                prev_f = x_tr[pp + cur_ind[i + 1] * p];
                l += pi[cur_ind[i]];
                if (fabs(l) > fabs(r_opt)) {
                    r_opt = l;
                    i_opt = pp;
                    theta_opt = prev_f;
                    q_def_opt = 0;
                }
                printf("%.6f ", l);
            }
            printf("\n");
        }
        re_theta[tt] = theta_opt;
        re_i[tt] = i_opt;
        re_q_def[tt] = q_def_opt;
        // Go back to Figure 2 of RankBoost.B
        if (fabs(fabs(r_opt) - 1.0) < 1e-10) {
            r_opt = (r_opt > 0 ? 1. : -1.) * (1. - 1e-10);
        }
        double alpha_t = .5 * log((1. + r_opt) / (1. - r_opt));
        re_alpha[tt] = alpha_t;
        printf("rmax: %.6f alpha[%d]: %.6f threshold[%d]: %.6f rankfeat[%d]: %d\n",
               r_opt, tt, re_alpha[tt], tt, re_theta[tt], tt, re_i[tt]);
        // to update the v vector.
        double z_posi = 0.0, z_nega = 0.0;
        for (int i = 0; i < t_posi; i++) {
            double hi = x_tr[posi_inds[i] * p + i_opt] > theta_opt ? 1.0 : 0.0;
            v[posi_inds[i]] *= exp(alpha_t * hi);
            z_posi += v[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            double hi = x_tr[nega_inds[j] * p + i_opt] > theta_opt ? 1.0 : 0.0;
            v[nega_inds[j]] *= exp(-alpha_t * hi);
            z_nega += v[nega_inds[j]];
        }
        if (z_posi != 0.0) {
            for (int i = 0; i < t_posi; i++) {
                v[posi_inds[i]] /= z_posi;
            }
        }
        if (z_nega != 0.0) {
            for (int j = 0; j < t_nega; j++) {
                v[nega_inds[j]] /= z_nega;
            }
        }
        printf("v: ");
        for (int i = 0; i < n; i++) {
            printf("%.6f ", v[i]);
        }
        printf("\n");
        re_alpha[tt] = -re_alpha[tt];
    }
    printf("\nre_alpha: \n");
    for (int i = 0; i < para_t; i++) {
        printf("%.6f ", re_alpha[i]);
    }
    printf("\nre_threshold: \n");
    for (int i = 0; i < para_t; i++) {
        printf("%.6f ", re_theta[i]);
    }
    printf("\nre_rankfeat: \n");
    for (int i = 0; i < para_t; i++) {
        printf("%d ", re_i[i]);
    }
    printf("\n");
    free(sorted_ind);
    free(sorted_f);
    free(d);
    free(pi);
    free(v);
    free(posi_inds);
    free(nega_inds);
    return 1;
}

/**
 * A more efficient version of RankBoost for bipartite feedback.
 * It implements RankBoost.B shown in Page 943 of [1].
 *
 * ---
 * [1]  Freund, Yoav, et al. "An efficient boosting algorithm for
 *      combining preferences." Journal of machine learning
 *      research 4.Nov (2003): 933-969.
 * @param x_tr
 * @param y_tr
 * @param n
 * @param p
 * @param para_t
 * @param re_alpha
 * @param re_threshold
 * @param re_rank_feat
 * @return
 */
int algo_rank_boost(const double *x_tr, const double *y_tr,
                    int n, int p, int para_t,
                    double *re_alpha, double *re_theta, int *re_q_def, int *re_i) {
    // http://asi.insa-rouen.fr/enseignants/~arakoto/toolbox/index.html
    // obtain the positive/negative training samples.
    int t_posi = 0, t_nega = 0;
    for (int i = 0; i < n; i++) {
        t_posi += y_tr[i] > 0 ? 1 : 0;
        t_nega += y_tr[i] < 0 ? 1 : 0;
    }
    // if it contains only one class, nothing needs to do.
    // or if it has zero iteration, just return.
    if (t_posi == 0 || t_nega == 0 || para_t <= 0) {
        return false;
    }
    double p_gran = 1. / (double) (t_posi);
    double n_gran = 1. / (double) (t_nega);
    int *posi_inds = malloc(sizeof(int) * t_posi);
    int *nega_inds = malloc(sizeof(int) * t_nega);
    t_posi = 0;
    t_nega = 0;
    // initialize v function.
    double *v = malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++) {
        if (y_tr[i] > 0) {
            posi_inds[t_posi] = i;
            t_posi++;
            v[i] = p_gran;
        } else {
            nega_inds[t_nega] = i;
            t_nega++;
            v[i] = n_gran;
        }
    }
    // getting the sorted features.
    double *sorted_f = malloc(sizeof(double) * (n * p));
    int *sorted_ind = malloc(sizeof(int) * (n * p));
    double *eps_f = malloc(sizeof(double) * p);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < n; j++) {
            sorted_f[i * n + j] = x_tr[i + j * p];
        }
        double *cur_f = sorted_f + i * n;
        int *cur_ind = sorted_ind + i * n;
        arg_sort_descend_modify(cur_f, cur_ind, n);
        double cur_min_eps = INFINITY;
        for (int j = 0; j < n - 1; j++) {
            cur_min_eps = fmin(cur_min_eps, fabs(cur_f[j + 1] - cur_f[j]));
        }
        eps_f[i] = cur_min_eps;
        for (int j = 0; j < n; j++) {
            cur_f[j] -= eps_f[i] / 2.;
        }
    }
    // implementing the weak learner by using the third method.
    double v_posi, v_nega;
    double *d = malloc(sizeof(double) * n);
    double *pi = malloc(sizeof(double) * n);
    for (int tt = 0; tt < para_t; tt++) {
        // check the Equation (6) in [1]
        v_posi = 0.0, v_nega = 0.0;
        for (int i = 0; i < t_posi; i++) {
            v_posi += v[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            v_nega += v[nega_inds[j]];
        }
        // calculate d in Equation (8) and at the same time
        // calculate pi, in our binary classification case,
        // we can combine Equation (8) and (10), then
        // pi = d * s. shown in Page 945 of [1]
        // at the initial stage, r is the sum of pi.
        for (int i = 0; i < t_posi; i++) {
            d[posi_inds[i]] = v[posi_inds[i]] * v_nega;
            pi[posi_inds[i]] = -d[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            d[nega_inds[j]] = v[nega_inds[j]] * v_posi;
            pi[nega_inds[j]] = d[nega_inds[j]];
        }
        // Go to Figure 3 of Page 947 to learn the
        // weak learner: it is uniquely defined by
        // i_opt, q_def_opt, and theta_opt
        // TODO, this can improved by only considering
        // different features.
        double l, r_opt = 0.0, theta_opt = INFINITY;
        int i_opt = 0, q_def_opt = 0;
        for (int pp = 0; pp < p; pp++) {
            l = 0.0; // r has been calculated.
            int *cur_ind = sorted_ind + pp * n;
            double prev_f = x_tr[pp + cur_ind[0] * p];
            for (int i = 0; i < n - 1; i++) {
                // to handle the same values.
                if (prev_f == x_tr[pp + cur_ind[i + 1] * p]) {
                    prev_f = x_tr[pp + cur_ind[i + 1] * p];
                    l += pi[cur_ind[i]];
                    continue;
                }
                prev_f = x_tr[pp + cur_ind[i + 1] * p];
                l += pi[cur_ind[i]];
                if (fabs(l) > fabs(r_opt)) {
                    r_opt = l;
                    i_opt = pp;
                    theta_opt = prev_f;
                    q_def_opt = 0;
                }
            }
        }
        re_theta[tt] = theta_opt;
        re_i[tt] = i_opt;
        re_q_def[tt] = q_def_opt;
        // Go back to Figure 2 of RankBoost.B
        if (fabs(fabs(r_opt) - 1.0) < 1e-10) {
            r_opt = (r_opt > 0 ? 1. : -1.) * (1. - 1e-10);
        }
        double alpha_t = .5 * log((1. + r_opt) / (1. - r_opt));
        re_alpha[tt] = alpha_t;
        // to update the v vector.
        double z_posi = 0.0, z_nega = 0.0;
        for (int i = 0; i < t_posi; i++) {
            double hi = x_tr[posi_inds[i] * p + i_opt] > theta_opt ? 1.0 : 0.0;
            v[posi_inds[i]] *= exp(alpha_t * hi);
            z_posi += v[posi_inds[i]];
        }
        for (int j = 0; j < t_nega; j++) {
            double hi = x_tr[nega_inds[j] * p + i_opt] > theta_opt ? 1.0 : 0.0;
            v[nega_inds[j]] *= exp(-alpha_t * hi);
            z_nega += v[nega_inds[j]];
        }
        if (z_posi != 0.0) {
            for (int i = 0; i < t_posi; i++) {
                v[posi_inds[i]] /= z_posi;
            }
        }
        if (z_nega != 0.0) {
            for (int j = 0; j < t_nega; j++) {
                v[nega_inds[j]] /= z_nega;
            }
        }
        re_alpha[tt] = -re_alpha[tt];
    }
    free(sorted_ind);
    free(sorted_f);
    free(eps_f);
    free(d);
    free(pi);
    free(v);
    free(posi_inds);
    free(nega_inds);
    return 1;
}


bool decision_function(const double *x_te, int n, int p, int t,
                       const double *alpha, const double *theta,
                       const int *q_def, const int *i,
                       double *re_scores) {
    for (int ii = 0; ii < n; ii++) {
        const double *cur_x = x_te + ii * p;
        re_scores[ii] = 0.0;
        for (int jj = 0; jj < t; jj++) {
            double ht = cur_x[i[jj]] > theta[jj] ? 1.0 : 0.0;
            re_scores[ii] += alpha[jj] * ht;
        }
    }
    return true;
}

static PyObject *wrap_rank_boost(PyObject *self, PyObject *args) {
    if (self == NULL) { printf("some error!"); }
    double start_time = clock();
    PyArrayObject *x_tr, *y_tr;
    int para_t;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &x_tr,
                          &PyArray_Type, &y_tr, &para_t)) { return NULL; }
    int n = x_tr->dimensions[0];
    int d = x_tr->dimensions[1];
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost((double *) PyArray_DATA(x_tr),
                    (double *) PyArray_DATA(y_tr),
                    n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
    PyObject *results = PyTuple_New(5);
    PyObject *alpha = PyList_New(para_t);
    PyObject *theta = PyList_New(para_t);
    PyObject *feat = PyList_New(para_t);
    PyObject *q_def = PyList_New(para_t);
    for (int i = 0; i < para_t; i++) {
        PyList_SetItem(alpha, i, PyFloat_FromDouble(re_alpha[i]));
        PyList_SetItem(theta, i, PyFloat_FromDouble(re_theta[i]));
        PyList_SetItem(feat, i, PyFloat_FromDouble(re_i[i]));
        PyList_SetItem(q_def, i, PyFloat_FromDouble(re_q_def[i]));
    }
    double run_time = (clock() - start_time) / CLOCKS_PER_SEC;
    PyTuple_SetItem(results, 0, alpha);
    PyTuple_SetItem(results, 1, theta);
    PyTuple_SetItem(results, 2, feat);
    PyTuple_SetItem(results, 3, q_def);
    PyTuple_SetItem(results, 4, PyFloat_FromDouble(run_time));

    free(re_alpha);
    free(re_theta);
    free(re_q_def);
    free(re_i);

    return results;
}


void test_cases_0() {
    /**
     * test some duplicated points
     */
    int n = 7, d = 2;
    double x_tr[14] = {2.5, 0.5,
                       3.5, 0.5,
                       4.5, 1.5,
                       0.5, 1.5,
                       0.5, 2.5,
                       1.5, 2.5,
                       2.5, 2.5};
    double y_tr[7] = {1., 1., 1., -1., -1., -1., -1.};
    int para_t = 5;
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost(x_tr, y_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
    double *scores = malloc(sizeof(double) * n);
    decision_function(x_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i, scores);
    for (int i = 0; i < n; i++) {
        printf("y[%d]: %.5f score[%d]: %.5f\n", i, y_tr[i], i, scores[i]);
    }
    double auc = roc_auc_score(y_tr, scores, n);
    printf("final auc: %.4f\n", auc);
    free(re_alpha);
    free(re_theta);
    free(re_q_def);
    free(re_i);
}


void test_cases_1() {
    /**
     * test some duplicated points
     */
    int n = 10, d = 2;
    double re_wt[2], re_auc;
    double x_tr[20] = {0.0, 0.0,
                       1.0, 1.0,
                       0.1, 0.3,
                       0.6, 0.9,
                       0.2, 0.7,
                       0.1, 0.8,
                       0.1, 0.8,
                       0.1, 0.8,
                       0.1, 0.8,
                       0.1, 0.8};
    double y_tr[10] = {-1, -1., 1., 1., -1., 1., 1., 1., 1., 1.};
    int para_t = 3;
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost(x_tr, y_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
    double *scores = malloc(sizeof(double) * n);
    decision_function(x_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i, scores);
    for (int i = 0; i < n; i++) {
        printf("y[%d]: %.5f score[%d]: %.5f\n", i, y_tr[i], i, scores[i]);
    }
    double auc = roc_auc_score(y_tr, scores, n);
    printf("final auc: %.4f\n", auc);
    free(re_alpha);
    free(re_theta);
    free(re_q_def);
    free(re_i);
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
    int para_t = 10;
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost(x_tr, y_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
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
    int para_t = 10;
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost(x_tr, y_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
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
    int para_t = 10;
    double *re_alpha = malloc(sizeof(double) * para_t);
    double *re_theta = malloc(sizeof(double) * para_t);
    int *re_q_def = malloc(sizeof(int) * para_t);
    int *re_i = malloc(sizeof(int) * para_t);
    algo_rank_boost(x_tr, y_tr, n, d, para_t, re_alpha, re_theta, re_q_def, re_i);
    printf("optimal auc: %.6f -- w: [ %.8f %.8f]\n",
           re_auc, re_wt[0], re_wt[1]);
}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        // pass
    } else {
        test_cases_0();
    }
    return 0;
}

static PyMethodDef list_methods[] = {
        {"c_rank_boost", (PyCFunction) wrap_rank_boost, METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};

// This is the interface for Python-3.7
#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "rank_boost_3",
        "This is a module", -1, list_methods,      /* m_methods */
        NULL, NULL, NULL, NULL,};

PyMODINIT_FUNC PyInit_librank_boost_3(void) {
    Py_Initialize();
    import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}

#else
// define your own python-2.7 interface here
#endif