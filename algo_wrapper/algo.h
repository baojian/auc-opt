//
// Created by baojian on 2/20/20.
//

#ifndef AUC_LOGISTIC_ALGO_H
#define AUC_LOGISTIC_ALGO_H

#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

#define PI 3.14159265358979323846
#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }
#define is_posi(x) ( x > 0.0 ? 1.0 : 0.0)
#define is_nega(x) ( x < 0.0 ? 1.0 : 0.0)


typedef struct {
    double val;
    int index;
} data_pair;

typedef struct {
    double *wt;
    double *wt_prev;
    double *aucs;
    double *rts;
    int auc_len; // how many auc evaluated.
    int total_iterations; // total iterations
    int total_epochs; // total epochs executed.
} AlgoResults;

typedef struct {
    int num_passes;
    int verbose;
    int step_len;
    int record_aucs;
    double stop_eps;
} GlobalParas;

typedef struct {
    const double *x_tr_vals;
    const int *x_tr_inds;
    const int *x_tr_poss;
    const int *x_tr_lens;
    const double *y_tr;
    bool is_sparse;
    int n;
    int p;
} Data;

static inline int _comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len);

double _auc_score(const double *true_labels, const double *scores, int len);

void _evaluate_aucs(Data *data, double *y_pred, AlgoResults *re, double start_time);

#endif //AUC_LOGISTIC_ALGO_H
