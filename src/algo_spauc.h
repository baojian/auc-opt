//
// Created by --- on ---.
//

#ifndef AUC_LOGISTIC_ALGO_SPAUC_H
#define AUC_LOGISTIC_ALGO_SPAUC_H


#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

#define sign(x) (x > 0) - (x < 0)

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


void _algo_spauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_mu,
                 double para_l1, double para_l2);

#endif //AUC_LOGISTIC_ALGO_SPAUC_H
