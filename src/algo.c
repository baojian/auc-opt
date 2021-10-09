//
// Created by baojian on 2/20/20.
//

#include "algo.h"

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
