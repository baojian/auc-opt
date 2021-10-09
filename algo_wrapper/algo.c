//
// Created by baojian on 2/20/20.
//

#include "algo.h"

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
    for (int i = 0; i < len; i++) {
        tpr[i] = tpr[i] / num_posi;
        fpr[i] = fpr[i] / num_nega;
    }
    // AUC score
    double auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < len; i++) {
        auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
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
