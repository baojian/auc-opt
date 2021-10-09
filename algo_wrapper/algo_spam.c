//
// Created by baojian on 2/20/20.
//

#include "algo_spam.h"

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