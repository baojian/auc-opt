//
// Created by baojian on 2/21/20.
//

#include "algo_spauc.h"


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