//
// Created by --- on ---.
//

#ifndef AUC_LOGISTIC_ALGO_OPT_AUC_H
#define AUC_LOGISTIC_ALGO_OPT_AUC_H

void roc_curve(const double *true_labels, const double *scores,
               int n, double *tpr, double *fpr, double *auc);

double roc_auc_score(const double *t_labels,
                     const double *scores, int len);

int algo_opt_auc(const double *x_tr, const double *y_tr,
                 int n, int d, double para_eps,
                 double *re_wt, double *re_auc, int para_verbose);

#endif //AUC_LOGISTIC_ALGO_OPT_AUC_H
