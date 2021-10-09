#!/bin/bash
ROOT="/config/your/path"
PYTHON_PATH="${ROOT}python-3.7/include/python3.7m/"
NUMPY_PATH="${ROOT}python-3.7/lib/python3.7/site-packages/numpy/core/include/"
PYTHON_LIB="${ROOT}python-3.7/lib/"
FLAGS="-g -shared  -Wall -fPIC -std=c11 -O3"
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH}"
gcc ${FLAGS} ${INCLUDE} -L${PYTHON_LIB} algo_opt_auc.c algo_opt_auc.h -o libopt_auc_3.so -lm
gcc ${FLAGS} ${INCLUDE} -L${PYTHON_LIB} algo_spam.c algo_spam.h -o libspam_l2.so -lm
gcc ${FLAGS} ${INCLUDE} -L${PYTHON_LIB} algo_spauc.c algo_spauc.h -o libspauc_l2.so -lm
gcc ${FLAGS} ${INCLUDE} -L${PYTHON_LIB} algo_rank_boost.c algo_rank_boost.h  -o librank_boost_3.so -lm
