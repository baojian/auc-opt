cmake_minimum_required(VERSION 3.15)
project(auc_logistic C)
include_directories(algo_wrapper/)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython3.7m -pthread")

set(NUMPY_INCLUDE "/home/baojian/anaconda3/envs/auc-logistic/lib/python3.7/site-packages/numpy/core/include")
set(PYTHON_LIBRARY "/home/baojian/anaconda3/envs/auc-logistic/lib/libpython3.7m.so")
set(PYTHON_INCLUDE "/home/baojian/anaconda3/envs/auc-logistic/include/python3.7m/")

find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(auc_logistic main.c algo_wrapper/algo_spam.c algo_wrapper/algo_spam.h
        algo_wrapper/algo.c algo_wrapper/algo.h algo_wrapper/main_wrapper.c
        algo_wrapper/algo_spauc.c algo_wrapper/algo_spauc.h)

add_library(sparse_module SHARED algo_wrapper/main_wrapper.c
        algo_wrapper/algo_spam.c algo_wrapper/algo_spam.h
        algo_wrapper/algo.c algo_wrapper/algo.h)
target_link_libraries(sparse_module python3.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -lm -lpython3.7m -lpthread)