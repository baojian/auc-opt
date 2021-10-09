cmake_minimum_required(VERSION 3.12)
project(auc-logistic C)
include_directories(auc-logistic/src/)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(ROOT "/network/rit/lab/ceashpc/bz383376/opt/anaconda3/envs/")
set(ROOT "/home/baojian/anaconda3/envs/")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython3.7m -pthread")

set(NUMPY_INCLUDE "${ROOT}python-3.7/lib/python3.7/site-packages/numpy/core/include")
set(PYTHON_LIBRARY "${ROOT}python-3.7/lib/")
set(PYTHON_INCLUDE "${ROOT}python-3.7/include/python3.7m/")

set(NUMPY_INCLUDE2 "/home/baojian/anaconda3/envs/python-2.7/lib/python2.7/site-packages/numpy/core/include")
set(PYTHON_LIBRARY2 "/home/baojian/anaconda3/envs/python-2.7/lib/")
set(PYTHON_INCLUDE2 "/home/baojian/anaconda3/envs/python-2.7/include/python2.7/")

find_package(PythonLibs 3.7 REQUIRED)
include_directories(${PYTHON_INCLUDE})

set(MAIN_FILE src/main.c)

add_executable(test_opt_auc src/algo_opt_auc.c src/algo_opt_auc.h)
target_link_libraries(test_opt_auc -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_library(opt_auc_3 SHARED src/algo_opt_auc.c src/algo_opt_auc.h)
target_link_libraries(opt_auc_3 -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_executable(test_rank_boost src/algo_rank_boost.c src/algo_rank_boost.h)
target_link_libraries(test_rank_boost -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_library(rank_boost_3 SHARED src/algo_rank_boost.c src/algo_rank_boost.h)
target_link_libraries(rank_boost_3 -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_executable(test_main src/main.c)
target_link_libraries(test_main -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_library(test_module_3 SHARED src/main.c)
target_link_libraries(test_module_3 PUBLIC -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})

add_library(test_module_2 SHARED src/main.c)
target_link_libraries(test_module_2 PUBLIC -std=c11 -Wall -Wextra -O3 -lm -lpython2.7 -lpthread
        -I${NUMPY_INCLUDE2} -I${PYTHON_INCLUDE2} -L${PYTHON_LIBRARY2})

add_library(auc_module SHARED src/main_wrapper.c
        src/algo_spam.c src/algo_spam.h src/algo.c src/algo.h src/algo_spauc.c src/algo_spauc.h)
target_link_libraries(auc_module -std=c11 -Wall -Wextra -O3 -lm -lpython3.7m -lpthread
        -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${PYTHON_LIBRARY})