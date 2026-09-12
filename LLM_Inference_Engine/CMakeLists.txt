cmake_minimum_required(VERSION 3.8)
project(LLM_Inference_Engine)

find_package(Eigen3 REQUIRED)

add_executable(main main.cpp)
