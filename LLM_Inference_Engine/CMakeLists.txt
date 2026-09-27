cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0141 NEW)
project(LLM_Inference_Engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(CMAKE_PREFIX_PATH "C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/Lib/site-packages/torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(include)
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG v0.2.2
)
FetchContent_MakeAvailable(sentencepiece)

add_executable(llm_engine
 src/tokenizer.cpp
)
target_include_directories(llm_engine PRIVATE
    "C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/Lib/site-packages/torch/include"
    "C:/Users/LENOVO/AppData/Local/Programs/Python/Python313/Lib/site-packages/torch/include/torch/csrc/api/include"
)

target_link_libraries(llm_engine PRIVATE
    sentencepiece::sentencepiece
    "${TORCH_LIBRARIES}"
)



