cmake_minimum_required(VERSION 3.14)
project(LLM_Inference_Engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


include_directories(include)
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG master
)
FetchContent_MakeAvailable(sentencepiece)

add_executable(llm_engine
 src/tokenizer.cpp
)
target_link_libraries(llm_engine PRIVATE sentencepiece::sentencepiece)



