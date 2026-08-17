cmake_minimum_required(VERSION 3.18)
project(FineLocationCpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to the extracted ONNX Runtime package")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to an extracted ONNX Runtime C/C++ package")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64"
    NO_DEFAULT_PATH REQUIRED)

add_library(fineloc STATIC src/fineloc.cpp)
target_include_directories(fineloc
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(fineloc PUBLIC "${ONNXRUNTIME_LIBRARY}")

add_executable(fineloc_cli src/main.cpp)
target_link_libraries(fineloc_cli PRIVATE fineloc)

if(MSVC)
    target_compile_options(fineloc PRIVATE /W4 /permissive-)
else()
    target_compile_options(fineloc PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    add_custom_command(TARGET fineloc_cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:fineloc_cli>)
endif()

install(TARGETS fineloc fineloc_cli ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
