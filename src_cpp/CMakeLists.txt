cmake_minimum_required(VERSION 3.14)
project(battlefield_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 include directory via Python
execute_process(
    COMMAND python -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Find Python include & library directory
execute_process(
    COMMAND python -c "import sysconfig; print(sysconfig.get_path('include'))"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

# Find OpenCV via CMake or Python cv2 package location
find_package(OpenCV QUIET)

if(NOT OpenCV_FOUND)
    execute_process(
        COMMAND python -c "import cv2, os; print(os.path.dirname(cv2.__file__))"
        OUTPUT_VARIABLE OPENCV_PYTHON_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(OPENCV_PYTHON_DIR)
        set(OpenCV_INCLUDE_DIRS "${OPENCV_PYTHON_DIR}/include")
        file(GLOB OpenCV_LIBS "${OPENCV_PYTHON_DIR}/*.lib")
    endif()
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PYBIND11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
)

if(OpenCV_INCLUDE_DIRS)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Build C++ Zero-Copy pybind11 module
add_library(battlefield_core SHARED
    src/pybind_bridge.cpp
    src/video_ingestor.cpp
    src/anomaly_calculus.cpp
)

set_target_properties(battlefield_core PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../src_python"
)

if(OpenCV_LIBS)
    target_link_libraries(battlefield_core PRIVATE ${OpenCV_LIBS})
endif()
