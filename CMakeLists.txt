cmake_minimum_required(VERSION 3.10)
project(
  zester-smart-traffic
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OPEN CV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# enable cuda if needed
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.15.1)

if (WIN32)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (LINUX)
    if (USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")
    else ()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}")
    endif ()
elseif (APPLE)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}")
    # Apple X64 binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-x64-${ONNXRUNTIME_VERSION}")
    # Apple Universal binary
    # set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}")
else ()
    message(SEND_ERROR "Variable ONNXRUNTIME_ROOT is not set properly. Please check if your cmake project \
    is not compiled with `-D WIN32=TRUE`, `-D LINUX=TRUE`, or `-D APPLE=TRUE`!")
endif ()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)

# for the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(src)

set(PROJECT_SOURCES
        src/main.cpp
        src/inference.h
        src/inference.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

find_package(SFML 3 REQUIRED COMPONENTS Graphics Audio Network)

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib SFML::Graphics SFML::Audio SFML::Network)
endif ()

# For windows system, copy onnxruntime.dll to the same folder of the executable file
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

# Copy assets to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/assets
	 DESTINATION ${CMAKE_BINARY_DIR}
	 FILES_MATCHING PATTERN "*")

add_definitions(-DASSETS="${CMAKE_BINARY_DIR}/assets")
