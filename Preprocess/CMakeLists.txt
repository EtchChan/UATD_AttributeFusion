cmake_minimum_required(VERSION 3.25)
project(Preprocess)

set(CMAKE_CXX_STANDARD 23)
set(OpenCV_DIR "D:/Tools/opencv/mingw-build/install")

include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

add_executable(Preprocess FusingImagesDisplay.cpp)

target_link_libraries(Preprocess ${OpenCV_LIBS})
