cmake_minimum_required(VERSION 3.10)
project(ReadDICOM)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${ITK_LIBRARIES} ${OpenCV_LIBS})
