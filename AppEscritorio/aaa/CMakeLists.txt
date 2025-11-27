cmake_minimum_required(VERSION 3.16)
project(AppIMA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(main 
    main.cpp 
    ImageProcessor.cpp
)

target_link_libraries(main ${OpenCV_LIBS} ${ITK_LIBRARIES})
