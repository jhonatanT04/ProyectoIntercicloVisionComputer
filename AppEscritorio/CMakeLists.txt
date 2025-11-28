cmake_minimum_required(VERSION 3.16)
project(AppEscritorio VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ---------- RUTA QT6 PARA ARCH LINUX ----------
set(CMAKE_PREFIX_PATH "/usr/lib/cmake/Qt6")

# ---------- Qt6 ----------
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# ---------- OpenCV ----------
find_package(OpenCV REQUIRED)

# ---------- ITK ----------
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# ---------- FUENTES ----------
set(SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    pipelinedialog.cpp  
    pipelinedialog.h
    ProcesadoIMGc/CTProcessorSimple.cpp
    aaa/ImageProcessor.cpp
)

add_executable(AppEscritorio ${SOURCES})

target_link_libraries(AppEscritorio
    Qt6::Widgets
    ${OpenCV_LIBS}
    ${ITK_LIBRARIES}
)
