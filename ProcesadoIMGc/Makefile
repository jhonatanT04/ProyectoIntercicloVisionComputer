# Nombre del ejecutable
TARGET = vision.bin

# Compilador
CXX = g++
CXXFLAGS = -std=c++17

# Include paths
INCLUDES = -I/usr/include/ITK-5.3 \
           -I/usr/local/include/opencv4/

# Librerías
LIBDIRS = -L/usr/lib/x86_64-linux-gnu \
          -L/usr/local/lib/

LIBS = \
    -lITKCommon-5.3 \
    -litkvnl-5.3 \
    -litkvnl_algo-5.3 \
    -litkv3p_netlib-5.3 \
    -litkvcl-5.3 \
    -lITKVNLInstantiation-5.3 \
    -lITKIOImageBase-5.3 \
    -lITKIOGDCM-5.3 \
    -lITKIOJPEG-5.3 \
    -lITKIOPNG-5.3 \
    -lITKIONIFTI-5.3 \
    -litksys-5.3 \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -lopencv_video \
    -lopencv_videoio

# Archivos fuente
SRC = main.cpp

# Regla principal
all:
	$(CXX) $(SRC) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) $(LIBS) -o $(TARGET)

# Regla para ejecutar
run: all
	./$(TARGET)

# Limpiar binarios
clean:
	rm -f $(TARGET)
