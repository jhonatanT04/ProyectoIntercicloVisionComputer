#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// ITK
#include <itkImage.h>
using namespace cv;
class ImageProcessor {
public:
    using PixelType = signed short;
    using ImageType = itk::Image<PixelType, 2>;

    ImageProcessor();
    ~ImageProcessor();

    bool loadImage(const std::string& filePath);
    Mat deteccionHuesos(int a, int b);
    Mat deteccionPulmones(int a, int b);
    Mat deteccionMuscular(int a, int b);
    Mat imgEcualizada();
    // Opcionalmente puedes agregar getters
    const Mat& getOriginalImage() const { return m_originalImage; }
    const Mat& getRawImage() const { return m_rawImage; }

    void analyzeRawImage();

private:
    Mat m_originalImage;  // Imagen lista para visualizar (8 bits)
    Mat m_rawImage;       // Imagen cruda cargada (16 bits)
};

