#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>
#include <itkImageRegionConstIterator.h>

#include <iostream>
#include <filesystem>
#include <string>
#include "CTProcessorSimple.h"
#include <filesystem>
namespace fs = std::filesystem;

using PixelType = signed short;
using ImageType = itk::Image<PixelType, 2>;

// Constructor
CTImageProcessor::CTImageProcessor(const std::string& outputFolder) 
    : m_outputFolder(outputFolder), m_imageCounter(0) 
{
    fs::create_directories(outputFolder);
    std::cout << "Output folder: " << outputFolder << std::endl;
}

// ============ CARGA DE IMAGEN ============
bool CTImageProcessor::loadImage(const std::string& filePath) {
    std::cout << "Loading: " << filePath << std::endl;
    
    // Detectar si es DICOM/IMA o imagen estándar
    std::string ext = fs::path(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".ima" || ext == ".dcm") {
        // Cargar con ITK
        try {
            using ReaderType = itk::ImageFileReader<ImageType>;
            using ImageIOType = itk::GDCMImageIO;
            
            auto dicomIO = ImageIOType::New();
            auto reader = ReaderType::New();
            reader->SetFileName(filePath);
            reader->SetImageIO(dicomIO);
            reader->Update();
            
            auto itkImage = reader->GetOutput();
            auto region = itkImage->GetLargestPossibleRegion();
            auto size = region.GetSize();
            
            int width = size[0];
            int height = size[1];
            
            m_rawImage = cv::Mat(height, width, CV_16SC1);
            
            itk::ImageRegionConstIterator<ImageType> it(itkImage, region);
            int idx = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
                int y = idx / width;
                int x = idx % width;
                m_rawImage.at<short>(y, x) = it.Get();
            }
            
            // Convertir a 8 bits para visualización
            cv::normalize(m_rawImage, m_originalImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
            
            std::cout << "DICOM/IMA loaded: " << width << "x" << height << std::endl;
            return true;
            
        } catch (const itk::ExceptionObject& ex) {
            std::cerr << "ITK Error: " << ex << std::endl;
            return false;
        }
    } else {
        // Cargar imagen estándar con OpenCV
        m_originalImage = cv::imread(filePath, cv::IMREAD_GRAYSCALE);
        if (m_originalImage.empty()) {
            std::cerr << "Error: Could not load image with OpenCV" << std::endl;
            return false;
        }
        
        // Convertir a 16 bits para compatibilidad
        m_originalImage.convertTo(m_rawImage, CV_16SC1);
        
        std::cout << "Standard image loaded: " << m_originalImage.cols 
                  << "x" << m_originalImage.rows << std::endl;
        return true;
    }
}

// ============ GUARDAR IMAGEN ============
void CTImageProcessor::saveImage(const cv::Mat& img, const std::string& stepName) {
    m_imageCounter++;
    std::string filename = m_outputFolder + "/" + 
                          std::to_string(m_imageCounter) + "_" + 
                          stepName + ".png";
    
    cv::Mat toSave;
    if (img.depth() != CV_8U) {
        cv::normalize(img, toSave, 0, 255, cv::NORM_MINMAX);
        toSave.convertTo(toSave, CV_8UC1);
    } else {
        toSave = img;
    }
    
    cv::imwrite(filename, toSave);
    std::cout << "  Saved: " << filename << std::endl;
}

// ============ WINDOW/LEVEL ============
cv::Mat CTImageProcessor::applyWindowLevel(int center, int width) {
    if (m_rawImage.empty()) return m_originalImage.clone();
    
    cv::Mat output;
    double minVal = center - width / 2.0;
    double maxVal = center + width / 2.0;
    
    cv::Mat floatImg;
    m_rawImage.convertTo(floatImg, CV_64F);
    
    floatImg = (floatImg - minVal) / (maxVal - minVal) * 255.0;
    cv::threshold(floatImg, floatImg, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(floatImg, floatImg, 0, 0, cv::THRESH_TOZERO);
    
    floatImg.convertTo(output, CV_8UC1);
    return output;
}

// ============ NORMALIZACIÓN Y CONTRASTE ============
cv::Mat CTImageProcessor::normalize(const cv::Mat& input) {
    cv::Mat output;
    cv::normalize(input, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return output;
}

cv::Mat CTImageProcessor::contrastStretching(const cv::Mat& input) {
    return normalize(input);
}

cv::Mat CTImageProcessor::applyCLAHE(const cv::Mat& input, double clipLimit) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    auto clahe = cv::createCLAHE(clipLimit, cv::Size(8, 8));
    clahe->apply(gray, output);
    return output;
}

cv::Mat CTImageProcessor::histogramEqualization(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::equalizeHist(gray, output);
    return output;
}

// ============ THRESHOLDING ============
cv::Mat CTImageProcessor::threshold(const cv::Mat& input, int threshValue) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::threshold(gray, output, threshValue, 255, cv::THRESH_BINARY);
    return output;
}

cv::Mat CTImageProcessor::thresholdOtsu(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::threshold(gray, output, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return output;
}

cv::Mat CTImageProcessor::thresholdAdaptive(const cv::Mat& input, int blockSize) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    if (blockSize % 2 == 0) blockSize++;  // Asegurar que es impar
    cv::adaptiveThreshold(gray, output, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, blockSize, 2);
    return output;
}

// ============ OPERACIONES LÓGICAS ============
cv::Mat CTImageProcessor::applyNOT(const cv::Mat& input) {
    cv::Mat output;
    cv::bitwise_not(input, output);
    return output;
}

cv::Mat CTImageProcessor::applyAND(const cv::Mat& input1, const cv::Mat& input2) {
    cv::Mat output;
    cv::bitwise_and(input1, input2, output);
    return output;
}

cv::Mat CTImageProcessor::applyOR(const cv::Mat& input1, const cv::Mat& input2) {
    cv::Mat output;
    cv::bitwise_or(input1, input2, output);
    return output;
}

cv::Mat CTImageProcessor::applyXOR(const cv::Mat& input1, const cv::Mat& input2) {
    cv::Mat output;
    cv::bitwise_xor(input1, input2, output);
    return output;
}

// ============ DETECCIÓN DE BORDES ============
cv::Mat CTImageProcessor::edgeCanny(const cv::Mat& input, int low, int high) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.4);
    cv::Canny(gray, output, low, high);
    return output;
}

cv::Mat CTImageProcessor::edgeSobel(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);
    
    cv::Mat absGradX, absGradY;
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
    return output;
}

cv::Mat CTImageProcessor::edgeLaplacian(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_16S, 3);
    cv::convertScaleAbs(laplacian, output);
    return output;
}

// ============ FILTROS DE SUAVIZADO ============
cv::Mat CTImageProcessor::filterGaussian(const cv::Mat& input, int ksize) {
    if (ksize % 2 == 0) ksize++;  // Asegurar impar
    cv::Mat output;
    cv::GaussianBlur(input, output, cv::Size(ksize, ksize), 0);
    return output;
}

cv::Mat CTImageProcessor::filterMedian(const cv::Mat& input, int ksize) {
    if (ksize % 2 == 0) ksize++;  // Asegurar impar
    cv::Mat output;
    cv::medianBlur(input, output, ksize);
    return output;
}

cv::Mat CTImageProcessor::filterBilateral(const cv::Mat& input, int d) {
    cv::Mat output, gray;
    if (input.depth() != CV_8U) input.convertTo(gray, CV_8UC1);
    else gray = input.clone();
    
    cv::bilateralFilter(gray, output, d, 75, 75);
    return output;
}

cv::Mat CTImageProcessor::filterMean(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::blur(input, output, cv::Size(ksize, ksize));
    return output;
}

cv::Mat CTImageProcessor::filterNLMeans(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::fastNlMeansDenoising(gray, output, 10, 7, 21);
    return output;
}

// ============ MORFOLOGÍA ============
cv::Mat CTImageProcessor::morphErosion(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::erode(input, output, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphDilation(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::dilate(input, output, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphOpening(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(input, output, cv::MORPH_OPEN, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphClosing(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(input, output, cv::MORPH_CLOSE, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphGradient(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(input, output, cv::MORPH_GRADIENT, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphTopHat(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(input, output, cv::MORPH_TOPHAT, kernel);
    return output;
}

cv::Mat CTImageProcessor::morphBlackHat(const cv::Mat& input, int ksize) {
    cv::Mat output;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
    cv::morphologyEx(input, output, cv::MORPH_BLACKHAT, kernel);
    return output;
}

// ============ VISUALIZACIÓN ============
cv::Mat CTImageProcessor::createColorOverlay(const cv::Mat& original, const cv::Mat& mask, 
                                              cv::Scalar color, double alpha) {
    cv::Mat output, bgr;
    if (original.channels() == 1) cv::cvtColor(original, bgr, cv::COLOR_GRAY2BGR);
    else bgr = original.clone();
    
    cv::Mat overlay = bgr.clone();
    overlay.setTo(color, mask);
    cv::addWeighted(bgr, 1 - alpha, overlay, alpha, 0, output);
    return output;
}

cv::Mat CTImageProcessor::createHeatmap(const cv::Mat& input) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::applyColorMap(gray, output, cv::COLORMAP_JET);
    return output;
}

cv::Mat CTImageProcessor::segmentByIntensity(const cv::Mat& input, int lower, int upper) {
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    cv::inRange(gray, cv::Scalar(lower), cv::Scalar(upper), output);
    return output;
}

// ============ RESALTAR REGIÓN (NUEVO) ============
cv::Mat CTImageProcessor::highlightRegion(const cv::Mat& mask, const cv::Mat& background) {
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat maskCopy = mask.clone();
    cv::findContours(maskCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    cv::Mat output;
    if (background.empty()) {
        if (mask.channels() == 1) {
            cv::cvtColor(mask, output, cv::COLOR_GRAY2BGR);
        } else {
            output = mask.clone();
        }
    } else {
        if (background.channels() == 1) {
            cv::cvtColor(background, output, cv::COLOR_GRAY2BGR);
        } else {
            output = background.clone();
        }
    }
    
    // Dibujar contornos amarillos
    cv::drawContours(output, contours, -1, cv::Scalar(0, 255, 255), 2);
    
    // Dibujar rectángulos y etiquetas
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect boundRect = cv::boundingRect(contours[i]);
        double area = cv::contourArea(contours[i]);
        
        // Filtrar contornos muy pequeños
        if (area > 100) {
            cv::rectangle(output, boundRect, cv::Scalar(0, 255, 0), 2);
            
            std::string label = "Area: " + std::to_string((int)area);
            cv::putText(output, label, cv::Point(boundRect.x, boundRect.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
    }
    
    return output;
}

// ============ DEEP LEARNING - DENOISING DNN ============
cv::Mat CTImageProcessor::applyDenoisingDNN(const cv::Mat& input) {
    // Implementación simple usando fastNlMeansDenoising como fallback
    // Para usar DNN real, necesitarías cargar un modelo pre-entrenado
    
    cv::Mat output, gray;
    if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    // Denoising avanzado con múltiples pasadas
    cv::Mat temp;
    cv::fastNlMeansDenoising(gray, temp, 15, 7, 21);
    cv::bilateralFilter(temp, output, 9, 75, 75);
    
    return output;
}

// ============ PIPELINE COMPLETO ============
void CTImageProcessor::processComplete() {
    std::cout << "\n=== Iniciando procesamiento completo ===\n\n";
    // ... (mantener el código del CTProcessorSimple.cpp original)
}