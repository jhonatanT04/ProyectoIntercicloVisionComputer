// CTProcessorSimple.cpp
// Procesador de imagen CT individual (archivo IMA/DICOM)
// Compila: g++ -o ct_process CTProcessorSimple.cpp `pkg-config --cflags --libs opencv4` -lITKCommon -lITKIOGDCM -lITKIOImageBase -litkgdcmDICT -litkgdcmMSFF -std=c++17

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// Tipos ITK
using PixelType = signed short;
using ImageType = itk::Image<PixelType, 2>;
using OutputPixelType = unsigned char;
using OutputImageType = itk::Image<OutputPixelType, 2>;

class CTImageProcessor {
private:
    cv::Mat m_originalImage;      // Imagen original en formato OpenCV
    cv::Mat m_rawImage;           // Imagen raw (16 bits)
    std::string m_outputFolder;
    int m_imageCounter;

public:
    CTImageProcessor(const std::string& outputFolder) 
        : m_outputFolder(outputFolder), m_imageCounter(0) {
        // Crear carpeta de salida
        fs::create_directories(outputFolder);
        std::cout << "Output folder: " << outputFolder << std::endl;
    }

    // Cargar archivo IMA/DICOM
    bool loadImage(const std::string& filePath) {
        std::cout << "Loading: " << filePath << std::endl;
        
        try {
            using ReaderType = itk::ImageFileReader<ImageType>;
            using ImageIOType = itk::GDCMImageIO;
            
            auto dicomIO = ImageIOType::New();
            auto reader = ReaderType::New();
            reader->SetFileName(filePath);
            reader->SetImageIO(dicomIO);
            reader->Update();
            
            // Convertir ITK a OpenCV
            auto itkImage = reader->GetOutput();
            auto region = itkImage->GetLargestPossibleRegion();
            auto size = region.GetSize();
            
            int width = size[0];
            int height = size[1];
            
            // Crear imagen de 16 bits
            m_rawImage = cv::Mat(height, width, CV_16SC1);
            
            itk::ImageRegionConstIterator<ImageType> it(itkImage, region);
            int idx = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
                int y = idx / width;
                int x = idx % width;
                m_rawImage.at<short>(y, x) = it.Get();
            }
            
            std::cout << "Image loaded: " << width << "x" << height << std::endl;
            return true;
            
        } catch (const itk::ExceptionObject& ex) {
            std::cerr << "ITK Error: " << ex << std::endl;
            return false;
        }
    }

    // Guardar imagen con nombre descriptivo
    void saveImage(const cv::Mat& img, const std::string& stepName) {
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

    // ===== FUNCIONES DE PROCESAMIENTO =====

    // 1. Window/Level (ajuste de ventana CT)
    cv::Mat applyWindowLevel(int center, int width) {
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

    // 2. Normalización
    cv::Mat normalize(const cv::Mat& input) {
        cv::Mat output;
        cv::normalize(input, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        return output;
    }

    // 3. Thresholding simple
    cv::Mat threshold(const cv::Mat& input, int threshValue) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::threshold(gray, output, threshValue, 255, cv::THRESH_BINARY);
        return output;
    }

    // 4. Threshold Otsu
    cv::Mat thresholdOtsu(const cv::Mat& input) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::threshold(gray, output, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        return output;
    }

    // 5. Threshold Adaptativo
    cv::Mat thresholdAdaptive(const cv::Mat& input, int blockSize = 11) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::adaptiveThreshold(gray, output, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, blockSize, 2);
        return output;
    }

    // 6. Contrast Stretching
    cv::Mat contrastStretching(const cv::Mat& input) {
        cv::Mat output;
        cv::normalize(input, output, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        return output;
    }

    // 7. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv::Mat applyCLAHE(const cv::Mat& input, double clipLimit = 2.0) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        auto clahe = cv::createCLAHE(clipLimit, cv::Size(8, 8));
        clahe->apply(gray, output);
        return output;
    }

    // 8. Histogram Equalization
    cv::Mat histogramEqualization(const cv::Mat& input) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::equalizeHist(gray, output);
        return output;
    }

    // 9. Operación NOT
    cv::Mat applyNOT(const cv::Mat& input) {
        cv::Mat output;
        cv::bitwise_not(input, output);
        return output;
    }

    // 10. Operación AND
    cv::Mat applyAND(const cv::Mat& input1, const cv::Mat& input2) {
        cv::Mat output;
        cv::bitwise_and(input1, input2, output);
        return output;
    }

    // 11. Operación OR
    cv::Mat applyOR(const cv::Mat& input1, const cv::Mat& input2) {
        cv::Mat output;
        cv::bitwise_or(input1, input2, output);
        return output;
    }

    // 12. Operación XOR
    cv::Mat applyXOR(const cv::Mat& input1, const cv::Mat& input2) {
        cv::Mat output;
        cv::bitwise_xor(input1, input2, output);
        return output;
    }

    // 13. Detección de bordes Canny
    cv::Mat edgeCanny(const cv::Mat& input, int low = 50, int high = 150) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.4);
        cv::Canny(gray, output, low, high);
        return output;
    }

    // 14. Detección de bordes Sobel
    cv::Mat edgeSobel(const cv::Mat& input) {
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

    // 15. Detección de bordes Laplacian
    cv::Mat edgeLaplacian(const cv::Mat& input) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_16S, 3);
        cv::convertScaleAbs(laplacian, output);
        return output;
    }

    // 16. Filtro Gaussiano
    cv::Mat filterGaussian(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::GaussianBlur(input, output, cv::Size(ksize, ksize), 0);
        return output;
    }

    // 17. Filtro Mediana
    cv::Mat filterMedian(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::medianBlur(input, output, ksize);
        return output;
    }

    // 18. Filtro Bilateral
    cv::Mat filterBilateral(const cv::Mat& input, int d = 9) {
        cv::Mat output, gray;
        if (input.depth() != CV_8U) input.convertTo(gray, CV_8UC1);
        else gray = input.clone();
        
        cv::bilateralFilter(gray, output, d, 75, 75);
        return output;
    }

    // 19. Filtro de Media
    cv::Mat filterMean(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::blur(input, output, cv::Size(ksize, ksize));
        return output;
    }

    // 20. NL-Means Denoising
    cv::Mat filterNLMeans(const cv::Mat& input) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::fastNlMeansDenoising(gray, output, 10, 7, 21);
        return output;
    }

    // 21. Erosión
    cv::Mat morphErosion(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::erode(input, output, kernel);
        return output;
    }

    // 22. Dilatación
    cv::Mat morphDilation(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::dilate(input, output, kernel);
        return output;
    }

    // 23. Opening
    cv::Mat morphOpening(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(input, output, cv::MORPH_OPEN, kernel);
        return output;
    }

    // 24. Closing
    cv::Mat morphClosing(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(input, output, cv::MORPH_CLOSE, kernel);
        return output;
    }

    // 25. Gradiente Morfológico
    cv::Mat morphGradient(const cv::Mat& input, int ksize = 5) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(input, output, cv::MORPH_GRADIENT, kernel);
        return output;
    }

    // 26. Top Hat
    cv::Mat morphTopHat(const cv::Mat& input, int ksize = 15) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(input, output, cv::MORPH_TOPHAT, kernel);
        return output;
    }

    // 27. Black Hat
    cv::Mat morphBlackHat(const cv::Mat& input, int ksize = 15) {
        cv::Mat output;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
        cv::morphologyEx(input, output, cv::MORPH_BLACKHAT, kernel);
        return output;
    }

    // 28. Crear overlay de color
    cv::Mat createColorOverlay(const cv::Mat& original, const cv::Mat& mask, 
                               cv::Scalar color, double alpha = 0.5) {
        cv::Mat output, bgr;
        if (original.channels() == 1) cv::cvtColor(original, bgr, cv::COLOR_GRAY2BGR);
        else bgr = original.clone();
        
        cv::Mat overlay = bgr.clone();
        overlay.setTo(color, mask);
        cv::addWeighted(bgr, 1 - alpha, overlay, alpha, 0, output);
        return output;
    }

    // 29. Crear heatmap
    cv::Mat createHeatmap(const cv::Mat& input) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::applyColorMap(gray, output, cv::COLORMAP_JET);
        return output;
    }

    // 30. Segmentación por intensidad
    cv::Mat segmentByIntensity(const cv::Mat& input, int lower, int upper) {
        cv::Mat output, gray;
        if (input.channels() > 1) cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else gray = input.clone();
        if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
        
        cv::inRange(gray, cv::Scalar(lower), cv::Scalar(upper), output);
        return output;
    }

    // ===== PIPELINE PRINCIPAL =====
    void processComplete() {
        std::cout << "\n=== Iniciando procesamiento completo ===\n\n";

        // 1. IMAGEN ORIGINAL CON DIFERENTES VENTANAS CT
        std::cout << "1. Aplicando Window/Level...\n";
        cv::Mat softTissue = applyWindowLevel(40, 400);   // Tejido blando
        cv::Mat lung = applyWindowLevel(-600, 1500);       // Pulmón
        cv::Mat bone = applyWindowLevel(400, 1800);        // Hueso
        
        saveImage(softTissue, "01_window_soft_tissue");
        saveImage(lung, "01_window_lung");
        saveImage(bone, "01_window_bone");
        
        // Usar tejido blando como base
        m_originalImage = softTissue.clone();

        // 2. NORMALIZACIÓN
        std::cout << "2. Normalizando imagen...\n";
        cv::Mat normalized = normalize(m_rawImage);
        saveImage(normalized, "02_normalized");

        // 3. MEJORA DE CONTRASTE
        std::cout << "3. Mejorando contraste...\n";
        cv::Mat contrastStr = contrastStretching(m_originalImage);
        cv::Mat histEq = histogramEqualization(m_originalImage);
        cv::Mat claheImg = applyCLAHE(m_originalImage);
        
        saveImage(contrastStr, "03_contrast_stretching");
        saveImage(histEq, "03_histogram_equalization");
        saveImage(claheImg, "03_CLAHE");

        // 4. FILTROS DE SUAVIZADO
        std::cout << "4. Aplicando filtros de suavizado...\n";
        cv::Mat gaussianF = filterGaussian(claheImg);
        cv::Mat medianF = filterMedian(claheImg);
        cv::Mat bilateralF = filterBilateral(claheImg);
        cv::Mat meanF = filterMean(claheImg);
        cv::Mat nlmeansF = filterNLMeans(claheImg);
        
        saveImage(gaussianF, "04_filter_gaussian");
        saveImage(medianF, "04_filter_median");
        saveImage(bilateralF, "04_filter_bilateral");
        saveImage(meanF, "04_filter_mean");
        saveImage(nlmeansF, "04_filter_nlmeans");

        // 5. THRESHOLDING
        std::cout << "5. Aplicando thresholding...\n";
        cv::Mat thresh128 = threshold(bilateralF, 128);
        cv::Mat threshOtsu = thresholdOtsu(bilateralF);
        cv::Mat threshAdapt = thresholdAdaptive(bilateralF);
        
        saveImage(thresh128, "05_threshold_128");
        saveImage(threshOtsu, "05_threshold_otsu");
        saveImage(threshAdapt, "05_threshold_adaptive");

        // 6. BINARIZACIÓN POR UMBRAL DE COLOR/INTENSIDAD
        std::cout << "6. Segmentación por intensidad...\n";
        cv::Mat segLow = segmentByIntensity(bilateralF, 0, 80);
        cv::Mat segMid = segmentByIntensity(bilateralF, 80, 180);
        cv::Mat segHigh = segmentByIntensity(bilateralF, 180, 255);
        
        saveImage(segLow, "06_segment_low_intensity");
        saveImage(segMid, "06_segment_mid_intensity");
        saveImage(segHigh, "06_segment_high_intensity");

        // 7. OPERACIONES DE PUNTOS (NOT, AND, OR, XOR)
        std::cout << "7. Operaciones de puntos...\n";
        cv::Mat notImg = applyNOT(threshOtsu);
        cv::Mat andImg = applyAND(segMid, segHigh);
        cv::Mat orImg = applyOR(segLow, segMid);
        cv::Mat xorImg = applyXOR(segMid, segHigh);
        
        saveImage(notImg, "07_NOT");
        saveImage(andImg, "07_AND_mid_high");
        saveImage(orImg, "07_OR_low_mid");
        saveImage(xorImg, "07_XOR_mid_high");

        // 8. DETECCIÓN DE BORDES
        std::cout << "8. Detectando bordes...\n";
        cv::Mat cannyEdge = edgeCanny(bilateralF);
        cv::Mat sobelEdge = edgeSobel(bilateralF);
        cv::Mat laplacianEdge = edgeLaplacian(bilateralF);
        
        saveImage(cannyEdge, "08_edge_canny");
        saveImage(sobelEdge, "08_edge_sobel");
        saveImage(laplacianEdge, "08_edge_laplacian");

        // 9. OPERACIONES MORFOLÓGICAS
        std::cout << "9. Operaciones morfológicas...\n";
        cv::Mat erosion = morphErosion(threshOtsu);
        cv::Mat dilation = morphDilation(threshOtsu);
        cv::Mat opening = morphOpening(threshOtsu);
        cv::Mat closing = morphClosing(threshOtsu);
        cv::Mat gradient = morphGradient(threshOtsu);
        cv::Mat tophat = morphTopHat(bilateralF);
        cv::Mat blackhat = morphBlackHat(bilateralF);
        
        saveImage(erosion, "09_morph_erosion");
        saveImage(dilation, "09_morph_dilation");
        saveImage(opening, "09_morph_opening");
        saveImage(closing, "09_morph_closing");
        saveImage(gradient, "09_morph_gradient");
        saveImage(tophat, "09_morph_tophat");
        saveImage(blackhat, "09_morph_blackhat");

        // 10. MÁSCARAS DE RESALTADO Y VISUALIZACIÓN FINAL
        std::cout << "10. Creando máscaras de resaltado...\n";
        
        // Máscara limpia (opening + closing)
        cv::Mat cleanMask = morphClosing(morphOpening(threshOtsu, 3), 5);
        saveImage(cleanMask, "10_mask_cleaned");
        
        // Overlay verde sobre área de interés
        cv::Mat overlayGreen = createColorOverlay(m_originalImage, cleanMask, 
                                                   cv::Scalar(0, 255, 0), 0.4);
        saveImage(overlayGreen, "10_overlay_green");
        
        // Overlay con diferentes colores por región
        cv::Mat multiOverlay = m_originalImage.clone();
        cv::cvtColor(multiOverlay, multiOverlay, cv::COLOR_GRAY2BGR);
        
        // Rojo para alta intensidad (posible hueso)
        cv::Mat redOverlay = multiOverlay.clone();
        redOverlay.setTo(cv::Scalar(0, 0, 255), segHigh);
        cv::addWeighted(multiOverlay, 0.7, redOverlay, 0.3, 0, multiOverlay);
        
        // Azul para baja intensidad
        cv::Mat blueOverlay = multiOverlay.clone();
        blueOverlay.setTo(cv::Scalar(255, 0, 0), segLow);
        cv::addWeighted(multiOverlay, 0.8, blueOverlay, 0.2, 0, multiOverlay);
        
        saveImage(multiOverlay, "10_overlay_multicolor");
        
        // Heatmap
        cv::Mat heatmap = createHeatmap(bilateralF);
        saveImage(heatmap, "10_heatmap");
        
        // Contornos sobre imagen original
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(cleanMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Mat contourImg;
        cv::cvtColor(m_originalImage, contourImg, cv::COLOR_GRAY2BGR);
        cv::drawContours(contourImg, contours, -1, cv::Scalar(0, 255, 255), 2);
        saveImage(contourImg, "10_contours_yellow");

        // 11. IMAGEN FINAL COMBINADA
        std::cout << "11. Generando imagen final...\n";
        
        // Combinar bordes con imagen mejorada
        cv::Mat finalCombined;
        cv::cvtColor(claheImg, finalCombined, cv::COLOR_GRAY2BGR);
        
        // Añadir bordes en color
        cv::Mat edgeColor;
        cv::cvtColor(cannyEdge, edgeColor, cv::COLOR_GRAY2BGR);
        edgeColor.setTo(cv::Scalar(0, 255, 0), cannyEdge);
        cv::addWeighted(finalCombined, 0.8, edgeColor, 0.5, 0, finalCombined);
        
        saveImage(finalCombined, "11_FINAL_enhanced_with_edges");

        std::cout << "\n=== Procesamiento completado ===\n";
        std::cout << "Total de imágenes guardadas: " << m_imageCounter << std::endl;
        std::cout << "Carpeta de salida: " << m_outputFolder << std::endl;
    }
};

// ===== MAIN =====
int main(int argc, char** argv) {
    std::string inputFile = "L19.IMA";
    std::string outputFolder = "./output_L19";
    
    // Permitir parámetros opcionales
    if (argc > 1) inputFile = argv[1];
    if (argc > 2) outputFolder = argv[2];
    
    std::cout << "========================================\n";
    std::cout << "   CT Image Processor - Single File    \n";
    std::cout << "========================================\n";
    std::cout << "Input file: " << inputFile << "\n";
    std::cout << "Output folder: " << outputFolder << "\n\n";
    
    CTImageProcessor processor(outputFolder);
    
    if (!processor.loadImage(inputFile)) {
        std::cerr << "Error: Could not load " << inputFile << std::endl;
        return 1;
    }
    
    processor.processComplete();
    
    return 0;
}