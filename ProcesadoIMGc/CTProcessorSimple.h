#ifndef CTPROCESSORSIMPLE_H
#define CTPROCESSORSIMPLE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>          // Para DNN denoising
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>         // Para DICOM
#include <itkImageRegionConstIterator.h>
#include <itkRescaleIntensityImageFilter.h>
#include <filesystem>
#include <string>
#include <vector>

class CTImageProcessor
{
public:
    CTImageProcessor(const std::string& outputFolder);

    // ================ LECTURA/ESCRITURA ================
    bool loadImage(const std::string& filePath);
    void saveImage(const cv::Mat& img, const std::string& stepName);

    // ================ GETTERS ==========================
    cv::Mat getOriginalImage() const { return m_originalImage.clone(); }
    cv::Mat getRawImage() const { return m_rawImage.clone(); }
    int getImageCounter() const { return m_imageCounter; }

    // ================ WINDOW/LEVEL (CT) ================
    cv::Mat applyWindowLevel(int center, int width);

    // ================ NORMALIZACIÓN Y CONTRASTE ========
    cv::Mat normalize(const cv::Mat& input);
    cv::Mat contrastStretching(const cv::Mat& input);
    cv::Mat applyCLAHE(const cv::Mat& input, double clipLimit = 2.0);
    cv::Mat histogramEqualization(const cv::Mat& input);

    // ================ THRESHOLDING =====================
    cv::Mat threshold(const cv::Mat& input, int threshValue);
    cv::Mat thresholdOtsu(const cv::Mat& input);
    cv::Mat thresholdAdaptive(const cv::Mat& input, int blockSize = 11);

    // ================ OPERADORES LÓGICOS ===============
    cv::Mat applyNOT(const cv::Mat& input);
    cv::Mat applyAND(const cv::Mat& input1, const cv::Mat& input2);
    cv::Mat applyOR(const cv::Mat& input1, const cv::Mat& input2);
    cv::Mat applyXOR(const cv::Mat& input1, const cv::Mat& input2);

    // ================ DETECCIÓN DE BORDES ==============
    cv::Mat edgeCanny(const cv::Mat& input, int low = 50, int high = 150);
    cv::Mat edgeSobel(const cv::Mat& input);
    cv::Mat edgeLaplacian(const cv::Mat& input);

    // ================ FILTROS DE SUAVIZADO =============
    cv::Mat filterGaussian(const cv::Mat& input, int ksize = 5);
    cv::Mat filterMedian(const cv::Mat& input, int ksize = 5);
    cv::Mat filterBilateral(const cv::Mat& input, int d = 9);
    cv::Mat filterMean(const cv::Mat& input, int ksize = 5);
    cv::Mat filterNLMeans(const cv::Mat& input);

    // ================ MORFOLOGÍA =======================
    cv::Mat morphErosion(const cv::Mat& input, int ksize = 5);
    cv::Mat morphDilation(const cv::Mat& input, int ksize = 5);
    cv::Mat morphOpening(const cv::Mat& input, int ksize = 5);
    cv::Mat morphClosing(const cv::Mat& input, int ksize = 5);
    cv::Mat morphGradient(const cv::Mat& input, int ksize = 5);
    cv::Mat morphTopHat(const cv::Mat& input, int ksize = 15);
    cv::Mat morphBlackHat(const cv::Mat& input, int ksize = 15);

    // ================ SEGMENTACIÓN =====================
    cv::Mat segmentByIntensity(const cv::Mat& input, int lower, int upper);

    // ================ VISUALIZACIÓN ====================
    cv::Mat createColorOverlay(const cv::Mat& original, const cv::Mat& mask,
                               cv::Scalar color, double alpha = 0.5);
    cv::Mat createHeatmap(const cv::Mat& input);
    
    // ⭐ NUEVO: Resaltar región con contornos, áreas y rectángulos
    cv::Mat highlightRegion(const cv::Mat& mask, const cv::Mat& background = cv::Mat());
    
    // ⭐ NUEVO: Extraer huesos usando umbral de Hounsfield
    cv::Mat extractBones(const cv::Mat& input, int minHU = 200, int maxHU = 3000);
    
    // ⭐ NUEVO: Overlay multicolor por rangos de intensidad
    cv::Mat createMultiColorOverlay(const cv::Mat& original);
    
    // ⭐ NUEVO: Combinar imagen con bordes
    cv::Mat combineWithEdges(const cv::Mat& input, const cv::Mat& edges, 
                             cv::Scalar edgeColor = cv::Scalar(0, 255, 0));

    // ================ DEEP LEARNING ====================
    // ⭐ MEJORADO: Denoising con DNN o fallback a filtros avanzados
    cv::Mat applyDenoisingDNN(const cv::Mat& input);

    // ================ PIPELINE COMPLETO ================
    void processComplete();

private:
    std::string m_outputFolder;
    int m_imageCounter;

    cv::Mat m_rawImage;         // Imagen raw (16 bits para CT)
    cv::Mat m_originalImage;    // Imagen procesada (8 bits para display)
    
    // ⭐ NUEVO: Funciones auxiliares privadas
    cv::Mat ensureGrayscale(const cv::Mat& input);
    cv::Mat ensureUInt8(const cv::Mat& input);
    void validateKernelSize(int& ksize);
};

#endif // CTPROCESSORSIMPLE_