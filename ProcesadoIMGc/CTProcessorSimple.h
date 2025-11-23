#ifndef CTPROCESSORSIMPLE_H
#define CTPROCESSORSIMPLE_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkRescaleIntensityImageFilter.h>
#include <filesystem>
#include <string>

class CTImageProcessor
{
public:
    CTImageProcessor(const std::string& outputFolder);

    // ---------------- Lectura/Escritura ----------------
    bool loadImage(const std::string& filePath);
    void saveImage(const cv::Mat& img, const std::string& stepName);

    // ----------------- Getters ------------------------
    cv::Mat getOriginalImage() const { return m_originalImage; }

    // ----------------- Procesamiento ------------------
    cv::Mat applyWindowLevel(int center, int width);
    cv::Mat normalize(const cv::Mat& input);
    cv::Mat threshold(const cv::Mat& input, int threshValue);
    cv::Mat thresholdOtsu(const cv::Mat& input);
    cv::Mat thresholdAdaptive(const cv::Mat& input, int blockSize);
    cv::Mat contrastStretching(const cv::Mat& input);

    // ---------------- Filtros ------------------------
    cv::Mat applyCLAHE(const cv::Mat& input, double clipLimit);
    cv::Mat histogramEqualization(const cv::Mat& input);

    // -------------- Operadores lógicos ---------------
    cv::Mat applyNOT(const cv::Mat& input);
    cv::Mat applyAND(const cv::Mat& i1, const cv::Mat& i2);
    cv::Mat applyOR(const cv::Mat& i1, const cv::Mat& i2);
    cv::Mat applyXOR(const cv::Mat& i1, const cv::Mat& i2);

    // -------------- Detección de bordes -------------
    cv::Mat edgeCanny(const cv::Mat& input, int low, int high);
    cv::Mat edgeSobel(const cv::Mat& input);
    cv::Mat edgeLaplacian(const cv::Mat& input);

    // ---------------- Filtros medios -----------------
    cv::Mat filterGaussian(const cv::Mat& input, int k);
    cv::Mat filterMedian(const cv::Mat& input, int k);
    cv::Mat filterBilateral(const cv::Mat& input, int d);
    cv::Mat filterMean(const cv::Mat& input, int k);
    cv::Mat filterNLMeans(const cv::Mat& input);

    // ---------------- Morfología ---------------------
    cv::Mat morphErosion(const cv::Mat& input, int k);
    cv::Mat morphDilation(const cv::Mat& input, int k);
    cv::Mat morphOpening(const cv::Mat& input, int k);
    cv::Mat morphClosing(const cv::Mat& input, int k);
    cv::Mat morphGradient(const cv::Mat& input, int k);
    cv::Mat morphTopHat(const cv::Mat& input, int k);
    cv::Mat morphBlackHat(const cv::Mat& input, int k);

    // ---------------- Segmentación -------------------
    cv::Mat segmentByIntensity(const cv::Mat& input, int low, int high);

    // ---------------- Overlay ------------------------
    cv::Mat createColorOverlay(const cv::Mat& original, const cv::Mat& mask,
                               cv::Scalar color, double alpha);
    cv::Mat createHeatmap(const cv::Mat& input);

    // ---------------- Pipeline completo -------------
    void processComplete();

    // ----------------- Stubs / Qt -------------------
    cv::Mat extractBones(const cv::Mat& input);
    cv::Mat highlightRegion(const cv::Mat& input);
    cv::Mat applyDenoisingDNN(const cv::Mat& input);

private:
    std::string m_outputFolder;
    int m_imageCounter;

    cv::Mat m_rawImage;
    cv::Mat m_originalImage;
};

#endif // CTPROCESSORSIMPLE_H
