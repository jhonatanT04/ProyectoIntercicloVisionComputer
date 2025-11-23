#include "CTProcessorSimple.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace cv;

// ======================================================
// Constructor
// ======================================================
CTImageProcessor::CTImageProcessor(const std::string& outputFolder)
    : m_outputFolder(outputFolder), m_imageCounter(0) {

    if (!std::filesystem::exists(outputFolder)) {
        std::filesystem::create_directory(outputFolder);
    }
}

// ======================================================
// Lectura de imagen (DICOM o PNG/JPG)
// ======================================================
bool CTImageProcessor::loadImage(const std::string& filePath) {
    std::string ext = std::filesystem::path(filePath).extension().string();

    if (ext == ".IMA" || ext == ".dcm") {
        using ImageType = itk::Image<short, 2>;
        using ReaderType = itk::ImageFileReader<ImageType>;
        using RescaleType = itk::RescaleIntensityImageFilter<ImageType, ImageType>;

        try {
            ReaderType::Pointer reader = ReaderType::New();
            reader->SetFileName(filePath);
            reader->Update();

            RescaleType::Pointer rescaler = RescaleType::New();
            rescaler->SetInput(reader->GetOutput());
            rescaler->SetOutputMinimum(0);
            rescaler->SetOutputMaximum(255);
            rescaler->Update();

            m_rawImage = Mat(
                rescaler->GetOutput()->GetLargestPossibleRegion().GetSize()[1],
                rescaler->GetOutput()->GetLargestPossibleRegion().GetSize()[0],
                CV_16S,
                (void*)rescaler->GetOutput()->GetBufferPointer()
            ).clone();

            m_rawImage.convertTo(m_originalImage, CV_8U);
            return true;

        } catch (...) {
            std::cerr << "Error leyendo DICOM\n";
            return false;
        }

    } else {
        m_originalImage = imread(filePath, IMREAD_GRAYSCALE);
        return !m_originalImage.empty();
    }
}

// ======================================================
// Guardar imagen
// ======================================================
void CTImageProcessor::saveImage(const Mat& img, const std::string& stepName) {
    std::string filename = m_outputFolder + "/" +
                           std::to_string(++m_imageCounter) + "_" + stepName + ".png";
    imwrite(filename, img);
}

// ======================================================
// Funciones de procesamiento
// ======================================================
Mat CTImageProcessor::applyWindowLevel(int center, int width) {
    Mat out;
    double low = center - width / 2;
    double high = center + width / 2;
    cv::threshold(m_originalImage, out, high, 255, THRESH_TRUNC);
    cv::threshold(out, out, low, 0, THRESH_TOZERO);
    cv::normalize(out, out, 0, 255, NORM_MINMAX);
    return out;
}

Mat CTImageProcessor::normalize(const Mat& input) {
    Mat out;
    cv::normalize(input, out, 0, 255, NORM_MINMAX);
    return out;
}

Mat CTImageProcessor::threshold(const Mat& input, int threshValue) {
    Mat out;
    cv::threshold(input, out, threshValue, 255, THRESH_BINARY);
    return out;
}

Mat CTImageProcessor::thresholdOtsu(const Mat& input) {
    Mat out;
    cv::threshold(input, out, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return out;
}

Mat CTImageProcessor::thresholdAdaptive(const Mat& input, int blockSize) {
    Mat out;
    adaptiveThreshold(input, out, 255, ADAPTIVE_THRESH_MEAN_C,
                      THRESH_BINARY, blockSize, 2);
    return out;
}

Mat CTImageProcessor::contrastStretching(const Mat& input) {
    Mat out;
    cv::normalize(input, out, 0, 255, NORM_MINMAX);
    return out;
}

// --------------------- Filtros ------------------------
Mat CTImageProcessor::applyCLAHE(const Mat& input, double clipLimit) {
    Ptr<CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clipLimit);
    Mat out;
    clahe->apply(input, out);
    return out;
}

Mat CTImageProcessor::histogramEqualization(const Mat& input) {
    Mat out;
    equalizeHist(input, out);
    return out;
}

// ---------------- Operadores lógicos -------------------
Mat CTImageProcessor::applyNOT(const Mat& input) {
    Mat out;
    bitwise_not(input, out);
    return out;
}

Mat CTImageProcessor::applyAND(const Mat& i1, const Mat& i2) {
    Mat out;
    bitwise_and(i1, i2, out);
    return out;
}

Mat CTImageProcessor::applyOR(const Mat& i1, const Mat& i2) {
    Mat out;
    bitwise_or(i1, i2, out);
    return out;
}

Mat CTImageProcessor::applyXOR(const Mat& i1, const Mat& i2) {
    Mat out;
    bitwise_xor(i1, i2, out);
    return out;
}

// ---------------- Detección de bordes ------------------
Mat CTImageProcessor::edgeCanny(const Mat& input, int low, int high) {
    Mat out;
    Canny(input, out, low, high);
    return out;
}

Mat CTImageProcessor::edgeSobel(const Mat& input) {
    Mat gradX, gradY, out;
    Sobel(input, gradX, CV_16S, 1, 0);
    Sobel(input, gradY, CV_16S, 0, 1);
    convertScaleAbs(gradX, gradX);
    convertScaleAbs(gradY, gradY);
    addWeighted(gradX, 0.5, gradY, 0.5, 0, out);
    return out;
}

Mat CTImageProcessor::edgeLaplacian(const Mat& input) {
    Mat out16, out8;
    Laplacian(input, out16, CV_16S);
    convertScaleAbs(out16, out8);
    return out8;
}

// ------------------- Filtros medios --------------------
Mat CTImageProcessor::filterGaussian(const Mat& input, int k) {
    Mat out;
    GaussianBlur(input, out, Size(k, k), 0);
    return out;
}

Mat CTImageProcessor::filterMedian(const Mat& input, int k) {
    Mat out;
    medianBlur(input, out, k);
    return out;
}

Mat CTImageProcessor::filterBilateral(const Mat& input, int d) {
    Mat out;
    bilateralFilter(input, out, d, d * 2, d / 2);
    return out;
}

Mat CTImageProcessor::filterMean(const Mat& input, int k) {
    Mat out;
    blur(input, out, Size(k, k));
    return out;
}

Mat CTImageProcessor::filterNLMeans(const Mat& input) {
    Mat out;
    fastNlMeansDenoising(input, out);
    return out;
}

// ------------------- Morfología ------------------------
Mat CTImageProcessor::morphErosion(const Mat& input, int k) {
    Mat out;
    erode(input, out, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphDilation(const Mat& input, int k) {
    Mat out;
    dilate(input, out, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphOpening(const Mat& input, int k) {
    Mat out;
    morphologyEx(input, out, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphClosing(const Mat& input, int k) {
    Mat out;
    morphologyEx(input, out, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphGradient(const Mat& input, int k) {
    Mat out;
    morphologyEx(input, out, MORPH_GRADIENT, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphTopHat(const Mat& input, int k) {
    Mat out;
    morphologyEx(input, out, MORPH_TOPHAT, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

Mat CTImageProcessor::morphBlackHat(const Mat& input, int k) {
    Mat out;
    morphologyEx(input, out, MORPH_BLACKHAT, getStructuringElement(MORPH_RECT, Size(k, k)));
    return out;
}

// ------------------- Segmentación ----------------------
Mat CTImageProcessor::segmentByIntensity(const Mat& input, int low, int high) {
    Mat mask;
    inRange(input, Scalar(low), Scalar(high), mask);
    return mask;
}

// ------------------- Overlay ---------------------------
Mat CTImageProcessor::createColorOverlay(const Mat& original, const Mat& mask,
                                         Scalar color, double alpha) {
    Mat out, col;
    cvtColor(original, out, COLOR_GRAY2BGR);
    out.copyTo(col);
    col.setTo(color, mask);
    addWeighted(out, 1 - alpha, col, alpha, 0, out);
    return out;
}

Mat CTImageProcessor::createHeatmap(const Mat& input) {
    Mat heat;
    applyColorMap(input, heat, COLORMAP_JET);
    return heat;
}

// ======================================================
// PROCESS COMPLETE (pipeline resumido para Qt)
// ======================================================
void CTImageProcessor::processComplete() {
    Mat clahe = applyCLAHE(m_originalImage, 3.0);
    saveImage(clahe, "clahe");

    Mat otsu = thresholdOtsu(clahe);
    saveImage(otsu, "otsu");

    Mat edges = edgeCanny(clahe, 50, 120);
    saveImage(edges, "canny");

    Mat seg = segmentByIntensity(clahe, 100, 200);
    saveImage(seg, "segment");

    Mat overlay = createColorOverlay(clahe, seg, Scalar(0, 255, 0), 0.5);
    saveImage(overlay, "overlay");
}

// ==================== STUBS para Qt ===================
Mat CTImageProcessor::extractBones(const Mat& input) {
    // Segmentación por intensidad simple
    return segmentByIntensity(input, 100, 200);
}

Mat CTImageProcessor::highlightRegion(const Mat& input) {
    // Resaltado verde
    cv::Mat dummy = cv::Mat::zeros(input.size(), CV_8UC1);
    return createColorOverlay(dummy, input, Scalar(0,255,0), 0.5);
}

Mat CTImageProcessor::applyDenoisingDNN(const Mat& input) {
    // Solo aplicamos denoise rápido
    cv::Mat out;
    fastNlMeansDenoising(input, out);
    return out;
}
